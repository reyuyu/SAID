# Phase 0 — Reproducing the official SmartCLIP baseline

This document records how the official SmartCLIP code was made reproducible on the
SAID server, what was verified, and what is still missing. No new algorithm
(SALU / SAID / Said Router / Prototype Bank / Unsaid Explorer) is implemented in
this phase — the code is the official SmartCLIP baseline plus the small,
backwards-compatible changes listed at the bottom.

## Status

* **Phase 0 infrastructure/smoke reproduction: completed** - environment, data pipeline,
  OpenAI CLIP ViT-B/16 initialisation, 4-GPU training smoke, checkpoint save/reload and
  the COCO retrieval evaluation were all verified on the server.
* **Full 1.246M SmartCLIP training reproduction: blocked by missing SAM subset** - the
  official ShareGPT4V-PT mixture needs 570,486 SAM images that are currently not
  obtainable, so no full-dataset training run has been performed in this phase. The
  verified smoke runs use an official-format subset JSON (676,415 COCO + LLaVA records),
  which is **not** the full official training set.

## 1. Baseline provenance

| item | value |
| --- | --- |
| origin | `git@github.com:reyuyu/SAID.git` |
| upstream | `https://github.com/Mid-Push/SmartCLIP.git` |
| baseline commit | `f4e69832b94d084d950271cf0cf9582cbfc2c5a4` (`main` == `upstream/main`) |
| baseline tag | `smartclip-official-baseline-f4e69832` |
| development branch | `codex/phase0-smartclip-reproduction` |

## 2. Verified environment

* Ubuntu 22.04.4 LTS (container devbox), kernel 4.18
* 4 × NVIDIA A800-SXM4-80GB (79.33 GiB usable each), driver 535.129.03, CUDA 12.4
* conda env `said-smartclip`: Python 3.10, PyTorch 2.5.1+cu124, torchvision 0.20.1+cu124,
  openai `clip` @ `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` (full list:
  `docs/environment_freeze.txt`)
* `setuptools<81` is required: `model/longclip.py` imports `pkg_resources`, which was
  removed from setuptools 83.
* Create the environment with:

```bash
conda create -n said-smartclip python=3.10 -y
conda run -n said-smartclip pip install -r requirements.txt
```

## 3. Data layout

The official code resolves its data relative to the working directory (`train/`),
i.e. `<repo>/datasets/...`. Three environment variables were added so the data can
live anywhere; **the official defaults are unchanged when the variables are unset**:

| variable | official default | used on this server |
| --- | --- | --- |
| `SHARE4V_DATA_ROOT` | `../datasets/ShareGPT4V/` | `/root/datasets/ShareGPT4V/` |
| `SHARE4V_JSON` | `share-captioner_coco_lcs_sam_1246k_1107.json` | `debug/share4v_smoke_nosam.json` (smoke only) |
| `COCO_DATA_ROOT` | `../datasets/coco/` (`train_utils.eval_coco`), `../../datasets/coco/` (`eval/retrieval/coco.py`) | `/root/datasets/coco/` |

Expected contents (all paths relative to `SHARE4V_DATA_ROOT`):

```text
ShareGPT4V/
├── share-captioner_coco_lcs_sam_1246k_1107.json   # 1.39 GB, 1,246,901 records
├── coco/train2017/                                # 118,287 images
├── llava/llava_pretrain/images/<5-digit>/<9-digit>.jpg   # 558,128 images
└── sam/images/sa_XXXXXX.jpg                       # 570,486 images (missing, see §6)
coco/
├── val2017/                                       # 5,000 images
└── annotations/captions_val2017.json
```

### Downloads actually used

| dataset | size | source |
| --- | --- | --- |
| caption JSON | 1.39 GB | `https://hf-mirror.com/datasets/Lin-Chen/ShareGPT4V/...` |
| LLaVA-Pretrain `images.zip` | 27.3 GB | `https://hf-mirror.com/datasets/liuhaotian/LLaVA-Pretrain/...` |
| COCO train2017 | 19.3 GB | `http://images.cocodataset.org/zips/train2017.zip` |
| COCO val2017 | 0.78 GB | `http://images.cocodataset.org/zips/val2017.zip` |
| COCO annotations | 0.25 GB | `http://images.cocodataset.org/annotations/annotations_trainval2017.zip` |

Note: `huggingface.co` is **not reachable** from this server; `hf-mirror.com` is.
`unzip` is not installed in the container — use `python3 -m zipfile -e <zip> <dir>`.

## 4. Multi-GPU note (required in this container)

NCCL fails with the default network interface
(`socketStartConnect ... Software caused connection abort`), because it picks an
interface that cannot connect inside the container. Export before launching:

```bash
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
```

Verified with a 4-rank `all_reduce` test.

## 5. Commands

Official 4-GPU training (unchanged algorithm):

```bash
cd train
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
SHARE4V_DATA_ROOT=/root/datasets/ShareGPT4V/ COCO_DATA_ROOT=/root/datasets/coco/ \
torchrun --nproc_per_node=4 --master_port=25960 train.py --base_model=B16 --lambda_sparse=2
```

Smoke test (same command + early stop; `--max_steps` defaults to `None`, i.e. the
official full-training behaviour):

```bash
... torchrun ... train.py --base_model=B16 --lambda_sparse=2 --max_steps=250
```

Retrieval evaluation and checkpoint reload:

```bash
cd eval/retrieval
COCO_DATA_ROOT=/root/datasets/coco/ python coco.py --checkpoint=<ckpt.pt>
```

```python
from model import longclip
model, preprocess = longclip.load('<ckpt.pt>', device='cuda')
```

## 6. Verified smoke results (2026-09-09)

Run: 4 x A800-80GB, `--max_steps=120` (plus a 250-step run), on an official-format
**subset** JSON containing COCO + LLaVA records only (676,415 records) because the SAM
images are missing; batch 256/GPU, accumulation 1 -> effective batch 1024.

This is the **Phase 0 infrastructure/smoke reproduction: completed** result. It is NOT
the full 1.246 M official training run, which is **blocked by the missing SAM subset**
(see Status above and section 7).

| check | result |
| --- | --- |
| 4 ranks started, NCCL ok | yes (`NCCL_SOCKET_IFNAME=lo`) |
| loss finite | `sidm` 3.76 -> 0.45, `dism` 1.40 -> 0.45, `sparsity` 0.46 -> 0.90 over 120 steps |
| optimizer | AdamW (backbone + mask net) stepped every iteration |
| throughput | 0.70 s/step -> ~1,463 samples/s (effective batch 1024); 100 steps ~ 70 s |
| peak memory / GPU | 51.4 GiB during training; 65.2 GiB on rank 0 during the batch-1000 val test |
| GPU utilization | ~81 % avg (GPU 1-3) during training; ~64 % on GPU 0 (rank 0 also logs/evals) |
| checkpoint save | `smartclip_epoch00.pt` (612 MB), exactly once per run |
| checkpoint reload | `longclip.load(ckpt)` OK, 317 keys, `mask_net` present, no NaN/Inf |
| ShareGPT4V val retrieval | 0.84 (120 steps) / 0.88 (250 steps) |
| COCO retrieval pipeline | i2t R@1 0.562 / R@5 0.800 / R@10 0.874; t2i R@1 0.373 / R@5 0.627 / R@10 0.727 |
| DataLoader | 2 batches verified, 0 missing files in 300 random samples |

These numbers come from a smoke checkpoint on a data subset; they are not paper results.

## 7. Known gaps / next steps

* **SAM images are missing (570,486 files, `sa_000001` … `sa_570590`).**
  The JSON references them as `sam/images/sa_XXXXXX.jpg`. The official source
  (`https://dl.fbaipublicfiles.com/segment_anything/...`) now returns `403`, and the
  public mirrors checked (`zhangtao-whu/used_sam_images`, `kkkkkcm/SA-1B-400k`,
  `hdtech/SA-1B`, `DavidNguyen/ShareGPT4V-Sam`) do not contain this contiguous id
  slice in matching file names. Therefore the smoke run used an **official-format
  subset JSON** (COCO + LLaVA records only, 676,415 records) at
  `datasets/ShareGPT4V/debug/share4v_smoke_nosam.json`. Therefore
  **Full 1.246M SmartCLIP training reproduction: blocked by missing SAM subset**.
* Upstream behaviour kept as-is: every DDP rank creates its own `runs/<id>_...`
  directory and TensorBoard writer, and `get_run_id()` races across ranks, so a
  4-GPU run produces four run directories (rank 0 writes `loss.txt` / `metric.txt`).
* `train_utils.eval_coco` iterates COCO val one image at a time, so it takes minutes
  for 5,000 images. Left unchanged.

## 8. Source changes on this branch

| file | change | reason |
| --- | --- | --- |
| `.gitignore` | ignore datasets/checkpoints/outputs/runs/wandb/weights/caches/.env | never upload data, weights or credentials |
| `train/train.py` | removed `from eval.classification.cifar.smartcifar10 import ...` | module does not exist upstream → `import train` failed |
| `train/train.py` | added `--max_steps` (default `None`) | early-stop for smoke tests; `None` = official behaviour |
| `train/sharegpt4v.py` | `SHARE4V_DATA_ROOT` / `SHARE4V_JSON` env overrides | data lives outside the repo; defaults unchanged |
| `train/train_utils.py` | `COCO_DATA_ROOT` env override in `eval_coco` | same |
| `eval/retrieval/coco.py` | `COCO_DATA_ROOT` env override | same |
| `train/sharegpt4v.py` | `os.path.join` instead of string concatenation | `SHARE4V_DATA_ROOT` works with or without a trailing `/` |
| `requirements.txt` | pinned `torchvision==0.20.1` and CLIP commit `d05afc4...` | exact reproducible environment |
| `docs/environment_freeze.txt` | `pip freeze` of the working env | full environment record |

Caption sampling, loss, optimizer, model and DDP logic are **untouched**.
