# SmartCLIP 3-Epoch Baseline Reproduction (Phase 3.0A baseline validation)

Independent reproduction of the **original** SmartCLIP training path
(`train/train.py`, class `CLIP_Clean_Train`) on full ShareGPT4V, evaluated with the **same
canonical evaluator** used for the Said-only and Full-Base arms.

No SALU method change, no Gap core change, no USS, no `train_salu.py`
for SmartCLIP. Training loss is untouched:
`lambda_sparse * loss_sparsity + lambda_align * (loss_sidm + loss_dism)`.

Artifacts (small, tracked) in this directory:

| file | content |
|---|---|
| `smartclip_3epoch_canonical.json` | canonical retrieval, all 4 checkpoints + unified table + deltas |
| `smartclip_reproduction_report.md` | this report |

Checkpoints, logs and score matrices are not committed.

---

## 1. Training configuration (official-faithful)

| item | value |
|---|---|
| trainer | `train/train.py` → `CLIP_Clean_Train` |
| base_model | **ViT-B/16** |
| batch | 256 / GPU × 4 GPU → `accumulation_steps = 1024//256//4 = 1` → **effective global batch 1024** |
| epochs | **3** |
| lr / mask_lr / weight_decay | 1e-6 / 1e-3 / 1e-2 |
| lambda_sparse / lambda_align | 2 / 10 |
| warmup_length | 200 |
| soft_mask | **0** (the repo default: hard / straight-through mask) |
| seed | 0 (opt-in flag added; the original had none) |
| run dir | `runs_smartclip/smartclip_B16_full3ep_repro_20260911/` |
| wall time | 13:55 → 14:53 ≈ **58 min**, `SMARTCLIP_EXIT=0` |

## 2. Dataset / Full Data Gate

| item | measured |
|---|---|
| dataset JSON | `share-captioner_coco_lcs_sam_1246k_1107.json` |
| JSON sha256 | printed by the launcher at start-up (recorded in the run log) |
| JSON size | 1,492,479,912 bytes |
| **dataset_size** | **1,245,901** |
| **steps_per_epoch** | **1217** |
| effective global batch | 1024 |

The Full Data Gate passes: `dataset_size = 1,245,901`, i.e. the full ShareGPT4V set, no debug
subset. Note `steps_per_epoch = 1217` here versus `1216` in the Said-only arms: the original
`train.py` uses a **plain DataLoader without `drop_last`**, so its last batch is partial. This
is a property of the original implementation and was deliberately left unchanged; the 3-epoch
LR horizon is therefore `3 × 1217 = 3651` steps (the Said-only arms used 3648).

## 3. Training health

Loss terms at the same iteration within each epoch (`sidm`, `dism`, `sparsity`):

| epoch | iter | sidm | dism | sparsity | num_mean (min/max) |
|---|---|---|---|---|---|
| 0 | 999 | 0.2025 | 0.2694 | 0.7429 | 380 (276/450) |
| 1 | 999 | 0.1059 | 0.0988 | 0.7335 | 375 (290/461) |
| 2 | 1216 | **0.0409** | **0.0576** | 0.6769 | 346 (237/420) |

Both alignment terms fall monotonically; no NaN/Inf, no OOM, no crash. The trainer's own
legacy `eval_coco` (reference only, **not** used for the comparison) reports
I2T R@1 0.6158 → 0.6188 → 0.6204 and T2I R@1 0.4213 → 0.4253 → 0.4263 over epochs 0→2.

## 4. SmartCLIP canonical retrieval (same evaluator as A / C)

`initial` here is a fresh pretrained ViT-B/16 with the same configuration as the A/C
`initial`; `epoch1/2/3` are `smartclip_epoch00/01/02.pt`.

| checkpoint | COCO I2T / T2I R@1 | 1K first_sentence | 1K fixed_sparse | 1K full_dense |
|---|---|---|---|---|
| initial | 0.5170 / 0.3269 | 0.5440 / 0.5140 | 0.7460 / 0.7400 | 0.7580 / 0.7760 |
| epoch1 | 0.6156 / 0.4214 | 0.7070 / 0.6760 | 0.9170 / 0.8950 | 0.9780 / 0.9710 |
| epoch2 | 0.6190 / 0.4253 | 0.7110 / 0.6920 | 0.9260 / 0.9060 | 0.9830 / 0.9790 |
| **epoch3** | **0.6198 / 0.4263** | **0.7100 / 0.6920** | **0.9280 / 0.9080** | **0.9830 / 0.9780** |

Full R@1/R@5/R@10 for every variant and both directions are in
`smartclip_3epoch_canonical.json`.

**Independent cross-check.** The canonical evaluator gives COCO I2T R@1 = **0.6198** at
epoch 3; the trainer's own legacy evaluator gave **0.6204**. Two different protocols agreeing
to 0.0006 is strong evidence that the checkpoint loaded correctly and the evaluation is sound.

## 5. Unified comparison (same canonical evaluator for all rows)

| model | COCO I2T R@1 | COCO T2I R@1 | 1K first I2T/T2I | 1K sparse I2T/T2I | 1K full I2T/T2I |
|---|---|---|---|---|---|
| Initial CLIP | 0.5170 | 0.3269 | 0.5440 / 0.5140 | 0.7460 / 0.7400 | 0.7580 / 0.7760 |
| **Original SmartCLIP 3ep** | **0.6198** | **0.4263** | **0.7100 / 0.6920** | **0.9280 / 0.9080** | **0.9830 / 0.9780** |
| A Said-only 3ep | 0.5224 | 0.3423 | 0.6180 / 0.6190 | 0.8280 / 0.8460 | 0.9220 / 0.9280 |
| C Full Base 3ep | 0.0144 | 0.2102 | 0.0900 / 0.4740 | 0.0510 / 0.7040 | 0.3740 / 0.7710 |

Deltas in **percentage points** (I2T / T2I R@1):

| comparison | COCO | 1K first_sentence | 1K fixed_sparse | 1K full_dense |
|---|---|---|---|---|
| SmartCLIP − Initial | +10.28 / +9.94 | +16.60 / +17.80 | +18.20 / +16.80 | +22.50 / +20.20 |
| A Said-only − Initial | +0.54 / +1.54 | +7.40 / +10.50 | +8.20 / +10.60 | +16.40 / +15.20 |
| **A Said-only − SmartCLIP** | **−9.74 / −8.40** | **−9.20 / −7.30** | **−10.00 / −6.20** | **−6.10 / −5.00** |
| C Full Base − Initial | −50.26 / −11.68 | −45.40 / −4.00 | −69.50 / −3.60 | −38.40 / −0.50 |
| C Full Base − A | −50.80 / −13.21 | −52.80 / −14.50 | −77.70 / −14.20 | −54.80 / −15.70 |

## 6. Image embedding geometry (64-image fixed probe)

Same 64 cohort images, each with its own `C_S`:

| model | CLS mean norm | min | max | inter-image pairwise cosine | std | 64-way I2T@1 |
|---|---|---|---|---|---|---|
| Initial CLIP | 10.3076 | 9.0970 | 11.5959 | 0.52042 | 0.08269 | 0.9375 |
| SmartCLIP initial | 9.3324 | 8.7050 | 10.3173 | 0.42204 | 0.09393 | **1.0000** |
| SmartCLIP epoch3 | 9.1889 | 8.5772 | 10.0703 | **0.40769** | 0.09567 | **1.0000** |
| A Said-only 3ep | 8.2811 | 7.6420 | 9.3546 | 0.50002 | 0.09927 | 0.9531 |
| **C Full Base 3ep** | **15.9233** | 15.4940 | 16.3219 | **0.97690** | 0.00750 | **0.1875** |

**SmartCLIP shows no collapse.** Over 3 epochs its inter-image cosine *decreases* (0.422 →
0.408), its CLS norm stays stable (9.33 → 9.19) and its 64-way I2T@1 stays at 1.0000. The
Full-Base arm is the opposite (cosine 0.977, norm 15.9, I2T@1 0.1875).

## 7. Answers to the four required questions

**A. SmartCLIP reproduction healthy? — PASS.**
`SMARTCLIP_EXIT=0`, three complete epochs (3651 steps) on the full 1,245,901-sample dataset,
all loss terms falling monotonically, all three epoch checkpoints saved with 317 tensors each,
and two independent evaluators agreeing on COCO I2T R@1 to within 0.0006. The canonical
metrics improve monotonically and then plateau (0.6156 → 0.6190 → 0.6198), which is the
expected shape.

**B. SmartCLIP improves over initial? — YES.** +10.28 pp COCO I2T R@1, +9.94 pp COCO T2I R@1,
and +16.6 … +22.5 pp on the ShareGPT4V-1K variants. It is the strongest of the four models on
every single metric measured.

**C. Said-only improves over SmartCLIP? — NO (consistently worse).**
`A Said-only − SmartCLIP` is **negative on all four scopes and both directions**
(−9.74/−8.40, −9.20/−7.30, −10.00/−6.20, −6.10/−5.00 pp). The Said-only arm does improve over
Initial on every scope, but it does not reach the original SmartCLIP baseline.

**D. Any representation collapse? — NO for SmartCLIP; YES for the Full-Base arm.**
SmartCLIP's inter-image cosine is 0.408 with 64-way I2T@1 = 1.0000. The Full-Base (C) arm
collapses: inter-image cosine 0.9769, CLS norm 15.9, 64-way I2T@1 = 0.1875, and COCO
I2T R@1 0.0144 ≈ chance.

---

## 8. Two defects found while doing this reproduction (recorded for honesty)

**Defect 1 — my evaluator silently loaded nothing (evaluation side, fixed).**
The SmartCLIP `train.py` saves the state dict of a **bare `CLIP`** module
(`visual.*`, `transformer.*`, `ln_final`, …), while the canonical evaluator builds a
`SALUModel` whose keys are `clip.*`. With `load_state_dict(..., strict=False)` this matched
**zero** tensors (missing 321 / unexpected 317) and the first evaluation therefore ran on a
**randomly initialised** model: all three SmartCLIP checkpoints produced numbers identical to
`initial`, which is exactly how the bug surfaced. Fixed by an explicit bidirectional
prefix-mapping step plus a hard failure on any unexpected key, with `said_router.*` (absent by
construction in a bare-CLIP checkpoint, and unused by CLS retrieval) as the only permitted
omission. Every checkpoint now reports
`loaded 317 tensors, skipped 0, router-absent 4`.

**Defect 2 — the original trainer starts from a non-reproducible initial state (in the
upstream code, documented not changed).**
`train.py` calls `setup_distributed()` *before* any seeding, and the model is then built by
`longclip.load_from_clip`, which uses the unseeded numpy RNG for parts of the initialisation.
Measured consequence: of the 317 tensors shared between the SmartCLIP initial checkpoint and
the `Initial CLIP` state used by the other arms, **316 differ** (only 1 is bit-identical), and
the encoded CLS differs as well (probe mean norm 9.33 vs 10.31). The reproduction is
internally consistent and its own `initial → epoch3` trajectory is valid, but its *starting
point* is not the same pretrained state as the `Initial CLIP` row, so the
`SmartCLIP − Initial` numbers in Section 5 are **upper-bounded** by that fact and should be
read as "SmartCLIP's trained result versus the shared reference initial", not as a clean
controlled delta.

A minimal, opt-in `--init_state` flag was added to `train.py` to remove this confound (load a
frozen initial state dict before training, hard-failing on any mismatch). It was verified on a
fresh launch: `INIT_STATE_LOADED ... tensors=317 mask_net_absent=0`. **Re-running the
3-epoch training with it was not performed** (stopped on request), so the numbers in this
report come from the reproduction run that completed.

### Non-invasiveness of the `train.py` change

`git diff` is **+123 lines, −0 lines** — every change is an addition:
opt-in `--output_dir`, `--seed`, `--init_state`, two helpers (`seed_everything`,
`state_digest`) and read-only records (`reproducibility.json`, `data_scale.json`).
No line touching the loss, the mask behaviour, the optimizer grouping, the scheduler, the
model forward or the dataset caption sampling was modified; with none of the new flags the
trainer behaves exactly as before.
