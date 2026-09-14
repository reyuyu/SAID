# S0 Dual-Mask Suffix — Clean v0.1

完整实验已完成3651步：[中文实验入口与最终结果](../../experiments/s0_dualmask_masked_3epoch/README.md)。下文保留原实现与验收说明；在 main 复现历史 Clean 轨迹时请使用入口文档中的固定 Clean SHA。

This clean implementation starts from base commit `5676666` on branch
`codex/s0-dualmask-clean-v01`. It keeps the base S0 mask and SmartCLIP terms and adds one suffix
task. The original caption is split as `P = '. '.join(parts[:K])` and
`R = '. '.join(parts[K:last_nonempty])`; the final non-empty sentence is reserved, while the
dataset's original `K` and prefix stream are preserved exactly.

Erratum for commit `3137140`: that implementation used the image row's prefix mask for every
candidate column. This fix uses the S0 prefix mask for candidate text column `j`, gathered in the
same rank order as the suffix text bank. The original eight tests did not cover that protocol
difference; candidate-column and same-image counterexamples now do.

`native` scores normalized image features against separately encoded suffix features. `masked`
uses `mS` from the unchanged S0 helper and a new `Linear(1024,512) -> GELU -> Linear(512,512)`
gate. Its first layer is Xavier initialized with seed 0, the final weight is zero and final bias is
`log(8)`, so the initial hard mask is all open. Gate inputs are exactly
`concat(stop_grad(g_i), stop_grad(g_i) * stop_grad(mS_j))`; suffix tokens never enter the gate.

For each local image row, masked scores are computed in image/text blocks and then gathered by row.
Invalid suffix candidates are filled with `-inf` only after selecting valid queries, while labels
remain their original global `rank * B + arange(B)` values. The two CE sums use `W/V` once, and
`V < 2` returns a connected zero. Both directions use standard PyTorch normalization, CE and
autograd-aware gather; no custom backward or compact production candidate pool is used.

## Commands

CPU unit tests:

```bash
python -m pytest -q tests/test_dual_mask_suffix.py
```

The completed formal experiment used masked mode only, with loopback NCCL settings:

```bash
NCCL_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 GLOO_SOCKET_IFNAME=lo \
torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29610 \
  train/train_dual_mask_suffix.py --suffix-mode masked --run-type formal \
  --batch-size 256 --epochs 3 --max-steps 500 --seed 0 --num-workers 8 --save-every 100 \
  --init-state /root/SAID-gap-completion/runs_salu/said_cls_cvssl/shared_init/cvssl_initial.pt \
  --output-dir /root/SAID-s0-dualmask-clean-v01/runs_salu/dual_mask_suffix_masked_formal500_flatgather_v01
```

No native comparison arm was formally trained. Earlier short acceptance runs use
`DEBUG_NOT_FORMAL` directories and are not initialization checkpoints for the formal experiment.
The saved checkpoint contains `clip_state`, `suffix_gate_state` (`None` for
native), optimizer states, `completed_steps`, `config` and `provenance`, and is atomically replaced.

The frozen COCO canonical (5000 images / 25000 captions) and Urban-1k (1000 / 1000)
evaluations are complete using only normalized native student image/text embeddings.
The suffix gate is not part of the exported bare student or either retrieval scorer.
Full results are in [masked_formal500_report.md](masked_formal500_report.md), with
configuration, hashes, per-rank streams, memory and exit codes in
[masked_formal500_report.json](masked_formal500_report.json).

To create that evaluation artifact from a checkpoint:

```bash
python -c "from train.train_dual_mask_suffix import export_bare_student; export_bare_student('CHECKPOINT.pt', 'bare_student.pt')"
```

## Acceptance evidence

Historical evidence at `455013f`/`bb2d259` is preserved in `acceptance_evidence.json`.
Its finite-only assertions did not establish DDP gradient equivalence; the old combined
absolute/relative error field was ambiguous. Those checks are superseded by the fixed,
elementwise full-total-objective assertions and separate error reports in
`ddp_total_gloo.json` and `ddp_total_nccl.json`.

The suffix-only communication layout fix is explained in [gather_layout_fix.md](gather_layout_fix.md).
It passes 10 focused tests and the Gloo/NCCL production DDP comparisons with non-open test gates,
unequal/zero valid counts, complete parameter gradients, and AdamW updates.
The formal training SHA is `ff5ad1d4b918d56c6bfa48a2870dc5223e757237`.
The unique masked formal trajectory completed exactly 500 updates, followed by strict export
and both native retrieval evaluations; all four exit codes are zero. No update 501 was executed.
