# S0 Dual-Mask Suffix — Clean v0.1

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
`concat(stop_grad(g), stop_grad(g * mS))`; suffix tokens never enter the gate.

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

The real formal entry is prepared but is not launched in this round:

```bash
torchrun --nproc_per_node=4 train/train_dual_mask_suffix.py --suffix-mode masked \
  --batch-size 256 --epochs 3 --max-steps 500 \
  --init-state /root/SAID-gap-completion/runs_salu/said_cls_cvssl/shared_init/cvssl_initial.pt \
  --output-dir /root/SAID-s0-dualmask-clean-v01/runs_salu/dual_mask_suffix_masked_v01
```

Use `--suffix-mode native` for the native comparison. A short acceptance run must use a
`DEBUG_NOT_FORMAL` output directory and is reported separately; it is never a formal 500-step
starting point. The saved checkpoint contains `clip_state`, `suffix_gate_state` (`None` for
native), optimizer states, `completed_steps`, `config` and `provenance`, and is atomically replaced.

The complete COCO/Urban evaluations are deliberately not run here. Future calls should use the
base repository's `tools/phase30a_fixed_cohort_eval.py` and `tools/eval_urban1k_cls.py` against the
bare student exported from `clip_state`; the suffix gate is not part of that bare student.

To create that evaluation artifact from a checkpoint:

```bash
python -c "from train.train_dual_mask_suffix import export_bare_student; export_bare_student('CHECKPOINT.pt', 'bare_student.pt')"
```

## Acceptance evidence

Evidence for code SHA `455013f47d3363c6d1725cc6f6d918d98091b699` is recorded in
`acceptance_evidence.json`. The nine focused tests pass, and native/masked real-CLIP runs both
completed three steps at 4 GPUs × 256 (global batch 1024), with `formal_optimizer_updates=0`.
The DDP global-reference probe reports equal scalar loss but exposes a visual/text gradient scaling
difference (maximum absolute/relative error `56.679512`, one-step parameter error `0.002000004`);
the result is preserved as evidence and no tolerance was relaxed or production reduction changed.
The first NCCL launch failed before forward; the retry with loopback NCCL settings succeeded for
both modes. Full 500-step training and COCO/Urban evaluation remain unrun.
