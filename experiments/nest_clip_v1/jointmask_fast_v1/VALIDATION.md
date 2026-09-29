# NEST JointMask fast configuration validation

Date: 2026-09-29 UTC

Base performance commit: `f614bda2eab5ad2b71a08ff9f62abdc68edb593a`.

The three formal configurations instantiate `image_chunk=128`, `text_chunk=128`,
`checkpoint_pair_blocks=false`, and `checkpoint_encoders=true`. The trainer asserts these
values against the actual `NestedSemanticMask` instance and verifies that both encoder
transformer forwards use the non-reentrant checkpoint implementation. Runtime values are
saved in each run's `config.json`.

The focused suite passes 42 tests. It includes fixed-shape statistics, discarded T2I
summary, nonzero joint-adapter visual corrections, tail geometry (180 local queries and 720
global candidates), RandomK, resume boundaries, and the existing semantic-mask tests.

The two-rank NCCL reference passes all-valid, a zero-valid rank, global V=0, global V=1,
shuffle, partial local batches, nonzero adapter output, and pair checkpoint on/off. It
checks gradients by parameter name, including exact `None` states. The maximum observed
gradient error is `9.307861328125e-4` at `clip.visual.proj[7,6]`; the maximum SGD reference
update error is `9.313225746154785e-8` at the same location.

A separate four-rank full ViT-B/16 test compares the real project AdamW update on the same
real 4×256 batch and initial state. T and nonzero-adapter TI have exactly equal losses
between the diagnosed 32×64/cp-on setting and the 128×128/cp-off setting, and all gradient
`None` states match. The largest gradient differences are:

- T: `0.1328125` at `clip.transformer.resblocks.3.attn.in_proj_weight[1204,174]`, or
  `7.4771e-4` relative to that parameter's reference maximum absolute gradient.
- TI: `0.08203125` at `clip.transformer.resblocks.1.attn.in_proj_weight[1350,174]`, or
  `4.8566e-4` relative to that parameter's reference maximum absolute gradient.

Because AdamW normalizes the first-step gradient, a small sign-sensitive near-zero mask
bias gradient produces maximum one-step parameter differences of `2.9026653e-4` for T and
`1.6791874e-4` for TI, both at `clip.mask_net.attn_pool.attention.bias[0]`. Therefore the
cross-chunk comparison is not claimed to be stepwise equivalent.

The isolation comparison at fixed 128×128 changes only pair checkpoint on/off. For both T
and nonzero-adapter TI, loss, every named gradient, `None` state, and the formal AdamW update
are bitwise identical. The three new formal groups all use this same 128×128/cp-off setting,
so their main comparison remains controlled.

Raw results are in `evidence/adamw-one-step.json`, `evidence/ddp-validation.console.txt`,
and the preserved failed-threshold logs. The nonzero adapter state exists only inside the
validation processes; formal initialization retains zero `WO`.
