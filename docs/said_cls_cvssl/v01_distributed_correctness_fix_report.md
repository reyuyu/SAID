# SAID-CLS-CVSSL v0.1 — distributed-correctness fix report

Scope: **training correctness only**. No algorithmic change: `m_S`, `m_U`, the SmartCLIP
positive/negative construction, `lambda_U`, `rho`, `tau_U`, the augmentation and the four arms are
untouched. The only behavioural change outside the trainer's plumbing is the R0 random-mask seeding
(§5), which restores the arm's documented meaning.

Repo state for this report: branch `codex/said-cls-cvssl-v01`, committed tip
`9a7c08621414f035bf0b8ad0b323fb9ed61bdee6`, all of this round's work **uncommitted** in the working
tree. Nothing was pushed, merged, rebased, squashed or force-pushed.

Measured numbers live in `v01_ddp_debug_matrix.md`; the raw JSON artifacts are under
`/tmp/cvssl_diag/stage1`…`stage5` on the remote box.

## 1. What was wrong with `9a7c086`

`9a7c086` computed the whole objective inside a plain `nn.Module` and never wrapped it in
`DistributedDataParallel`. The only cross-rank operation was the **feature-level** autograd
all-gather inherited from SmartCLIP. Consequently:

* the image/text losses agreed across ranks (their candidates came from the gather),
* but **no parameter gradient was ever synchronised**, so the four ranks stepped four different
  models from their own local anchors;
* the U term's `× W` factor had been introduced against a hypothetical DDP average that did not
  exist in the code.

The previous report also claimed the reference trainer does not use DDP. That claim was wrong:
`train/train.py:171` wraps the model in `DistributedDataParallel` and `:173` calls
`_set_static_graph()`. The claim has been corrected in `v01_implementation_report.md`.

## 2. What the corrected trainer does

* `SaidClsCvsslTrainModule` owns the CLIP model exactly once and runs the whole objective in
  `forward`, so DDP sees every parameter, `mask_net` included.
* The trainer wraps it in `DistributedDataParallel(..., find_unused_parameters=True)` and keeps
  `_set_static_graph()` (measured compatible, §6).
* `cvssl_train_step` is the production step: DDP forward → `loss_total_for_backward.backward()` →
  both AdamW steps. It also (a) asserts equal per-rank batch sizes **before** the first collective,
  and (b) exposes `capture_grads` for the acceptance tests.
* The forward always goes through the DDP call. `ddp_model.module` is used only for
  checkpoints/`state_dict`/optimizer groups/diagnostics, never for training.
* `ddp_gradient_averaging=1` (default) multiplies the CVSSL term by the world size; the SmartCLIP
  term is **not** rescaled.

## 3. Are the four ranks updating the same parameter state? — YES (measured)

The acceptance sentence was "prove that rank0/1/2/3 really update the same parameter state, then
look at whether C0 beats S0/G0/R0". Measured on the production step function, 2 ranks, gloo/CPU:

* **Forward**: `mean(rank loss) == single-process global loss` to `0.0` for `loss_smart`,
  `loss_sparsity` and `loss_vssl_global_mean`, and to rel `6.7e-08` / `4.0e-08` for `loss_sidm` /
  `loss_dism` (different anchors per rank is correct; only the mean is globally defined).
* **Backward**: `max|g_rank0 - g_rank1| == 0.0` for every parameter and every term
  (`sidm`, `dism`, `sparse`, `smart`, `u`, `total`). Bit-identical, not just close.
* **Update**: `param_digest`, `backbone_opt_digest` and `mask_opt_digest` are equal across ranks at
  steps 1, 5 and 20, for all four arms plus the duplicate-image and repeated-caption cases.
* **Against the single process**: `maxabs(param) = 1.49e-08` on a parameter scale of `6.12e-01`
  (rel `2.4e-08`), and identical digests across ranks at every step.

## 4. Why the DDP-vs-single residual is not a scaling bug

fp32 residual on the S0/C0 total gradient: `6.10e-05` on `max|g| = 6.43e+02` (rel `1.26e-07`).
The same experiment in fp64 (identical math, only the accumulation precision changes) gives
`8.30e-12` (rel `1.29e-14`). The residual therefore tracks the order of accumulation, not a factor:
a missing or duplicated `W` produces a norm ratio of exactly `0.5` or `2.0`, i.e. rel `0.5`–`1.0`.

The U term's factor was measured separately, with the gradient of the U term taken through a real
`.backward()`:

| configuration | norm ratio (DDP mean / single) |
| --- | ---: |
| `--ddp_gradient_averaging 1` (default) | `1.000000` |
| `--ddp_gradient_averaging 0` | `0.500000` (min = median = max) |

`loss_vssl_global_mean` stays `0.03550041` in both, so the reported metric is independent of the
switch. Keeping the factor is the correct reading of the global objective; it was not chosen because
"it looks closer".

## 5. Fixes applied

1. **Parameter-gradient synchronisation** — the train module is DDP-wrapped and the production step
   back-propagates through the DDP forward (`train/train_said_cls_cvssl.py`,
   `model/said_cls_cvssl.py`).
2. **Ragged-batch guard hoisted** — `assert_equal_local_batch` now runs before the first cross-rank
   gather. Before this, a genuinely different per-rank batch size crashed gloo with
   `op.preamble.length <= op.nbytes. 96 vs 64` and SIGABRT instead of raising; now every rank
   participates in the size collective, learns the mismatch and raises the same message, which is
   also written to stderr so torchrun cannot swallow it.
3. **`capture_grads` no longer assumes a DDP wrapper** — `getattr(ddp_model, 'module', ddp_model)`,
   so the single-process validation path works.
4. **R0 random mask is per sample, not per row** — `build_random_mask` seeds from
   `mask_seed + 1000003 * sample_id` instead of consuming a generator row by row. Before the fix a
   sample's random mask depended on the local batch layout: the global U loss was `0.03308536` in
   the single-process global-batch run and `0.00889252` under 2 ranks (rel `0.73`). This restores
   the arm's documented "per sample independent permutation" semantics and makes R0 comparable
   between the single-process reference and the distributed run. R0 only. After the fix:
   `loss_vssl_global_mean` is `0.0259756036` in the single-process run and on **both** ranks
   (`mean - single = 0.0`), and it is distinct from C0's `0.0254811142`, so the permutation is
   really applied. Before the fix it was `0.03308536` vs `0.00889252`.
5. **`override` hook** (default `None`) so the degenerate all-zero-mask path can be exercised at
   trainer level; **docstring correction** for the DDP claim.

## 6. Engineering switches

* `_set_static_graph()`: ON and OFF give the same gradients and the same update, and ranks stay
  bit-identical in both. It is **kept**, so this repository stays as close to the reference trainer
  as possible while checkpointing both views with `use_reentrant=False`.
* Activation checkpointing ON/OFF (single process): `param_digest` identical,
  `backbone_opt_digest` / `mask_opt_digest` identical, `maxabs(param) = 0.0`. Under 2 ranks the
  checkpointed run is bit-identical across ranks and differs from the single-process run by
  `1.49e-08` on a scale of `6.12e-01`. It is a memory switch only.

## 7. Test status

| suite | result |
| --- | --- |
| `tests/test_said_cls_cvssl.py` + `tests/test_complement_visual_ssl_ddp.py` | 30 passed |
| `tests/test_cvssl_ddp_trainer.py` (rewritten, real production step) | 25 passed |
| `tests/test_cvssl_dtype_audit.py` (new) | 4 passed |
| `pytest tests/ -q` | **528 passed, 1 skipped** |

### Dtype audit (new)

* every trainable parameter and every AdamW state is fp32, and the two `build_optimizers` groups
  contain exactly the backbone / `mask_net` parameters;
* `fp32_context` really disables autocast inside the block and the masked-cosine core returns
  fp32 tensors;
* the documented multipliers hold exactly in the production step
  (`loss_smart == 10*(sidm+dism) + 2*sparsity`) and every captured gradient is fp32 and finite.
* **Declared limitation:** the bf16 autocast numerics of the two-view forward are **NOT RUN** on
  this host. Measured: CPU autocast reports enabled inside `torch.autocast('cpu', bf16)` while some
  ops warn that the dtype is unsupported and fall back, and in the trainer's call
  (`torch.autocast(..., dtype=torch.float32, enabled=True)`) torch warns and disables autocast
  outright. The audit on CPU therefore certifies the dtype contract, not the bf16 numerics; that has
  to be exercised on the GPU run, and is listed as open.

### Tolerance policy (declared, not tuned to pass)

* Cross-rank comparisons are asserted **bit-equal** (`maxabs == 0`), because DDP's reducer is
  deterministic and the measurements above really are bit-equal. A synchronisation bug is orders of
  magnitude larger than fp32 round-off.
* DDP-averaged vs single-process uses a **relative** criterion
  `maxabs <= ATOL + RTOL * max|reference|` with `ATOL = 0`, `RTOL = 1e-5`. The earlier `ATOL = 1e-6`
  **absolute** bound was unachievable for fp32 gradients of magnitude `6.4e+02` (an fp32 ulp there
  is `~3e-05`), and fp32 `sha256` digest equality is not a valid numerical criterion either. The
  relative bound is still ~100× tighter than one ulp of the reference and would fail a 2× scaling
  error by four orders of magnitude, so it is discriminating rather than permissive.
* The rewritten worker also fixes two harness defects that had nothing to do with the algorithm:
  `build_optimizers` is called with `train_module.clip` (exactly as production does), and every rank
  now **slices** the same deterministic global batch instead of regenerating one.

### Coverage declared explicitly

`test_ranks_stay_identical_across_20_steps` covers all four arms plus duplicate-image and
repeated-caption cases, at steps 1/5/20. It asserts cross-rank equality of parameters, both
optimizer states and the reduced U loss — it does **not** assert that per-rank `loss_sidm` /
`loss_dism` are equal, because those are local-anchor means and are supposed to differ.

## 8. Status of the pre-fix artifacts

The 20-step outputs produced before this fix (`runs_salu/said_cls_cvssl/smoke_*`) are **engineering
records only**: the ranks were training four different models, so they say nothing about the
objective. They are preserved and must not be used for any algorithmic comparison. All reruns use
new `ddpfix_*` directories.

## 9. Consolidated status

| item | status |
| --- | --- |
| Implementation (same parameter state on every rank) | **PASS** — measured, §3 |
| Implementation (DDP gradient == single-process global gradient) | **PASS** — fp32 round-off, fp64 confirms, §4 |
| `× W` factor | **PASS** — exact `1.000000` vs `0.500000`, §4 |
| Retrieval (native CLS canonical COCO 1K/5K) | **NOT RUN** in this round |
| 4-arm 20-step / 500-step reruns | **NOT RUN** in this round |
| `weighted_vssl_to_smart_grad_ratio`, `cos(grad_visual L_smart, grad_visual lambda_U L_U)` | **NOT RUN** |
| 3-epoch runs, hyperparameter search | **NOT RUN** (not started, as instructed) |
| Complementary semantic preservation | **NOT ESTABLISHED** — no withheld/concept benchmark was run, and none of the reported diagnostics (U cross-view top-1, falling U loss, mask differences, low `cos(U, Global)`) is evidence of it |
