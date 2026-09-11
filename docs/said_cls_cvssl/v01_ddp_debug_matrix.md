# SAID-CLS-CVSSL v0.1 — DDP debug matrix (measured)

Every cell in this document is a **measured** number, produced on the remote box with the
diagnostic workers listed in §0. Nothing here is inferred from theory.

Branch `codex/said-cls-cvssl-v01`, uncommitted working tree, committed tip
`9a7c08621414f035bf0b8ad0b323fb9ed61bdee6`.

## 0. What was run

| tool | purpose |
| --- | --- |
| `tmp_said_cvssl_diag.py` | layered worker: single-process global batch, 2-rank DDP, 2-rank without DDP, per-term backward through the real DDP path, dtype control, optimizer step |
| `tmp_said_clip_realddp.py` | golden reference: the **real** `model.model_longclip.CLIP.forward`, real `mask_net`, real tokenizer, real `torch.distributed.nn.all_gather`, with and without real `DistributedDataParallel` |
| `tmp_patch_capture_grads.py`, `tmp_patch_engineering_fixes.py`, `tmp_patch_mask_override.py` | the production fixes recorded in §7 |
| raw artifacts | `/tmp/cvssl_diag/stage1` … `/tmp/cvssl_diag/stage5` |

Model stand-in for the trainer-level runs: identical `StubClip` in every mode (fixed
`torch.manual_seed(1234)`), global batch 6 pairs (3 per rank), `DIM=16`, fp32, gloo, CPU.
The single canonical global batch is generated once and **sliced** per rank.

## 1. The 11 original failures, sorted into root-cause classes

| # | failing tests | root cause | class |
| --- | --- | --- | --- |
| 1 | `test_gradients_per_rank_match…[S0,A]`, `test_one_step_update…[S0,A]`, `test_ranks_stay_identical…[A..F]`, `test_unequal_local_batches…` | worker called `build_optimizers(module, args)` on the `SaidClsCvsslTrainModule`; `build_optimizers` reads `model.mask_net`, which the train module does not expose. Every ranked run died at setup with `AttributeError: 'SaidClsCvsslTrainModule' object has no attribute 'mask_net'` — **test-harness defect**, production passes `train_module.clip` | test harness |
| 2 | `test_unequal_local_batches…` (after fix 1) | the ragged-batch guard ran *after* the first cross-rank `all_gather`, so gloo aborted with `op.preamble.length <= op.nbytes. 96 vs 64` / SIGABRT and the intended message was never printed | **production defect** |
| 3 | all gradient/update comparisons (after fix 1) | tolerance policy: `ATOL = 1e-6` **absolute** on fp32 gradients of magnitude `6.4e+02`, and fp32 `sha256` digests compared for equality. Both are guaranteed to fail for reasons unrelated to synchronisation | test tolerance |
| 4 | `test_forward_losses…` (new) | my own assertion that per-rank `loss_sidm`/`loss_dism` are equal across ranks: they are **local-anchor means** and must differ; only their mean is the global value | test assertion (author error) |
| 5 | R0 single-vs-rank | `build_random_mask` drew from a generator consumed over **local rows**, so the random mask of a given sample depended on the local batch layout. Measured: global U loss single `0.03308536` vs rank mean `0.00889252`, rel `0.73` | **production defect (R0 semantics)** |
| 6 | (new) `capture_grads` | `ddp_model.module.named_parameters()` fails when the bare module is passed (single-process path) | **production defect** |

`m_U` / `m_S` definitions, SmartCLIP positives/negatives, `lambda_U`, `rho`, `tau_U`,
augmentation, arms: **not touched**.

## 2. Forward aggregation — `mean(rank local loss) == single-process global loss`

| quantity | single | rank0 | rank1 | `mean - single` |
| --- | ---: | ---: | ---: | ---: |
| `loss_sidm` | 1.7917660475 | 1.7920621634 | 1.7914701700 | `1.19e-07` (rel `6.7e-08`) |
| `loss_dism` | 23.7193279266 | 23.4954376221 | 23.9432163239 | `9.54e-07` (rel `4.0e-08`) |
| `loss_sparsity` | 0.5625000000 | 0.5625000000 | 0.5625000000 | `0.0` |
| `loss_smart` | 256.2359313965 | 254.0000000000 | 258.4718627930 | `0.0` |
| `loss_vssl_global_mean` | 0.0254811142 | 0.0254811142 | 0.0254811142 | `0.0` |
| initial parameter digest | identical in single and both ranks | | | — |

Conclusion: the forward aggregation and the candidate ordering are correct; per-rank losses
*should* differ (different anchors) and only the mean is globally defined.

## 3. Backward — gradients taken through the real DDP path

Rank0 vs rank1 after `loss.backward()`, per term, **all bit-identical**:

| term | rank0 vs rank1 `maxabs` | DDP mean vs single `maxabs` | rel |
| --- | ---: | ---: | ---: |
| S0 `sidm` | `0.0` | `8.35e-07` | `1.70e-02` (on `max|g|=1.25e+00`) |
| S0 `dism` | `0.0` | `7.63e-06` | `1.57e-07` |
| S0 `sparse` | `0.0` | `9.31e-10` | `1.00e-07` |
| S0 `smart` | `0.0` | `6.10e-05` | `1.26e-07` |
| S0 `total` | `0.0` | `6.10e-05` | `1.26e-07` |
| C0 `u` | `0.0` | `7.45e-09` | `6.60e-08` |
| C0 `total` | `0.0` | `6.10e-05` | `1.26e-07` |

**The residual is not a scaling bug.** Same experiment in fp64:

| term (dtype) | `max|g|` | worst `maxabs` | worst rel |
| --- | ---: | ---: | ---: |
| S0 `sidm` (fp32) | 1.2538e+00 | 8.345e-07 | 1.700e-02 |
| S0 `sidm` (**fp64**) | 1.2541e+00 | **3.553e-13** | **6.562e-11** |
| S0 `smart` (fp32) | 6.4295e+02 | 6.104e-05 | 1.258e-07 |
| S0 `smart` (**fp64**) | 6.4296e+02 | **8.299e-12** | **1.291e-14** |
| C0 `total` (fp32) | 6.4295e+02 | 6.104e-05 | 1.258e-07 |
| C0 `total` (**fp64**) | 6.4296e+02 | **8.299e-12** | **1.291e-14** |

The fp32 residual scales with `max|g|` and vanishes when the only difference left is the order of
accumulation ⇒ it is fp32 round-off, not a missing or duplicated factor. A 2×/0.5× scaling error
would show ratio `2.0`/`0.5`, i.e. rel `0.5`–`1.0`, five to seven orders of magnitude larger.

**Measurement caveat (recorded on purpose):** term gradients obtained with `torch.autograd.grad`
are *not* DDP-synchronised — `.backward()` is what triggers the reducer. An earlier pass of this
matrix used `autograd.grad` and showed `rank0 != rank1` with differences up to `1.06e+03`; those
numbers were local gradients, not a synchronisation failure. Every number in the table above comes
from a real `.backward()` of exactly one term.

## 4. U-only — is the world-size factor right?

Arm C0, `lambda_U = 1`, gradient of the U term only, through the real DDP path.

| configuration | rank losses | `loss_scale` | DDP mean vs single | norm ratio |
| --- | --- | ---: | ---: | ---: |
| single (world 1) | 0.03550041 | 1.00 | reference | — |
| 2-rank, `--ddp_gradient_averaging 1` | `[0.03826138, 0.03273943]` | **2.00** | worst_abs `7.45e-09`, rel `3.72e-08` | `1.000000` |
| 2-rank, `--ddp_gradient_averaging 0` | `[0.01913069, 0.01636972]` | 1.00 | worst_abs `5.65e-02`, rel `2.82e-01` | **`0.500000`** |

With the factor off, the DDP-averaged parameter gradient is **exactly half** the global-mean
gradient (`norm ratio 0.500000`, min = median = max). With the factor on it is `1.000000`.
`loss_vssl_global_mean` is `0.03550041` in every configuration, i.e. the reported global metric is
independent of the scaling switch, as designed.

Verdict: keeping `ddp_gradient_averaging=1` (the default) is **required**; it is not a fudge factor.
The SmartCLIP term is *not* rescaled, and §3 confirms its DDP gradient already equals the
single-process global gradient.

## 5. Engineering switches

| switch | rank0 vs rank1 | DDP vs single | veredict |
| --- | --- | --- | --- |
| `_set_static_graph()` ON | gradients bit-identical | same as OFF (rel `1.26e-07`, identical to the OFF run) | compatible |
| `_set_static_graph()` OFF | gradients bit-identical | rel `1.26e-07` | — |
| activation checkpoint ON (single process, 1 step) | — | `param_digest` **identical** to OFF, `backbone_opt_digest`/`mask_opt_digest` identical, `maxabs(param) = 0.0` | equivalent |
| activation checkpoint ON (2-rank) | `param_digest` identical; `maxabs(param)` vs single = `1.49e-08` on scale `6.12e-01` (rel `2.4e-08`) | | equivalent |

`_set_static_graph()` is therefore kept (it stays compatible with non-reentrant two-view
checkpointing), and activation checkpointing is numerically free — it is a memory switch only.

## 6. Original SmartCLIP `CLIP.forward` under real DDP (golden reference)

Tiny **real** `model.model_longclip.CLIP` (embed_dim 16, width 128, 1 vision block, 1 text block,
context 8, real `MaskNetwork`, real tokenizer), same global batch of 6, same seed.

| quantity | single (world 1) | 2-rank DDP mean | diff |
| --- | ---: | ---: | ---: |
| `loss_sidm` | 26.91943932 | 26.91943550 | `3.8e-06` |
| `loss_dism` | 3.43399334 | 3.43399405 | `7.2e-07` |
| `loss_sparsity` | 0.53125000 | 0.53125000 | `0.0` |
| `loss_total` | 304.59680176 | 304.59680176 | `0.0` |

Init digest identical (`24aea6cd32244de4`) in all three runs. Conclusion: this repository's
SmartCLIP objective **does** aggregate across ranks under DDP exactly as its all-gather construction
implies, and the new `S0` train module reproduces that same forward/backward chain (§2, §3). The
earlier note in `v01_implementation_report.md` claiming the reference trainer does not use DDP was
wrong on two counts: `train/train.py:171` wraps the model in DDP, and the CVSSL trainer added in
`9a7c086` performed no parameter-gradient synchronisation at all.

## 7. Production fixes applied in this round

1. `train/train_said_cls_cvssl.py::cvssl_train_step` — `assert_equal_local_batch` moved **before**
   the first cross-rank gather; the guard writes its message to stderr before raising so torchrun's
   error summary cannot swallow it.
2. `train/train_said_cls_cvssl.py::cvssl_train_step` — `capture_grads` uses
   `getattr(ddp_model, 'module', ddp_model)` so the single-process path works.
3. `model/complement_visual_ssl.py::build_random_mask` — the R0 permutation is seeded **per sample**
   (`mask_seed + 1000003 * sample_id`) instead of consuming a row stream, so a sample's random mask
   no longer depends on the local batch layout. R0 only; C0/S0/G0 and every loss definition are
   untouched.
4. `model/complement_visual_ssl.py::build_arm_mask` — optional `override` (test-only, default
   `None`) so the degenerate all-zero-mask path can be exercised at trainer level;
   `cvssl_train_step` accepts the matching `mask_override=None`.
5. `model/complement_visual_ssl.py` — docstring corrected (it claimed the trainer does not use DDP).

### 7.1 R0 verification after the seeding fix

| configuration | `loss_vssl_global_mean` |
| --- | ---: |
| single-process global batch | `0.0259756036` |
| 2-rank, rank0 | `0.0259756036` |
| 2-rank, rank1 | `0.0259756036` |
| `mean - single` | `0.0` (rel `0.0`) |

Before the fix the same comparison was `0.03308536` vs `0.00889252` (rel `0.73`). The R0 U loss is
also distinct from C0's (`0.0254811142`), i.e. the permutation really is applied and is not the
complement.

## 8. Resolved / still open

| question | status |
| --- | --- |
| forward aggregation correct | **PASS** (§2) |
| rank0 == rank1 after DDP backward | **PASS**, bit-identical for every term (§3) |
| DDP gradient == single-process global gradient | **PASS** at fp32 round-off, fp64 confirms (§3) |
| `× W` on the U term justified | **PASS**, exact `1.000000` vs `0.500000` (§4) |
| original SmartCLIP DDP == new S0 DDP | **PASS** (§2–§4) |
| static graph / activation checkpoint safe | **PASS** (§5) |
| R0 per-sample matched independence | **fixed and measured** (§7.3) |
| equal local batches incl. ragged last batch | guard verified; ragged *last* batch still to be exercised on real data |
| 20-step / 500-step runs, retrieval | **NOT RUN** (this round stops at the corrected tests) |
| `weighted_vssl_to_smart_grad_ratio`, `cos(grad_visual L_smart, grad_visual lambda_U L_U)` | **NOT RUN** |
| "complementary semantic preservation" | **NOT ESTABLISHED** (no withheld/concept benchmark) |
