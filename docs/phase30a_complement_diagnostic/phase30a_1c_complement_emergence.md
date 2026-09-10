# Phase 3.0A.1c — Complement Emergence Diagnostic

Base objective only: **Said-Conditioned Visual Complement Discovery** on `(I, C_S)`.
No `C_F`, no `C_U`, no USS, no Global-text InfoNCE, no legacy Unsaid. The gap math in
`model/gap_completion.py` was not modified.

Artifacts (all small, tracked):

| file | content |
|---|---|
| `temperature_sweep_summary.json` | forward-only sweep, 2 feature sources x 5 temperatures |
| `matched_100step_summary.json` | two matched 100-step arms, per-step trajectory + stream identity |

Both were produced from the raw run outputs under `runs_salu/phase30a_1c_*` and
`outputs/phase30a_complement_diagnostic/` (checkpoints and raw logs are not committed).

## 1. Forward-only sweep (no optimizer step)

One fixed `(I, C_S)` batch of 64 real ShareGPT4V images, one fixed initial checkpoint.
`batch_sha256 = b04bde11e5c4e772…`; both feature sources started from the *same* weights
(`weights_sha256 = 6695aec5f3f2697e…`). `optimizer_steps = 0`.

### residual (`said_feature_source=residual`)

| tau | attn JSD | overlap | cos(zS,zU) | novel_norm | closure | clos_pos | gap_after |
|---|---|---|---|---|---|---|---|
| 2.00 | 0.050118 | 0.747062 | 0.994447 | 0.101122 | 0.006761 | 0.6250 | 0.370093 |
| 1.00 | 0.085208 | 0.669161 | 0.991014 | 0.129112 | 0.005412 | 0.5938 | 0.370553 |
| 0.50 | 0.144376 | 0.568009 | 0.986215 | 0.160311 | 0.002523 | 0.5156 | 0.371578 |
| 0.25 | 0.209363 | 0.476875 | 0.982382 | 0.181602 | **-0.000266** | 0.5000 | 0.372588 |
| 0.10 | 0.267583 | 0.402745 | 0.980125 | 0.193117 | **-0.002290** | 0.4688 | 0.373340 |

### attention_delta (`said_feature_source=attention_delta`)

| tau | attn JSD | overlap | cos(zS,zU) | novel_norm | closure | clos_pos | gap_after |
|---|---|---|---|---|---|---|---|
| 2.00 | 0.017541 | 0.849925 | 0.999428 | 0.029497 | 0.003238 | 0.5625 | 0.407731 |
| 1.00 | 0.041314 | 0.770300 | 0.998778 | 0.043650 | 0.004253 | 0.5469 | 0.407271 |
| 0.50 | 0.089260 | 0.666466 | 0.997719 | 0.059921 | 0.005203 | 0.5469 | 0.406834 |
| 0.25 | 0.148816 | 0.573938 | 0.996785 | 0.071249 | 0.005799 | 0.5312 | 0.406558 |
| 0.10 | 0.203694 | 0.502454 | 0.996269 | 0.076817 | 0.006126 | 0.5312 | 0.406411 |

### patch geometry (tau independent)

| feature_source | patch_pair_cos | patch_pair_cos_std | patch_to_global_cos | patch_centered_energy |
|---|---|---|---|---|
| residual | 0.588580 | 0.099359 | 0.484401 | 0.634828 |
| attention_delta | **0.953875** | 0.025475 | 0.578211 | **0.203838** |

Raw pool magnitudes (the normalisation hides these): `residual`
`raw_S = 5.545007`, `raw_U = 5.587…5.672` (ratio 1.008 → 1.023, rising as tau falls);
`attention_delta` `raw_S = 8.090343`, `raw_U = 8.070…8.052` (ratio ≈ 1.00).

### Section 13 counterexample — confirmed twice, independently

1. In the residual sweep, lowering tau buys diversity monotonically and **destroys
   closure**: `closure` goes `+0.0068 (tau=2)` → `+0.0025 (0.5)` → `-0.00027 (0.25)` →
   `-0.0023 (0.10)`, while `gap_after` worsens monotonically. tau 0.25 and 0.10 are
   therefore **rejected** ("different" is not "useful").
2. In the 100-step arms, the same divergence appears *during training* (Section 3 below).

**Selected stronger temperature: `tau = 0.5`.** It is the strongest temperature that still
keeps closure positive in the forward-only sweep, and it is not the rejected 0.1/0.25.

## 2. Matched 100-step diagnostic

`objective_mode=gap_completion`, `lambda_global=0`, `lambda_unsaid=0`, `lambda_said=1`,
`lambda_gap_discover=1`, `lambda_global_absorb=1`, identifiable, residual, ViT-B/16,
4x A800, batch_size 256/GPU (global 1024), bf16, seed 0, warmup_length 200,
lr_total_steps 3648, `max_steps=100`, `log_every=10`, full ShareGPT4V + Full Data Gate
(dataset_size 1245901, steps_per_epoch 1216). Checkpoints: initial / step20 / step50 /
step100 (+ last). Both arms exited `EXIT=0`, no non-finite value in either log.

### Stream identity (Section 10) — all matched

| quantity | tau1.0 | tau0.5 |
|---|---|---|
| `initial_state_sha256` | `4621b8d027f482ea…` | `4621b8d027f482ea…` |
| `sampler_order_sha256` | `953402b3c3d7322e…` | `953402b3c3d7322e…` |
| `caption_stream_sha256` | `45f3b25bdc52499e…` | `45f3b25bdc52499e…` |
| `full_caption_stream_sha256` | `null` | `null` |
| `unsaid_caption_stream_sha256` | `null` | `null` |

Per-logged-step `batch_image_sha256` and `batch_caption_sha256`: **identical at all
compared steps** (0, 10, 20, 30, 40, 50, 60, 70, 80, 90). No mismatch.

### Core trajectory (steps 0 / 20 / 50 / 99)

| metric | tau1.0 @0 | tau1.0 @20 | tau1.0 @50 | tau1.0 @99 | tau0.5 @0 | tau0.5 @20 | tau0.5 @50 | tau0.5 @99 |
|---|---|---|---|---|---|---|---|---|
| L_S | 3.333893 | 3.176914 | 3.061692 | 2.333001 | 3.333893 | 3.176355 | 3.064637 | 2.339711 |
| L_gap | 0.373392 | 0.357686 | 0.350081 | 0.290356 | 0.373889 | 0.359969 | 0.355117 | 0.297152 |
| L_absorb | 0.373392 | 0.357686 | 0.350081 | 0.290356 | 0.373889 | 0.359969 | 0.355117 | 0.297152 |
| L_total | 4.080677 | 3.892287 | 3.761854 | 2.913712 | 4.081670 | 3.896294 | 3.774871 | 2.934016 |
| route_top1 | 0.003906 | 0.007812 | 0.011719 | 0.207031 | 0.003906 | 0.007812 | 0.015625 | 0.222656 |
| evidence_top1 | 0.746094 | 0.812500 | 0.839844 | 0.882812 | 0.746094 | 0.796875 | 0.835938 | 0.886719 |
| attn JSD | 0.084738 | 0.088203 | 0.096713 | 0.161663 | 0.145344 | 0.149859 | 0.159283 | 0.229320 |
| attn overlap | 0.666616 | 0.658767 | 0.643637 | 0.538038 | 0.562368 | 0.554431 | 0.541803 | 0.445765 |
| cos(zS,zU) | 0.989322 | 0.988401 | 0.986612 | 0.971789 | 0.983524 | 0.982154 | 0.979924 | 0.963650 |
| novel_norm | 0.139766 | 0.144790 | 0.156214 | 0.226859 | 0.173741 | 0.179775 | 0.191424 | 0.257803 |
| gap_before | 0.379174 | 0.355737 | 0.334673 | 0.257896 | 0.379174 | 0.355732 | 0.334561 | 0.257702 |
| gap_after | 0.373392 | 0.357686 | 0.350081 | 0.290356 | 0.373889 | 0.359969 | 0.355117 | 0.297152 |
| gap_reduction | 0.005782 | -0.001949 | -0.015409 | -0.032460 | 0.005285 | -0.004237 | -0.020556 | -0.039450 |
| closure ratio | 0.015799 | -0.006369 | -0.048104 | -0.142845 | 0.014351 | -0.013290 | -0.064165 | -0.172825 |
| closure pos frac | 0.652344 | 0.414062 | 0.132812 | 0.066406 | 0.625000 | 0.382812 | 0.117188 | 0.050781 |
| patch_pair_cos | 0.607376 | 0.592255 | 0.587639 | 0.565071 | 0.607376 | 0.592545 | 0.587845 | 0.566895 |
| patch_centered_energy | 0.619673 | 0.631726 | 0.635826 | 0.653154 | 0.619673 | 0.631548 | 0.635556 | 0.651814 |
| said_raw_pool_norm | 5.644986 | 5.550265 | 5.542434 | 5.601494 | 5.644986 | 5.550484 | 5.543123 | 5.597416 |
| unsaid_raw_pool_norm | 5.749986 | 5.615976 | 5.567132 | 5.431604 | 5.792939 | 5.650276 | 5.595164 | 5.453646 |
| cos(g,zS) | 0.620826 | 0.644263 | 0.665328 | 0.742104 | 0.620826 | 0.644268 | 0.665439 | 0.742297 |
| cos(g,zU) | 0.626606 | 0.642184 | 0.649548 | 0.707819 | 0.626024 | 0.639693 | 0.644108 | 0.699960 |
| grad_norm_backbone | 30.162227 | 16.299406 | 14.821697 | 49.165486 | 30.116057 | 16.673387 | 25.156857 | 43.509371 |
| grad_norm_said_router | 0.824409 | 0.593358 | 0.504944 | 0.838437 | 0.824409 | 0.590728 | 0.503406 | 0.833698 |
| grad_gap contribution | 2.156739 | 2.161206 | 2.195177 | n/a | 2.103041 | 2.122165 | 2.157845 | n/a |
| grad_absorb contribution | 4.071493 | 3.841933 | 3.784838 | n/a | 4.073606 | 3.918039 | 3.713472 | n/a |

`grad_*_contribution` is measured at steps 0/20/50/100 only (each needs its own backward
pass); at step 100 the log record is written without the digest/gradient block, so the
values are `n/a` there. Per-term backwards are run on a **fresh forward graph** inside
`no_sync` — reusing the training graph is impossible because DDP's reducer frees it.

### Paired comparison over the 13 logged steps

| metric | tau0.5 vs tau1.0 |
|---|---|
| attn JSD (higher = more diverse) | higher at **13/13** |
| attn overlap (lower = more diverse) | lower at **13/13** |
| cos(zS,zU) (lower = more diverse) | lower at **13/13** |
| novel_norm (higher = more diverse) | higher at **13/13** |
| gap_closure_ratio (higher = more useful) | higher at **0/13** |
| closure_positive_fraction (higher = more useful) | higher at **0/13** |
| gap_after (lower = better) | worse at **13/13** |

## 3. Hypotheses

### H1 — training insufficient: **INCONCLUSIVE**

The within-arm cross-step numbers below come from **different batches at every logged step**
(the training loop draws a new batch each step), so they are an optimization log, not a
fixed-cohort longitudinal measurement. In addition `max_steps = 100` sits entirely **inside**
`warmup_length = 200`, so the learning rate never leaves its first ~1.5 % of the schedule.
A claim about whether training helps or hurts therefore requires fixed-cohort checkpoint
evaluation; the trajectory alone cannot settle it.

What the training log does show at face value:

* `cos(z_S,z_U)`: 0.9893 → 0.9718 (tau1.0) and 0.9835 → 0.9637 (tau0.5) — moves down.
* `novel_norm`: 0.1398 → 0.2269 (tau1.0) and 0.1737 → 0.2578 (tau0.5) — rises.
* `gap_after` (the optimized term): 0.3734 → 0.2904 (tau1.0) and 0.3739 → 0.2972 (tau0.5)
  — **decreases**, i.e. `L_gap` is being optimized.
* `gap_before` (the detached Said-only reference): 0.3792 → 0.2579 (tau1.0) and
  0.3792 → 0.2577 (tau0.5) — decreases *faster* than `gap_after`.
* `closure ratio`: 0.0158 → −0.1428 (tau1.0) and 0.0144 → −0.1728 (tau0.5).

**The optimized absolute completion distance decreases, but the moving Said-only reference
decreases faster, causing relative closure to deteriorate.** The optimizer is doing its job
on its own objective; the defect is that this objective is not the closure diagnostic.

### H2 — `tau = 1.0` too soft: **NOT SUPPORTED**

The first half of the H2 criterion holds completely: `tau = 0.5` is more diverse than
`tau = 1.0` from step 0 and at **every** logged step (JSD up 13/13, overlap down 13/13,
`cos(z_S,z_U)` down 13/13, `novel_norm` up 13/13). The decisive half fails: the H2
criterion also required `gap closure >= tau=1`, and `tau = 0.5` closes *less* at **0/13**
steps, has a worse `gap_after` at **13/13** steps, and ends at a worse closure ratio
(-0.1728 vs -0.1428). Lower temperature buys diversity and pays for it in usefulness —
exactly the Section 13 counterexample, now inside training rather than only at step 0.

This arm-to-arm comparison is the one longitudinal claim the training log *does* support:
both arms see the **same batch at the same step** (verified in Section 2), the only
difference is the temperature, so the paired comparison is matched rather than confounded.

### H3 — CLIP patch features homogeneous: **INCONCLUSIVE / does not explain the cosine**

* The residual patch set is only *moderately* homogeneous: `patch_pair_cosine_mean`
  0.5886 (std 0.0994), `patch_centered_energy` 0.6348. Those are not "high / low" values.
* The decisive control is `attention_delta`: its patches are *far more* homogeneous
  (`patch_pair_cosine_mean` **0.9539**, `patch_centered_energy` **0.2038**) yet it reaches
  *better* closure at every temperature (0.0032–0.0061 vs 0.0025–0.0068 at equal tau, and
  the whole `attention_delta` column stays positive while residual goes negative at
  tau ≤ 0.25). High patch homogeneity therefore does **not** force high
  `cos(z_S,z_U)` and does not by itself prevent closure.
* What the data does show is a different mechanism: the two attention distributions are
  genuinely different (JSD 0.085–0.268, overlap 0.40–0.67) yet the pooled features stay
  near-parallel (0.98–0.99). With `patch_pair_cosine_mean ≈ 0.59` and comparatively small
  `patch_centered_energy ≈ 0.63`, two *different* convex combinations of the same patch set
  are still ≈0.98 parallel, because the patches share a large common component. The
  collapse therefore happens in the **pooling/geometry**, not because the patches are
  near-duplicates.

Per the brief, no redesign is attempted in this round: `attention_delta`, earlier-layer
patch features and local-projection strategies are left to the next phase.

## 4. Strict separation of the three notions

| notion | evidence | verdict |
|---|---|---|
| **attention diversity** | JSD 0.085 → 0.162 (tau1.0), 0.145 → 0.229 (tau0.5); overlap 0.667 → 0.538 / 0.562 → 0.446 | improves with training and with a stronger tau |
| **feature diversity** | `cos(z_S,z_U)` 0.989 → 0.972 / 0.984 → 0.964; `novel_norm` 0.140 → 0.227 / 0.174 → 0.258 | improves with training and with a stronger tau |
| **useful gap closure** | closure ratio 0.016 → **-0.143** / 0.014 → **-0.173**; positive fraction 0.65 → 0.07 / 0.63 → 0.05 | **degrades** on both axes, and degrades *more* with the stronger tau |

The base objective, as currently specified, reliably produces the first two and reliably
loses the third. Any claim that the base method "forms a semantic complement" is not
supported by this round; the honest statement is that it forms a *different* pooled feature
that does not make the global embedding better explained by `Said + Unsaid`.
