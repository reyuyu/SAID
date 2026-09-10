# Phase 3.0A.1d — Fixed-Cohort Semantic Validation & Loss Attribution

Base objective only: **Said-Conditioned Visual Complement Discovery** on `(I, C_S)`.
No `C_F`, no `C_U` in training, no USS, no Gap core change, no full training. In every
evaluation below the training weights are frozen and no optimizer step is taken.

Artifacts (small, tracked) in this directory:

| file | content |
|---|---|
| `fixed_cohort_gap_summary.json` | internal gap trajectory on the frozen cohort |
| `fixed_cohort_usr_summary.json` | target-independent Phase 3 USR per checkpoint |
| `loss_attribution_summary.json` | A / B / C / D comparison and stream identity |
| `canonical_retrieval_summary.json` | ShareGPT4V-1K x3 variants and COCO val2017 |

Raw per-arm evaluation JSONs live in `outputs/phase30a_fixed_semantic_eval/`
(checkpoints, score matrices and raw run logs are not committed).

## 0. Why this round exists

Phase 1c's H1 verdict was **corrected to INCONCLUSIVE**. Two defects made its longitudinal
claim unusable:

* the training log reports a **different batch at every step**, so step-to-step change is an
  optimization log, not a fixed-cohort measurement;
* `max_steps = 100` sits entirely **inside** `warmup_length = 200`, so the LR never leaves
  its first ~1.5 %.

Everything below is therefore measured on **one frozen cohort** evaluated for every
checkpoint: same images, same `C_S`, same `C_U` targets, same candidate pool.

## 1. Frozen cohort

| quantity | value |
|---|---|
| protocol | `sharegpt4v1k-usr-v1` (frozen, reused; never re-sampled) |
| manifest sha256 | `ba2298ad836be0d9…` |
| queries `Q` | **868** |
| candidate pool | **868** (always the full pool) |
| cohort identity sha256 | `20c06542df13c309…` |
| random chance | R@1 = 0.001152, R@5 = 0.005760, R@10 = 0.011521 |

Every checkpoint in every arm is evaluated against this identical cohort.

## 2. Fixed-cohort internal gap trajectory (Q1)

`A` = `L_S` only, `B` = `L_S + L_gap`, `C` = `L_S + L_gap + L_absorb`, all at tau = 1.0 and
identical streams; `D` = `C` at tau = 0.5.

| metric | arm | initial | step20 | step50 | step100 |
|---|---|---|---|---|---|
| `gap_before_mean` | A | 0.369116 | 0.365648 | 0.351529 | 0.333956 |
| | B | 0.369116 | 0.364661 | 0.345066 | 0.307272 |
| | C | 0.369116 | 0.360961 | 0.322807 | 0.231664 |
| | D | 0.369116 | 0.360924 | 0.322665 | 0.231665 |
| `gap_after_mean` | A | 0.365496 | 0.366770 | 0.371582 | **0.381952** |
| | B | 0.365496 | 0.365823 | 0.364957 | **0.348446** |
| | C | 0.365496 | 0.361966 | 0.341230 | **0.266045** |
| | D | 0.366370 | 0.363849 | 0.346612 | **0.272954** |
| `gap_reduction_mean` | A | +0.003620 | −0.001122 | −0.020053 | −0.047996 |
| | B | +0.003620 | −0.001162 | −0.019891 | −0.041173 |
| | C | +0.003620 | −0.001005 | −0.018422 | −0.034382 |
| | D | +0.002746 | −0.002925 | −0.023947 | −0.041289 |
| `gap_closure_ratio_mean` | A | +0.009460 | −0.003556 | −0.059122 | −0.154535 |
| | B | +0.009460 | −0.003678 | −0.059778 | −0.145074 |
| | C | +0.009460 | −0.003286 | −0.059416 | −0.161686 |
| | D | +0.006884 | −0.008875 | −0.077248 | −0.193045 |
| `gap_closure_positive_fraction` | A | 0.595622 | 0.455069 | 0.072581 | 0.041475 |
| | B | 0.595622 | 0.451613 | 0.074885 | 0.061060 |
| | C | 0.595622 | 0.461982 | 0.084101 | 0.046083 |
| | D | 0.557604 | 0.419355 | 0.069124 | 0.034562 |
| `cos_said_unsaid` | A | 0.991055 | 0.991130 | 0.987781 | 0.963336 |
| | B | 0.991055 | 0.991134 | 0.987882 | 0.967140 |
| | C | 0.991055 | 0.991181 | 0.988254 | 0.970103 |
| | D | 0.986223 | 0.986361 | 0.982332 | 0.962063 |
| `unsaid_novel_component_norm` | A | 0.129301 | 0.128800 | 0.149425 | 0.256593 |
| | B | 0.129301 | 0.128778 | 0.148916 | 0.243790 |
| | C | 0.129301 | 0.128431 | 0.146586 | 0.232682 |
| | D | 0.160591 | 0.159774 | 0.179924 | 0.262971 |
| `said_unsaid_attention_jsd` | A | 0.084292 | 0.084396 | 0.096070 | 0.176490 |
| | B | 0.084292 | 0.084410 | 0.095941 | 0.170997 |
| | C | 0.084292 | 0.084304 | 0.095382 | 0.170813 |
| | D | 0.144327 | 0.144439 | 0.157276 | 0.238227 |
| `said_unsaid_attention_overlap` | A | 0.668115 | 0.667768 | 0.645808 | 0.517368 |
| | B | 0.668115 | 0.667742 | 0.645914 | 0.524978 |
| | C | 0.668115 | 0.667954 | 0.646967 | 0.525016 |
| | D | 0.564557 | 0.564277 | 0.545273 | 0.434343 |
| `cos_global_said` | A | 0.630884 | 0.634352 | 0.648471 | 0.666044 |
| | B | 0.630884 | 0.635339 | 0.654934 | 0.692728 |
| | C | 0.630884 | 0.639039 | 0.677193 | **0.768336** |
| | D | 0.630884 | 0.639076 | 0.677335 | **0.768335** |
| `cos_global_unsaid` | A | 0.634482 | 0.633154 | 0.627966 | 0.614817 |
| | B | 0.634482 | 0.634100 | 0.634599 | 0.649015 |
| | C | 0.634482 | 0.637959 | 0.658360 | **0.731857** |
| | D | 0.633532 | 0.635953 | 0.652570 | 0.723866 |
| `patch_pair_cosine_mean` | A | 0.594486 | 0.592446 | 0.580773 | 0.526191 |
| | C | 0.594486 | 0.593299 | 0.586696 | 0.556129 |
| `patch_centered_energy` | A | 0.630509 | 0.632142 | 0.641411 | 0.682971 |
| | C | 0.630509 | 0.631469 | 0.636787 | 0.660642 |

**Q1 answer.** On the fixed cohort, relative closure **degrades** in every arm, including
the Said-only control that never sees `L_gap` or `L_absorb` at all (A: +0.0095 → −0.1545).
The gap losses do reduce the *absolute* completion distance much faster than the control
(`gap_after` at step100: A 0.3820, B 0.3484, C 0.2660), i.e. they optimize their own term
successfully — but `gap_before` falls even faster (A 0.3340, C 0.2317), so the ratio gets
worse. `L_absorb` is clearly active on its own target: `cos_global_said` reaches 0.768 in C
vs 0.666 in A (and `cos_global_unsaid` 0.732 vs 0.615).

## 3. Target-independent USR (Q2, Q3)

`z_U` is precomputed from `(I, C_S)` **before any candidate exists**; the score matrix is
then the plain inner product over the full 868-candidate pool. `C_U` appears only as
candidate. `complete` = `normalize(s_ref + u_new)`.

`z_unsaid` scorer (does `z_U` contain withheld semantics?):

| arm | step | R@1 | R@5 | R@10 | MRR | mean_rank | median_rank |
|---|---|---|---|---|---|---|---|
| A (L_S) | initial | 0.0588 | 0.1256 | 0.1728 | 0.1025 | 189.9 | 126.0 |
| A | step20 | 0.0553 | 0.1187 | 0.1636 | 0.0974 | 192.1 | 131.0 |
| A | step50 | 0.0472 | 0.0933 | 0.1302 | 0.0803 | 203.0 | 148.0 |
| A | step100 | **0.0380** | 0.0726 | 0.0899 | 0.0629 | 214.4 | 166.0 |
| B (L_S+L_gap) | step100 | **0.0449** | 0.0829 | 0.1071 | 0.0722 | 207.6 | 159.0 |
| C (L_S+L_gap+L_abs) | step100 | **0.0472** | 0.0806 | 0.1060 | 0.0723 | 207.7 | 159.0 |
| D (C, tau 0.5) | step100 | 0.0449 | 0.0783 | 0.0956 | 0.0696 | 213.0 | 162.0 |

`global` scorer (does the CLS carry withheld semantics?):

| arm | initial | step20 | step50 | step100 |
|---|---|---|---|---|
| A | 0.2650 / 0.3514 | 0.2661 / 0.3521 | 0.2661 / 0.3564 | **0.2719 / 0.3634** |
| B | 0.2650 / 0.3514 | 0.2650 / 0.3514 | 0.2673 / 0.3565 | 0.2684 / 0.3592 |
| C | 0.2650 / 0.3514 | 0.2650 / 0.3516 | 0.2673 / 0.3576 | **0.2742 / 0.3610** |
| D | 0.2650 / 0.3514 | 0.2661 / 0.3522 | 0.2684 / 0.3580 | 0.2719 / 0.3592 |

(cells are `R@1 / MRR`)

`complete` scorer at step100: A 0.0403, B 0.0449, C 0.0449, D 0.0461 (R@1) — i.e. adding
`u_new` to `z_S` does **not** beat `z_U` alone, so the composition adds nothing over the
complement itself.

**Q2 answer.** A genuine but weak withheld-semantic signal exists in `z_U` at
initialisation: R@1 = 0.0588 is **51x** the 0.001152 chance rate. A pure Said-only control
then **destroys** it (0.0588 → 0.0380, MRR 0.1025 → 0.0629). Adding `L_gap_discover`
recovers and exceeds the initial value (B 0.0449, C 0.0472 at step100, both above A's
0.0380 and close to the frozen initial 0.0588).

**Q3 answer.** The CLS keeps its withheld-semantic content and slightly improves in every
arm (0.2650 → 0.2684…0.2742 R@1), so the base objective does not damage the global
representation — but `z_U` is still ~6x worse than the CLS on the same queries, and the
`L_absorb` arm's global gain over the control (+0.0023 R@1) is well inside the noise of a
868-query cohort (±0.015 at 1σ).

## 4. Loss attribution (Q4)

Stream identity is **fully matched** (verified):

| arm | `initial_state_sha256` | `sampler_order_sha256` | `caption_stream_sha256` | full | unsaid |
|---|---|---|---|---|---|
| A | `4621b8d027f4…` | `953402b3c3d7…` | `45f3b25bdc52…` | null | null |
| B | `4621b8d027f4…` | `953402b3c3d7…` | `45f3b25bdc52…` | null | null |
| C | `4621b8d027f4…` | `953402b3c3d7…` | `45f3b25bdc52…` | null | null |

Per-step `batch_image_sha256` / `batch_caption_sha256` are identical across arms at every
logged step (checked for A/B/C at steps 0/10/…/90). Causal attribution is therefore legal.

| step | arm | `z_U` USR R@1 / MRR | global USR R@1 / MRR | internal closure |
|---|---|---|---|---|
| initial | A / B / C | 0.0588 / 0.1025 | 0.2650 / 0.3514 | +0.009460 |
| step20 | A | 0.0553 / 0.0974 | 0.2661 / 0.3521 | −0.003556 |
| step20 | B | 0.0553 / 0.0975 | 0.2650 / 0.3514 | −0.003678 |
| step20 | C | 0.0553 / 0.0975 | 0.2650 / 0.3516 | −0.003286 |
| step50 | A | 0.0472 / 0.0803 | 0.2661 / 0.3564 | −0.059122 |
| step50 | B | 0.0484 / 0.0812 | 0.2673 / 0.3565 | −0.059778 |
| step50 | C | 0.0472 / 0.0811 | 0.2673 / 0.3576 | −0.059416 |
| step100 | A | 0.0380 / 0.0629 | 0.2719 / 0.3634 | −0.154535 |
| step100 | B | 0.0449 / 0.0722 | 0.2684 / 0.3592 | −0.145074 |
| step100 | C | 0.0472 / 0.0723 | 0.2742 / 0.3610 | −0.161686 |
| step100 | D (tau 0.5) | 0.0449 / 0.0696 | 0.2719 / 0.3592 | −0.193045 |

**Q4 answer.**

* `B > A` on `z_U` withheld-semantic retrieval (+0.0069 R@1, +0.0093 MRR at step100): the
  Discovery term does deliver a real, measurable benefit over the pure Said-only control.
* `C > B` on `global` withheld-semantic retrieval (+0.0058 R@1, +0.0018 MRR) and on
  `cos_global_said` (0.693 vs 0.768): the Absorption term is what moves the CLS toward the
  completion, and its global-retrieval gain is the largest of the three arms — but it is
  small relative to the 868-query noise, so this is a directional result, not an established
  effect.
* Neither term rescues the internal closure metric: closure is negative in **all** arms
  including the control.

## 5. Canonical retrieval (Q6)

Standard CLIP CLS retrieval via `encode_image()`. Note the base objective contains **no
Global-text InfoNCE**, so the question is only whether it damages canonical retrieval.

ShareGPT4V-1K `image2text_R1` / `text2image_R1`:

| arm | step | first_sentence | fixed_sparse | full_dense |
|---|---|---|---|---|
| A | initial | 0.544 / 0.514 | 0.746 / 0.740 | 0.758 / 0.776 |
| A | step100 | 0.561 / 0.547 | 0.818 / 0.820 | 0.886 / 0.884 |
| B | step100 | 0.550 / 0.544 | 0.823 / 0.817 | 0.883 / 0.886 |
| C | step100 | 0.566 / 0.553 | 0.815 / 0.810 | 0.882 / 0.872 |
| D | step100 | 0.561 / 0.557 | 0.811 / 0.808 | 0.884 / 0.872 |

COCO val2017 (5-caption protocol):

| arm | step | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 |
|---|---|---|---|---|---|---|---|
| A | initial | 0.5170 | 0.7662 | 0.8428 | 0.3269 | 0.5776 | 0.6823 |
| A | step100 | 0.5360 | 0.7870 | 0.8622 | 0.3448 | 0.5944 | 0.6996 |
| B | step100 | 0.5340 | 0.7836 | 0.8600 | 0.3422 | 0.5926 | 0.6974 |
| C | step100 | 0.5344 | 0.7828 | 0.8596 | 0.3495 | 0.5993 | 0.7050 |
| D | step100 | 0.5322 | 0.7814 | 0.8594 | 0.3502 | 0.5996 | 0.7055 |

All four arms start from the identical checkpoint (every `initial` row is byte-identical).

Canonical retrieval **improves** over the 100 steps in every arm, on every 1K variant and on
COCO: 1K `first_sentence` I2T R@1 0.544 → 0.550–0.566, `fixed_sparse` 0.746 → 0.811–0.823,
`full_dense` 0.758 → 0.882–0.886; COCO I2T R@1 0.5170 → 0.5322–0.5360 and T2I R@1
0.3269 → 0.3422–0.3502. The four arms are within ~0.01 of each other everywhere, and the
`A` control is not worse than the arms that carry the gap losses. The base objective, which
contains **no Global-text InfoNCE at all**, therefore does not break standard CLIP retrieval.

## 6. Patch common mode (Q5)

Because attention weights sum to 1, `p_S = mu + delta_S` and `p_U = mu + delta_U` with
`mu = mean_p h_p`; the shared centroid therefore enters both convex combinations and the
centred deviations are the only parts that can carry a difference.

| metric | arm | initial | step20 | step50 | step100 |
|---|---|---|---|---|---|
| `raw_pool_cosine` | A | 0.991055 | 0.991129 | 0.987781 | 0.963336 |
| | B | 0.991055 | 0.991134 | 0.987882 | 0.967140 |
| | C | 0.991055 | 0.991181 | 0.988254 | 0.970103 |
| | D | 0.986223 | 0.986361 | 0.982332 | 0.962063 |
| `centered_pool_cosine` | A | −0.982105 | −0.982667 | −0.980765 | **−0.925009** |
| | B | −0.982105 | −0.982652 | −0.980915 | **−0.926361** |
| | C | −0.982105 | −0.982686 | −0.981137 | **−0.931089** |
| | D | −0.973897 | −0.974809 | −0.973840 | **−0.903606** |
| `common_mode_ratio` | A | 0.772551 | 0.771208 | 0.763514 | 0.726668 |
| | C | 0.772551 | 0.771770 | 0.767444 | 0.747350 |
| `common_mode_fraction_said` | A | 1.007503 | 1.006687 | 1.001901 | 0.969519 |
| | C | 1.007503 | 1.006609 | 1.001111 | 0.968834 |
| `common_mode_fraction_unsaid` | A | 0.988975 | 0.989796 | 0.993328 | 0.998010 |
| | C | 0.988975 | 0.989897 | 0.994174 | **1.001740** |
| `patch_centroid_norm` | A | 5.578143 | 5.557345 | 5.447015 | 4.983030 |
| | C | 5.578143 | 5.569482 | 5.528144 | 5.357076 |
| `centered_said_norm` | A | 0.359404 | 0.355158 | 0.434375 | **0.937125** |
| | C | 0.359404 | 0.354665 | 0.430479 | **0.901360** |
| `centered_unsaid_norm` | A | 0.380199 | 0.377261 | 0.396678 | **0.428815** |
| | C | 0.380199 | 0.377163 | 0.396588 | **0.429740** |
| `said_raw_pool_norm` | A | 5.537613 | 5.521386 | 5.437736 | 5.146204 |
| | C | 5.537613 | 5.533868 | 5.523110 | 5.535686 |
| `unsaid_raw_pool_norm` | A | 5.640216 | 5.614505 | 5.483744 | 4.993910 |
| | C | 5.640216 | 5.626190 | 5.560655 | 5.348605 |

`common_mode_diagnosis` at step100 is `{"raw_pool_cosine": 0.9633, "centered_pool_cosine":
-0.9250, "drop_raw_minus_centered": 1.8883, "centered_is_materially_lower": true,
"common_mode_dominates_supported": true}` for A, and the same shape for B, C and D.

**Q5 answer — the strongest mechanism finding of this round.** The two *centred* pools are
strongly **anti-aligned** (`centered_pool_cosine` −0.90 … −0.98) while the *raw* pools are
almost identical (0.962 … 0.991). The decomposition explains it exactly: since
`||mu|| ≈ 5.0–5.6` and `||delta|| ≈ 0.36–0.94`, the dominant centroid term cancels the
opposing deviation term in the sum, so the raw cosine is high *because of* the cancellation,
not because the poolings are the same. Formally,

    cos(p_S, p_U) = (||mu||^2 + mu·(d_S+d_U) + d_S·d_U) / (||p_S|| ||p_U||)

with `mu·(d_S+d_U) = ||mu|| ||d_S|| cos(theta_S) + ||mu|| ||d_U|| cos(theta_U)` strongly
negative, which offsets the positive `||mu||^2`.

Two additional consequences are visible in the table:

* `common_mode_fraction_unsaid` is ≈ 0.99–1.00 in every arm and every checkpoint, i.e.
  `p_U` is essentially the patch centroid: the anti-Said pooling is almost a **plain mean
  pool** of the patches, whereas `common_mode_fraction_said` drifts from 1.008 down to
  0.969 as `z_S` becomes genuinely differentiated.
* `centered_said_norm` grows 0.359 → 0.90–0.94 over training while `centered_unsaid_norm`
  stays flat at 0.38 → 0.43. The Said pooling moves away from the mean patch far more than
  the Unsaid pooling does, which is consistent with `z_U` carrying less withheld-semantic
  content than the CLS.

This is direct mechanism evidence and it supersedes the indirect `patch_pair_cosine_mean`
argument: `patch_pair_cosine_mean` is only 0.526–0.556 with `patch_centered_energy`
0.660–0.683, so the patch set is *not* near-degenerate, yet the poolings stay ≈0.96–0.99
parallel purely because of the shared centroid.

## 7. Revised hypotheses

**H1 — training insufficient: INCONCLUSIVE (unchanged, now by design).** The 1c verdict was
corrected to INCONCLUSIVE because its evidence was batch-varying and warmup-clipped. This
round supplies the fixed-cohort measurement that replaces it, and the fixed-cohort result is
*not* "training is insufficient": representational diversity rises (attention JSD 0.084 →
0.170–0.238, `cos(z_S,z_U)` 0.991 → 0.962–0.970) and withheld-semantic retrieval in `z_U`
is **recovered** by the discovery term relative to the control. What fails is specifically
relative closure. Since 100 steps still sit inside a 200-step warmup, and since the
mechanism's benefit is small compared with cohort noise, H1 stays INCONCLUSIVE rather than
being flipped to SUPPORTED or NOT SUPPORTED.

**H2 — `tau = 1.0` too soft: NOT SUPPORTED.** Fixed cohort, C (tau 1.0) vs D (tau 0.5) at
step100: D has higher attention JSD (0.2382 vs 0.1708), lower overlap (0.4343 vs 0.5250) and
a lower raw pool cosine (0.9621 vs 0.9701) — but D is *worse* on `z_U` withheld-semantic
retrieval (R@1 0.0449 vs 0.0472, MRR 0.0696 vs 0.0723) and on closure (−0.1930 vs −0.1617).
Lower temperature again buys diversity and pays in usefulness.

**H3 — patch features homogeneous: NOT SUPPORTED as the mechanism; replaced by the
common-mode explanation.** On the fixed cohort the residual patch set is only moderately
homogeneous (`patch_pair_cosine_mean` 0.526–0.556, `patch_centered_energy` 0.660–0.683), and
the earlier `attention_delta` control was *far more* homogeneous yet closed better. The
near-parallel pools are explained by the shared patch centroid (common mode), not by
near-duplicate patches.

## 8. Scientific verdict

**C. Complement differs geometrically but not semantically.**
**D. Internal gap metric is a poor proxy for withheld semantics.**

Evidence for **C**: the complement is geometrically real — attention JSD rises from 0.084 to
0.171–0.238, overlap falls from 0.668 to 0.434–0.525, `cos(z_S,z_U)` falls from 0.991 to
0.962–0.970, and the centred pools are strongly anti-aligned (`centered_pool_cosine`
−0.90 … −0.98) even though the raw pools stay at 0.962–0.991. Semantically the gain is
thin: `z_U` reaches only R@1 0.0449–0.0472 at step100 against a CLS baseline of 0.2684–0.2742
on the very same queries and candidate pool, i.e. ~6x worse, and the `complete`
(`z_S + u_new`) scorer does not beat `z_U` alone.

Evidence for **D**: on the fixed cohort the internal closure ratio is negative in **every**
arm, including the Said-only control that never optimizes it, and it becomes *more* negative
precisely in the arms where `z_U` withheld-semantic retrieval is *better* (A −0.1545 /
R@1 0.0380; B −0.1451 / 0.0449; C −0.1617 / 0.0472). A metric that moves opposite to the
semantic measurement on the same cohort cannot be used as a proxy for it.

**A (Discovery + Absorption both supported) is NOT claimed.** The directional pattern is
right — `B > A` on `z_U` retrieval, `C > B` on global retrieval and on `cos_global_said` —
but the effect sizes (+0.0069 and +0.0058 R@1 on 868 queries, ±0.015 at 1σ) are inside
cohort noise, and no arm separates from the others on the internal metric. **B (Discovery
supported, Absorption not supported) is not claimed either**, for the same noise reason.
**E (Inconclusive)** is the honest qualifier on the *attribution sizes*; the two selected
verdicts are the ones the fixed-cohort evidence actually establishes.

## 9. Note on the gap objective's maths (Section 12)

The objective optimises `L_gap = gap_after` while the diagnostic of interest is
`closure = (gap_before − gap_after) / (gap_before + eps)`. Because `gap_before` is detached,
simply rewriting the loss as `L = gap_after − gap_before` would **not** change the direct
gradient:

    grad(gap_after − gap_before) = grad(gap_after)

since `gap_before` carries no gradient. That rewrite is therefore **not** a fix and must not
be presented as one. What the fixed cohort shows is that both quantities decrease — the
optimized term and the detached reference — with the reference falling faster, so the ratio
deteriorates. Changing that requires a different objective, which this phase deliberately
does not attempt (no centred-pooling loss, no earlier layer, no new projection, no new
residual target, no relative gap loss, no USS).
