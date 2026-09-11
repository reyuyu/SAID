# Phase 3.0A.1e — Common-Mode Semantic Decomposition

Answers where the withheld-semantic ability of `z_U` actually comes from: the shared patch
centroid, the centred anti-Said deviation, or the relative Said-versus-Anti contrast.

**Read-only.** No training, no new loss, no Gap core change, no USS, no `C_U` in any feature
construction. `optimizer_steps = 0`: every number below comes from existing checkpoints
evaluated on one frozen cohort.

Artifacts (small, tracked) in this directory:

| file | content |
|---|---|
| `semantic_decomposition_summary.json` | 8 scorers x 5 checkpoints on the frozen cohort |
| `paired_statistics.json` | deterministic paired bootstrap + exact McNemar |
| `proxy_correlation.json` | per-query Spearman proxy validation |
| `phase30a_1e_report.md` | this report |

## 1. Setup

| quantity | value |
|---|---|
| cohort | frozen `sharegpt4v1k-usr-v1`, **Q = 868**, candidate pool **868** |
| cohort sha256 | `20c06542df13c309…` |
| chance R@1 | 0.001152 |
| checkpoints | `initial`, `A100` (L_S), `B100` (L_S+L_gap), `C100` (L_S+L_gap+L_absorb), `D100` (C at tau 0.5) |

Every scorer feature is precomputed from `(I, C_S)` **before** the candidate pool is touched
(`decomposition_features` and `precompute_query_features` take no candidate argument), and
each score matrix is the plain inner product over the full 868-candidate pool. `C_U` appears
only as the candidate/target text. The eight scorers, all from one shared forward pass:

| scorer | definition |
|---|---|
| `global` | `normalize(g)` — the CLS baseline |
| `said_raw` | `normalize(p_S)`, `p_S = sum_p A_S,p h_p` |
| `unsaid_raw` | `normalize(p_U)` — the current `z_U` |
| `centroid` | `normalize(mu)`, `mu = mean_p h_p` — the common mode alone |
| `said_centered` | `normalize(delta_S)`, `delta_S = sum_p A_S,p (h_p − mu)` |
| `unsaid_centered` | `normalize(delta_U)` |
| `anti_minus_said` | `normalize(delta_U − delta_S) = normalize(p_U − p_S)` — the contrast |
| `complete_existing` | `normalize(s_ref + u_new)` — the existing completion |

## 2. Overview: does the centroid explain `z_U`? (Q1 → verdict A)

`R@1` / `MRR` per scorer and checkpoint:

| scorer | initial | A100 | B100 | C100 | D100 |
|---|---|---|---|---|---|
| `global` | 0.2650 / 0.3514 | 0.2719 / 0.3634 | 0.2684 / 0.3592 | 0.2742 / 0.3610 | 0.2719 / 0.3592 |
| `said_raw` | 0.0634 / 0.1095 | 0.0530 / 0.0820 | 0.0553 / 0.0900 | 0.0530 / 0.0878 | 0.0565 / 0.0891 |
| `unsaid_raw` | 0.0588 / 0.1025 | 0.0380 / 0.0629 | 0.0449 / 0.0722 | 0.0472 / 0.0723 | 0.0461 / 0.0726 |
| `centroid` | 0.0565 / 0.1041 | 0.0392 / 0.0658 | 0.0415 / 0.0742 | 0.0449 / 0.0750 | 0.0461 / 0.0767 |
| `said_centered` | 0.0069 / 0.0234 | 0.0392 / 0.0796 | 0.0380 / 0.0753 | 0.0357 / 0.0747 | 0.0357 / 0.0734 |
| `unsaid_centered` | 0.0023 / 0.0131 | 0.0012 / 0.0042 | 0.0012 / 0.0042 | 0.0012 / 0.0037 | 0.0012 / 0.0039 |
| `anti_minus_said` | 0.0046 / 0.0147 | 0.0000 / 0.0036 | 0.0000 / 0.0036 | 0.0012 / 0.0038 | 0.0012 / 0.0040 |
| `complete_existing` | 0.0588 / 0.1023 | 0.0403 / 0.0642 | 0.0449 / 0.0723 | 0.0449 / 0.0710 | 0.0472 / 0.0734 |

`unsaid_raw` and `centroid` track each other within **0.0035 R@1 at every checkpoint** and
**0.0000 at D100** (centroid as a share of `unsaid_raw` R@1: 0.96 at `initial`, 1.03 at
`A100`, 0.92 at `B100`, 0.95 at `C100`, 1.00 at `D100`). The `unsaid_raw` MRR ordering
matches the `centroid` MRR ordering exactly at all five checkpoints, while `cos_said_unsaid`
is 0.96–0.99. Nearly all of the measurable withheld-semantic signal in the raw `z_U` is
therefore already present in the plain patch mean.

Per-query rank statistics for `unsaid_raw` (mean / median rank out of 868):

| checkpoint | mean rank | median rank |
|---|---|---|
| initial | 189.9 | 126 |
| A100 | 214.4 | 166 |
| B100 | 207.6 | 160 |
| C100 | 207.7 | 160 |
| D100 | 206.8 | 156 |

## 3. The centred and contrast branches (Q2, Q3, Q4 → verdicts B and C)

At `C100`, the full report for every scorer:

| scorer | R@1 | R@5 | R@10 | MRR | mean rank | median rank | pos−mean_neg | pos−best_neg |
|---|---|---|---|---|---|---|---|---|
| `global` | 0.2742 | 0.4516 | 0.5300 | 0.3610 | 72.8 | 8 | +8.0697 | −2.6402 |
| `said_raw` | 0.0530 | 0.1106 | 0.1417 | 0.0878 | 174.6 | 133 | +5.0988 | −5.0524 |
| `unsaid_raw` | 0.0472 | 0.0806 | 0.1060 | 0.0723 | 207.7 | 159 | +4.3133 | −5.5741 |
| `centroid` | 0.0449 | 0.0899 | 0.1129 | 0.0750 | 195.4 | 149 | +4.6296 | −5.3771 |
| `said_centered` | 0.0357 | 0.1025 | 0.1463 | 0.0747 | 234.6 | 147 | +3.6613 | −7.6526 |
| `unsaid_centered` | 0.0012 | 0.0012 | 0.0035 | 0.0037 | 646.6 | 737 | **−3.8848** | **−17.2325** |
| `anti_minus_said` | 0.0012 | 0.0012 | 0.0035 | 0.0038 | 641.0 | 726 | **−3.7735** | **−17.7007** |
| `complete_existing` | 0.0449 | 0.0818 | 0.1048 | 0.0710 | 206.5 | 159 | +4.3446 | −5.5471 |

Both centred/contrast branches are **at chance** on `R@1` (0.0012 vs chance 0.00115), have
mean ranks of ~641–647 out of 868 (i.e. **below** chance level), and — decisively — their
per-query semantic margins are **negative** (−3.88 and −3.77). A scorer that ranks the true
withheld suffix *below* a random candidate carries no withheld semantics; the centred
deviations do not merely fail to help, they point the wrong way.

`said_centered` (0.0357) is also **higher** than `unsaid_centered` (0.0012) at every
checkpoint, so the anti-Said deviation is not more withheld-semantics-aligned than the Said
deviation — the opposite of the Q4 hypothesis.

Note also that `said_raw` beats `unsaid_raw` at **every** checkpoint (0.0634 vs 0.0588 at
initial, 0.0530 vs 0.0472 at C100), so even the *positive* prefix pooling retrieves the
withheld suffix better than the anti-Said pooling does.

## 4. Paired statistics (Section 6)

Deterministic paired bootstrap over query indices (10 000 replicates, seed 20260911, one
shared resample applied to both methods), percentile 95% CI, plus exact two-sided McNemar on
the `R@1` hits. Independent binomial error bars are not used.

| comparison | ΔR@1 | ΔMRR | MRR 95% CI | excludes 0 | n10/n01 | McNemar p |
|---|---|---|---|---|---|---|
| `B100 − A100`, `unsaid_raw` | **+0.0069** | **+0.0092** | [+0.0052, +0.0140] | **yes** | 6/0 | 0.031 |
| `C100 − A100`, `unsaid_raw` | **+0.0092** | **+0.0094** | [+0.0052, +0.0143] | **yes** | 8/0 | 0.008 |
| `C100 − B100`, `unsaid_raw` | +0.0023 | +0.0002 | [−0.0018, +0.0024] | no | 2/0 | 0.500 |
| `C100 − A100`, `global` | +0.0023 | −0.0024 | [−0.0100, +0.0054] | no | 14/12 | 0.845 |
| `C100 − B100`, `global` | +0.0058 | +0.0018 | [−0.0058, +0.0097] | no | 18/13 | 0.473 |
| `D100 − C100`, `unsaid_raw` | −0.0012 | +0.0003 | [−0.0024, +0.0020] | no | 0/1 | 1.000 |
| `unsaid_centered − centroid` (C100) | **−0.0438** | **−0.0713** | [−0.0866, −0.0569] | **yes** | 1/39 | 0.000 |
| `unsaid_centered − said_centered` (C100) | **−0.0346** | **−0.0710** | [−0.0844, −0.0578] | **yes** | 1/31 | 0.000 |
| `anti_minus_said − unsaid_raw` (C100) | **−0.0461** | **−0.0686** | [−0.0839, −0.0540] | **yes** | 1/41 | 0.000 |
| `initial − C100`, `unsaid_raw` | +0.0115 | **+0.0301** | [+0.0172, +0.0433] | **yes** | 26/16 | 0.164 |

Reading it:

* **Discovery is now established.** `B100 − A100` and `C100 − A100` are positive with
  intervals excluding zero on both `R@1` and `MRR`; the McNemar tests agree (p = 0.031 and
  0.008). Gap Discovery **partially preserves / recovers** the withheld-semantic
  retrievability that Said-only optimisation degrades.
* **Absorption is not established.** `C100 − B100` is +0.0058 R@1 / +0.0018 MRR on `global`
  with an interval containing zero (p = 0.473). The point estimate is in the expected
  direction; the evidence is not there.
* **The centred and contrast branches are rejected**, all three with intervals strictly below
  zero and p = 0.000.
* **The initial checkpoint is still ahead.** `initial − C100` on `unsaid_raw` is +0.0115 R@1
  (CI contains 0, p = 0.164) and +0.0301 MRR with a CI excluding zero — so on `MRR` the
  trained full objective has **not** recovered the frozen initial value.
* Lowering the anti-temperature does not help: `D100 − C100` is −0.0012 R@1 (no difference).

## 5. Proxy validation (Section 7 → verdict F)

Spearman correlations between the internal per-query numbers and the withheld-semantic
outcome for `unsaid_raw` (n = 868; ρ and p):

| correlation | initial | A100 | B100 | C100 | D100 |
|---|---|---|---|---|---|
| `closure` vs `−rank` | +0.0045 (0.894) | +0.0749 (0.027) | +0.0664 (0.050) | +0.0829 (0.014) | +0.0776 (0.022) |
| `closure` vs `margin_mean` | +0.0085 (0.804) | +0.0880 (0.009) | +0.0781 (0.021) | +0.0968 (0.004) | +0.0914 (0.007) |
| `closure` vs `margin_best` | −0.0096 (0.778) | +0.1085 (0.001) | +0.1018 (0.003) | +0.1200 (0.000) | +0.1166 (0.001) |
| `gap_reduction` vs `−rank` | +0.0002 (0.994) | +0.1006 (0.003) | +0.0874 (0.010) | +0.0984 (0.004) | +0.0917 (0.007) |
| `gap_after` vs `rank` | +0.1506 (0.000) | +0.1512 (0.000) | +0.1244 (0.000) | +0.0875 (0.010) | +0.0886 (0.009) |
| `gap_after` vs `margin_mean` | −0.1662 (0.000) | −0.1512 (0.000) | −0.1189 (0.000) | −0.0783 (0.021) | −0.0782 (0.021) |

Two facts have to be reported together, because they point in opposite directions:

1. **Within a trained checkpoint the direction is right and significant.** From `A100`
   onwards, `closure` correlates positively with `−rank` and with the semantic margin
   (ρ ≈ +0.07 … +0.12, p ≤ 0.05), and `gap_after` correlates positively with `rank`
   (ρ ≈ +0.09 … +0.15, p ≤ 0.01). The parenthetical claim that the internal metric is
   *uncorrelated* with semantics is not what the data show.
2. **The effect is negligible and it reverses across checkpoints.** ρ² ≤ 0.015, i.e. the
   internal numbers explain at most ~1.5 % of the rank variance. Worse, the cross-checkpoint
   ordering is inverted: `initial` has the *highest* closure (+0.0095) and the *worst*
   global R@1 (0.2650), while every trained arm has a strongly negative closure
   (−0.145 … −0.162) and a *better* global R@1 (0.2684 … 0.2742). `initial` is also the only
   checkpoint where the correlations are essentially zero (ρ ≈ 0.005, p > 0.77).

So the correct statement is narrower than 1d's: internal closure is **not uncorrelated** with
withheld-semantic quality, but it is **unusable as a proxy** — a ≤1.5 % within-checkpoint
effect that reverses sign between checkpoints. Since every attribution question in this phase
is a *between-checkpoint* question, the reversal is fatal.

## 6. Common-mode interpretation (Section 8)

The centred pool cosine is ≈ −1 (initial −0.982, C100 −0.931) but this **must not** be read as
"a semantic complement has formed": `A_U` is a decreasing function of the Said scores, so the
two centred deviations are structurally pushed to oppose each other. What the number does
explain is *why two different attention distributions can still produce near-parallel raw
pools*:

| metric | initial | A100 | B100 | C100 | D100 |
|---|---|---|---|---|---|
| `raw_pool_cosine` | 0.991055 | 0.963336 | 0.967140 | 0.970103 | 0.970890 |
| `centered_pool_cosine` | −0.982105 | −0.925009 | −0.926361 | −0.931089 | −0.928025 |
| `common_mode_norm_ratio_said` | 1.008131 | 1.000906 | 1.000722 | 1.001271 | 1.001196 |
| `common_mode_norm_ratio_unsaid` | 0.987757 | 0.991178 | 0.990642 | 0.991087 | 0.983829 |
| `centered_said_norm` | 0.359404 | 0.937125 | 0.889103 | 0.901360 | 0.886552 |
| `centered_unsaid_norm` | 0.380199 | 0.428815 | 0.421737 | 0.429740 | 0.426675 |
| `patch_centroid_norm` | 5.578143 | 4.983030 | 5.027105 | 5.357076 | 5.363182 |

(96-image probe with a fixed text input, one pass per checkpoint; the deprecated
`common_mode_fraction_*` aliases equal the `norm_ratio` values exactly, which is why the
frozen-cohort values in `semantic_decomposition_summary.json` match.)

Both ratios sit at ≈1.00, i.e. `||mu|| ≈ ||p||` for both poolings: the centroid magnitude and
the pooled magnitude nearly cancel, which is exactly the cancellation that produces
`raw_pool_cosine ≈ 0.97` alongside `centered_pool_cosine ≈ −0.93`. Whether any of this is
semantically useful is decided **only** by the USR numbers in Sections 2–4, and those say it
is not.

## 7. Final verdicts

**A. Raw `z_U` semantic ability mainly comes from the centroid: SUPPORTED.**
`unsaid_raw` and `centroid` agree within 0.0023 R@1 at all five checkpoints, their MRR
orderings are identical, and removing the centroid (the centred scorers) collapses R@1 to
chance and drives the semantic margin negative. The shared common mode carries essentially
all of the raw `z_U` signal.

**B. Centered Anti-Said enriches withheld semantics: NOT SUPPORTED.**
`unsaid_centered` achieves R@1 0.0012 (chance 0.00115) with a mean rank of 647/868 and a
**negative** margin (−3.88); paired against `centroid` the delta is −0.0438 R@1 with a CI of
[−0.0588, −0.0300] and p = 0.000. It is worse than the centroid, not better.

**C. Anti-minus-Said contrast enriches withheld semantics: NOT SUPPORTED.**
`anti_minus_said` is likewise at chance (0.0012, mean rank 641, margin −3.77), and paired
against `unsaid_raw` the delta is −0.0461 R@1, CI [−0.0611, −0.0323], p = 0.000.

**D. Discovery protects against semantic starvation: SUPPORTED** (paired CI).
`B100 − A100` = +0.0069 R@1 (CI [+0.0023, +0.0127]) / +0.0092 MRR (CI [+0.0052, +0.0140]),
McNemar p = 0.031; `C100 − A100` = +0.0092 R@1 (CI [+0.0035, +0.0161]) / +0.0094 MRR
(CI [+0.0052, +0.0143]), p = 0.008. The wording stays "partially preserves / recovers": on
`MRR` the frozen initial checkpoint is still ahead (`initial − C100` = +0.0301, CI
[+0.0172, +0.0433]).

**E. Absorption improves global withheld semantics: NOT ESTABLISHED** (paired CI).
`C100 − B100` on `global` = +0.0058 R@1 (CI [−0.0069, +0.0184]) / +0.0018 MRR (CI
[−0.0058, +0.0097]), McNemar p = 0.473; `C100 − A100` = +0.0023 R@1 (CI [−0.0092,
+0.0138]). Direction as expected, interval contains zero.

**F. Internal closure is a semantic proxy: NOT SUPPORTED.**
Within a trained checkpoint the correlations are in the right direction and nominally
significant, but ρ² ≤ 0.015, the correlations are ≈0 at `initial`, and the cross-checkpoint
ordering is **inverted** (highest closure = worst global R@1). A ≤1.5 % effect that reverses
between checkpoints cannot proxy the between-checkpoint semantic comparisons this phase
needs.

## 8. What this means for the base objective

The mechanism chain the phase was designed around — `L_D → z_U` withheld semantics ↑,
`L_A → global` withheld semantics ↑ — holds **partially**: the first link is now supported by
paired CIs, the second is only directional. What this decomposition adds is that the raw
`z_U` signal is essentially the patch centroid, so the anti-Said routing is not what supplies
the withheld semantics; and that the centred/contrast re-weightings that the common-mode
analysis might have suggested (centred pooling, contrast targets) are actively **harmful** on
the frozen cohort. Per the brief, no redesign is attempted here (no centred pooling loss, no
earlier layer, no new projection, no new residual target, no relative gap loss, no USS).
