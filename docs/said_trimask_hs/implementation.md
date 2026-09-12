# S0-TriMask-HS v0.2 -- implementation record

Objective `smartclip_trimask_hs`, arm `S0_TriMask_HS`, phase `s0-trimask-hs-v0.2`, branch
`codex/s0-trimask-hs-v02`, base commit `56b45a4a04654c0230e311a6225fb6590b05a8b2` (the frozen
soft-gate S0-TriMask v0.1 result), worktree `/root/SAID-s0-trimask-hs-v02`.

## What changed relative to v0.1

One trainer serves both modes; the default is still the v0.1 soft gate, so every existing checkpoint
and launcher keeps its exact meaning.

```
soft    (v0.1, default)  mT = 0.1 + 0.9 * sigmoid(aT)
                         L = 10*L1 + 1*L2 + 1*L3 + 2*L_sparse_I
hard_st (v0.2)           pT = sigmoid(aT),  hT = (pT >= 0.5),  mT = hT + (pT - pT.detach())
                         L = 10*L1 + 1*L2 + 1*L3 + 2*L_sparse_I + 0.2*L_sparse_T
```

* The hard gate's **forward value is exactly 0 or 1**; the backward pass is the sigmoid
  straight-through approximation. That is a training device for a discrete gate, not the true
  derivative of the threshold, and it is described that way everywhere.
* The **0.1 floor is gone** in hard mode; the normalisation `eps = 1e-6` is untouched.
* `L_sparse_T = mean(abs(mT))` uses the same form as the visual side and keeps the straight-through
  graph. It is attached to `mT`, never to `hT.detach()`. `abs` has a zero subgradient at exactly 0, so
  a coordinate the gate currently closes receives no sparsity gradient -- that asymmetry is the
  definition this version adopts, it is pinned by a test against a manual reference, and it is *not*
  silently replaced by `mean(mT)`.
* Both sparsity terms are means over this rank's own captions, counted once each; the path weights are
  unchanged at 10 / 1 / 1 and no extra `world_size` or `512` factor is applied. `lambda_adv` stays 0.
* The gate mode is part of the experiment identity: it is written into `objective`, `arm`, `phase`,
  `config` and the checkpoint, and `check_checkpoint_compatibility` refuses to resume a v0.1 payload
  in hard mode (and vice versa), so a soft checkpoint can never be silently reinterpreted.
* The new text branch is built inside `torch.random.fork_rng(devices=<current device>)`, i.e. the CPU
  generator **and** the CUDA generator this process uses; a test asserts both states survive
  construction. Forking only the CPU generator (the v0.1 code) would not have supported that claim.
* A **gradient-finiteness gate** runs before every optimizer step: all ranks evaluate the same
  predicate and `all_reduce(MIN)` it, and if any rank sees a non-finite gradient every rank refuses
  the update and the run aborts. No clipping, no loss rescaling, no silent skip.

## Formula, unchanged from v0.1

```
Q1[i,j] = 100 * dot(Norm(v_i * mI_j), t_hat_j)      mI_j, mT_j from candidate caption j
Q2[i,j] = 100 * dot(v_hat_i,          rT_j)         t_hat = Norm(t_raw), rT = Norm(t_raw * mT)
Q3[i,j] = 100 * dot(Norm(v_i * mI_j), rT_j)
Lk = CE(Qk, targets) + CE(Qk^T, targets)            both directions, no 0.5
```

One visual mask and one text mask are generated per caption and reused by all three paths; `mI.square()`
stays in the graph; the masked-norm denominator is clamped before the square root.

## Logging (task section 8)

Scalars every 10 steps, the heavier detached statistics and the mask snapshot every 25 steps (the
heavy block is skipped entirely on light steps, which saves the quantiles and the variance
decomposition). Added this round:

* per path: `margin_max_negative_i2t/t2i` (does one negative overtake the positive) and
  `margin_logsumexp_i2t/t2i` (the negative mass the cross-entropy actually drives down) -- two
  different quantities under two different names;
* `adv_gap_positive_part`: explicitly the mean of `relu(ell3 - best_single)`, so paths that are
  already better cannot cancel paths that are worse;
* `L_sparse_T`, `0.2*L_sparse_T`, `text_gate_hT_zero_fraction`, `text_gate_hT_is_binary`,
  `text_gate_produces_zeros`, and the `pT` distribution (`p10/p50/p90/min/max/near-threshold`);
* intersection diagnostics: `K_intersection`, `K_union`, intersection percentiles, empty-intersection
  fraction, "both non-empty but the intersection is empty", and the Jaccard index which is `null`
  when the union is empty together with the count of undefined rows;
* within-caption (across coordinates) versus across-caption variance, the per-coordinate profile and
  the cross-caption standard deviation. The documentation states that a large within-caption share
  alone does **not** prove caption adaptation;
* `statistics_scope: "rank0/local_batch"` on every record, and no collective is ever issued from a
  rank-0-only branch;
* non-finite floats are written as `null`, never as the non-standard `NaN` token a strict JSON reader
  would reject.

## Tests

`tests/test_said_trimask_hs.py` (18 cases) covers the seven groups the task requires: gate type
(binary forward, all-open initialisation, controlled logits, no top-k, the floor present in soft and
absent in hard), gradients (sigmoid proxy, the `abs` subgradient including the closed coordinates and
the divergence from `mean(mT)`), reuse and responsibilities (one mask generation, tensor identity by
recomputation, per-path and per-sparsity-term gradient ownership with the other term switched off),
forward+backward against an independent per-pair reference, boundaries (all open / all closed / empty
intersection / near-zero norms, undefined Jaccard as `null`, and the update refusal on non-finite
gradients), real two-rank DDP with the text sparsity active on a **non-trivial** hard mask
(`tests/_trimask_hs_ddp_worker.py`), and regression (v0.1 soft default unchanged, `lambda_sparse_t`
reaching loss and gradients, checkpoint mode mismatch rejected, contradictory CLI flags rejected,
export keys). The v0.1 suite still passes unchanged.

## Source facts recorded rather than assumed

* `compute_smartclip_terms` applies no duplicate-caption filtering; none is added here either.
* The visual mask keeps the reference execution strategy (ambient bf16 autocast); the gate, the
  normalisation, the masked norms, the similarities and both cross-entropies run in explicit fp32.
* At step 0 `pT = 8/9`, so `hT` is all ones, `Norm(t*mT) == Norm(t)`, `Q3 == Q1` and `Q2` equals the
  unmasked global score. The *default* total loss is nevertheless not the S0 loss, because paths 2
  and 3 carry weight 1 each and the text sparsity term is now active.
* Hard gates producing zeros is not evidence that the right semantics were removed, and increased
  sparsity is not evidence of better retrieval; this version changes the gate type and adds a sparsity
  term at the same time, so no single-item causal conclusion is drawn from the result.
