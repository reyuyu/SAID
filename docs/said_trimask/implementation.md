# S0-TriMask v0.1 -- implementation record

Objective `smartclip_trimask`, arm `S0_TriMask`, base commit
`894599dc56944920eba53e5396b294b37a74a140`, worktree `/root/SAID-s0-trimask-v01`,
branch `codex/s0-trimask-v01`, phase `s0-trimask-v0.1`.

## Formula actually implemented

One forward produces **one** visual mask and **one** text mask per caption; all three score
matrices reuse those same tensors and their computation graph.

```
mI_j  = ST(mask_net(H_j.detach()))                      # reference Said mask, unchanged
fT_j  = text_stem(H_j.detach())                         # new MaskNetwork(512, 1 layer, 8 heads)
mT_j  = 0.1 + 0.9 * sigmoid(text_gate_projection(fT_j)) # soft suppression gate in [0.1, 1]

t_hat_j = Norm(t_raw_j)                                 v_hat_i = Norm(v_i)
rT_j    = Norm(t_raw_j * mT_j)                          den_i,j = sqrt(clamp_min(sum_d v_i,d^2 * mI_j,d^2, eps^2))

Q1[i, j] = 100 * (v_i . (mI_j * t_hat_j)) / den_i,j      # MASK image  - native text
Q2[i, j] = 100 * (v_hat_i . rT_j)                        # native image - MASK text
Q3[i, j] = 100 * (v_i . (mI_j * rT_j))   / den_i,j       # MASK image  - MASK text

Lk = CE(Qk, targets) + CE(Qk^T, targets)                 # both directions, no 0.5
L_sparse_I = mean_d |mI_i,d|                             # real matched pairs only, counted once
L_total = 10*L1 + 1*L2 + 1*L3 + 2*L_sparse_I             # lambda_sparse_T = lambda_adv = 0
```

`eps = 1e-6`; the denominator is clamped **before** the square root, so a fully closed mask gives
`scores == 0` with finite gradients instead of `sqrt(0)`. `mI.square()` is deliberately kept in the
graph: for a straight-through mask its forward value equals `mI`, its gradient does not.

## Initialisation equivalence

`text_gate_projection.weight = 0` and `bias = log(8)` give `mT == 0.9` exactly for every caption,
so `Norm(t_raw * mT) == Norm(t_raw)`, `Q2` equals the unmasked global score and `Q3 == Q1`. The
whole new branch is built inside `torch.random.fork_rng(devices=[])` seeded with `module_seed = 0`,
so construction consumes neither the ambient torch RNG nor the caption/data streams. Only the
output layer is zeroed -- `text_stem` keeps its normal random initialisation, so its first-step
gradient is zero (a real model measurement, not an assumption) and becomes non-zero after the first
update.

## Distributed route and scaling

Per rank: encode the local images and texts once, build `mI`/`mT` once, gather the text side with the
reference's autograd-aware `torch.distributed.nn.all_gather`, compute the local image rows against
all global text columns for all three paths, then gather the three `[n_local, N]` score matrices
row-wise to obtain the global `[N, N]`. I2T uses the local rows, T2I uses the columns owned by this
rank, and the sparse term uses the local matched masks. Nothing is detached.

Per-path cross-entropies are means over the rank's own anchors and DDP averages parameter gradients,
so `mean_r dL_r/dp == d(global anchor mean)/dp`: **no `world_size` factor is applied**, which is the
reference S0 convention of the baseline this run is compared against.
`tests/test_said_trimask.py::test_two_rank_trimask_step_matches_the_single_process_reference`
verifies that on a real 2-process gloo run against a single-process run of the same global batch
(gradients, cross-rank bit identity, and one AdamW update).

## Optimizer groups

Two AdamW groups, checked by parameter id: the student backbone (`lr = 1e-6`, `wd = 1e-2`,
`warmup = 200`) and the mask group = original visual `mask_net` (14 tensors) + the whole new text
branch (16 tensors, 3,415,553 parameters), `lr = 1e-3`, `wd = 0`, `warmup = 0`. The unused
`logit_scale` keeps its compatibility key, is frozen and never enters an optimizer;
`positional_embedding` is frozen by the repository's own CLIP builder and is the only other frozen
tensor. `--max_steps` truncates the run; the LR horizon stays at the three-epoch plan
(`3 * len(loader)` = 3651).

## Tests (13, `tests/test_said_trimask.py`)

Path-1 degeneration to `compute_smartclip_terms` (loss + shared gradients), text-gate
initialisation equivalence, single mask generation and reuse (forward-call counts, tensor identity
by recomputation, caption/other-sample invariance, determinism), per-path gradient responsibilities
including the measured fact that the sparse term keeps training `mI` even when path 1 is off, an
independent per-pair reference for forward *and* mask-logit gradients, the zero-mask norm lower
bound, cross-entropy properties (`log N` at equal scores, shift invariance, monotonicity), the
two-rank DDP step, the export key guard, the optimizer-group audit, RNG isolation and the CLI.

## Source facts recorded rather than assumed

* `compute_smartclip_terms` applies **no duplicate-caption filtering**: every global caption is a
  candidate. No filter is added here, and the earlier T1/FP0 duplicate rules are not migrated.
* The visual mask keeps the reference execution strategy (ambient bf16 autocast); the new text
  branch, the gating, the normalised scores and both cross-entropies run in explicit fp32
  (`torch.autocast(enabled=False)`), because autocast would otherwise turn the fp32 matmuls into
  bf16.
* `view_b` is still constructed by the loader so the reference caption stream is untouched, but it
  never enters the model: this experiment has one image view and no visual SSL term.
* At step 0 the text gate is the constant `0.9`, so paths 2 and 3 are numerically near-copies of
  path 1 while still contributing gradient. The default total loss is therefore *not* the S0 loss;
  only `lambda_2 = lambda_3 = 0` degenerates to it exactly.
