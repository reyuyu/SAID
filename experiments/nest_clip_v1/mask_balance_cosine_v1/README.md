# Balanced Stack / Cosine CrossScore: Two Fixed 500-Update Repairs

Parent experiment: `codex/nest-stack-crossscore-v1`, result commit
`40d6f44d68a6df89b3e47d78adbcaeda5a5b049c`.

Balanced-Stack-Patch keeps the full 248 text / 196 patch token branches, but
separately pools each with the shared original AttentionPool. Its bias-free
1024-to-512 gate has zero initial weights. Every candidate pair gets
`g=sigmoid(W_g[u_T;u_V])`, followed by `g*u_T+(1-g)*u_V`, sigmoid and hard-ST.
The gate belongs to the existing adapter AdamW group (LR 1e-4, no warmup).
This follows the previous fusion-head grouping; B_T/B_V/pool retain LR 1e-3.
The concatenated linear gate is computed as two exactly equivalent linear terms.

Cosine-CrossScore-CLS normalizes projected 64-dimensional Q/K with eps 1e-6.
It uses raw cosine relations with no sqrt(64), temperature or softmax. Its
248-to-512 readout uses the standard PyTorch Linear initialization, including
the standard nonzero uniform bias initialization permitted by the request.
Q/K and visual adapter use the same isolated seeds as the parent experiment.
Normalization is before the exact readout/key contraction, retaining gradients.
The original unused AttentionPool stays frozen in this arm.

Common CLIP/B_T/B_V/A_V initialization, detach, losses, AdamW, precision,
RandomK, data, 4x256 global candidates and horizon3651 stay fixed. Each stage
starts independently from the shared untrained checkpoint. Only two arms run.

Order: unit/two-rank reference, 5-update real four-rank smoke, 5+30 real-DataLoader
full-update speed gate, then independent 500-update formal training for each
passed arm. Every measured gate update must be <=3 seconds and peak allocated
<=65 GiB/rank. No tokens, candidates, batch or losses are reduced to pass.

Diagnostics at completed steps 0/99/199/299/399/499 (updates 1/100/200/300/400/500)
include all-pair logit moments/saturation and cosine QK moments/absolute maximum.
Balanced gate moments/quantiles refer to global valid positive pairs, with
F/P/R gate absolute differences recorded on the same valid positive examples.
Positive/negative keep, full-open/full-closed and image-condition replacement
switches remain available. Saturation is p<.01 or p>.99. If any sampled cosine
view reaches 99% saturation, its full state is saved and the arm stops; no
temperature, LayerNorm, clipping or automatic retraining is added.

Only formal step500 students run native COCO, Urban-1k, Flickr30k test1K and
DOCCI test5K. Compare with parent S-PATCH/C-CLS and TI-fast@500. No DCI,
Long-DCI, combined architecture, new seeds or full-epoch continuation runs.

Output root: `/root/lk_projects/SAID-nest-clip-v1/mask_balance_cosine_v1/`.
Code/small evidence branch: `codex/nest-mask-balance-cosine-v1`. Checkpoints,
full per-step token logs, data and caches remain on the server.

After the original four-set results completed, the user explicitly requested a
separate Long-DCI supplement for both new step500 students. Launch it with
`python -m experiments.nest_clip_v1.mask_balance_cosine_v1.long_dci`.
It uses the same 7602-pair manifest as TI-fast@500, normalized native embeddings,
batch64 on cuda:0, and a complete candidate pool. Progress is recorded separately
in `long-dci-status.json`. It writes LONG_DCI_REPORT.md / LONG_DCI_RESULTS.json
and pushes only compact evidence. The original REPORT.md / RESULTS.json and
their fixed Urban/DOCCI J_long remain unchanged. DCI Full is still excluded.
