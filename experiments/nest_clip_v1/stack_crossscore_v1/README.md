# Stack-Pool / CrossScore-Flat Exploration

The four fixed candidates are S-CLS, C-CLS, S-PATCH and C-PATCH. Each has the
unchanged NEST A3-RandomK backbone, complete 248-slot text branch, independent
visual branch and native embedding-channel hard mask. The visual condition is
the final-layer post-LN/pre-projection CLS or all 196 patches excluding CLS.

Stack-Pool concatenates the independently processed token sequences and applies
the original AttentionPool. Its implementation uses exactly equivalent separate
logsumexp/weighted-average summaries and a scalar mixture. It adds no pairwise
token attention. CrossScore uses raw QK/sqrt(64), a fixed visual-outer/text-inner
flattening order and a dense learned readout, without softmax or residual logits.
Both CLS and patch CrossScore contract dense readout weights with local text keys
once per view before image-query contraction. The patch head is not low-rank:
it has 24,887,296 readout weights plus 512 biases.

New modules initialize under isolated fixed CPU RNG scopes: visual adapter seed
1763, Q/K seed 1787, dense readout seed 1789. B_V is a deep copy of step-0 B_T.
B_T and B_V use mask LR 1e-3; A_V and CrossScore use adapter LR 1e-4. CrossScore
freezes the unused original pool. Optimizer membership is unique by parameter ID.

The model uses the VCP text-owner organization: local text columns score all
global image candidates, differentiable score gathering supplies local image
queries, and each rank's CE sum is scaled exactly once by W/N. No full Z_T
communication is needed. The zero-valid-rank path connects all gather tensors
through finite zeros before candidate logits are masked with negative infinity.

Every candidate needs its own real four-rank resource gate: 5 warmup + 30 measured
updates, each at most 3 seconds, peak allocated at most 65 GiB. A failed measured
step stops the probe immediately rather than completing an already-failed gate.
Exceptionally slow full-update feasibility probes can also stop after the second
update, with their exact timing preserved. Resource failure produces no Recall.
No method, precision, batch, tokens or candidate-pool reduction is allowed.

Passed candidates independently run 5-update smoke then 500-update formal from
the shared untrained checkpoint. The scheduler horizon remains 3651. Formal
updates are measured end to end including DataLoader waiting and ordinary logs;
three consecutive slow updates stop the candidate after saving its full state.
Initializations and checkpoint writes are logged separately from regular steps.

Only step500 bare students are evaluated on COCO canonical, Urban-1k, Flickr30k
test1K and DOCCI test5K. No DCI or Long-DCI is scheduled. The primary native
comparison is against the exact TI-fast@500 reference JSON from 6bafa8a. The
fixed summary is the mean of Urban/DOCCI I2T/T2I R@1. No 3651-update model enters
the 500-update ranking; no new seeds or mechanisms are scheduled.

Scripts resolve their actual worktree. Full weights/data stay outside Git at
`/root/lk_projects/SAID-nest-clip-v1/stack_crossscore_v1/`. Compact report/evidence
is committed to `codex/nest-stack-crossscore-v1` after pipeline completion.
