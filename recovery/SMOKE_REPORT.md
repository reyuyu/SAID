# Recovery smoke report

**READY_TO_START_S02_FULL**.

The final Summary+RandomDetail S=0.2 smoke executed exactly5 updates on four
A10080GB GPUs, from common step0, weights `[1.4,0.2,1.4]`, horizon4868.
All losses/gradients finite; DDP parameter difference0;5120 sample/text/token
streams exactly replayed; AdamW counters5; strict bare export and reloaded
native image/text similarity forward passed. This is the S02 candidate smoke,
not a newly executed canonical RandomK smoke.

Median post-first full-cycle time2.040678s; peak allocated27.759881GiB per rank.
First-cycle startup/warmup is investigated in `S02_SMOKE_REPORT.md`.
Full per-step loss/CE/weighted alignment/sparsity/inclusion/gate/LR/gradnorm
metrics and acceptance evidence are linked there.

Smoke artifacts are quarantined, marked `SMOKE_ONLY_DO_NOT_RESUME.md`.
No formal500/4868 training has started or been authorized. Any full S02 run
must start from shared common step0 after another explicit user instruction.
