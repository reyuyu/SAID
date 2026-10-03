# R-SentenceDrop Four-Epoch Confirmation

The user authorized a full four-epoch confirmation after the completed500-step
experiment. Continue the exact R-SentenceDrop500 full checkpoint to4868 updates
with its original horizon4868, optimizer, four rank RNG states and loader cursor.
The complete trajectory starts at the common shared step0. Reuse the completed
compact old-R4868 baseline; never retrain the baseline or repeat the first500.

Use the identical B16 Balanced recipe, seeds, training/data/model code and
whole-sentence subset construction. Only the stopping point is extended;
save epoch-boundary checkpoints1217,2434,3651 and final4868 locally. Five frozen
native evaluations run once at4868, followed by the complete Recall/delta report,
whole-trajectory sampling/resource statistics, strict export evidence and hashes.
Compare the sample/F/K/prefix diagnostics against the complete old-R4868 stream.
That historical stream predates explicit P-token hashing; preserve this limitation
and reuse the matched P-token evidence over the first500 updates.

Runtime: `/root/lk_projects/SAID-nest-clip-v1/balanced_rdrop_4epoch_v1`.
Branch: `codex/nest-balanced-rdrop-4epoch-v1`.
Report: `RDROP_4EPOCH_REPORT.md`; raw metrics/evidence in `evidence/`.
Monitor progress using runtime `state.json`, `execution/` and per-update logs.
The detached supervisor publishes compact evidence automatically when complete.
Failures are preserved without automatic retries. Stop after4868 and five evals;
no further tuning or seeds. Large weights, data and caches stay server-local.
