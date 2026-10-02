# Position-Preserving Remainder:500-Step Test

Reference:14653c92c6da9d552a2b624ab169eaaa275cdde8, ViT-B/16.
Branch:codex/nest-balanced-rmask-500-v1.
Runtime:/root/lk_projects/SAID-nest-clip-v1/balanced_rmask_500_v1/.

Only R changes: compact tokenized suffix becomes F.clone() with the selected
prefix's exact BPE token span, including its trailing separator, replaced by PAD0.
Boundary IDs are validated against F before masking. SOT, suffix tokens and their
absolute positions, original F EOT and trailing padding remain unchanged. No
attention mask, extra view, shuffle, structural change or loss change is added.

F/P/K, seed0, sampling_seed0, data order, deterministic preprocessing, Balanced
coefficients and initialization remain identical. Split fallback behavior is
unchanged. The model/loss source is byte-identical to the frozen reference.

Both arms use the common B16 step0,4x256,global1024, BF16 encoder/FP32 other paths,
encoder checkpoint ON, pair checkpoint OFF, blocks128x128 and horizon4868.
Each has independent5-step smoke and a fresh formal500 run. The exact earlier
four-epoch run did not save500, so the authorized sole matched compact-R baseline
is run first, followed by RMask; each receives all five native evaluations.
Historical horizon3651 results are not the matched baseline.

Resource policy remains<=3s complete regular updates and allocated<=65GiB/rank.
Five-dataset raw Recall, raw Score5/J summaries and all ten R1 deltas are in
RMASK_500_REPORT.md and RESULTS.json. Readable token-position proofs are in
evidence/real-samples.json, and complete configs/raw results are separated by arm.
Stop after both500 evaluations. A gain may justify suggesting full4epoch confirmation,
but no4868 training, extra hyperparameter values or new seeds are scheduled.
