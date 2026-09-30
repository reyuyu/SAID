# VCP-Mask: Full Three-Epoch Continuation

Continue the existing VCP-Mask formal checkpoint at update 500 to update 3651,
restoring the model, VCP query adapter, AdamW state and each rank's RNG state.
The unchanged scheduler horizon is 3651, with 1217 updates per epoch.

The model, objective, dataset, precision, pair blocks and optimizer settings are
identical to the reviewed 500-update run. The trainer-only change allows skipping
its startup checkpoint. The new config sets the save interval to 3651, so this
continuation creates only `step003651.pt`, a complete final training checkpoint.
Existing 500-update checkpoints and results remain in their original directory.

Run with `bash experiments/nest_clip_v1/vcp_mask_3epoch_v1/run_training.sh`.
The script resolves its own worktree and rejects any existing output directory.

Server output: `/root/lk_projects/SAID-nest-clip-v1/vcp_mask_3epoch_v1/`.
`console.txt` and `status.json` track the detached run. Four ranks use all four
GPUs, batch 256 per rank. The continuation records updates 501-3651.

Only training is launched here. Native evaluation will be performed when queried
after the final checkpoint is available; no DCI evaluation is scheduled.
