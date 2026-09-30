# Balanced-Stack-Patch: Full Three-Epoch Continuation

Continue the audited formal step500 checkpoint to step3651 (three full epochs),
restoring CLIP, the complete visual/gate branch, AdamW moments and all four RNG
states. Keep the original scheduler horizon3651, data, losses, precision, batch,
128x128 tiles and all model/trainer source files byte-identical to the parent.

Changes are only the experiment/output name, stop update3651, checkpoint interval
3651 and disabled startup checkpoint. The continuation writes only step003651.pt
normally; genuine sustained resource/numerical failures may still save an
incident state. Historical 500-update checkpoints and metrics stay untouched.

An independent supervisor waits for the previous two-model Long-DCI supplement
to finish successfully before launching training, then checks all GPUs are idle.
It never interrupts or shares GPUs with the previous evaluator.

Run: `bash experiments/nest_clip_v1/balanced_stack_3epoch_v1/run_training.sh`.
Detached supervision: `python -m experiments.nest_clip_v1.balanced_stack_3epoch_v1.background`.
State/logs: `/root/lk_projects/SAID-nest-clip-v1/balanced_stack_3epoch_v1/`.

All four ranks use batch256, seed0 and the original 1217-update DataLoader per
epoch. Steps501-3651 represent 3151 additional shared optimizer updates, not a
restart or a new 3651 updates. The full final training checkpoint retains the
fusion gate and optimizer for strict export and native evaluation afterward.
