# S0.2 four-epoch continuation

Authorized continuation of the existing independently initialized S0.2 trial from500 to4868 total optimizer updates. The original500 updates already use the complete four-epoch scheduler horizon4868. Restore model/fusion weights, full AdamW moments/steps, rank-specific Python/NumPy/Torch/CUDA RNG and the epoch-start DataLoader generator state. No optimizer, warmup or scheduler restart is performed.

All production model/trainer/data/optimizer/scheduler sources remain byte-identical to the500 checkpoint manifest. The resume validator verifies exact hyperparameters, trial identity, data/sampling/config, code hashes, horizon and epoch/batch cursor. The only configuration changes are experiment label and checkpoint I/O interval1217, which saves epoch-boundary recovery checkpoints and does not alter the objective.

28 continuation/scheduler/sampling regression tests passed. Runtime compares all4368×4 resumed sample-ID streams, epoch indices and batch sizes against the official seed0 DistributedSampler. First resumed update is501; final is4868. Original tail batches remain180/rank (global720) at each1217-update epoch boundary; ordinary batches remain256/rank (global1024).

The final4868 checkpoint is strictly exported and evaluated with the unchanged native5 protocol. Comparison uses the published RandomK four-epoch/H4868 reference. The parent500 result is separately labeled as a shorter budget. Only this S0.2 method is trained; no extra weights or fifth epoch.
