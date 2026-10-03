# Balanced runtime optimization audit

Reference commit c93248a; mathematics reference is compact old-R, RandomK,
Balanced-Stack-Patch B16 and fusion2e-4/visual-scale1/view1:1:1/sparse1/inc1.
Use4 A10080GB,256/rank,global1024,accum1,224,context248,workers8,
BF16 encoders and FP32 auxiliary/scoring/loss. Every performance candidate uses
5 warmup +30 real DataLoader updates from the identical seed0 sampler prefix.
Profiler uses5 warmup +10 traced updates and is excluded from speed results.
No retrieval evaluation, no500-step model, no formal L14 four-epoch training.

Current immutable source snapshots and original SHA256 manifest are in reference/.
Runtime, fixed real inputs and profiler traces stay outside Git at
/root/lk_projects/SAID-nest-clip-v1/runtime_optimization_v1/.
Compact metrics, correctness evidence, actual profiler summaries and recommended
B16/L14 configurations will be written here. Gate-cache/fused-view/scoring/checkpoint/
DDP/optimizer candidates require named forward, gradient and AdamW regression.
TF32 has a separate RELAXED classification and cannot replace the exact path.
Full/sparse audit policies preserve training and checkpoint state, and benchmark
mode performs no per-update disk writes or nonessential object collective.
