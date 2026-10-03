# Original Balanced Runtime Restored

Optimization was cancelled by the user. Training/model/data code is byte-identical
to c93248a, before runtime optimizations. The archived experiment branch and local
probe results remain available; none of the optimized paths is adopted.

Use configs/nest_balanced_runtime_original_v1.json: compact old-R, RandomK,
fusion2e-4, 4x256/global1024,128x128 chunks, full encoder checkpoint ON,
pair checkpoint OFF, original full audit, BF16 encoder and FP32 auxiliary/loss.
Original measured mean2.061873 seconds/update, allocated27.76GiB/GPU.
All optimization/probe processes have stopped. No new training run is started.
