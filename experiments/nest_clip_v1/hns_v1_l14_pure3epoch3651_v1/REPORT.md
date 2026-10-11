# HNS-v1 ViT-L/14 Pure3 resource-gate report

The isolated L14 Step0 was created from the official OpenAI ViT-L/14 checkpoint and passed the architecture, loader, HNS forward/backward, and finite-value checks. The Step0 checkpoint is retained locally with SHA256 `8fa8f7819f41f01635b95c0c7aed3c2a51f8dd58317666cacd74d3bc162fc912`.

The mandatory four-GPU gate was then run with the production batch (`256` samples per rank, global `1024`), BF16 encoder autocast, checkpointing, and the original HNS-v1 objective. NCCL initialization and the gradient synchronization check passed. The first five updates were warmup. The first measured updates were 4.2874 s, 4.2352 s, 4.3615 s, and 5.0773 s for the four-rank end-to-end cycle. The fourth measured cycle exceeded the fixed 5-second limit, so the gate failed after update 9. Peak allocated memory was approximately 47.36 GiB on every GPU, below the 65 GiB limit.

The worker-abort message occurred while the failed probe was being shut down. It is retained as evidence and does not change the primary gate result. No formal smoke, 3651-step training, checkpoint continuation, or evaluation was started. No batch-size, precision, threshold, or objective change was attempted.

Per the pre-authorized stop rule, this experiment is stopped as `STOPPED_RESOURCE_GATE_FAILURE`. The existing B16 HNS experiment and all historical checkpoints remain untouched.
