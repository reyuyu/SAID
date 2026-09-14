# Masked dual-mask formal 500-step result

Training SHA: `ff5ad1d4b918d56c6bfa48a2870dc5223e757237`. Exactly 500 updates from the common initialization; 4 GPUs × 256, global batch 1024. No native arm was trained.

The contiguous one-dimensional suffix gather passes the fixed Gloo/NCCL full-total-objective gradient and AdamW reference checks. S0 and W/V remain unchanged.

| Dataset | Direction | S0@500 R@1 / R@5 / R@10 (%) | Masked@500 R@1 / R@5 / R@10 (%) | R@1 change (pp) |
|---|---|---|---|---|
| COCO canonical | I2T | 60.580 / 82.200 / 89.060 | 60.540 / 82.480 / 89.180 | -0.040 |
| COCO canonical | T2I | 41.236 / 67.092 / 76.620 | 41.464 / 66.672 / 76.552 | +0.228 |
| Urban-1k | I2T | 87.000 / 97.100 / 98.800 | 88.200 / 97.500 / 99.200 | +1.200 |
| Urban-1k | T2I | 84.200 / 96.700 / 98.100 | 85.800 / 97.500 / 99.100 | +1.600 |

Full configuration, per-rank stream digests, first three online checks, checkpoint hashes, real exit codes, and frozen evaluator provenance are in `masked_formal500_report.json`.

Training wall time: 934 seconds. All train/export/COCO/Urban exit codes: 0.
Maximum per-GPU allocated/reserved memory: 65.83 / 68.65 GiB.

Only the combined method is compared with the existing S0 result. This experiment does not isolate the new gate versus ordinary suffix supervision. No model weights, data, or large logs are committed. Training stopped at update 500.
