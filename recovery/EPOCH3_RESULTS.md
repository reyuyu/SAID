# S=0.2 epoch3 strict native evaluation

Same formal trajectory, step3651 (3 ×1217 updates), horizon4868. Evaluation only; no training.

| Dataset | I2T R@1 / R@5 / R@10 (%) | T2I R@1 / R@5 / R@10 (%) |
|---|---|---|
| COCO | 62.320000 / 84.260000 / 90.200000 | 43.252000 / 68.968000 / 78.420000 |
| Urban-1k | 92.600006 / 98.800004 / 99.500006 | 91.900003 / 99.000007 / 99.300003 |
| Flickr30k-test1k | 89.700000 / 98.600000 / 99.600000 | 73.720000 / 92.100000 / 95.860000 |
| DOCCI | 79.000000 / 95.860000 / 98.280000 | 79.900000 / 95.780000 / 98.160000 |
| Long-DCI | 58.405683 / 77.255985 / 83.004473 | 60.326230 / 78.308340 / 83.346488 |

| Score (%) | Epoch3 | Epoch4 | Delta vs epoch4 (pp) | Delta vs RandomK4epoch (pp) |
|---|---:|---:|---:|---:|
| Score5 | 73.112392 | 73.209163 | -0.096771 | +0.344245 |
| J_long3 | 77.021987 | 77.137938 | -0.115951 | -0.048257 |
| J_long | 85.850002 | 85.945002 | -0.095000 | +0.364999 |
| Short4 | 67.248000 | 67.316000 | -0.068000 | +0.933000 |

RandomK comparison uses its 4epoch reference; it is not a matched epoch3 baseline.

Checkpoint: `/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/s02-local500-20261006/step4868/step003651.pt`; SHA256 `d3fbe825c70ba403fb867223d234d3fb6cab5276cdca5a4ed3bdec1f8c968a3a`; bytes 1884189690.
Bare student: `/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/s02-local500-20261006/epoch3-eval-step3651/student_step3651.pt`; SHA256 `f6d38b7b0e2a18f9e4a01586f8fd89edd343ab6ed5989d68e852100b38a0a1cc`.

Strict model load; native image/text embedding equivalence exact. Checkpoint SHA unchanged before/after evaluation. All five results share the bare SHA. Long-DCI uses 7602 images/captions and the frozen manifest. No mask/gate/rerank/ensemble/Summary/Detail inference.

Detailed dataset R1 deltas, command provenance and local raw-log paths/sizes/SHA/time ranges: `EPOCH3_RESULTS.json`. Checkpoints, bare student, dataset and raw logs remain local.
