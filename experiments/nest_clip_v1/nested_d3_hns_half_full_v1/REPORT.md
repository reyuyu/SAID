# HNS-Half E3 / E4 continuation

Exact evaluated Half@2434 parent restored. Hierarchy stays0.5; no SG; beta2/2; old soft inclusion0.
All production trainer/model/data/export/evaluator sources unchanged. Complete optimizer, scheduler, RNG and loader cursor restored per rank.
Training pauses at3651 for strict native five-dataset evaluation, then restores3651 and ends exactly4868.
GPU0 COCO; GPU1 DOCCI; GPU2 Long-DCI; GPU3 Flickr then Urban. Evaluation batch64 unchanged.

| Step | Score5 | J_long3 | J_long | Short4 |
|---|---:|---:|---:|---:|
| 500 | 71.202056 | 75.152094 | 84.160001 | 65.277000 |
| 1217 | 72.869675 | 77.012792 | 85.925003 | 66.655000 |
| 2434 | 73.575684 | 77.894807 | 86.735002 | 67.097000 |
| 3651 | 73.706809 | 78.066015 | 86.880002 | 67.168000 |

## Step3651

| Dataset | I2T R1/R5/R10 | T2I R1/R5/R10 |
|---|---|---|
| COCO | 61.860000 / 83.620000 / 89.820000 | 42.992000 / 68.600000 / 78.172000 |
| Urban-1k | 94.200003 / 99.000007 / 99.600005 | 92.300004 / 99.000007 / 99.200004 |
| Flickr30k-test1k | 89.900000 / 98.200000 / 99.200000 | 73.920000 / 92.060000 / 95.840000 |
| DOCCI | 80.180000 / 96.380000 / 98.560000 | 80.840000 / 96.100000 / 98.400000 |
| Long-DCI | 60.181531 / 78.150487 / 84.148908 | 60.694554 / 78.663510 / 83.438569 |

D3 Balanced deltas(pp): `{"Score5": -0.12337242271175342, "J_long3": -0.13962070451958652, "J_long": -0.04500064373016244, "Short4": -0.09899999999998954}`.

HNS-v1 deltas(pp): `{"Score5": -0.06760844168513813, "J_long3": -0.12201406947522742, "J_long": -0.16000084638596945, "Short4": 0.014000000000010004}`.

All30 recall deltas: RESULTS.json. Resource statistics: RUNTIME_STATS.json. Full stream gates: VALIDATION.json.
Checkpoints/bare/raw logs remain in persistent canonical runtime and are never uploaded. /root is disposable cache; NFS originals retained.
Ordinary slow steps warn; true I/O/CUDA/DDP/nonfinite/OOM-kill/>60s/supervisor failures stop. No additional experiments.
