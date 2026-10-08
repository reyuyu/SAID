# HNS-Half E2 validation

Status: `PROMISING`. Stop exactly2434. No3651/full/third experiment.
One seed; retrieval is primary. Material long decline threshold0.05pp; promising Score5 gap0.05pp, declared before launch.
Checkpoint boundaries pause all DDP ranks for strict export and five-dataset evaluation, then restore complete state without changing the schedule.
Production trainer/model/data/export/evaluator sources equal the respective mother. B changes only the process-local ramp200 to500 with completed-before-current-update convention.
Native normalized image/full-caption embedding retrieval only. GPU0 COCO;1 DOCCI;2 Long;3 Flickr then Urban. Batch64 unchanged.

| Step | Score5 | J_long3 | J_long | Short4 |
|---|---:|---:|---:|---:|
| 500 | 71.202056 | 75.152094 | 84.160001 | 65.277000 |
| 1217 | 72.869675 | 77.012792 | 85.925003 | 66.655000 |
| 2434 | 73.575684 | 77.894807 | 86.735002 | 67.097000 |

## Step2434

| Dataset | I2T R1/R5/R10 | T2I R1/R5/R10 |
|---|---|---|
| COCO | 61.960000 / 83.680000 / 89.580000 | 42.808000 / 68.564000 / 77.936000 |
| Urban-1k | 93.600005 / 99.000007 / 99.600005 | 92.400002 / 99.200004 / 99.400002 |
| Flickr30k-test1k | 89.600000 / 98.500000 / 99.200000 | 74.020000 / 92.020000 / 95.720000 |
| DOCCI | 80.140000 / 96.280000 / 98.520000 | 80.800000 / 96.060000 / 98.340000 |
| Long-DCI | 59.878979 / 78.097869 / 83.925283 | 60.549855 / 78.755591 / 83.570113 |

D3 Balanced delta(pp): `{"Score5": -0.02253177831194364, "J_long3": -0.05288629718656068, "J_long": -0.03000032186508861, "Short4": 0.023000000000010346}`.

HNS-v1 delta(pp): `{"Score5": 0.09985464763832397, "J_long3": 0.07042441273053157, "J_long": 0.029998629093171303, "Short4": 0.14400000000000546}`.

All30 recalls and deltas: RESULTS.json. Keep/violation/IoU/CE/share/group-gradient/gate: TRAINING_DIAGNOSTICS.json.
No inference mask/gate/rerank/ensemble/TTA. GradScaler remains disabled under original BF16 autocast.
Source/checkpoint/RNG/cursor identity and full actual sample/text/token/index/LR gates are recorded. No cross-run bitwise-loss requirement.
`/root` disposable image cache; persistent NFS originals retained. Binaries/raw logs stay in canonical runtime, never Git.
Recommendation only; this runner cannot continue to3651 or launch any extra experiment.
