# Balanced-Ramp500 E2 validation

Status: `NEGATIVE`. Stop exactly2434. No3651/full/third experiment.
One seed; retrieval is primary. Material long decline threshold0.05pp; promising Score5 gap0.05pp, declared before launch.
Checkpoint boundaries pause all DDP ranks for strict export and five-dataset evaluation, then restore complete state without changing the schedule.
Production trainer/model/data/export/evaluator sources equal the respective mother. B changes only the process-local ramp200 to500 with completed-before-current-update convention.
Native normalized image/full-caption embedding retrieval only. GPU0 COCO;1 DOCCI;2 Long;3 Flickr then Urban. Batch64 unchanged.

| Step | Score5 | J_long3 | J_long | Short4 |
|---|---:|---:|---:|---:|
| 500 | 71.057836 | 74.830393 | 83.980001 | 65.399000 |
| 1217 | 72.740282 | 76.967803 | 85.920003 | 66.399000 |
| 2434 | 73.439212 | 77.750019 | 86.725003 | 66.973000 |

## Step500

| Dataset | I2T R1/R5/R10 | T2I R1/R5/R10 |
|---|---|---|
| COCO | 60.700000 / 82.660000 / 89.200000 | 41.496000 / 67.240000 / 76.948000 |
| Urban-1k | 91.000003 / 98.700005 / 99.700004 | 89.600003 / 98.500007 / 99.500006 |
| Flickr30k-test1k | 87.500000 / 97.700000 / 99.000000 | 71.900000 / 91.540000 / 95.200000 |
| DOCCI | 77.860000 / 95.380000 / 98.060000 | 77.460000 / 95.000000 / 97.740000 |
| Long-DCI | 56.590371 / 75.361747 / 81.675875 | 56.471981 / 76.032623 / 81.873191 |

D3 Balanced delta(pp): `{"Score5": -0.005764062088914557, "J_long3": -0.12427343681487457, "J_long": -0.045000000000001705, "Short4": 0.17199999999999704}`.

HNS-v1 delta(pp): `{"Score5": -0.14422060757445365, "J_long3": -0.32170101262408934, "J_long": -0.1799995589256298, "Short4": 0.12199999999999989}`.

HNS-Half delta(pp): `{"Score5": -0.14422060757445365, "J_long3": -0.32170101262408934, "J_long": -0.1799995589256298, "Short4": 0.12199999999999989}`.

INC0 delta(pp): `{"Score5": -0.10047365296927069, "J_long3": -0.14812275494880112, "J_long": -0.015001811981207425, "Short4": -0.028999999999996362}`.

## Step1217

| Dataset | I2T R1/R5/R10 | T2I R1/R5/R10 |
|---|---|---|
| COCO | 61.480000 / 83.300000 / 89.560000 | 42.436000 / 68.192000 / 77.756000 |
| Urban-1k | 93.500006 / 98.900002 / 99.600005 | 91.800004 / 98.800004 / 99.500006 |
| Flickr30k-test1k | 88.500000 / 98.300000 / 99.300000 | 73.180000 / 91.980000 / 95.500000 |
| DOCCI | 78.880000 / 95.960000 / 98.320000 | 79.500000 / 95.700000 / 98.080000 |
| Long-DCI | 58.418837 / 77.150750 / 82.820310 | 59.707972 / 78.097869 / 83.188635 |

D3 Balanced delta(pp): `{"Score5": -0.07237543311566697, "J_long3": -0.03395905519278131, "J_long": -0.1199993562698296, "Short4": -0.12999999999999545}`.

HNS-v1 delta(pp): `{"Score5": -0.09616033315170114, "J_long3": -0.0036005552528166618, "J_long": -0.034998307228093495, "Short4": -0.23499999999999943}`.

HNS-Half delta(pp): `{"Score5": -0.12939347046720684, "J_long3": -0.044989117445339843, "J_long": -0.005000119209285003, "Short4": -0.2560000000000002}`.

## Step2434

| Dataset | I2T R1/R5/R10 | T2I R1/R5/R10 |
|---|---|---|
| COCO | 61.660000 / 83.680000 / 89.820000 | 42.852000 / 68.648000 / 78.052000 |
| Urban-1k | 93.600005 / 99.200004 / 99.600005 | 92.600006 / 99.100006 / 99.400002 |
| Flickr30k-test1k | 89.300000 / 98.400000 / 99.400000 | 74.080000 / 92.100000 / 95.680000 |
| DOCCI | 79.940000 / 96.280000 / 98.520000 | 80.760000 / 96.040000 / 98.240000 |
| Long-DCI | 59.221258 / 77.808471 / 83.385951 | 60.378848 / 78.453039 / 83.649040 |

D3 Balanced delta(pp): `{"Score5": -0.15900436793435802, "J_long3": -0.19767394655724502, "J_long": -0.03999947547912086, "Short4": -0.10099999999999909}`.

HNS-v1 delta(pp): `{"Score5": -0.03661794198409041, "J_long3": -0.07436323664015276, "J_long": 0.01999947547913905, "Short4": 0.01999999999999602}`.

All30 recalls and deltas: RESULTS.json. Keep/violation/IoU/CE/share/group-gradient/gate: TRAINING_DIAGNOSTICS.json.
No inference mask/gate/rerank/ensemble/TTA. GradScaler remains disabled under original BF16 autocast.
Source/checkpoint/RNG/cursor identity and full actual sample/text/token/index/LR gates are recorded. No cross-run bitwise-loss requirement.
`/root` disposable image cache; persistent NFS originals retained. Binaries/raw logs stay in canonical runtime, never Git.
Recommendation only; this runner cannot continue to3651 or launch any extra experiment.
