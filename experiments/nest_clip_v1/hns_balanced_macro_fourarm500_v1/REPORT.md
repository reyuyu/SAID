# HNS + D3 Balanced macro-loss four-arm search @500

| Model | lambdaA | lambdaS | lambdaH | Score5 | J_long3 | J_long | Short4 | Urban I2T/T2I |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| HNS-v1 | 10 | 1 | 1 | 71.202056 | 75.152094 | 84.160001 | 65.277000 | 91.500 / 89.700 |
| E1-HNS-A8 | 8.0 | 1.0 | 1.0 | 71.124115 | 74.933526 | 83.990002 | 65.410000 | 91.000 / 89.500 |
| D3 Balanced | 10 | 1 | 1 | 71.063600 | 74.954666 | 84.025001 | 65.227000 | 90.900 / 89.700 |
| E2-Balanced-A8 | 8.0 | 1.0 | 1.0 | 70.978999 | 74.777664 | 83.845002 | 65.281000 | 90.900 / 89.000 |
| E3-Balanced-S12 | 10.0 | 1.2 | 1.0 | 71.099370 | 74.943616 | 84.015003 | 65.333000 | 91.300 / 89.300 |
| E4-Balanced-H125 | 10.0 | 1.0 | 1.25 | 71.075560 | 74.914599 | 84.080002 | 65.317000 | 91.500 / 89.300 |

| Arm | Baseline | Delta Score5 | Delta J_long3 | Delta J_long | Delta Short4 | Delta Urban I2T/T2I | Delta vs HNS Score5 |
|---|---|---:|---:|---:|---:|---|---:|
| E1-HNS-A8 | HNS | -0.077941 | -0.218568 | -0.169999 | +0.133000 | -0.500 / -0.200 | -0.077941 |
| E2-Balanced-A8 | Balanced | -0.084601 | -0.177002 | -0.179999 | +0.054000 | +0.000 / -0.700 | -0.223058 |
| E3-Balanced-S12 | Balanced | +0.035770 | -0.011050 | -0.009999 | +0.106000 | +0.400 / -0.400 | -0.102687 |
| E4-Balanced-H125 | Balanced | +0.011960 | -0.040067 | +0.055001 | +0.090000 | +0.600 / -0.400 | -0.126497 |

Selection: `{"BEST_HNS_ARM": "HNS-v1", "BEST_BALANCED_ARM": "E3-Balanced-S12", "BEST_OVERALL_500": "HNS-v1", "RECOMMENDED_NEXT_VALIDATION": "KEEP_HNS_V1", "overall_gain_pp": 0.0, "weak_single_seed_signal": false, "automatic_continuation": false}`.
Every dataset/direction R@1/5/10 and all deltas are in per-arm RESULTS.json and the aggregate models/comparisons.
Single seed; improvement below0.05pp is a weak signal.500 rankings need not scale toE3/full training.
All arms stopped500; GPUs idle. No fifth arm, combinations, seed sweep or1217/2434/3651/4868 continuation.
