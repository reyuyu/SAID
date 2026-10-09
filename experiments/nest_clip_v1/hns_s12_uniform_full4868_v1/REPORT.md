# E2-Uniform full 4868 validation

Uniform sparsity [5/3,5/3,5/3], effective [2,2,2], fixed macro loss [10,1.2,1].

| Step | Model | Score5 | J_long3 | J_long | Short4 | Urban I2T/T2I | Urban Mean |
|---:|---|---:|---:|---:|---:|---|---:|
| 500 | E2-Uniform | 71.314371 | 75.120618 | 84.195003 | 65.605000 | 91.100 / 90.000 | 90.550 |
| 1217 | E2-Uniform | 72.794967 | 76.922278 | 85.855003 | 66.604000 | 93.000 / 91.800 | 92.400 |
| 2434 | E2-Uniform | 73.479512 | 77.715187 | 86.630002 | 67.126000 | 93.100 / 92.400 | 92.750 |
| 2434 | HNS_S12 | 73.647115 | 77.969192 | 86.840002 | 67.164000 | 93.500 / 92.400 | 92.950 |
| 2434 | HNS_v1 | 73.475830 | 77.824383 | 86.705003 | 66.953000 | 94.000 / 92.500 | 93.250 |
| 2434 | D3_Balanced | 73.598216 | 77.947693 | 86.765002 | 67.074000 | 93.500 / 92.400 | 92.950 |

## Step 2434

| Dataset | I2T R1/R5/R10 (%) | T2I R1/R5/R10 (%) |
|---|---|---|
| COCO | 62.040000 / 83.940000 / 90.020000 | 42.924000 / 68.680000 / 78.044000 |
| Urban-1k | 93.100005 / 98.800004 / 99.600005 | 92.400002 / 99.300003 / 99.400002 |
| Flickr30k-test1k | 89.700000 / 98.600000 / 99.200000 | 73.840000 / 92.140000 / 95.680000 |
| DOCCI | 79.880000 / 96.420000 / 98.560000 | 81.140000 / 96.120000 / 98.340000 |
| Long-DCI | 59.444883 / 77.769008 / 83.675349 | 60.326230 / 78.466193 / 83.372797 |

HNS_S12 deltas (pp): {"J_long": -0.21000020265580588, "J_long3": -0.2540052214846469, "Score5": -0.16760313289080386, "Short4": -0.038000000000010914, "Urban_I2T": -0.40000081062316895, "Urban_T2I": 0.0}

HNS_v1 deltas (pp): {"J_long": -0.07500125169754313, "J_long3": -0.1091957831627468, "Score5": 0.0036825301023526436, "Short4": 0.1729999999999876, "Urban_I2T": -0.9000003337860107, "Urban_T2I": -0.10000467300415039}

D3_Balanced deltas (pp): {"J_long": -0.13500020265580304, "J_long3": -0.23250649307983906, "Score5": -0.11870389584791496, "Short4": 0.0519999999999925, "Urban_I2T": -0.40000081062316895, "Urban_T2I": 0.0}

All production training/loss/sampling/evaluator sources remain checkpoint-manifest identical.
Every node uses complete optimizer/scheduler/RNG/sampler restoration and stops only at 4868.
No fifth epoch, parameter search, arm combination or additional continuation is authorized.
