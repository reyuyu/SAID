# E1-AlignMatched: HNS-S12 sparse-ratio @500

Fresh common0, smoke5 and formal500. Normalized ratio `4.5:4.5:1`; weights `[2.25, 2.25, 0.5]`; effective coefficients `[2.6999999999999997, 2.6999999999999997, 0.6]`.
HNS beta2/2, Hard-ST, no-SG, K3, lambda_align10, lambda_sparse1.2, lambda_hierarchy1, original ramp200.

| Dataset | I2T R1/R5/R10 (%) | T2I R1/R5/R10 (%) |
|---|---|---|
| COCO | 60.740000 / 83.000000 / 89.340000 | 41.568000 / 67.324000 / 76.892000 |
| Urban-1k | 91.000003 / 98.600006 / 99.600005 | 89.700001 / 98.500007 / 99.500006 |
| Flickr30k-test1k | 87.700000 / 97.400000 / 99.200000 | 71.820000 / 91.620000 / 95.100000 |
| DOCCI | 78.000000 / 95.460000 / 98.120000 | 77.960000 / 95.120000 / 97.840000 |
| Long-DCI | 57.340174 / 75.835306 / 81.781110 | 56.721915 / 75.887924 / 81.965272 |

Scores: `{"Score5": 71.25500930247671, "J_long3": 75.12034883746117, "J_long": 84.1650010251999, "Short4": 65.457, "Urban_I2T": 91.00000262260437, "Urban_T2I": 89.70000147819519}`.
Delta vs HNS-S12 baseline(pp): `{"Score5": -0.05245400864119176, "J_long3": -0.0400900144019829, "J_long": -0.07000084638595183, "Short4": -0.07099999999999795, "Urban_I2T": -0.40000081062316895, "Urban_T2I": 0.1999974250793457, "Urban_Mean": -0.10000169277191162}`.
Weighted per-view sparsity contribution and HNS hierarchy telemetry are in TRAINING_DIAGNOSTICS.json; gradients/cosines are in GRADIENT_AUDIT.json.
All512000 sample IDs, F/Dall/D3 text/tokens/indices and LR matched. Native full-caption inference only.
Single seed500 exploratory result; no continuation, third arm, coefficient combination or other K.
