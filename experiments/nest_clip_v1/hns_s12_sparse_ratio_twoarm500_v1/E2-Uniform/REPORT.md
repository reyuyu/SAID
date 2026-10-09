# E2-Uniform: HNS-S12 sparse-ratio @500

Fresh common0, smoke5 and formal500. Normalized ratio `1:1:1`; weights `[1.6666666666666667, 1.6666666666666667, 1.6666666666666667]`; effective coefficients `[2.0, 2.0, 2.0]`.
HNS beta2/2, Hard-ST, no-SG, K3, lambda_align10, lambda_sparse1.2, lambda_hierarchy1, original ramp200.

| Dataset | I2T R1/R5/R10 (%) | T2I R1/R5/R10 (%) |
|---|---|---|
| COCO | 60.580000 / 82.940000 / 89.380000 | 41.560000 / 67.304000 / 76.984000 |
| Urban-1k | 91.100007 / 98.600006 / 99.700004 | 90.000004 / 98.600006 / 99.500006 |
| Flickr30k-test1k | 88.100000 / 97.600000 / 99.000000 | 72.180000 / 91.560000 / 95.240000 |
| DOCCI | 78.020000 / 95.320000 / 98.180000 | 77.660000 / 95.080000 / 97.740000 |
| Long-DCI | 57.156012 / 75.585372 / 81.675875 | 56.787687 / 76.032623 / 81.925809 |

Scores: `{"Score5": 71.31437098984591, "J_long3": 75.12061831640985, "J_long": 84.1950027179718, "Short4": 65.605, "Urban_I2T": 91.10000729560852, "Urban_T2I": 90.00000357627869}`.
Delta vs HNS-S12 baseline(pp): `{"Score5": 0.006907678728012456, "J_long3": -0.039820535453301886, "J_long": -0.0399991536140476, "Short4": 0.07700000000001239, "Urban_I2T": -0.29999613761901855, "Urban_T2I": 0.4999995231628418, "Urban_Mean": 0.10000169277191162}`.
Weighted per-view sparsity contribution and HNS hierarchy telemetry are in TRAINING_DIAGNOSTICS.json; gradients/cosines are in GRADIENT_AUDIT.json.
All512000 sample IDs, F/Dall/D3 text/tokens/indices and LR matched. Native full-caption inference only.
Single seed500 exploratory result; no continuation, third arm, coefficient combination or other K.
