# E1-HNS-A8: HNS macro-loss @500

Fresh common0, smoke5 independently, exactly500 formal updates. Macro coefficients: (8.0, 1.0, 1.0).
View weights1.35/1.35/.30, sparsity1/2/2, fixedK3, original200-step ramp, original native protocol frozen.
Hierarchy: Hard-ST hard support, beta2/2, no-SG, soft inclusion0.

| Dataset | I2T R1/R5/R10 (%) | T2I R1/R5/R10 (%) |
|---|---|---|
| COCO | 60.660000 / 82.860000 / 89.400000 | 41.540000 / 67.252000 / 76.844000 |
| Urban-1k | 91.000003 / 98.900002 / 99.800003 | 89.500004 / 98.800004 / 99.400002 |
| Flickr30k-test1k | 87.400000 / 97.500000 / 99.100000 | 72.040000 / 91.680000 / 95.140000 |
| DOCCI | 77.740000 / 95.520000 / 98.100000 | 77.720000 / 95.200000 / 97.860000 |
| Long-DCI | 56.998158 / 75.716917 / 81.728493 | 56.642989 / 76.151013 / 82.083662 |

Scores: `{"Score5": 71.12411537422815, "J_long3": 74.93352562371362, "J_long": 83.99000166893005, "Short4": 65.41, "Urban_I2T": 91.00000262260437, "Urban_T2I": 89.50000405311584}`.
Delta vs own baseline(pp): `{"Score5": -0.07794097343511908, "J_long3": -0.21856828905849568, "J_long": -0.16999923706055142, "Short4": 0.13299999999999557, "Urban_I2T": -0.4999995231628418, "Urban_T2I": -0.1999974250793457}`.
Delta vs HNS-v1(pp): `{"Score5": -0.07794097343511908, "J_long3": -0.21856828905849568, "J_long": -0.16999923706055142, "Short4": 0.13299999999999557, "Urban_I2T": -0.4999995231628418, "Urban_T2I": -0.1999974250793457}`.
All30 recall deltas in RESULTS.json. Actual differentiated component/view gradients and cosines: GRADIENT_AUDIT.json.
Samples/text/tokens/indices/LR matched512000 records. No mask/rerank/ensemble/TTA inference.
Single seed500; gains below0.05pp are weak signals. No automatic full, fifth arm, combination or additional seed.
/root cache disposable; persistent NFS originals retained. Weights, images and large raw logs remain local.
