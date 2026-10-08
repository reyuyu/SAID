# E3-Balanced-S12: Balanced macro-loss @500

Fresh common0, smoke5 independently, exactly500 formal updates. Macro coefficients: (10.0, 1.2, 1.0).
View weights1.35/1.35/.30, sparsity1/2/2, fixedK3, original200-step ramp, original native protocol frozen.
Hierarchy: Soft probability detached-child adjacent chain; HNS disabled; inclusion_max1 scaled once.

| Dataset | I2T R1/R5/R10 (%) | T2I R1/R5/R10 (%) |
|---|---|---|
| COCO | 60.340000 / 82.760000 / 89.280000 | 41.592000 / 67.324000 / 76.952000 |
| Urban-1k | 91.300005 / 98.600006 / 99.700004 | 89.300007 / 98.600006 / 99.500006 |
| Flickr30k-test1k | 87.600000 / 97.400000 / 99.000000 | 71.800000 / 91.400000 / 95.140000 |
| DOCCI | 77.680000 / 95.420000 / 98.040000 | 77.780000 / 95.180000 / 97.760000 |
| Long-DCI | 56.998158 / 75.493291 / 81.623257 | 56.603525 / 75.822152 / 82.057353 |

Scores: `{"Score5": 71.0993695116154, "J_long3": 74.94361585269233, "J_long": 84.0150028371811, "Short4": 65.333, "Urban_I2T": 91.30000472068787, "Urban_T2I": 89.3000066280365}`.
Delta vs own baseline(pp): `{"Score5": 0.035769709437659, "J_long3": -0.011050484270569427, "J_long": -0.00999850988387152, "Short4": 0.10599999999999454, "Urban_I2T": 0.40000081062316895, "Urban_T2I": -0.3999948501586914}`.
Delta vs HNS-v1(pp): `{"Score5": -0.10268683604788009, "J_long3": -0.2084780600797842, "J_long": -0.14499806880949961, "Short4": 0.055999999999997385, "Urban_I2T": -0.1999974250793457, "Urban_T2I": -0.3999948501586914}`.
All30 recall deltas in RESULTS.json. Actual differentiated component/view gradients and cosines: GRADIENT_AUDIT.json.
Samples/text/tokens/indices/LR matched512000 records. No mask/rerank/ensemble/TTA inference.
Single seed500; gains below0.05pp are weak signals. No automatic full, fifth arm, combination or additional seed.
/root cache disposable; persistent NFS originals retained. Weights, images and large raw logs remain local.
