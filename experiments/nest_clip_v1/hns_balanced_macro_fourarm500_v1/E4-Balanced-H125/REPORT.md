# E4-Balanced-H125: Balanced macro-loss @500

Fresh common0, smoke5 independently, exactly500 formal updates. Macro coefficients: (10.0, 1.0, 1.25).
View weights1.35/1.35/.30, sparsity1/2/2, fixedK3, original200-step ramp, original native protocol frozen.
Hierarchy: Soft probability detached-child adjacent chain; HNS disabled; inclusion_max1 scaled once.

| Dataset | I2T R1/R5/R10 (%) | T2I R1/R5/R10 (%) |
|---|---|---|
| COCO | 60.480000 / 82.820000 / 89.180000 | 41.508000 / 67.292000 / 76.912000 |
| Urban-1k | 91.500002 / 98.500007 / 99.800003 | 89.300007 / 98.600006 / 99.500006 |
| Flickr30k-test1k | 87.600000 / 97.500000 / 99.200000 | 71.680000 / 91.440000 / 95.240000 |
| DOCCI | 77.760000 / 95.500000 / 98.060000 | 77.760000 / 95.160000 / 97.820000 |
| Long-DCI | 56.735070 / 75.348592 / 81.636411 | 56.432518 / 75.940542 / 81.938963 |

Scores: `{"Score5": 71.07555962507834, "J_long3": 74.91459937513058, "J_long": 84.08000219345094, "Short4": 65.31700000000001, "Urban_I2T": 91.50000214576721, "Urban_T2I": 89.3000066280365}`.
Delta vs own baseline(pp): `{"Score5": 0.011959822900607264, "J_long3": -0.0400669618323235, "J_long": 0.055000846385965474, "Short4": 0.09000000000000341, "Urban_I2T": 0.5999982357025146, "Urban_T2I": -0.3999948501586914}`.
Delta vs HNS-v1(pp): `{"Score5": -0.12649672258493183, "J_long3": -0.23749453764153827, "J_long": -0.07999871253966262, "Short4": 0.04000000000000625, "Urban_I2T": 0.0, "Urban_T2I": -0.3999948501586914}`.
All30 recall deltas in RESULTS.json. Actual differentiated component/view gradients and cosines: GRADIENT_AUDIT.json.
Samples/text/tokens/indices/LR matched512000 records. No mask/rerank/ensemble/TTA inference.
Single seed500; gains below0.05pp are weak signals. No automatic full, fifth arm, combination or additional seed.
/root cache disposable; persistent NFS originals retained. Weights, images and large raw logs remain local.
