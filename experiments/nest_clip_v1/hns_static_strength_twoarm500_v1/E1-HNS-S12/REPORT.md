# E1-HNS-S12: HNS macro-loss @500

Fresh common0, smoke5 independently, exactly500 formal updates. Macro coefficients: (10.0, 1.2, 1.0).
View weights1.35/1.35/.30, sparsity1/2/2, fixedK3, original200-step ramp, original native protocol frozen.
Hierarchy: Hard-ST hard support, beta2/2, no-SG, soft inclusion0.

| Dataset | I2T R1/R5/R10 (%) | T2I R1/R5/R10 (%) |
|---|---|---|
| COCO | 60.560000 / 83.020000 / 89.280000 | 41.552000 / 67.288000 / 76.920000 |
| Urban-1k | 91.400003 / 98.600006 / 99.600005 | 89.500004 / 98.700005 / 99.500006 |
| Flickr30k-test1k | 87.800000 / 97.600000 / 98.900000 | 72.200000 / 91.560000 / 95.220000 |
| DOCCI | 78.120000 / 95.540000 / 98.100000 | 77.920000 / 95.220000 / 97.940000 |
| Long-DCI | 57.142857 / 75.887924 / 81.978427 | 56.879768 / 76.098395 / 82.031044 |

Scores: `{"Score5": 71.3074633111179, "J_long3": 75.16043885186315, "J_long": 84.23500187158585, "Short4": 65.52799999999999, "Urban_I2T": 91.40000343322754, "Urban_T2I": 89.50000405311584}`.
Delta vs own baseline(pp): `{"Score5": 0.10540696345462663, "J_long3": 0.008344939091031733, "J_long": 0.07500096559525105, "Short4": 0.25099999999999056, "Urban_I2T": -0.09999871253967285, "Urban_T2I": -0.1999974250793457}`.
Delta vs HNS-v1(pp): `{"Score5": 0.10540696345462663, "J_long3": 0.008344939091031733, "J_long": 0.07500096559525105, "Short4": 0.25099999999999056, "Urban_I2T": -0.09999871253967285, "Urban_T2I": -0.1999974250793457}`.
All30 recall deltas in RESULTS.json. Actual differentiated component/view gradients and cosines: GRADIENT_AUDIT.json.
Samples/text/tokens/indices/LR matched512000 records. No mask/rerank/ensemble/TTA inference.
Single seed500; gains below0.05pp are weak signals. No automatic full, third arm, combination or additional seed.
/root cache disposable; persistent NFS originals retained. Weights, images and large raw logs remain local.

Pure coefficient control and trained-state audit are reported separately. H4 was fixed4 throughout; no intermediate retrieval-based adjustment.
