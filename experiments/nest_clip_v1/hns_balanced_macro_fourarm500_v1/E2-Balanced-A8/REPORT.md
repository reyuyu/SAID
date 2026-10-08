# E2-Balanced-A8: Balanced macro-loss @500

Fresh common0, smoke5 independently, exactly500 formal updates. Macro coefficients: (8.0, 1.0, 1.0).
View weights1.35/1.35/.30, sparsity1/2/2, fixedK3, original200-step ramp, original native protocol frozen.
Hierarchy: Soft probability detached-child adjacent chain; HNS disabled; inclusion_max1 scaled once.

| Dataset | I2T R1/R5/R10 (%) | T2I R1/R5/R10 (%) |
|---|---|---|
| COCO | 60.500000 / 82.800000 / 89.460000 | 41.524000 / 67.260000 / 76.888000 |
| Urban-1k | 90.900004 / 98.800004 / 99.700004 | 89.000005 / 98.600006 / 99.400002 |
| Flickr30k-test1k | 87.300000 / 97.400000 / 99.300000 | 71.800000 / 91.300000 / 95.120000 |
| DOCCI | 77.920000 / 95.440000 / 98.120000 | 77.560000 / 95.100000 / 97.800000 |
| Long-DCI | 56.800842 / 75.466982 / 81.754801 | 56.485135 / 75.940542 / 81.728493 |

Scores: `{"Score5": 70.97899858143928, "J_long3": 74.77766430239882, "J_long": 83.84500211000443, "Short4": 65.281, "Urban_I2T": 90.9000039100647, "Urban_T2I": 89.000004529953}`.
Delta vs own baseline(pp): `{"Score5": -0.084601220738449, "J_long3": -0.17700203456408303, "J_long": -0.17999923706054233, "Short4": 0.054000000000002046, "Urban_I2T": 0.0, "Urban_T2I": -0.6999969482421875}`.
Delta vs HNS-v1(pp): `{"Score5": -0.2230577662239881, "J_long3": -0.3744296103732978, "J_long": -0.3149987959861704, "Short4": 0.0040000000000048885, "Urban_I2T": -0.5999982357025146, "Urban_T2I": -0.6999969482421875}`.
All30 recall deltas in RESULTS.json. Actual differentiated component/view gradients and cosines: GRADIENT_AUDIT.json.
Samples/text/tokens/indices/LR matched512000 records. No mask/rerank/ensemble/TTA inference.
Single seed500; gains below0.05pp are weak signals. No automatic full, fifth arm, combination or additional seed.
/root cache disposable; persistent NFS originals retained. Weights, images and large raw logs remain local.
