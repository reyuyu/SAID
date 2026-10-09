# E2-HNS-H4: HNS macro-loss @500

Fresh common0, smoke5 independently, exactly500 formal updates. Macro coefficients: (10.0, 1.0, 4.0).
View weights1.35/1.35/.30, sparsity1/2/2, fixedK3, original200-step ramp, original native protocol frozen.
Hierarchy: Hard-ST hard support, beta2/2, no-SG, soft inclusion0.

| Dataset | I2T R1/R5/R10 (%) | T2I R1/R5/R10 (%) |
|---|---|---|
| COCO | 60.380000 / 83.040000 / 89.240000 | 41.616000 / 67.392000 / 76.916000 |
| Urban-1k | 91.100007 / 98.600006 / 99.800003 | 89.300007 / 98.600006 / 99.500006 |
| Flickr30k-test1k | 87.300000 / 97.500000 / 99.100000 | 71.880000 / 91.480000 / 95.160000 |
| DOCCI | 77.800000 / 95.380000 / 98.100000 | 77.580000 / 95.180000 / 97.920000 |
| Long-DCI | 56.906077 / 75.598527 / 81.767956 | 56.629834 / 76.072086 / 81.820574 |

Scores: `{"Score5": 71.04919255258551, "J_long3": 74.8859875876425, "J_long": 83.94500348091125, "Short4": 65.294, "Urban_I2T": 91.10000729560852, "Urban_T2I": 89.3000066280365}`.
Delta vs own baseline(pp): `{"Score5": -0.1528637950777636, "J_long3": -0.26610632512961274, "J_long": -0.21499742507934627, "Short4": 0.016999999999995907, "Urban_I2T": -0.3999948501586914, "Urban_T2I": -0.3999948501586914}`.
Delta vs HNS-v1(pp): `{"Score5": -0.1528637950777636, "J_long3": -0.26610632512961274, "J_long": -0.21499742507934627, "Short4": 0.016999999999995907, "Urban_I2T": -0.3999948501586914, "Urban_T2I": -0.3999948501586914}`.
All30 recall deltas in RESULTS.json. Actual differentiated component/view gradients and cosines: GRADIENT_AUDIT.json.
Samples/text/tokens/indices/LR matched512000 records. No mask/rerank/ensemble/TTA inference.
Single seed500; gains below0.05pp are weak signals. No automatic full, third arm, combination or additional seed.
/root cache disposable; persistent NFS originals retained. Weights, images and large raw logs remain local.

Pure coefficient control and trained-state audit are reported separately. H4 was fixed4 throughout; no intermediate retrieval-based adjustment.
