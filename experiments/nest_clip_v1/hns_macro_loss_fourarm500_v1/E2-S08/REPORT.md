# E2-S08: HNS macro-loss500

Macro scales: `{'lambda_align': 10.0, 'lambda_sparse': 0.8, 'lambda_hierarchy': 1.0}`. Independent smoke5 and fresh common0 formal500. Horizon4868; local-only.
Internal alignment1.35/1.35/.30, sparsity1/2/2, beta2/2, no SG, ramp200, fixedK3 unchanged.

| Dataset | I2T R1/R5/R10 | T2I R1/R5/R10 |
|---|---|---|
| COCO | 60.480000 / 82.880000 / 89.260000 | 41.372000 / 67.332000 / 76.944000 |
| Urban-1k | 91.200006 / 98.700005 / 99.700004 | 89.200002 / 98.700005 / 99.500006 |
| Flickr30k-test1k | 87.200000 / 97.300000 / 99.300000 | 71.800000 / 91.540000 / 95.200000 |
| DOCCI | 78.060000 / 95.480000 / 98.100000 | 77.800000 / 95.300000 / 97.920000 |
| Long-DCI | 56.932386 / 75.887924 / 81.767956 | 56.682452 / 75.953696 / 81.873191 |

Quality: `{"Score5": 71.07268461636541, "J_long3": 74.97914102727569, "J_long": 84.06500199079514, "Short4": 65.213, "Urban_I2T": 91.2000060081482, "Urban_T2I": 89.20000195503235}`.
Deltas vs original HNS-v1(pp): `{"Score5": -0.12937173129786572, "J_long3": -0.17295288549642862, "J_long": -0.09499891519546111, "Short4": -0.06400000000000716, "Urban_I2T": -0.29999613761901855, "Urban_T2I": -0.4999995231628418}`.
All30 recall deltas: RESULTS.json. Actual component gradients/cosines: GRADIENT_AUDIT.json, matched immutable first global1024 batch atstep500.
Last50 CE/share/keep/IoU/violations/equality: TRAINING_DIAGNOSTICS.json and MASK_HIERARCHY_AUDIT.json.
No inference mask/gate/rerank/ensemble/TTA. No long continuation or combination. Single seed500 only; <0.05pp gains unconfirmed.
/root cache is ephemeral; persistent NFS originals retained. Weights/images/cache/raw large logs never uploaded.
