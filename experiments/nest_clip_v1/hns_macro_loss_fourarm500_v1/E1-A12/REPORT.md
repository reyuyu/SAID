# E1-A12: HNS macro-loss500

Macro scales: `{'lambda_align': 12.0, 'lambda_sparse': 1.0, 'lambda_hierarchy': 1.0}`. Independent smoke5 and fresh common0 formal500. Horizon4868; local-only.
Internal alignment1.35/1.35/.30, sparsity1/2/2, beta2/2, no SG, ramp200, fixedK3 unchanged.

| Dataset | I2T R1/R5/R10 | T2I R1/R5/R10 |
|---|---|---|
| COCO | 60.460000 / 82.760000 / 89.160000 | 41.448000 / 67.368000 / 76.844000 |
| Urban-1k | 91.200006 / 98.700005 / 99.700004 | 89.500004 / 98.700005 / 99.400002 |
| Flickr30k-test1k | 87.200000 / 97.600000 / 99.100000 | 71.680000 / 91.320000 / 95.020000 |
| DOCCI | 77.780000 / 95.400000 / 98.060000 | 77.820000 / 95.200000 / 97.760000 |
| Long-DCI | 56.564062 / 75.322284 / 81.557485 | 56.761379 / 76.019469 / 81.807419 |

Quality: `{"Score5": 71.0413450734771, "J_long3": 74.93757512246184, "J_long": 84.07500251531602, "Short4": 65.197, "Urban_I2T": 91.2000060081482, "Urban_T2I": 89.50000405311584}`.
Deltas vs original HNS-v1(pp): `{"Score5": -0.16071127418616982, "J_long3": -0.21451879031027943, "J_long": -0.08499839067458481, "Short4": -0.0799999999999983, "Urban_I2T": -0.29999613761901855, "Urban_T2I": -0.1999974250793457}`.
All30 recall deltas: RESULTS.json. Actual component gradients/cosines: GRADIENT_AUDIT.json, matched immutable first global1024 batch atstep500.
Last50 CE/share/keep/IoU/violations/equality: TRAINING_DIAGNOSTICS.json and MASK_HIERARCHY_AUDIT.json.
No inference mask/gate/rerank/ensemble/TTA. No long continuation or combination. Single seed500 only; <0.05pp gains unconfirmed.
/root cache is ephemeral; persistent NFS originals retained. Weights/images/cache/raw large logs never uploaded.
