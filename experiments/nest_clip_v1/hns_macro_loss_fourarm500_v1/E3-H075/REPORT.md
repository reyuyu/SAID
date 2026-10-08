# E3-H075: HNS macro-loss500

Macro scales: `{'lambda_align': 10.0, 'lambda_sparse': 1.0, 'lambda_hierarchy': 0.75}`. Independent smoke5 and fresh common0 formal500. Horizon4868; local-only.
Internal alignment1.35/1.35/.30, sparsity1/2/2, beta2/2, no SG, ramp200, fixedK3 unchanged.

| Dataset | I2T R1/R5/R10 | T2I R1/R5/R10 |
|---|---|---|
| COCO | 60.640000 / 82.780000 / 89.220000 | 41.560000 / 67.356000 / 76.944000 |
| Urban-1k | 91.200006 / 98.500007 / 99.800003 | 89.700001 / 98.500007 / 99.500006 |
| Flickr30k-test1k | 87.700000 / 97.600000 / 98.900000 | 71.740000 / 91.480000 / 95.240000 |
| DOCCI | 77.840000 / 95.440000 / 98.020000 | 77.660000 / 95.040000 / 97.860000 |
| Long-DCI | 57.077085 / 75.598527 / 81.754801 | 56.827151 / 75.980005 / 81.991581 |

Quality: `{"Score5": 71.19442432137835, "J_long3": 75.05070720229726, "J_long": 84.10000187158585, "Short4": 65.41, "Urban_I2T": 91.2000060081482, "Urban_T2I": 89.70000147819519}`.
Deltas vs original HNS-v1(pp): `{"Score5": -0.007632026284923654, "J_long3": -0.10138671047485559, "J_long": -0.05999903440475407, "Short4": 0.13299999999999557, "Urban_I2T": -0.29999613761901855, "Urban_T2I": 0.0}`.
All30 recall deltas: RESULTS.json. Actual component gradients/cosines: GRADIENT_AUDIT.json, matched immutable first global1024 batch atstep500.
Last50 CE/share/keep/IoU/violations/equality: TRAINING_DIAGNOSTICS.json and MASK_HIERARCHY_AUDIT.json.
No inference mask/gate/rerank/ensemble/TTA. No long continuation or combination. Single seed500 only; <0.05pp gains unconfirmed.
/root cache is ephemeral; persistent NFS originals retained. Weights/images/cache/raw large logs never uploaded.
