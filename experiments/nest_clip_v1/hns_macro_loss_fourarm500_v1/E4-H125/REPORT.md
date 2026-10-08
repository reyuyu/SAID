# E4-H125: HNS macro-loss500

Macro scales: `{'lambda_align': 10.0, 'lambda_sparse': 1.0, 'lambda_hierarchy': 1.25}`. Independent smoke5 and fresh common0 formal500. Horizon4868; local-only.
Internal alignment1.35/1.35/.30, sparsity1/2/2, beta2/2, no SG, ramp200, fixedK3 unchanged.

| Dataset | I2T R1/R5/R10 | T2I R1/R5/R10 |
|---|---|---|
| COCO | 60.460000 / 82.800000 / 89.400000 | 41.536000 / 67.376000 / 76.936000 |
| Urban-1k | 91.200006 / 98.600006 / 99.600005 | 89.700001 / 98.700005 / 99.500006 |
| Flickr30k-test1k | 87.500000 / 97.600000 / 99.100000 | 71.920000 / 91.540000 / 95.220000 |
| DOCCI | 77.660000 / 95.380000 / 98.040000 | 77.960000 / 95.160000 / 97.760000 |
| Long-DCI | 56.971850 / 75.559063 / 81.662720 | 56.629834 / 76.151013 / 82.057353 |

Quality: `{"Score5": 71.1537691253773, "J_long3": 75.02028187562884, "J_long": 84.13000187158585, "Short4": 65.354, "Urban_I2T": 91.2000060081482, "Urban_T2I": 89.70000147819519}`.
Deltas vs original HNS-v1(pp): `{"Score5": -0.04828722228597826, "J_long3": -0.13181203714327694, "J_long": -0.029999034404752933, "Short4": 0.07699999999999818, "Urban_I2T": -0.29999613761901855, "Urban_T2I": 0.0}`.
All30 recall deltas: RESULTS.json. Actual component gradients/cosines: GRADIENT_AUDIT.json, matched immutable first global1024 batch atstep500.
Last50 CE/share/keep/IoU/violations/equality: TRAINING_DIAGNOSTICS.json and MASK_HIERARCHY_AUDIT.json.
No inference mask/gate/rerank/ensemble/TTA. No long continuation or combination. Single seed500 only; <0.05pp gains unconfirmed.
/root cache is ephemeral; persistent NFS originals retained. Weights/images/cache/raw large logs never uploaded.
