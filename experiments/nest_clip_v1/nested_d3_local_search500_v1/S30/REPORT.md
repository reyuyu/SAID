# S30: isolated Nested D3 local500 search arm

Completed exactly500 fresh-common0 optimizer updates, horizon4868. Only changed axis:`sparsity`.
Alignment:[1.35, 1.35, 0.3]; sparsity coefficients:[0.8333333333333334, 1.6666666666666667, 2.5] (mass5); mode:nested_detail_d3.
Other construction,optimizer/LR,workers8,batch256/rank,inclusion chain/ramp200/max1,preprocess and native protocol frozen. No combination/full run.
Local-only:`/root/said_s02_stage500/ShareGPT4V`; missing/symlink/escape fails, NFS fallback forbidden. /root is disposable overlay; NFS originals retained.

| Dataset | I2T R@1 / R@5 / R@10 (%) | T2I R@1 / R@5 / R@10 (%) |
|---|---|---|
| COCO | 60.540000 / 82.640000 / 89.240000 | 41.592000 / 67.324000 / 76.916000 |
| Urban-1k | 90.900004 / 98.700005 / 99.600005 | 89.500004 / 98.500007 / 99.500006 |
| Flickr30k-test1k | 86.800000 / 97.600000 / 99.100000 | 71.860000 / 91.280000 / 95.240000 |
| DOCCI | 77.780000 / 95.420000 / 98.040000 | 77.740000 / 95.080000 / 97.760000 |
| Long-DCI | 56.906077 / 75.598527 / 81.860037 | 56.813996 / 76.072086 / 81.991581 |

| Metric | Arm | Delta vs Anchor(pp) |
|---|---:|---:|
| Score5 | 71.043208 | -0.020392 |
| J_long3 | 74.940014 | -0.014653 |
| J_long | 83.980002 | -0.044999 |
| Short4 | 65.198000 | -0.029000 |
| Urban_I2T | 90.900000 | +0.000000 |
| Urban_T2I | 89.500000 | -0.200000 |

| Dataset | Delta I2T R@1/5/10(pp) | Delta T2I R@1/5/10(pp) |
|---|---|---|
| COCO | +0.140000 / +0.020000 / -0.020000 | +0.004000 / +0.048000 / -0.016000 |
| Urban-1k | +0.000000 / +0.099999 / +0.000000 | -0.199997 / +0.000000 / +0.000000 |
| Flickr30k-test1k | -0.200000 / -0.200000 / -0.100000 | -0.060000 / -0.100000 / +0.120000 |
| DOCCI | -0.040000 / +0.080000 / +0.040000 | +0.060000 / -0.180000 / +0.040000 |
| Long-DCI | -0.118390 / +0.065772 / +0.210471 | +0.210471 / -0.013154 / +0.052618 |

| View | Combined CE | Weighted CE | Alignment share | Keep ratio |
|---|---:|---:|---:|---:|
| F | 0.110070 | 0.495316 | 17.347% | 0.866752 |
| Dall | 0.205748 | 0.925868 | 32.426% | 0.851811 |
| D3 | 1.434166 | 1.434166 | 50.227% | 0.754613 |

Raw gradient norms:`{'F': 11.753302216529846, 'Dall': 15.194415926933289, 'D3': 35.178797245025635}`.
Weighted gradient norms:`{'F': 52.889859974384315, 'Dall': 68.37487167119981, 'D3': 35.178797245025635}`; weighted D3/Dall ratio:`0.514499`.
Frozen gradient protocol:8 first seed0 epoch0 global1024 batches at fixed step500; production backbone group; raw directional-summed CE,all-reduce/4; no optimizer updates. Norm magnitudes are not a signed gradient decomposition.
Mask hierarchy:`{'inc': 0.008761275559663773, 'inc_weight': 1.0, 'F_Dall_mask_iou': 0.9541209137439728, 'Dall_D3_mask_iou': 0.8494160532951355, 'Dall_F_hard_violation': 0.012514383643865585, 'D3_Dall_hard_violation': 0.016895810328423976}`. Soft detached-child inclusion does not impose zero hard-mask violations.
Sampling:`{'records': 512000, 'valid_records': 511936, 'K_histogram_all': {'2': 3924, '3': 507507, '1': 505, '0': 64}, 'K_histogram_valid': {'2': 3924, '3': 507507, '1': 505}, 'K_histogram_by_m': {'8': {'3': 77266}, '5': {'3': 92074}, '6': {'3': 92953}, '9': {'3': 68110}, '4': {'3': 37167}, '10': {'3': 41151}, '12': {'3': 4210}, '7': {'3': 76923}, '11': {'3': 16257}, '3': {'2': 3924}, '13': {'3': 922}, '14': {'3': 232}, '1': {'1': 173}, '2': {'1': 332}, '24': {'3': 4}, '17': {'3': 18}, '15': {'3': 84}, '26': {'3': 5}, '16': {'3': 45}, '95': {'3': 1}, '23': {'3': 3}, '20': {'3': 4}, '21': {'3': 7}, '47': {'3': 2}, '18': {'3': 11}, '29': {'3': 1}, '41': {'3': 1}, '25': {'3': 4}, '51': {'3': 1}, '33': {'3': 2}, '34': {'3': 2}, '22': {'3': 4}, '19': {'3': 12}, '31': {'3': 1}, '30': {'3': 3}, '28': {'3': 3}, '39': {'3': 1}, '101': {'3': 1}, '92': {'3': 2}, '100': {'3': 1}, '32': {'3': 1}, '27': {'3': 3}, '82': {'3': 1}, '105': {'3': 2}, '44': {'3': 1}, '54': {'3': 1}, '79': {'3': 1}, '86': {'3': 1}, '89': {'3': 1}, '56': {'3': 1}, '40': {'3': 1}, '37': {'3': 1}, '99': {'3': 1}, '87': {'3': 1}, '77': {'3': 1}, '74': {'3': 1}}, 'm_histogram': {'3': 3924, '4': 37167, '5': 92074, '6': 92953, '7': 76923, '8': 77266, '9': 68110, '10': 41151, '11': 16257, '12': 4210, '13': 922, '14': 232, '1': 173, '2': 332, '24': 4, '17': 18, '15': 84, '0': 64, '26': 5, '16': 45, '95': 1, '23': 3, '20': 4, '21': 7, '47': 2, '18': 11, '29': 1, '41': 1, '25': 4, '51': 1, '33': 2, '34': 2, '22': 4, '19': 12, '31': 1, '30': 3, '28': 3, '39': 1, '101': 1, '92': 2, '100': 1, '32': 1, '27': 3, '82': 1, '105': 2, '44': 1, '54': 1, '79': 1, '86': 1, '89': 1, '56': 1, '40': 1, '37': 1, '99': 1, '87': 1, '77': 1, '74': 1}, 'mean_K_valid': 2.9903620765095638, 'mean_K_all': 2.98998828125, 'Dall_mean_sentences': 7.0531414083010375, 'strict_subset_ratio': 0.999537109375, 'mean_effective_tokens': {'F': 171.46915239404925, 'Dall': 153.89054100512564, 'D3': 66.38841183272909}, 'mean_content_tokens': {'F': 169.46915239404925, 'Dall': 151.89054100512564, 'D3': 64.38841183272909}, 'mean_lowest_Dall_content_token_coverage': 0.45601314895918665, 'pooled_lowest_Dall_content_token_coverage': 0.42391324309363193, 'coverage_definition': 'Actual EOT-delimited content tokens excluding SOT/EOT on identical valid samples'}`.
Full cycle(s):`{'count': 500, 'median': 2.1248598098754883, 'p95': 2.2881654858589173, 'p99': 2.3658046340942382, 'max': 27.475318908691406}`; slowest-rank data_wait(s):`{'count': 500, 'median': 0.000684637576341629, 'p95': 0.0009610153734683966, 'p99': 0.0014613378793001174, 'max': 24.39516371488571}`.
OOM kills:0; true training I/O errors:0; Pod anomaly:false. GPU peak:`{'0': 27.760382175445557, '1': 27.760382175445557, '2': 27.760382175445557, '3': 27.760382175445557}` GiB.
First5 gate passed before update6; all512000 actual IDs,F/Dall token hashes and selected-index digests checked. Weight/sparsity arms also match all D3 text/token hashes exactly to Anchor.
Strict bare/native inference only:normalized native image embedding @ normalized native full-caption text embedding.T. No masks/gates/local-view inference/rerank/ensemble/TTA.
Common0 SHA256:`54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`.
Checkpoint SHA256:`ee6aeba7aaa5ea49bbaa1c7f5c17e2d38d6ee074cfe99382b6ccc6a7a6902b7a`.
Bare SHA256:`5d4a2bc513842897a811ac2d9a038f04039066bf07209ce83086d974f96c6783`.
Checkpoints/bare/raw logs stay local. RUNTIME_STATS.json records raw paths,bytes,SHA256 and UTC containment windows.
Selection/classification and Pareto analysis occur after all seven arms, in SEARCH_SUMMARY.md. No arm-specific tuning or early metric stop.

Final search classification:`NO_IMPROVEMENT`.
