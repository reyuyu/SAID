# W20: isolated Nested D3 local500 search arm

Completed exactly500 fresh-common0 optimizer updates, horizon4868. Only changed axis:`alignment`.
Alignment:[1.4, 1.4, 0.2]; sparsity coefficients:[1.0, 2.0, 2.0] (mass5); mode:nested_detail_d3.
Other construction,optimizer/LR,workers8,batch256/rank,inclusion chain/ramp200/max1,preprocess and native protocol frozen. No combination/full run.
Local-only:`/root/said_s02_stage500/ShareGPT4V`; missing/symlink/escape fails, NFS fallback forbidden. /root is disposable overlay; NFS originals retained.

| Dataset | I2T R@1 / R@5 / R@10 (%) | T2I R@1 / R@5 / R@10 (%) |
|---|---|---|
| COCO | 60.400000 / 83.040000 / 89.140000 | 41.416000 / 67.312000 / 76.756000 |
| Urban-1k | 91.300005 / 98.400003 / 99.800003 | 89.400005 / 98.800004 / 99.500006 |
| Flickr30k-test1k | 87.600000 / 97.300000 / 99.200000 | 71.720000 / 91.440000 / 95.140000 |
| DOCCI | 77.620000 / 95.460000 / 98.080000 | 77.760000 / 95.260000 / 97.780000 |
| Long-DCI | 56.971850 / 75.730071 / 81.807419 | 56.695606 / 75.848461 / 81.794265 |

| Metric | Arm | Delta vs Anchor(pp) |
|---|---:|---:|
| Score5 | 71.088347 | +0.024747 |
| J_long3 | 74.957911 | +0.003245 |
| J_long | 84.020003 | -0.004999 |
| Short4 | 65.284000 | +0.057000 |
| Urban_I2T | 91.300000 | +0.400000 |
| Urban_T2I | 89.400000 | -0.300000 |

| Dataset | Delta I2T R@1/5/10(pp) | Delta T2I R@1/5/10(pp) |
|---|---|---|
| COCO | +0.000000 / +0.420000 / -0.120000 | -0.172000 / +0.036000 / -0.176000 |
| Urban-1k | +0.400001 / -0.200003 / +0.199997 | -0.299996 / +0.299996 / +0.000000 |
| Flickr30k-test1k | +0.600000 / -0.500000 / +0.000000 | -0.200000 / +0.060000 / +0.020000 |
| DOCCI | -0.200000 / +0.120000 / +0.080000 | +0.080000 / +0.000000 / +0.060000 |
| Long-DCI | -0.052618 / +0.197316 / +0.157853 | +0.092081 / -0.236780 / -0.144699 |

| View | Combined CE | Weighted CE | Alignment share | Keep ratio |
|---|---:|---:|---:|---:|
| F | 0.109777 | 0.512292 | 20.497% | 0.850709 |
| Dall | 0.207287 | 0.967339 | 38.703% | 0.828604 |
| D3 | 1.529599 | 1.019733 | 40.800% | 0.730469 |

Raw gradient norms:`{'F': 12.149347186088562, 'Dall': 15.94963026046753, 'D3': 36.39502143859863}`.
Weighted gradient norms:`{'F': 56.69695353507996, 'Dall': 74.43160788218181, 'D3': 24.263347625732425}`; weighted D3/Dall ratio:`0.325982`.
Frozen gradient protocol:8 first seed0 epoch0 global1024 batches at fixed step500; production backbone group; raw directional-summed CE,all-reduce/4; no optimizer updates. Norm magnitudes are not a signed gradient decomposition.
Mask hierarchy:`{'inc': 0.010012591928243638, 'inc_weight': 1.0, 'F_Dall_mask_iou': 0.9385275733470917, 'Dall_D3_mask_iou': 0.8364172017574311, 'Dall_F_hard_violation': 0.015324277505278587, 'D3_Dall_hard_violation': 0.020560262091457844}`. Soft detached-child inclusion does not impose zero hard-mask violations.
Sampling:`{'records': 512000, 'valid_records': 511936, 'K_histogram_all': {'2': 3924, '3': 507507, '1': 505, '0': 64}, 'K_histogram_valid': {'2': 3924, '3': 507507, '1': 505}, 'K_histogram_by_m': {'8': {'3': 77266}, '5': {'3': 92074}, '6': {'3': 92953}, '9': {'3': 68110}, '4': {'3': 37167}, '10': {'3': 41151}, '12': {'3': 4210}, '7': {'3': 76923}, '11': {'3': 16257}, '3': {'2': 3924}, '13': {'3': 922}, '14': {'3': 232}, '1': {'1': 173}, '2': {'1': 332}, '24': {'3': 4}, '17': {'3': 18}, '15': {'3': 84}, '26': {'3': 5}, '16': {'3': 45}, '95': {'3': 1}, '23': {'3': 3}, '20': {'3': 4}, '21': {'3': 7}, '47': {'3': 2}, '18': {'3': 11}, '29': {'3': 1}, '41': {'3': 1}, '25': {'3': 4}, '51': {'3': 1}, '33': {'3': 2}, '34': {'3': 2}, '22': {'3': 4}, '19': {'3': 12}, '31': {'3': 1}, '30': {'3': 3}, '28': {'3': 3}, '39': {'3': 1}, '101': {'3': 1}, '92': {'3': 2}, '100': {'3': 1}, '32': {'3': 1}, '27': {'3': 3}, '82': {'3': 1}, '105': {'3': 2}, '44': {'3': 1}, '54': {'3': 1}, '79': {'3': 1}, '86': {'3': 1}, '89': {'3': 1}, '56': {'3': 1}, '40': {'3': 1}, '37': {'3': 1}, '99': {'3': 1}, '87': {'3': 1}, '77': {'3': 1}, '74': {'3': 1}}, 'm_histogram': {'3': 3924, '4': 37167, '5': 92074, '6': 92953, '7': 76923, '8': 77266, '9': 68110, '10': 41151, '11': 16257, '12': 4210, '13': 922, '14': 232, '1': 173, '2': 332, '24': 4, '17': 18, '15': 84, '0': 64, '26': 5, '16': 45, '95': 1, '23': 3, '20': 4, '21': 7, '47': 2, '18': 11, '29': 1, '41': 1, '25': 4, '51': 1, '33': 2, '34': 2, '22': 4, '19': 12, '31': 1, '30': 3, '28': 3, '39': 1, '101': 1, '92': 2, '100': 1, '32': 1, '27': 3, '82': 1, '105': 2, '44': 1, '54': 1, '79': 1, '86': 1, '89': 1, '56': 1, '40': 1, '37': 1, '99': 1, '87': 1, '77': 1, '74': 1}, 'mean_K_valid': 2.9903620765095638, 'mean_K_all': 2.98998828125, 'Dall_mean_sentences': 7.0531414083010375, 'strict_subset_ratio': 0.999537109375, 'mean_effective_tokens': {'F': 171.46915239404925, 'Dall': 153.89054100512564, 'D3': 66.38841183272909}, 'mean_content_tokens': {'F': 169.46915239404925, 'Dall': 151.89054100512564, 'D3': 64.38841183272909}, 'mean_lowest_Dall_content_token_coverage': 0.45601314895918665, 'pooled_lowest_Dall_content_token_coverage': 0.42391324309363193, 'coverage_definition': 'Actual EOT-delimited content tokens excluding SOT/EOT on identical valid samples'}`.
Full cycle(s):`{'count': 500, 'median': 2.1264034509658813, 'p95': 2.2889534592628475, 'p99': 2.399579689502716, 'max': 41.50322723388672}`; slowest-rank data_wait(s):`{'count': 500, 'median': 0.0007222294807434082, 'p95': 0.001156299561262129, 'p99': 0.002776513546705233, 'max': 24.09991292655468}`.
OOM kills:0; true training I/O errors:0; Pod anomaly:false. GPU peak:`{'0': 27.760382175445557, '1': 27.760382175445557, '2': 27.760382175445557, '3': 27.760382175445557}` GiB.
First5 gate passed before update6; all512000 actual IDs,F/Dall token hashes and selected-index digests checked. Weight/sparsity arms also match all D3 text/token hashes exactly to Anchor.
Strict bare/native inference only:normalized native image embedding @ normalized native full-caption text embedding.T. No masks/gates/local-view inference/rerank/ensemble/TTA.
Common0 SHA256:`54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`.
Checkpoint SHA256:`812eb4e84f863ef9e332055de25c0927047f0bdd0e802bc91dad8cc5b2a0f3c7`.
Bare SHA256:`5877ba4b69e6b1ab8ee0bc984cc245e6ba166aa92a8dd34ca565db15997f2bc6`.
Checkpoints/bare/raw logs stay local. RUNTIME_STATS.json records raw paths,bytes,SHA256 and UTC containment windows.
Selection/classification and Pareto analysis occur after all seven arms, in SEARCH_SUMMARY.md. No arm-specific tuning or early metric stop.

Final search classification:`NO_IMPROVEMENT`.
