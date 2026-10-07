# W40: isolated Nested D3 local500 search arm

Completed exactly500 fresh-common0 optimizer updates, horizon4868. Only changed axis:`alignment`.
Alignment:[1.3, 1.3, 0.4]; sparsity coefficients:[1.0, 2.0, 2.0] (mass5); mode:nested_detail_d3.
Other construction,optimizer/LR,workers8,batch256/rank,inclusion chain/ramp200/max1,preprocess and native protocol frozen. No combination/full run.
Local-only:`/root/said_s02_stage500/ShareGPT4V`; missing/symlink/escape fails, NFS fallback forbidden. /root is disposable overlay; NFS originals retained.

| Dataset | I2T R@1 / R@5 / R@10 (%) | T2I R@1 / R@5 / R@10 (%) |
|---|---|---|
| COCO | 60.720000 / 82.620000 / 89.160000 | 41.580000 / 67.300000 / 76.936000 |
| Urban-1k | 91.200006 / 98.800004 / 99.700004 | 89.000005 / 98.500007 / 99.400002 |
| Flickr30k-test1k | 87.300000 / 97.400000 / 99.200000 | 72.080000 / 91.300000 / 95.260000 |
| DOCCI | 77.660000 / 95.400000 / 98.060000 | 77.680000 / 95.060000 / 97.840000 |
| Long-DCI | 56.853460 / 75.874770 / 82.004736 | 56.616680 / 75.927388 / 81.912655 |

| Metric | Arm | Delta vs Anchor(pp) |
|---|---:|---:|
| Score5 | 71.069015 | +0.005415 |
| J_long3 | 74.835025 | -0.119641 |
| J_long | 83.885003 | -0.139999 |
| Short4 | 65.420000 | +0.193000 |
| Urban_I2T | 91.200000 | +0.300000 |
| Urban_T2I | 89.000000 | -0.700000 |

| Dataset | Delta I2T R@1/5/10(pp) | Delta T2I R@1/5/10(pp) |
|---|---|---|
| COCO | +0.320000 / +0.000000 / -0.100000 | -0.008000 / +0.024000 / +0.004000 |
| Urban-1k | +0.300002 / +0.199997 / +0.099999 | -0.699997 / +0.000000 / -0.100005 |
| Flickr30k-test1k | +0.300000 / -0.400000 / +0.000000 | +0.160000 / -0.080000 / +0.140000 |
| DOCCI | -0.160000 / +0.060000 / +0.060000 | +0.000000 / -0.200000 / +0.120000 |
| Long-DCI | -0.171008 / +0.342015 / +0.355170 | +0.013154 / -0.157853 / -0.026309 |

| View | Combined CE | Weighted CE | Alignment share | Keep ratio |
|---|---:|---:|---:|---:|
| F | 0.114157 | 0.494680 | 15.455% | 0.859502 |
| Dall | 0.207301 | 0.898305 | 28.066% | 0.854039 |
| D3 | 1.355811 | 1.807748 | 56.479% | 0.816992 |

Raw gradient norms:`{'F': 10.890182852745056, 'Dall': 14.470597982406616, 'D3': 33.4169340133667}`.
Weighted gradient norms:`{'F': 47.19079236189525, 'Dall': 62.70592459042868, 'D3': 44.55591201782227}`; weighted D3/Dall ratio:`0.710553`.
Frozen gradient protocol:8 first seed0 epoch0 global1024 batches at fixed step500; production backbone group; raw directional-summed CE,all-reduce/4; no optimizer updates. Norm magnitudes are not a signed gradient decomposition.
Mask hierarchy:`{'inc': 0.010024321060627698, 'inc_weight': 1.0, 'F_Dall_mask_iou': 0.9657096362113953, 'Dall_D3_mask_iou': 0.9172745144367218, 'Dall_F_hard_violation': 0.012077293917536735, 'D3_Dall_hard_violation': 0.017389253582805395}`. Soft detached-child inclusion does not impose zero hard-mask violations.
Sampling:`{'records': 512000, 'valid_records': 511936, 'K_histogram_all': {'2': 3924, '3': 507507, '1': 505, '0': 64}, 'K_histogram_valid': {'2': 3924, '3': 507507, '1': 505}, 'K_histogram_by_m': {'8': {'3': 77266}, '5': {'3': 92074}, '6': {'3': 92953}, '9': {'3': 68110}, '4': {'3': 37167}, '10': {'3': 41151}, '12': {'3': 4210}, '7': {'3': 76923}, '11': {'3': 16257}, '3': {'2': 3924}, '13': {'3': 922}, '14': {'3': 232}, '1': {'1': 173}, '2': {'1': 332}, '24': {'3': 4}, '17': {'3': 18}, '15': {'3': 84}, '26': {'3': 5}, '16': {'3': 45}, '95': {'3': 1}, '23': {'3': 3}, '20': {'3': 4}, '21': {'3': 7}, '47': {'3': 2}, '18': {'3': 11}, '29': {'3': 1}, '41': {'3': 1}, '25': {'3': 4}, '51': {'3': 1}, '33': {'3': 2}, '34': {'3': 2}, '22': {'3': 4}, '19': {'3': 12}, '31': {'3': 1}, '30': {'3': 3}, '28': {'3': 3}, '39': {'3': 1}, '101': {'3': 1}, '92': {'3': 2}, '100': {'3': 1}, '32': {'3': 1}, '27': {'3': 3}, '82': {'3': 1}, '105': {'3': 2}, '44': {'3': 1}, '54': {'3': 1}, '79': {'3': 1}, '86': {'3': 1}, '89': {'3': 1}, '56': {'3': 1}, '40': {'3': 1}, '37': {'3': 1}, '99': {'3': 1}, '87': {'3': 1}, '77': {'3': 1}, '74': {'3': 1}}, 'm_histogram': {'3': 3924, '4': 37167, '5': 92074, '6': 92953, '7': 76923, '8': 77266, '9': 68110, '10': 41151, '11': 16257, '12': 4210, '13': 922, '14': 232, '1': 173, '2': 332, '24': 4, '17': 18, '15': 84, '0': 64, '26': 5, '16': 45, '95': 1, '23': 3, '20': 4, '21': 7, '47': 2, '18': 11, '29': 1, '41': 1, '25': 4, '51': 1, '33': 2, '34': 2, '22': 4, '19': 12, '31': 1, '30': 3, '28': 3, '39': 1, '101': 1, '92': 2, '100': 1, '32': 1, '27': 3, '82': 1, '105': 2, '44': 1, '54': 1, '79': 1, '86': 1, '89': 1, '56': 1, '40': 1, '37': 1, '99': 1, '87': 1, '77': 1, '74': 1}, 'mean_K_valid': 2.9903620765095638, 'mean_K_all': 2.98998828125, 'Dall_mean_sentences': 7.0531414083010375, 'strict_subset_ratio': 0.999537109375, 'mean_effective_tokens': {'F': 171.46915239404925, 'Dall': 153.89054100512564, 'D3': 66.38841183272909}, 'mean_content_tokens': {'F': 169.46915239404925, 'Dall': 151.89054100512564, 'D3': 64.38841183272909}, 'mean_lowest_Dall_content_token_coverage': 0.45601314895918665, 'pooled_lowest_Dall_content_token_coverage': 0.42391324309363193, 'coverage_definition': 'Actual EOT-delimited content tokens excluding SOT/EOT on identical valid samples'}`.
Full cycle(s):`{'count': 500, 'median': 2.1281174421310425, 'p95': 2.2843319773674007, 'p99': 2.382880744934082, 'max': 27.221223831176758}`; slowest-rank data_wait(s):`{'count': 500, 'median': 0.0007020384073257446, 'p95': 0.0010599095374345768, 'p99': 0.0014338889718055723, 'max': 23.83563318103552}`.
OOM kills:0; true training I/O errors:0; Pod anomaly:false. GPU peak:`{'0': 27.760382175445557, '1': 27.760382175445557, '2': 27.760382175445557, '3': 27.760382175445557}` GiB.
First5 gate passed before update6; all512000 actual IDs,F/Dall token hashes and selected-index digests checked. Weight/sparsity arms also match all D3 text/token hashes exactly to Anchor.
Strict bare/native inference only:normalized native image embedding @ normalized native full-caption text embedding.T. No masks/gates/local-view inference/rerank/ensemble/TTA.
Common0 SHA256:`54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`.
Checkpoint SHA256:`a6d9f22d76c5c0d1e179569c78d26d4b048779e612d3301e2a61157d0da42464`.
Bare SHA256:`a403ae6f6135caf3a4a0df87c34ce1feec2d3b45ebb9f99670a9b757f84b77f3`.
Checkpoints/bare/raw logs stay local. RUNTIME_STATS.json records raw paths,bytes,SHA256 and UTC containment windows.
Selection/classification and Pareto analysis occur after all seven arms, in SEARCH_SUMMARY.md. No arm-specific tuning or early metric stop.

Final search classification:`NO_IMPROVEMENT`.
