# W35: isolated Nested D3 local500 search arm

Completed exactly500 fresh-common0 optimizer updates, horizon4868. Only changed axis:`alignment`.
Alignment:[1.325, 1.325, 0.35]; sparsity coefficients:[1.0, 2.0, 2.0] (mass5); mode:nested_detail_d3.
Other construction,optimizer/LR,workers8,batch256/rank,inclusion chain/ramp200/max1,preprocess and native protocol frozen. No combination/full run.
Local-only:`/root/said_s02_stage500/ShareGPT4V`; missing/symlink/escape fails, NFS fallback forbidden. /root is disposable overlay; NFS originals retained.

| Dataset | I2T R@1 / R@5 / R@10 (%) | T2I R@1 / R@5 / R@10 (%) |
|---|---|---|
| COCO | 60.580000 / 82.600000 / 89.180000 | 41.556000 / 67.256000 / 76.940000 |
| Urban-1k | 90.900004 / 98.500007 / 99.700004 | 89.100003 / 98.600006 / 99.400002 |
| Flickr30k-test1k | 87.500000 / 97.500000 / 99.300000 | 71.720000 / 91.380000 / 95.180000 |
| DOCCI | 77.500000 / 95.420000 / 97.960000 | 77.600000 / 95.060000 / 97.640000 |
| Long-DCI | 57.090239 / 75.651144 / 81.938963 | 56.866614 / 76.216785 / 82.136280 |

| Metric | Arm | Delta vs Anchor(pp) |
|---|---:|---:|
| Score5 | 71.041286 | -0.022314 |
| J_long3 | 74.842810 | -0.111856 |
| J_long | 83.775002 | -0.250000 |
| Short4 | 65.339000 | +0.112000 |
| Urban_I2T | 90.900000 | +0.000000 |
| Urban_T2I | 89.100000 | -0.600000 |

| Dataset | Delta I2T R@1/5/10(pp) | Delta T2I R@1/5/10(pp) |
|---|---|---|
| COCO | +0.180000 / -0.020000 / -0.080000 | -0.032000 / -0.020000 / +0.008000 |
| Urban-1k | +0.000000 / -0.099999 / +0.099999 | -0.599998 / +0.099999 / -0.100005 |
| Flickr30k-test1k | +0.500000 / -0.300000 / +0.100000 | -0.200000 / +0.000000 / +0.060000 |
| DOCCI | -0.320000 / +0.080000 / -0.040000 | -0.080000 / -0.200000 / -0.080000 |
| Long-DCI | +0.065772 / +0.118390 / +0.289398 | +0.263089 / +0.131544 / +0.197316 |

| View | Combined CE | Weighted CE | Alignment share | Keep ratio |
|---|---:|---:|---:|---:|
| F | 0.112446 | 0.496639 | 16.372% | 0.853893 |
| Dall | 0.206169 | 0.910581 | 30.019% | 0.843538 |
| D3 | 1.393854 | 1.626163 | 53.609% | 0.781702 |

Raw gradient norms:`{'F': 10.507359266281128, 'Dall': 13.902645826339722, 'D3': 31.494255781173706}`.
Weighted gradient norms:`{'F': 46.40750342607498, 'Dall': 61.40335239966711, 'D3': 36.743298411369324}`; weighted D3/Dall ratio:`0.598392`.
Frozen gradient protocol:8 first seed0 epoch0 global1024 batches at fixed step500; production backbone group; raw directional-summed CE,all-reduce/4; no optimizer updates. Norm magnitudes are not a signed gradient decomposition.
Mask hierarchy:`{'inc': 0.010762291643768549, 'inc_weight': 1.0, 'F_Dall_mask_iou': 0.9563555824756622, 'Dall_D3_mask_iou': 0.8783126771450043, 'Dall_F_hard_violation': 0.013568027950823307, 'D3_Dall_hard_violation': 0.021792404875159262}`. Soft detached-child inclusion does not impose zero hard-mask violations.
Sampling:`{'records': 512000, 'valid_records': 511936, 'K_histogram_all': {'2': 3924, '3': 507507, '1': 505, '0': 64}, 'K_histogram_valid': {'2': 3924, '3': 507507, '1': 505}, 'K_histogram_by_m': {'8': {'3': 77266}, '5': {'3': 92074}, '6': {'3': 92953}, '9': {'3': 68110}, '4': {'3': 37167}, '10': {'3': 41151}, '12': {'3': 4210}, '7': {'3': 76923}, '11': {'3': 16257}, '3': {'2': 3924}, '13': {'3': 922}, '14': {'3': 232}, '1': {'1': 173}, '2': {'1': 332}, '24': {'3': 4}, '17': {'3': 18}, '15': {'3': 84}, '26': {'3': 5}, '16': {'3': 45}, '95': {'3': 1}, '23': {'3': 3}, '20': {'3': 4}, '21': {'3': 7}, '47': {'3': 2}, '18': {'3': 11}, '29': {'3': 1}, '41': {'3': 1}, '25': {'3': 4}, '51': {'3': 1}, '33': {'3': 2}, '34': {'3': 2}, '22': {'3': 4}, '19': {'3': 12}, '31': {'3': 1}, '30': {'3': 3}, '28': {'3': 3}, '39': {'3': 1}, '101': {'3': 1}, '92': {'3': 2}, '100': {'3': 1}, '32': {'3': 1}, '27': {'3': 3}, '82': {'3': 1}, '105': {'3': 2}, '44': {'3': 1}, '54': {'3': 1}, '79': {'3': 1}, '86': {'3': 1}, '89': {'3': 1}, '56': {'3': 1}, '40': {'3': 1}, '37': {'3': 1}, '99': {'3': 1}, '87': {'3': 1}, '77': {'3': 1}, '74': {'3': 1}}, 'm_histogram': {'3': 3924, '4': 37167, '5': 92074, '6': 92953, '7': 76923, '8': 77266, '9': 68110, '10': 41151, '11': 16257, '12': 4210, '13': 922, '14': 232, '1': 173, '2': 332, '24': 4, '17': 18, '15': 84, '0': 64, '26': 5, '16': 45, '95': 1, '23': 3, '20': 4, '21': 7, '47': 2, '18': 11, '29': 1, '41': 1, '25': 4, '51': 1, '33': 2, '34': 2, '22': 4, '19': 12, '31': 1, '30': 3, '28': 3, '39': 1, '101': 1, '92': 2, '100': 1, '32': 1, '27': 3, '82': 1, '105': 2, '44': 1, '54': 1, '79': 1, '86': 1, '89': 1, '56': 1, '40': 1, '37': 1, '99': 1, '87': 1, '77': 1, '74': 1}, 'mean_K_valid': 2.9903620765095638, 'mean_K_all': 2.98998828125, 'Dall_mean_sentences': 7.0531414083010375, 'strict_subset_ratio': 0.999537109375, 'mean_effective_tokens': {'F': 171.46915239404925, 'Dall': 153.89054100512564, 'D3': 66.38841183272909}, 'mean_content_tokens': {'F': 169.46915239404925, 'Dall': 151.89054100512564, 'D3': 64.38841183272909}, 'mean_lowest_Dall_content_token_coverage': 0.45601314895918665, 'pooled_lowest_Dall_content_token_coverage': 0.42391324309363193, 'coverage_definition': 'Actual EOT-delimited content tokens excluding SOT/EOT on identical valid samples'}`.
Full cycle(s):`{'count': 500, 'median': 2.1208786964416504, 'p95': 2.2680893421173094, 'p99': 2.360916049480438, 'max': 26.7146053314209}`; slowest-rank data_wait(s):`{'count': 500, 'median': 0.0006930120289325714, 'p95': 0.0010025985538959503, 'p99': 0.0013160440325736998, 'max': 23.065834507346153}`.
OOM kills:0; true training I/O errors:0; Pod anomaly:false. GPU peak:`{'0': 27.760382175445557, '1': 27.760382175445557, '2': 27.760382175445557, '3': 27.760382175445557}` GiB.
First5 gate passed before update6; all512000 actual IDs,F/Dall token hashes and selected-index digests checked. Weight/sparsity arms also match all D3 text/token hashes exactly to Anchor.
Strict bare/native inference only:normalized native image embedding @ normalized native full-caption text embedding.T. No masks/gates/local-view inference/rerank/ensemble/TTA.
Common0 SHA256:`54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`.
Checkpoint SHA256:`e765703a6147ca763c16cdeaa702d30ab8711b7483594abdbfaf2f5f1901504c`.
Bare SHA256:`aafaec755d24fa6fbe6c758dbb6e6a3f56b297d415cbd3c7692c764a2d065157`.
Checkpoints/bare/raw logs stay local. RUNTIME_STATS.json records raw paths,bytes,SHA256 and UTC containment windows.
Selection/classification and Pareto analysis occur after all seven arms, in SEARCH_SUMMARY.md. No arm-specific tuning or early metric stop.

Final search classification:`NO_IMPROVEMENT`.
