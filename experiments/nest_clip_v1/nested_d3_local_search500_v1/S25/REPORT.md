# S25: isolated Nested D3 local500 search arm

Completed exactly500 fresh-common0 optimizer updates, horizon4868. Only changed axis:`sparsity`.
Alignment:[1.35, 1.35, 0.3]; sparsity coefficients:[0.9090909090909091, 1.8181818181818181, 2.2727272727272725] (mass5); mode:nested_detail_d3.
Other construction,optimizer/LR,workers8,batch256/rank,inclusion chain/ramp200/max1,preprocess and native protocol frozen. No combination/full run.
Local-only:`/root/said_s02_stage500/ShareGPT4V`; missing/symlink/escape fails, NFS fallback forbidden. /root is disposable overlay; NFS originals retained.

| Dataset | I2T R@1 / R@5 / R@10 (%) | T2I R@1 / R@5 / R@10 (%) |
|---|---|---|
| COCO | 60.200000 / 82.840000 / 89.460000 | 41.472000 / 67.424000 / 76.988000 |
| Urban-1k | 91.400003 / 98.600006 / 99.800003 | 89.100003 / 98.800004 / 99.500006 |
| Flickr30k-test1k | 87.400000 / 97.400000 / 99.300000 | 71.780000 / 91.460000 / 95.240000 |
| DOCCI | 77.760000 / 95.360000 / 98.120000 | 77.460000 / 95.140000 / 97.820000 |
| Long-DCI | 57.169166 / 75.559063 / 81.833728 | 56.695606 / 76.058932 / 81.978427 |

| Metric | Arm | Delta vs Anchor(pp) |
|---|---:|---:|
| Score5 | 71.043678 | -0.019922 |
| J_long3 | 74.930797 | -0.023870 |
| J_long | 83.930002 | -0.095000 |
| Short4 | 65.213000 | -0.014000 |
| Urban_I2T | 91.400000 | +0.500000 |
| Urban_T2I | 89.100000 | -0.600000 |

| Dataset | Delta I2T R@1/5/10(pp) | Delta T2I R@1/5/10(pp) |
|---|---|---|
| COCO | -0.200000 / +0.220000 / +0.200000 | -0.116000 / +0.148000 / +0.056000 |
| Urban-1k | +0.500000 / +0.000000 / +0.199997 | -0.599998 / +0.299996 / +0.000000 |
| Flickr30k-test1k | +0.400000 / -0.400000 / +0.100000 | -0.140000 / +0.080000 / +0.120000 |
| DOCCI | -0.060000 / +0.020000 / +0.120000 | -0.220000 / -0.120000 / +0.100000 |
| Long-DCI | +0.144699 / +0.026309 / +0.184162 | +0.092081 / -0.026309 / +0.039463 |

| View | Combined CE | Weighted CE | Alignment share | Keep ratio |
|---|---:|---:|---:|---:|
| F | 0.110841 | 0.498786 | 17.538% | 0.862242 |
| Dall | 0.204112 | 0.918506 | 32.297% | 0.841025 |
| D3 | 1.426681 | 1.426681 | 50.165% | 0.765533 |

Raw gradient norms:`{'F': 12.013843774795532, 'Dall': 14.772582411766052, 'D3': 32.46286106109619}`.
Weighted gradient norms:`{'F': 54.06229698657991, 'Dall': 66.47662085294725, 'D3': 32.46286106109619}`; weighted D3/Dall ratio:`0.488335`.
Frozen gradient protocol:8 first seed0 epoch0 global1024 batches at fixed step500; production backbone group; raw directional-summed CE,all-reduce/4; no optimizer updates. Norm magnitudes are not a signed gradient decomposition.
Mask hierarchy:`{'inc': 0.01067000424489379, 'inc_weight': 1.0, 'F_Dall_mask_iou': 0.9391438317298889, 'Dall_D3_mask_iou': 0.8627802848815918, 'Dall_F_hard_violation': 0.015945013053715228, 'D3_Dall_hard_violation': 0.0213843984156847}`. Soft detached-child inclusion does not impose zero hard-mask violations.
Sampling:`{'records': 512000, 'valid_records': 511936, 'K_histogram_all': {'2': 3924, '3': 507507, '1': 505, '0': 64}, 'K_histogram_valid': {'2': 3924, '3': 507507, '1': 505}, 'K_histogram_by_m': {'8': {'3': 77266}, '5': {'3': 92074}, '6': {'3': 92953}, '9': {'3': 68110}, '4': {'3': 37167}, '10': {'3': 41151}, '12': {'3': 4210}, '7': {'3': 76923}, '11': {'3': 16257}, '3': {'2': 3924}, '13': {'3': 922}, '14': {'3': 232}, '1': {'1': 173}, '2': {'1': 332}, '24': {'3': 4}, '17': {'3': 18}, '15': {'3': 84}, '26': {'3': 5}, '16': {'3': 45}, '95': {'3': 1}, '23': {'3': 3}, '20': {'3': 4}, '21': {'3': 7}, '47': {'3': 2}, '18': {'3': 11}, '29': {'3': 1}, '41': {'3': 1}, '25': {'3': 4}, '51': {'3': 1}, '33': {'3': 2}, '34': {'3': 2}, '22': {'3': 4}, '19': {'3': 12}, '31': {'3': 1}, '30': {'3': 3}, '28': {'3': 3}, '39': {'3': 1}, '101': {'3': 1}, '92': {'3': 2}, '100': {'3': 1}, '32': {'3': 1}, '27': {'3': 3}, '82': {'3': 1}, '105': {'3': 2}, '44': {'3': 1}, '54': {'3': 1}, '79': {'3': 1}, '86': {'3': 1}, '89': {'3': 1}, '56': {'3': 1}, '40': {'3': 1}, '37': {'3': 1}, '99': {'3': 1}, '87': {'3': 1}, '77': {'3': 1}, '74': {'3': 1}}, 'm_histogram': {'3': 3924, '4': 37167, '5': 92074, '6': 92953, '7': 76923, '8': 77266, '9': 68110, '10': 41151, '11': 16257, '12': 4210, '13': 922, '14': 232, '1': 173, '2': 332, '24': 4, '17': 18, '15': 84, '0': 64, '26': 5, '16': 45, '95': 1, '23': 3, '20': 4, '21': 7, '47': 2, '18': 11, '29': 1, '41': 1, '25': 4, '51': 1, '33': 2, '34': 2, '22': 4, '19': 12, '31': 1, '30': 3, '28': 3, '39': 1, '101': 1, '92': 2, '100': 1, '32': 1, '27': 3, '82': 1, '105': 2, '44': 1, '54': 1, '79': 1, '86': 1, '89': 1, '56': 1, '40': 1, '37': 1, '99': 1, '87': 1, '77': 1, '74': 1}, 'mean_K_valid': 2.9903620765095638, 'mean_K_all': 2.98998828125, 'Dall_mean_sentences': 7.0531414083010375, 'strict_subset_ratio': 0.999537109375, 'mean_effective_tokens': {'F': 171.46915239404925, 'Dall': 153.89054100512564, 'D3': 66.38841183272909}, 'mean_content_tokens': {'F': 169.46915239404925, 'Dall': 151.89054100512564, 'D3': 64.38841183272909}, 'mean_lowest_Dall_content_token_coverage': 0.45601314895918665, 'pooled_lowest_Dall_content_token_coverage': 0.42391324309363193, 'coverage_definition': 'Actual EOT-delimited content tokens excluding SOT/EOT on identical valid samples'}`.
Full cycle(s):`{'count': 500, 'median': 2.1271138191223145, 'p95': 2.250055706501007, 'p99': 2.35835499048233, 'max': 25.50708770751953}`; slowest-rank data_wait(s):`{'count': 500, 'median': 0.000676419585943222, 'p95': 0.001009837538003921, 'p99': 0.0014221598207950583, 'max': 21.546190164983273}`.
OOM kills:0; true training I/O errors:0; Pod anomaly:false. GPU peak:`{'0': 27.760382175445557, '1': 27.760382175445557, '2': 27.760382175445557, '3': 27.760382175445557}` GiB.
First5 gate passed before update6; all512000 actual IDs,F/Dall token hashes and selected-index digests checked. Weight/sparsity arms also match all D3 text/token hashes exactly to Anchor.
Strict bare/native inference only:normalized native image embedding @ normalized native full-caption text embedding.T. No masks/gates/local-view inference/rerank/ensemble/TTA.
Common0 SHA256:`54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`.
Checkpoint SHA256:`67dfc54d8bfbff3d7120563d860b25579485b03bb9000dfbe61cf15c12ae83fc`.
Bare SHA256:`cd2ec340d37cac14b9767f954cd8361cd04ef08c5e9f411ae5095bccfe73789e`.
Checkpoints/bare/raw logs stay local. RUNTIME_STATS.json records raw paths,bytes,SHA256 and UTC containment windows.
Selection/classification and Pareto analysis occur after all seven arms, in SEARCH_SUMMARY.md. No arm-specific tuning or early metric stop.

Final search classification:`NO_IMPROVEMENT`.
