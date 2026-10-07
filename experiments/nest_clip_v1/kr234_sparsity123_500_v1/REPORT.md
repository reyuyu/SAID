# KR234-S123: isolated Nested D3 local500 search arm

Completed exactly500 fresh-common0 optimizer updates, horizon4868. Only changed axis:`sparsity`.
Alignment:[1.35, 1.35, 0.3]; absolute sparsity coefficients:[1.0, 2.0, 3.0] (mass6.0); mode:nested_detail_kr234.
Other construction,optimizer/LR,workers8,batch256/rank,inclusion chain/ramp200/max1,preprocess and native protocol frozen. No combination/full run.
Local-only:`/root/said_s02_stage500/ShareGPT4V`; missing/symlink/escape fails, NFS fallback forbidden. /root is disposable overlay; NFS originals retained.

| Dataset | I2T R@1 / R@5 / R@10 (%) | T2I R@1 / R@5 / R@10 (%) |
|---|---|---|
| COCO | 60.640000 / 82.680000 / 89.320000 | 41.592000 / 67.392000 / 76.908000 |
| Urban-1k | 91.000003 / 98.700005 / 99.700004 | 89.200002 / 98.700005 / 99.400002 |
| Flickr30k-test1k | 87.300000 / 97.500000 / 98.900000 | 71.900000 / 91.440000 / 95.240000 |
| DOCCI | 77.640000 / 95.460000 / 98.040000 | 77.500000 / 95.040000 / 97.660000 |
| Long-DCI | 56.353591 / 75.230203 / 81.583794 | 56.695606 / 75.953696 / 81.965272 |

| Metric | Arm | Delta vs KR234(pp) |
|---|---:|---:|
| Score5 | 70.982120 | -0.125781 |
| J_long3 | 74.731534 | -0.314968 |
| J_long | 83.835001 | -0.285001 |
| Short4 | 65.358000 | +0.158000 |
| Urban_I2T | 91.000000 | -0.600000 |
| Urban_T2I | 89.200000 | +0.000000 |

| Dataset | Delta I2T R@1/5/10(pp) | Delta T2I R@1/5/10(pp) |
|---|---|---|
| COCO | +0.060000 / +0.020000 / +0.120000 | +0.152000 / +0.152000 / -0.032000 |
| Urban-1k | -0.600004 / +0.000000 / -0.099999 | +0.000000 / +0.099999 / -0.100005 |
| Flickr30k-test1k | +0.100000 / -0.100000 / -0.300000 | +0.320000 / +0.040000 / -0.060000 |
| DOCCI | -0.120000 / -0.060000 / -0.140000 | -0.420000 / +0.040000 / -0.200000 |
| Long-DCI | -0.473560 / -0.105235 / -0.105235 | -0.276243 / -0.289398 / -0.197316 |

| View | Combined CE | Weighted CE | Alignment share | Keep ratio |
|---|---:|---:|---:|---:|
| F | 0.108274 | 0.487234 | 15.840% | 0.843459 |
| Dall | 0.204375 | 0.919687 | 29.900% | 0.818945 |
| Dk | 1.668965 | 1.668965 | 54.260% | 0.726462 |

Raw gradient norms:`{'F': 12.375751852989197, 'Dall': 15.534677386283875, 'Dk': 33.35106039047241}`.
Weighted gradient norms:`{'F': 55.6908833384514, 'Dall': 69.90604823827745, 'Dk': 33.35106039047241}`; weighted Dk/Dall ratio:`0.477084`.
Frozen gradient protocol:8 first seed0 epoch0 global1024 batches at fixed step500; production backbone group; raw directional-summed CE,all-reduce/4; no optimizer updates. Norm magnitudes are not a signed gradient decomposition.
Mask hierarchy:`{'inc': 0.010637992806732655, 'inc_weight': 1.0, 'F_Dall_mask_iou': 0.9306062269210815, 'Dall_Dk_mask_iou': 0.8328563964366913, 'Dall_F_hard_violation': 0.01726968728005886, 'Dk_Dall_hard_violation': 0.024379342757165433}`. Soft detached-child inclusion does not impose zero hard-mask violations.
Sampling:`{'records': 512000, 'valid_records': 511936, 'K_histogram_all': {'2': 179406, '3': 175403, '4': 156622, '1': 505, '0': 64}, 'K_histogram_valid': {'2': 179406, '3': 175403, '4': 156622, '1': 505}, 'K_histogram_by_m': {'8': {'4': 25556, '2': 25907, '3': 25803}, '5': {'4': 30664, '3': 30755, '2': 30655}, '6': {'3': 30947, '4': 31207, '2': 30799}, '9': {'2': 22754, '3': 22605, '4': 22751}, '4': {'3': 18498, '2': 18669}, '10': {'4': 13563, '2': 13827, '3': 13761}, '12': {'4': 1455, '3': 1417, '2': 1338}, '7': {'3': 25847, '4': 25500, '2': 25576}, '11': {'2': 5529, '3': 5300, '4': 5428}, '3': {'2': 3924}, '13': {'2': 286, '3': 311, '4': 325}, '14': {'4': 81, '2': 70, '3': 81}, '1': {'1': 173}, '2': {'1': 332}, '24': {'4': 1, '2': 1, '3': 2}, '17': {'4': 7, '3': 6, '2': 5}, '15': {'3': 28, '2': 30, '4': 26}, '26': {'3': 3, '2': 2}, '16': {'4': 19, '2': 12, '3': 14}, '95': {'2': 1}, '23': {'4': 2, '3': 1}, '20': {'2': 2, '4': 1, '3': 1}, '21': {'3': 4, '4': 1, '2': 2}, '47': {'3': 1, '2': 1}, '18': {'2': 3, '3': 3, '4': 5}, '29': {'4': 1}, '41': {'4': 1}, '25': {'3': 1, '4': 3}, '51': {'4': 1}, '33': {'3': 1, '4': 1}, '34': {'4': 1, '2': 1}, '22': {'3': 1, '2': 1, '4': 2}, '19': {'4': 6, '2': 4, '3': 2}, '31': {'4': 1}, '30': {'3': 2, '2': 1}, '28': {'3': 2, '4': 1}, '39': {'4': 1}, '101': {'4': 1}, '92': {'4': 1, '3': 1}, '100': {'2': 1}, '32': {'3': 1}, '27': {'4': 2, '3': 1}, '82': {'2': 1}, '105': {'2': 2}, '44': {'3': 1}, '54': {'3': 1}, '79': {'4': 1}, '86': {'4': 1}, '89': {'4': 1}, '56': {'2': 1}, '40': {'2': 1}, '37': {'4': 1}, '99': {'4': 1}, '87': {'4': 1}, '77': {'3': 1}, '74': {'4': 1}}, 'm_histogram': {'3': 3924, '4': 37167, '5': 92074, '6': 92953, '7': 76923, '8': 77266, '9': 68110, '10': 41151, '11': 16257, '12': 4210, '13': 922, '14': 232, '1': 173, '2': 332, '24': 4, '17': 18, '15': 84, '0': 64, '26': 5, '16': 45, '95': 1, '23': 3, '20': 4, '21': 7, '47': 2, '18': 11, '29': 1, '41': 1, '25': 4, '51': 1, '33': 2, '34': 2, '22': 4, '19': 12, '31': 1, '30': 3, '28': 3, '39': 1, '101': 1, '92': 2, '100': 1, '32': 1, '27': 3, '82': 1, '105': 2, '44': 1, '54': 1, '79': 1, '86': 1, '89': 1, '56': 1, '40': 1, '37': 1, '99': 1, '87': 1, '77': 1, '74': 1}, 'mean_K_valid': 2.9535215339417427, 'mean_K_all': 2.95315234375, 'Dall_mean_sentences': 7.0531414083010375, 'strict_subset_ratio': 0.999537109375, 'mean_effective_tokens': {'F': 171.46915239404925, 'Dall': 153.89054100512564, 'Dk': 65.51318914864358}, 'mean_content_tokens': {'F': 169.46915239404925, 'Dall': 151.89054100512564, 'Dk': 63.51318914864358}, 'mean_lowest_Dall_content_token_coverage': 0.4467718713244764, 'pooled_lowest_Dall_content_token_coverage': 0.41815104962000427, 'coverage_definition': 'Actual EOT-delimited content tokens excluding SOT/EOT on identical valid samples'}`.
Full cycle(s):`{'count': 500, 'median': 2.123536229133606, 'p95': 2.2873586416244507, 'p99': 2.3478581476211544, 'max': 28.31142234802246}`; slowest-rank data_wait(s):`{'count': 500, 'median': 0.000715397298336029, 'p95': 0.0008879415690898893, 'p99': 0.00128295861184597, 'max': 24.482991322875023}`.
OOM kills:0; true training I/O errors:0; Pod anomaly:false. GPU peak:`{'0': 27.760382175445557, '1': 27.760382175445557, '2': 27.760382175445557, '3': 27.760382175445557}` GiB.
First5 gate passed before update6; all512000 actual IDs,F/Dall token hashes and selected-index digests checked. Arms with frozen lowest-view sampling also match its text/token/K/index trajectory exactly to the declared reference.
Strict bare/native inference only:normalized native image embedding @ normalized native full-caption text embedding.T. No masks/gates/local-view inference/rerank/ensemble/TTA.
Common0 SHA256:`54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`.
Checkpoint SHA256:`519d1b79933868fe4cbaa91a2d35ee1bc894beefbdef48049dcd801ad10d2d47`.
Bare SHA256:`467e86ec7e1370f87c30a47fec1d419be1169d973608c0fc09602fed886a67a3`.
Checkpoints/bare/raw logs stay local. RUNTIME_STATS.json records raw paths,bytes,SHA256 and UTC containment windows.
This one arm is classified after complete native evaluation/review. No arm-specific tuning or early metric stop.

Actual sampling: median valid K=3.0; mean K/m=0.4505553911495799; pooled K/m=0.4187526327581723.

## Fixed-K Anchor and KR234 comparison

Classification: `NO_IMPROVEMENT`.
Only literal sparsity changes1/2/2 ->1/2/3; mass5 ->6; no normalization or extra loss scale.
Both the first5 gate and complete512000-record proof match KR234 IDs,F,Dall,K,Dk indices,text/tokens and LR exactly.

| Model | Sparsity | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I |
|---|---|---:|---:|---:|---:|---:|---:|
| Anchor fixedK3 | 1/2/2 | 71.063600 | 74.954666 | 84.025001 | 65.227000 | 90.900000 | 89.700000 |
| KR234 | 1/2/2 | 71.107901 | 75.046502 | 84.120002 | 65.200000 | 91.600000 | 89.200000 |
| KR234-S123 | 1/2/3 | 70.982120 | 74.731534 | 83.835001 | 65.358000 | 91.000000 | 89.200000 |

| Reference | ΔScore5 | ΔJ_long3 | ΔJ_long | ΔShort4 | ΔUrban I2T | ΔUrban T2I |
|---|---:|---:|---:|---:|---:|---:|
| Anchor | -0.081480 | -0.223133 | -0.190000 | +0.131000 | +0.100000 | -0.500000 |
| KR234 | -0.125781 | -0.314968 | -0.285001 | +0.158000 | -0.600000 | +0.000000 |

Dk keep ratio:0.782360 -> 0.726462.
Weighted Dk/Dall gradient ratio:0.524341 -> 0.477084.
Mask changes vs KR234:`{'inc': 0.00012092212215066012, 'inc_weight': 0.0, 'F_Dall_mask_iou': -0.028165982961654734, 'Dall_Dk_mask_iou': -0.04553719401359557, 'Dall_F_hard_violation': 0.004709498304873704, 'Dk_Dall_hard_violation': 0.0016347715631127371}`.
Operational thresholds were frozen in SEARCH_PLAN.json before launch. Strong hierarchy means a lower Dk keep, ordered F/Dall/Dk keeps, no smaller parent-child keep gap, and Dk outside-Dall violation increase<=.1pp. Soft inclusion remains unchanged; no hard projection is imposed.
Complete five-set/directional/recall deltas vs both references are in RESULTS.json. Raw/bare/full weights stay local.
Stopped at exactly500; no full,other sparsity/K/alignment/inclusion experiment or combination launched.
