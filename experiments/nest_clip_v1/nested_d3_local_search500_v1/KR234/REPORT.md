# KR234: isolated Nested D3 local500 search arm

Completed exactly500 fresh-common0 optimizer updates, horizon4868. Only changed axis:`granularity`.
Alignment:[1.35, 1.35, 0.3]; sparsity coefficients:[1.0, 2.0, 2.0] (mass5); mode:nested_detail_kr234.
Other construction,optimizer/LR,workers8,batch256/rank,inclusion chain/ramp200/max1,preprocess and native protocol frozen. No combination/full run.
Local-only:`/root/said_s02_stage500/ShareGPT4V`; missing/symlink/escape fails, NFS fallback forbidden. /root is disposable overlay; NFS originals retained.

| Dataset | I2T R@1 / R@5 / R@10 (%) | T2I R@1 / R@5 / R@10 (%) |
|---|---|---|
| COCO | 60.580000 / 82.660000 / 89.200000 | 41.440000 / 67.240000 / 76.940000 |
| Urban-1k | 91.600007 / 98.700005 / 99.800003 | 89.200002 / 98.600006 / 99.500006 |
| Flickr30k-test1k | 87.200000 / 97.600000 / 99.200000 | 71.580000 / 91.400000 / 95.300000 |
| DOCCI | 77.760000 / 95.520000 / 98.180000 | 77.920000 / 95.000000 / 97.860000 |
| Long-DCI | 56.827151 / 75.335438 / 81.689029 | 56.971850 / 76.243094 / 82.162589 |

| Metric | Arm | Delta vs Anchor(pp) |
|---|---:|---:|
| Score5 | 71.107901 | +0.044301 |
| J_long3 | 75.046502 | +0.091835 |
| J_long | 84.120002 | +0.095001 |
| Short4 | 65.200000 | -0.027000 |
| Urban_I2T | 91.600000 | +0.700000 |
| Urban_T2I | 89.200000 | -0.500000 |

| Dataset | Delta I2T R@1/5/10(pp) | Delta T2I R@1/5/10(pp) |
|---|---|---|
| COCO | +0.180000 / +0.040000 / -0.060000 | -0.148000 / -0.036000 / +0.008000 |
| Urban-1k | +0.700003 / +0.099999 / +0.199997 | -0.500000 / +0.099999 / +0.000000 |
| Flickr30k-test1k | +0.200000 / -0.200000 / +0.000000 | -0.340000 / +0.020000 / +0.180000 |
| DOCCI | -0.060000 / +0.180000 / +0.180000 | +0.240000 / -0.260000 / +0.140000 |
| Long-DCI | -0.197316 / -0.197316 / +0.039463 | +0.368324 / +0.157853 / +0.223625 |

| View | Combined CE | Weighted CE | Alignment share | Keep ratio |
|---|---:|---:|---:|---:|
| F | 0.111766 | 0.502948 | 16.586% | 0.852195 |
| Dall | 0.202867 | 0.912902 | 30.105% | 0.842106 |
| Dk | 1.616575 | 1.616575 | 53.310% | 0.782360 |

Raw gradient norms:`{'F': 11.163501501083374, 'Dall': 14.599633574485779, 'Dk': 34.448309659957886}`.
Weighted gradient norms:`{'F': 50.23575675487519, 'Dall': 65.69835108518602, 'Dk': 34.448309659957886}`; weighted Dk/Dall ratio:`0.524341`.
Frozen gradient protocol:8 first seed0 epoch0 global1024 batches at fixed step500; production backbone group; raw directional-summed CE,all-reduce/4; no optimizer updates. Norm magnitudes are not a signed gradient decomposition.
Mask hierarchy:`{'inc': 0.010517070684581995, 'inc_weight': 1.0, 'F_Dall_mask_iou': 0.9587722098827363, 'Dall_Dk_mask_iou': 0.8783935904502869, 'Dall_F_hard_violation': 0.012560188975185156, 'Dk_Dall_hard_violation': 0.022744571194052696}`. Soft detached-child inclusion does not impose zero hard-mask violations.
Sampling:`{'records': 512000, 'valid_records': 511936, 'K_histogram_all': {'2': 179406, '3': 175403, '4': 156622, '1': 505, '0': 64}, 'K_histogram_valid': {'2': 179406, '3': 175403, '4': 156622, '1': 505}, 'K_histogram_by_m': {'8': {'4': 25556, '2': 25907, '3': 25803}, '5': {'4': 30664, '3': 30755, '2': 30655}, '6': {'3': 30947, '4': 31207, '2': 30799}, '9': {'2': 22754, '3': 22605, '4': 22751}, '4': {'3': 18498, '2': 18669}, '10': {'4': 13563, '2': 13827, '3': 13761}, '12': {'4': 1455, '3': 1417, '2': 1338}, '7': {'3': 25847, '4': 25500, '2': 25576}, '11': {'2': 5529, '3': 5300, '4': 5428}, '3': {'2': 3924}, '13': {'2': 286, '3': 311, '4': 325}, '14': {'4': 81, '2': 70, '3': 81}, '1': {'1': 173}, '2': {'1': 332}, '24': {'4': 1, '2': 1, '3': 2}, '17': {'4': 7, '3': 6, '2': 5}, '15': {'3': 28, '2': 30, '4': 26}, '26': {'3': 3, '2': 2}, '16': {'4': 19, '2': 12, '3': 14}, '95': {'2': 1}, '23': {'4': 2, '3': 1}, '20': {'2': 2, '4': 1, '3': 1}, '21': {'3': 4, '4': 1, '2': 2}, '47': {'3': 1, '2': 1}, '18': {'2': 3, '3': 3, '4': 5}, '29': {'4': 1}, '41': {'4': 1}, '25': {'3': 1, '4': 3}, '51': {'4': 1}, '33': {'3': 1, '4': 1}, '34': {'4': 1, '2': 1}, '22': {'3': 1, '2': 1, '4': 2}, '19': {'4': 6, '2': 4, '3': 2}, '31': {'4': 1}, '30': {'3': 2, '2': 1}, '28': {'3': 2, '4': 1}, '39': {'4': 1}, '101': {'4': 1}, '92': {'4': 1, '3': 1}, '100': {'2': 1}, '32': {'3': 1}, '27': {'4': 2, '3': 1}, '82': {'2': 1}, '105': {'2': 2}, '44': {'3': 1}, '54': {'3': 1}, '79': {'4': 1}, '86': {'4': 1}, '89': {'4': 1}, '56': {'2': 1}, '40': {'2': 1}, '37': {'4': 1}, '99': {'4': 1}, '87': {'4': 1}, '77': {'3': 1}, '74': {'4': 1}}, 'm_histogram': {'3': 3924, '4': 37167, '5': 92074, '6': 92953, '7': 76923, '8': 77266, '9': 68110, '10': 41151, '11': 16257, '12': 4210, '13': 922, '14': 232, '1': 173, '2': 332, '24': 4, '17': 18, '15': 84, '0': 64, '26': 5, '16': 45, '95': 1, '23': 3, '20': 4, '21': 7, '47': 2, '18': 11, '29': 1, '41': 1, '25': 4, '51': 1, '33': 2, '34': 2, '22': 4, '19': 12, '31': 1, '30': 3, '28': 3, '39': 1, '101': 1, '92': 2, '100': 1, '32': 1, '27': 3, '82': 1, '105': 2, '44': 1, '54': 1, '79': 1, '86': 1, '89': 1, '56': 1, '40': 1, '37': 1, '99': 1, '87': 1, '77': 1, '74': 1}, 'mean_K_valid': 2.9535215339417427, 'mean_K_all': 2.95315234375, 'Dall_mean_sentences': 7.0531414083010375, 'strict_subset_ratio': 0.999537109375, 'mean_effective_tokens': {'F': 171.46915239404925, 'Dall': 153.89054100512564, 'Dk': 65.51318914864358}, 'mean_content_tokens': {'F': 169.46915239404925, 'Dall': 151.89054100512564, 'Dk': 63.51318914864358}, 'mean_lowest_Dall_content_token_coverage': 0.4467718713244764, 'pooled_lowest_Dall_content_token_coverage': 0.41815104962000427, 'coverage_definition': 'Actual EOT-delimited content tokens excluding SOT/EOT on identical valid samples'}`.
Full cycle(s):`{'count': 500, 'median': 2.126039743423462, 'p95': 2.273505413532257, 'p99': 2.3638276290893554, 'max': 40.18330764770508}`; slowest-rank data_wait(s):`{'count': 500, 'median': 0.0006797611713409424, 'p95': 0.0009411647915840149, 'p99': 0.0013911753892898555, 'max': 22.15341493487358}`.
OOM kills:0; true training I/O errors:0; Pod anomaly:false. GPU peak:`{'0': 27.760382175445557, '1': 27.760382175445557, '2': 27.760382175445557, '3': 27.760382175445557}` GiB.
First5 gate passed before update6; all512000 actual IDs,F/Dall token hashes and selected-index digests checked. Weight/sparsity arms also match all D3 text/token hashes exactly to Anchor.
Strict bare/native inference only:normalized native image embedding @ normalized native full-caption text embedding.T. No masks/gates/local-view inference/rerank/ensemble/TTA.
Common0 SHA256:`54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`.
Checkpoint SHA256:`a13ae36ce0d12ffc8b5d973bc045b31f3f5e9d717ec7edec7e0c97e053cafb2e`.
Bare SHA256:`ae30d4b27e70b319ce177fba371dff862100f155c0723d39d224d6b1724ed290`.
Checkpoints/bare/raw logs stay local. RUNTIME_STATS.json records raw paths,bytes,SHA256 and UTC containment windows.
Selection/classification and Pareto analysis occur after all seven arms, in SEARCH_SUMMARY.md. No arm-specific tuning or early metric stop.

Final search classification:`NO_IMPROVEMENT`.
