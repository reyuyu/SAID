# WeakSparse-051015: isolated Nested D3 local500 search arm

Completed exactly500 fresh-common0 optimizer updates, horizon4868. Only changed axis:`sparsity`.
Alignment:[1.35, 1.35, 0.3]; absolute sparsity coefficients:[0.5, 1.0, 1.5] (mass3.0); mode:nested_detail_d3.
Other construction,optimizer/LR,workers8,batch256/rank,inclusion chain/ramp200/max1,preprocess and native protocol frozen. No combination/full run.
Local-only:`/root/said_s02_stage500/ShareGPT4V`; missing/symlink/escape fails, NFS fallback forbidden. /root is disposable overlay; NFS originals retained.

| Dataset | I2T R@1 / R@5 / R@10 (%) | T2I R@1 / R@5 / R@10 (%) |
|---|---|---|
| COCO | 60.380000 / 82.740000 / 89.300000 | 41.368000 / 67.268000 / 76.840000 |
| Urban-1k | 90.700006 / 98.500007 / 99.800003 | 89.100003 / 98.600006 / 99.400002 |
| Flickr30k-test1k | 86.600000 / 97.600000 / 99.000000 | 71.760000 / 91.340000 / 95.280000 |
| DOCCI | 77.780000 / 95.300000 / 98.060000 | 77.900000 / 95.160000 / 97.800000 |
| Long-DCI | 56.866614 / 75.782689 / 81.781110 | 56.524599 / 75.848461 / 81.794265 |

| Metric | Arm | Delta vs Anchor(pp) |
|---|---:|---:|
| Score5 | 70.897922 | -0.165678 |
| J_long3 | 74.811870 | -0.142796 |
| J_long | 83.870002 | -0.154999 |
| Short4 | 65.027000 | -0.200000 |
| Urban_I2T | 90.700000 | -0.200000 |
| Urban_T2I | 89.100000 | -0.600000 |

| Dataset | Delta I2T R@1/5/10(pp) | Delta T2I R@1/5/10(pp) |
|---|---|---|
| COCO | -0.020000 / +0.120000 / +0.040000 | -0.220000 / -0.008000 / -0.092000 |
| Urban-1k | -0.199997 / -0.099999 / +0.199997 | -0.599998 / +0.099999 / -0.100005 |
| Flickr30k-test1k | -0.400000 / -0.200000 / -0.200000 | -0.160000 / -0.040000 / +0.160000 |
| DOCCI | -0.040000 / -0.040000 / +0.060000 | +0.220000 / -0.100000 / +0.080000 |
| Long-DCI | -0.157853 / +0.249934 / +0.131544 | -0.078927 / -0.236780 / -0.144699 |

| View | Combined CE | Weighted CE | Alignment share | Keep ratio |
|---|---:|---:|---:|---:|
| F | 0.107042 | 0.481690 | 16.843% | 0.902246 |
| Dall | 0.203550 | 0.915976 | 32.029% | 0.884958 |
| D3 | 1.462142 | 1.462142 | 51.127% | 0.809365 |

Raw gradient norms:`{'F': 12.400637865066528, 'Dall': 16.444295644760132, 'D3': 41.02921485900879}`.
Weighted gradient norms:`{'F': 55.80287039279939, 'Dall': 73.99933040142061, 'D3': 41.02921485900879}`; weighted D3/Dall ratio:`0.554454`.
Frozen gradient protocol:8 first seed0 epoch0 global1024 batches at fixed step500; production backbone group; raw directional-summed CE,all-reduce/4; no optimizer updates. Norm magnitudes are not a signed gradient decomposition.
Mask hierarchy:`{'inc': 0.008363506784662604, 'inc_weight': 1.0, 'F_Dall_mask_iou': 0.9590872192382812, 'Dall_D3_mask_iou': 0.8868564307689667, 'Dall_F_hard_violation': 0.009971975293010473, 'D3_Dall_hard_violation': 0.013274404685944319}`. Soft detached-child inclusion does not impose zero hard-mask violations.
Sampling:`{'records': 512000, 'valid_records': 511936, 'K_histogram_all': {'2': 3924, '3': 507507, '1': 505, '0': 64}, 'K_histogram_valid': {'2': 3924, '3': 507507, '1': 505}, 'K_histogram_by_m': {'8': {'3': 77266}, '5': {'3': 92074}, '6': {'3': 92953}, '9': {'3': 68110}, '4': {'3': 37167}, '10': {'3': 41151}, '12': {'3': 4210}, '7': {'3': 76923}, '11': {'3': 16257}, '3': {'2': 3924}, '13': {'3': 922}, '14': {'3': 232}, '1': {'1': 173}, '2': {'1': 332}, '24': {'3': 4}, '17': {'3': 18}, '15': {'3': 84}, '26': {'3': 5}, '16': {'3': 45}, '95': {'3': 1}, '23': {'3': 3}, '20': {'3': 4}, '21': {'3': 7}, '47': {'3': 2}, '18': {'3': 11}, '29': {'3': 1}, '41': {'3': 1}, '25': {'3': 4}, '51': {'3': 1}, '33': {'3': 2}, '34': {'3': 2}, '22': {'3': 4}, '19': {'3': 12}, '31': {'3': 1}, '30': {'3': 3}, '28': {'3': 3}, '39': {'3': 1}, '101': {'3': 1}, '92': {'3': 2}, '100': {'3': 1}, '32': {'3': 1}, '27': {'3': 3}, '82': {'3': 1}, '105': {'3': 2}, '44': {'3': 1}, '54': {'3': 1}, '79': {'3': 1}, '86': {'3': 1}, '89': {'3': 1}, '56': {'3': 1}, '40': {'3': 1}, '37': {'3': 1}, '99': {'3': 1}, '87': {'3': 1}, '77': {'3': 1}, '74': {'3': 1}}, 'm_histogram': {'3': 3924, '4': 37167, '5': 92074, '6': 92953, '7': 76923, '8': 77266, '9': 68110, '10': 41151, '11': 16257, '12': 4210, '13': 922, '14': 232, '1': 173, '2': 332, '24': 4, '17': 18, '15': 84, '0': 64, '26': 5, '16': 45, '95': 1, '23': 3, '20': 4, '21': 7, '47': 2, '18': 11, '29': 1, '41': 1, '25': 4, '51': 1, '33': 2, '34': 2, '22': 4, '19': 12, '31': 1, '30': 3, '28': 3, '39': 1, '101': 1, '92': 2, '100': 1, '32': 1, '27': 3, '82': 1, '105': 2, '44': 1, '54': 1, '79': 1, '86': 1, '89': 1, '56': 1, '40': 1, '37': 1, '99': 1, '87': 1, '77': 1, '74': 1}, 'mean_K_valid': 2.9903620765095638, 'mean_K_all': 2.98998828125, 'Dall_mean_sentences': 7.0531414083010375, 'strict_subset_ratio': 0.999537109375, 'mean_effective_tokens': {'F': 171.46915239404925, 'Dall': 153.89054100512564, 'D3': 66.38841183272909}, 'mean_content_tokens': {'F': 169.46915239404925, 'Dall': 151.89054100512564, 'D3': 64.38841183272909}, 'mean_lowest_Dall_content_token_coverage': 0.45601314895918665, 'pooled_lowest_Dall_content_token_coverage': 0.42391324309363193, 'coverage_definition': 'Actual EOT-delimited content tokens excluding SOT/EOT on identical valid samples'}`.
Full cycle(s):`{'count': 500, 'median': 2.1188801527023315, 'p95': 2.291295325756073, 'p99': 2.3653741407394406, 'max': 26.349651336669922}`; slowest-rank data_wait(s):`{'count': 500, 'median': 0.0006726793944835663, 'p95': 0.0008813027292489993, 'p99': 0.0012823705375194542, 'max': 22.613490112125874}`.
OOM kills:0; true training I/O errors:0; Pod anomaly:false. GPU peak:`{'0': 27.760382175445557, '1': 27.760382175445557, '2': 27.760382175445557, '3': 27.760382175445557}` GiB.
First5 gate passed before update6; all512000 actual IDs,F/Dall token hashes and selected-index digests checked. Weight/sparsity arms also match all D3 text/token hashes exactly to Anchor.
Strict bare/native inference only:normalized native image embedding @ normalized native full-caption text embedding.T. No masks/gates/local-view inference/rerank/ensemble/TTA.
Common0 SHA256:`54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`.
Checkpoint SHA256:`398db03c1cd9a7d1f7db645a15c108bac3928dd90c92a43b5441d8052bb13a5c`.
Bare SHA256:`0cb9bbf5083ec67821494c5fc53148a1a44d14df1450b6d810ef9a9fb21ba33c`.
Checkpoints/bare/raw logs stay local. RUNTIME_STATS.json records raw paths,bytes,SHA256 and UTC containment windows.
Selection/classification occurs after all declared arms, in SEARCH_SUMMARY.md. No arm-specific tuning or early metric stop.

Actual sampling: median valid K=3.0; mean K/m=0.4596849174826309; pooled K/m=0.42397591419195474.
