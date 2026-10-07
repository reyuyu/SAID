# CoupledNested: isolated Nested D3 local500 search arm

Completed exactly500 fresh-common0 optimizer updates, horizon4868. Only changed axis:`regularizer`.
Alignment:[1.35, 1.35, 0.3]; coupled edge coefficients:[1,2,2]; child global sparsity replaced by conditional inside terms; mode:nested_detail_d3.
Other construction,optimizer/LR,workers8,batch256/rank,preprocess and native protocol frozen. Inclusion max:1; zero bypasses the schedule and inclusion autograd graph. No combination/full run.
Local-only:`/root/said_s02_stage500/ShareGPT4V`; missing/symlink/escape fails, NFS fallback forbidden. /root is disposable overlay; NFS originals retained.

| Dataset | I2T R@1 / R@5 / R@10 (%) | T2I R@1 / R@5 / R@10 (%) |
|---|---|---|
| COCO | 60.620000 / 82.680000 / 89.360000 | 41.472000 / 67.288000 / 76.896000 |
| Urban-1k | 91.300005 / 98.700005 / 99.600005 | 89.500004 / 98.600006 / 99.500006 |
| Flickr30k-test1k | 87.400000 / 97.300000 / 99.100000 | 71.800000 / 91.380000 / 95.100000 |
| DOCCI | 77.760000 / 95.560000 / 98.120000 | 77.460000 / 95.140000 / 97.720000 |
| Long-DCI | 56.787687 / 75.348592 / 81.557485 | 56.629834 / 75.980005 / 81.754801 |

| Metric | Arm | Delta vs Anchor(pp) |
|---|---:|---:|
| Score5 | 71.072953 | +0.009353 |
| J_long3 | 74.906255 | -0.048411 |
| J_long | 84.005002 | -0.019999 |
| Short4 | 65.323000 | +0.096000 |
| Urban_I2T | 91.300000 | +0.400000 |
| Urban_T2I | 89.500000 | -0.200000 |

| Dataset | Delta I2T R@1/5/10(pp) | Delta T2I R@1/5/10(pp) |
|---|---|---|
| COCO | +0.220000 / +0.060000 / +0.100000 | -0.116000 / +0.012000 / -0.036000 |
| Urban-1k | +0.400001 / +0.099999 / +0.000000 | -0.199997 / +0.099999 / +0.000000 |
| Flickr30k-test1k | +0.400000 / -0.500000 / -0.100000 | -0.120000 / +0.000000 / -0.020000 |
| DOCCI | -0.060000 / +0.220000 / +0.120000 | -0.220000 / -0.120000 / +0.000000 |
| Long-DCI | -0.236780 / -0.184162 / -0.092081 | +0.026309 / -0.105235 / -0.184162 |

| View | Combined CE | Weighted CE | Alignment share | Keep ratio |
|---|---:|---:|---:|---:|
| F | 0.112004 | 0.504019 | 17.581% | 0.829079 |
| Dall | 0.207503 | 0.933761 | 32.571% | 0.801975 |
| D3 | 1.429064 | 1.429064 | 49.848% | 0.720364 |

Raw gradient norms:`{'F': 12.091317653656006, 'Dall': 14.566922307014465, 'D3': 34.33042573928833}`.
Weighted gradient norms:`{'F': 54.41092944145204, 'Dall': 65.55115038156511, 'D3': 34.33042573928833}`; weighted D3/Dall ratio:`0.523720`.
Frozen gradient protocol:8 first seed0 epoch0 global1024 batches at fixed step500; production backbone group; raw directional-summed CE,all-reduce/4; no optimizer updates. Norm magnitudes are not a signed gradient decomposition.
Mask hierarchy:`{'inc': 0.011966807506978512, 'inc_weight': 1.0, 'F_Dall_mask_iou': 0.9244786608219147, 'Dall_D3_mask_iou': 0.838025872707367, 'Dall_F_hard_violation': 0.018080187886953356, 'D3_Dall_hard_violation': 0.02648358706384897}`. Soft detached-child inclusion does not impose zero hard-mask violations.
Sampling:`{'records': 512000, 'valid_records': 511936, 'K_histogram_all': {'2': 3924, '3': 507507, '1': 505, '0': 64}, 'K_histogram_valid': {'2': 3924, '3': 507507, '1': 505}, 'K_histogram_by_m': {'8': {'3': 77266}, '5': {'3': 92074}, '6': {'3': 92953}, '9': {'3': 68110}, '4': {'3': 37167}, '10': {'3': 41151}, '12': {'3': 4210}, '7': {'3': 76923}, '11': {'3': 16257}, '3': {'2': 3924}, '13': {'3': 922}, '14': {'3': 232}, '1': {'1': 173}, '2': {'1': 332}, '24': {'3': 4}, '17': {'3': 18}, '15': {'3': 84}, '26': {'3': 5}, '16': {'3': 45}, '95': {'3': 1}, '23': {'3': 3}, '20': {'3': 4}, '21': {'3': 7}, '47': {'3': 2}, '18': {'3': 11}, '29': {'3': 1}, '41': {'3': 1}, '25': {'3': 4}, '51': {'3': 1}, '33': {'3': 2}, '34': {'3': 2}, '22': {'3': 4}, '19': {'3': 12}, '31': {'3': 1}, '30': {'3': 3}, '28': {'3': 3}, '39': {'3': 1}, '101': {'3': 1}, '92': {'3': 2}, '100': {'3': 1}, '32': {'3': 1}, '27': {'3': 3}, '82': {'3': 1}, '105': {'3': 2}, '44': {'3': 1}, '54': {'3': 1}, '79': {'3': 1}, '86': {'3': 1}, '89': {'3': 1}, '56': {'3': 1}, '40': {'3': 1}, '37': {'3': 1}, '99': {'3': 1}, '87': {'3': 1}, '77': {'3': 1}, '74': {'3': 1}}, 'm_histogram': {'3': 3924, '4': 37167, '5': 92074, '6': 92953, '7': 76923, '8': 77266, '9': 68110, '10': 41151, '11': 16257, '12': 4210, '13': 922, '14': 232, '1': 173, '2': 332, '24': 4, '17': 18, '15': 84, '0': 64, '26': 5, '16': 45, '95': 1, '23': 3, '20': 4, '21': 7, '47': 2, '18': 11, '29': 1, '41': 1, '25': 4, '51': 1, '33': 2, '34': 2, '22': 4, '19': 12, '31': 1, '30': 3, '28': 3, '39': 1, '101': 1, '92': 2, '100': 1, '32': 1, '27': 3, '82': 1, '105': 2, '44': 1, '54': 1, '79': 1, '86': 1, '89': 1, '56': 1, '40': 1, '37': 1, '99': 1, '87': 1, '77': 1, '74': 1}, 'mean_K_valid': 2.9903620765095638, 'mean_K_all': 2.98998828125, 'Dall_mean_sentences': 7.0531414083010375, 'strict_subset_ratio': 0.999537109375, 'mean_effective_tokens': {'F': 171.46915239404925, 'Dall': 153.89054100512564, 'D3': 66.38841183272909}, 'mean_content_tokens': {'F': 169.46915239404925, 'Dall': 151.89054100512564, 'D3': 64.38841183272909}, 'mean_lowest_Dall_content_token_coverage': 0.45601314895918665, 'pooled_lowest_Dall_content_token_coverage': 0.42391324309363193, 'coverage_definition': 'Actual EOT-delimited content tokens excluding SOT/EOT on identical valid samples'}`.
Full cycle(s):`{'count': 500, 'median': 2.119062662124634, 'p95': 2.29143363237381, 'p99': 2.4116548919677734, 'max': 24.94293785095215}`; slowest-rank data_wait(s):`{'count': 500, 'median': 0.000716283917427063, 'p95': 0.0009215135127305982, 'p99': 0.0021519242972135484, 'max': 21.160989969968796}`.
OOM kills:0; true training I/O errors:0; Pod anomaly:false. GPU peak:`{'0': 27.759899139404297, '1': 27.759899139404297, '2': 27.759899139404297, '3': 27.759899139404297}` GiB.
First5 gate passed before update6; all512000 actual IDs,F/Dall token hashes and selected-index digests checked. Arms with frozen lowest-view sampling also match its text/token/K/index trajectory exactly to the declared reference.
Strict bare/native inference only:normalized native image embedding @ normalized native full-caption text embedding.T. No masks/gates/local-view inference/rerank/ensemble/TTA.
Common0 SHA256:`54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`.
Checkpoint SHA256:`cb3751b1fe8029490140bd7a777fe218d00f81e01aaaa4f7d6c7e19f9008823b`.
Bare SHA256:`2f9cc44cfe2d564557eb6af3c456805e14c3a874aade567de852e98af2ea0e93`.
Checkpoints/bare/raw logs stay local. RUNTIME_STATS.json records raw paths,bytes,SHA256 and UTC containment windows.
One coupled arm only; classified after complete evaluation and audit. No arm-specific tuning or early metric stop.

Actual sampling: median valid K=3.0; mean K/m=0.4596849174826309; pooled K/m=0.42397591419195474.

## Coupled regularizer comparison

Classification: `COUPLED_NESTED_TRADEOFF`.
Only regularizer_mode changes. F global Hard-ST sparsity remains; child inside terms use soft detached parent and soft child. No extra child global sparsity or independent inclusion.
Outside expanded coefficient is 1/2*lambda per edge, identical to Anchor. Ramp uses original completed updates: update200=.995; update201 onwards=1.

| Model | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I |
|---|---:|---:|---:|---:|---:|---:|
| Anchor | 71.063600 | 74.954666 | 84.025001 | 65.227000 | 90.900000 | 89.700000 |
| INC0 | 71.158309 | 74.978516 | 83.995003 | 65.428000 | 91.100000 | 89.400000 |
| INC0.5 | 71.001869 | 74.818448 | 83.870003 | 65.277000 | 91.300000 | 88.800000 |
| CoupledNested | 71.072953 | 74.906255 | 84.005002 | 65.323000 | 91.300000 | 89.500000 |

| Reference | Delta Score5 | Delta J_long3 | Delta J_long | Delta Short4 | Delta Urban I2T | Delta Urban T2I |
|---|---:|---:|---:|---:|---:|---:|
| Anchor | +0.009353 | -0.048411 | -0.019999 | +0.096000 | +0.400000 | -0.200000 |
| INC0 | -0.085356 | -0.072261 | +0.009999 | -0.105000 | +0.200000 | +0.100000 |
| INC0.5 | +0.071084 | +0.087807 | +0.134999 | +0.046000 | +0.000000 | +0.700000 |

All native recalls vs Anchor (pp):

| Dataset | Delta I2T R@1/5/10 | Delta T2I R@1/5/10 |
|---|---|---|
| COCO | +0.220000 / +0.060000 / +0.100000 | -0.116000 / +0.012000 / -0.036000 |
| Urban-1k | +0.400001 / +0.099999 / +0.000000 | -0.199997 / +0.099999 / +0.000000 |
| Flickr30k-test1k | +0.400000 / -0.500000 / -0.100000 | -0.120000 / +0.000000 / -0.020000 |
| DOCCI | -0.060000 / +0.220000 / +0.120000 | -0.220000 / -0.120000 / +0.000000 |
| Long-DCI | -0.236780 / -0.184162 / -0.092081 | +0.026309 / -0.105235 / -0.184162 |

All native recalls vs INC0 (pp):

| Dataset | Delta I2T R@1/5/10 | Delta T2I R@1/5/10 |
|---|---|---|
| COCO | -0.060000 / -0.260000 / +0.120000 | -0.040000 / -0.040000 / -0.020000 |
| Urban-1k | +0.199997 / -0.099999 / -0.099999 | +0.099999 / -0.099999 / +0.000000 |
| Flickr30k-test1k | +0.000000 / -0.300000 / +0.000000 | -0.320000 / -0.120000 / -0.140000 |
| DOCCI | -0.100000 / +0.160000 / +0.080000 | -0.160000 / -0.080000 / -0.140000 |
| Long-DCI | -0.342015 / -0.394633 / -0.368324 | -0.131544 / -0.092081 / -0.289398 |

Last50 regularizer terms: `{'L_in_Dall_F': 0.6640025174617767, 'L_out_Dall_F': 0.0099195166118443, 'L_in_D3_Dall': 0.6254693531990051, 'L_out_D3_Dall': 0.014014098793268204, 'Omega_F': 0.8290791702270508, 'total_inside_sparsity': 0.8596479415893554, 'total_outside_penalty': 0.011966807506978512, 'total_nested_regularizer': 1.1479744601249695, 'old_global_sparse_counterfactual': 1.291252453327179, 'Dall_F_inside_parent_keep_ratio': 0.9448484158515931, 'D3_Dall_inside_parent_keep_ratio': 0.8656872355937958, 'Dall_F_outside_probability_mass': 102.85028015136719, 'D3_Dall_outside_probability_mass': 101.46624374389648, 'Dall_F_outside_mass_ratio': 0.31999850809574126, 'D3_Dall_outside_mass_ratio': 0.33489628255367276}`.
Matched actual Anchor magnitudes and same-current-mask counterfactual: `{'nested_regularizer': 1.1479744601249695, 'conditional_inside': 0.8596479415893554, 'outside_penalty': 0.011966807702556252, 'Omega_F': 0.8290791702270508, 'actual_Anchor_global_sparse': 1.375123951435089, 'actual_Anchor_inclusion': 0.008108068946748972, 'counterfactual_old_global_sparse_same_current_masks': 1.2912524064381918, 'delta_vs_matched_Anchor_total': -0.23525756025686861}`.
Gradient routing manual tensor audit: `{'passed': True, 'child': [[0.8, 0.2]], 'parent': [[0.1, 0.9]], 'inside': 0.25999974000026005, 'outside': 0.35000000000000003, 'outside_child_gradient': None, 'outside_parent_gradient': [[-0.5, -0.0]], 'inside_child_gradient': [[0.0999999000001, 0.8999991000009001]], 'inside_parent_gradient': None, 'epsilon': 1e-06, 'parent_detached_in_inside': True, 'child_detached_in_outside': True}`.
Hierarchy deltas (fractions; multiply violations by100 for pp): `{'Anchor': {'last50': {'inc': 0.0038587385602295403, 'inc_weight': 0.0, 'F_Dall_mask_iou': -0.04138521313667298, 'Dall_D3_mask_iou': -0.0620268392562866, 'Dall_F_hard_violation': 0.007718066163361074, 'D3_Dall_hard_violation': 0.009672408066689966}, 'keep_ratio': {'F': -0.023587570190429674, 'Dall': -0.04231418728828429, 'D3': -0.07169934511184695}}, 'INC0': {'last50': {'inc': -0.004703354462981224, 'inc_weight': 1.0, 'F_Dall_mask_iou': 0.0035231697559356956, 'Dall_D3_mask_iou': -0.011858916282653786, 'Dall_F_hard_violation': -0.009316344559192655, 'D3_Dall_hard_violation': -0.014775897525250915}, 'keep_ratio': {'F': 0.030323524475097674, 'Dall': 0.01291921019554143, 'D3': -0.02649802207946783}}, 'INC0.5': {'last50': {'inc': -0.0015309499017894263, 'inc_weight': 0.5, 'F_Dall_mask_iou': -0.003533600568771389, 'Dall_D3_mask_iou': -0.033145624399185225, 'Dall_F_hard_violation': -0.002175413891673087, 'D3_Dall_hard_violation': -0.0038174005225300803}, 'keep_ratio': {'F': -0.006219329833984322, 'Dall': -0.012705852985382071, 'D3': -0.04582251548767091}}}`.
Gradient deltas: `{'Anchor': {'weighted_D3_Dall_ratio': 0.01880263902480006, 'raw_norms': {'F': 1.3645546436309814, 'Dall': 0.24261748790740967, 'D3': 1.78379225730896}}, 'INC0': {'weighted_D3_Dall_ratio': -0.037889099488881195, 'raw_norms': {'F': 0.4769318103790283, 'Dall': -0.5027271509170532, 'D3': -3.754185914993286}}, 'INC0.5': {'weighted_D3_Dall_ratio': 0.018203235853399446, 'raw_norms': {'F': -0.02622044086456299, 'Dall': -0.7518998384475708, 'D3': -0.5171966552734375}}}`; mean coverage ordered:True.
Conditional support is normalized: uniformly tiny nonzero parent support does not guarantee tiny inside loss; relative low-support coordinates have small contributions.
Conditional/inside statistics are telemetry, not additional losses. Original last50 CE and fixed8-batch gradient protocol remain unchanged.
Stopped500; no full,other beta/budget/K/weights/optimizer or experiments. Binary/raw artifact paths,size,SHA/time ranges in RUNTIME_STATS.json.

Report-only recovery: historical Anchor logs lacked inclusion_loss. Applied inclusion was reconstructed as recorded inc_weight*inc. 27 correctness/report regression tests passed, including actual500-step Anchor log schema. Only report adapter/tests changed after launch; original launch/source archive remains intact. Training/evaluation were not repeated; checkpoint and bare SHA256 rechecked unchanged. See REGULARIZER_AUDIT.json report_only_recovery.
