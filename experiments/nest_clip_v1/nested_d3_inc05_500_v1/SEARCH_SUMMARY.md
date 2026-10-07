# INC0.5: isolated Nested D3 local500 search arm

Completed exactly500 fresh-common0 optimizer updates, horizon4868. Only changed axis:`inclusion`.
Alignment:[1.35, 1.35, 0.3]; absolute sparsity coefficients:[1.0, 2.0, 2.0] (mass5.0); mode:nested_detail_d3.
Other construction,optimizer/LR,workers8,batch256/rank,preprocess and native protocol frozen. Inclusion max:0.5; zero bypasses the schedule and inclusion autograd graph. No combination/full run.
Local-only:`/root/said_s02_stage500/ShareGPT4V`; missing/symlink/escape fails, NFS fallback forbidden. /root is disposable overlay; NFS originals retained.

| Dataset | I2T R@1 / R@5 / R@10 (%) | T2I R@1 / R@5 / R@10 (%) |
|---|---|---|
| COCO | 60.540000 / 82.700000 / 89.420000 | 41.588000 / 67.416000 / 76.944000 |
| Urban-1k | 91.300005 / 98.400003 / 99.600005 | 88.800007 / 98.700005 / 99.300003 |
| Flickr30k-test1k | 87.200000 / 97.600000 / 99.200000 | 71.780000 / 91.380000 / 95.200000 |
| DOCCI | 77.800000 / 95.460000 / 98.080000 | 77.580000 / 95.160000 / 97.820000 |
| Long-DCI | 56.958695 / 75.401210 / 81.715338 | 56.471981 / 75.966851 / 81.807419 |

| Metric | Arm | Delta vs Anchor(pp) |
|---|---:|---:|
| Score5 | 71.001869 | -0.061731 |
| J_long3 | 74.818448 | -0.136218 |
| J_long | 83.870003 | -0.154998 |
| Short4 | 65.277000 | +0.050000 |
| Urban_I2T | 91.300000 | +0.400000 |
| Urban_T2I | 88.800000 | -0.900000 |

| Dataset | Delta I2T R@1/5/10(pp) | Delta T2I R@1/5/10(pp) |
|---|---|---|
| COCO | +0.140000 / +0.080000 / +0.160000 | +0.000000 / +0.140000 / +0.012000 |
| Urban-1k | +0.400001 / -0.200003 / +0.000000 | -0.899994 / +0.199997 / -0.200003 |
| Flickr30k-test1k | +0.200000 / -0.200000 / +0.000000 | -0.140000 / +0.000000 / +0.080000 |
| DOCCI | -0.020000 / +0.120000 / +0.080000 | -0.100000 / -0.100000 / +0.100000 |
| Long-DCI | -0.065772 / -0.131544 / +0.065772 | -0.131544 / -0.118390 / -0.131544 |

| View | Combined CE | Weighted CE | Alignment share | Keep ratio |
|---|---:|---:|---:|---:|
| F | 0.112343 | 0.505544 | 17.603% | 0.835299 |
| Dall | 0.210895 | 0.949027 | 33.044% | 0.814681 |
| D3 | 1.417427 | 1.417427 | 49.353% | 0.766186 |

Raw gradient norms:`{'F': 12.117538094520569, 'Dall': 15.318822145462036, 'D3': 34.84762239456177}`.
Weighted gradient norms:`{'F': 54.528921425342574, 'Dall': 68.93469965457918, 'D3': 34.84762239456177}`; weighted D3/Dall ratio:`0.505516`.
Frozen gradient protocol:8 first seed0 epoch0 global1024 batches at fixed step500; production backbone group; raw directional-summed CE,all-reduce/4; no optimizer updates. Norm magnitudes are not a signed gradient decomposition.
Mask hierarchy:`{'inc': 0.013497757408767939, 'inc_weight': 0.5, 'F_Dall_mask_iou': 0.928012261390686, 'Dall_D3_mask_iou': 0.8711714971065522, 'Dall_F_hard_violation': 0.020255601778626443, 'D3_Dall_hard_violation': 0.03030098758637905}`. Soft detached-child inclusion does not impose zero hard-mask violations.
Sampling:`{'records': 512000, 'valid_records': 511936, 'K_histogram_all': {'2': 3924, '3': 507507, '1': 505, '0': 64}, 'K_histogram_valid': {'2': 3924, '3': 507507, '1': 505}, 'K_histogram_by_m': {'8': {'3': 77266}, '5': {'3': 92074}, '6': {'3': 92953}, '9': {'3': 68110}, '4': {'3': 37167}, '10': {'3': 41151}, '12': {'3': 4210}, '7': {'3': 76923}, '11': {'3': 16257}, '3': {'2': 3924}, '13': {'3': 922}, '14': {'3': 232}, '1': {'1': 173}, '2': {'1': 332}, '24': {'3': 4}, '17': {'3': 18}, '15': {'3': 84}, '26': {'3': 5}, '16': {'3': 45}, '95': {'3': 1}, '23': {'3': 3}, '20': {'3': 4}, '21': {'3': 7}, '47': {'3': 2}, '18': {'3': 11}, '29': {'3': 1}, '41': {'3': 1}, '25': {'3': 4}, '51': {'3': 1}, '33': {'3': 2}, '34': {'3': 2}, '22': {'3': 4}, '19': {'3': 12}, '31': {'3': 1}, '30': {'3': 3}, '28': {'3': 3}, '39': {'3': 1}, '101': {'3': 1}, '92': {'3': 2}, '100': {'3': 1}, '32': {'3': 1}, '27': {'3': 3}, '82': {'3': 1}, '105': {'3': 2}, '44': {'3': 1}, '54': {'3': 1}, '79': {'3': 1}, '86': {'3': 1}, '89': {'3': 1}, '56': {'3': 1}, '40': {'3': 1}, '37': {'3': 1}, '99': {'3': 1}, '87': {'3': 1}, '77': {'3': 1}, '74': {'3': 1}}, 'm_histogram': {'3': 3924, '4': 37167, '5': 92074, '6': 92953, '7': 76923, '8': 77266, '9': 68110, '10': 41151, '11': 16257, '12': 4210, '13': 922, '14': 232, '1': 173, '2': 332, '24': 4, '17': 18, '15': 84, '0': 64, '26': 5, '16': 45, '95': 1, '23': 3, '20': 4, '21': 7, '47': 2, '18': 11, '29': 1, '41': 1, '25': 4, '51': 1, '33': 2, '34': 2, '22': 4, '19': 12, '31': 1, '30': 3, '28': 3, '39': 1, '101': 1, '92': 2, '100': 1, '32': 1, '27': 3, '82': 1, '105': 2, '44': 1, '54': 1, '79': 1, '86': 1, '89': 1, '56': 1, '40': 1, '37': 1, '99': 1, '87': 1, '77': 1, '74': 1}, 'mean_K_valid': 2.9903620765095638, 'mean_K_all': 2.98998828125, 'Dall_mean_sentences': 7.0531414083010375, 'strict_subset_ratio': 0.999537109375, 'mean_effective_tokens': {'F': 171.46915239404925, 'Dall': 153.89054100512564, 'D3': 66.38841183272909}, 'mean_content_tokens': {'F': 169.46915239404925, 'Dall': 151.89054100512564, 'D3': 64.38841183272909}, 'mean_lowest_Dall_content_token_coverage': 0.45601314895918665, 'pooled_lowest_Dall_content_token_coverage': 0.42391324309363193, 'coverage_definition': 'Actual EOT-delimited content tokens excluding SOT/EOT on identical valid samples'}`.
Full cycle(s):`{'count': 500, 'median': 2.120060443878174, 'p95': 2.2663551688194272, 'p99': 2.343806364536285, 'max': 24.80042839050293}`; slowest-rank data_wait(s):`{'count': 500, 'median': 0.000739268958568573, 'p95': 0.0009896136820316311, 'p99': 0.0015614420175552337, 'max': 20.428772918879986}`.
OOM kills:0; true training I/O errors:0; Pod anomaly:false. GPU peak:`{'0': 27.760382652282715, '1': 27.760382652282715, '2': 27.760382652282715, '3': 27.760382652282715}` GiB.
First5 gate passed before update6; all512000 actual IDs,F/Dall token hashes and selected-index digests checked. Arms with frozen lowest-view sampling also match its text/token/K/index trajectory exactly to the declared reference.
Strict bare/native inference only:normalized native image embedding @ normalized native full-caption text embedding.T. No masks/gates/local-view inference/rerank/ensemble/TTA.
Common0 SHA256:`54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`.
Checkpoint SHA256:`41a620c18cbf1165ec79a2460a3154407fe0ac8f5bb5ac7cb98beeffd6b1abb1`.
Bare SHA256:`9afcc7e74754c47ee4e60da9712af7a1e531ebf5e6d65a99fcf1ebf45ceafb6f`.
Checkpoints/bare/raw logs stay local. RUNTIME_STATS.json records raw paths,bytes,SHA256 and UTC containment windows.
This one arm is classified after complete native evaluation/review. No arm-specific tuning or early metric stop.

Actual sampling: median valid K=3.0; mean K/m=0.4596849174826309; pooled K/m=0.42397591419195474.

## Inclusion max0/0.5/1 comparison

Classification: `SOFT_INCLUSION_TRADEOFF`.
Only inclusion_max1 ->0.5. Same detached-child two-edge chain, original200-completed-update linear ramp and hard masks. No model/data/sampling/objective source edit for INC0.5.
At update u, weight=0.5*min((u-1)/200,1): update200=.4975,update201 onwards=.5, exactly half the original Anchor convention.
Every512000 IDs,F/Dall/D3 strings/tokens,K,selected indices and LR match Anchor. Optimizer definition/group order,counters and scheduler are frozen; learned parameter/moment values may differ.

| Model | Inclusion max | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I |
|---|---:|---:|---:|---:|---:|---:|---:|
| INC0 | 0 | 71.158309 | 74.978516 | 83.995003 | 65.428000 | 91.100000 | 89.400000 |
| INC0.5 | 0.5 | 71.001869 | 74.818448 | 83.870003 | 65.277000 | 91.300000 | 88.800000 |
| Anchor | 1 | 71.063600 | 74.954666 | 84.025001 | 65.227000 | 90.900000 | 89.700000 |

| Reference | Delta Score5 | Delta J_long3 | Delta J_long | Delta Short4 | Delta Urban I2T | Delta Urban T2I |
|---|---:|---:|---:|---:|---:|---:|
| Anchor | -0.061731 | -0.136218 | -0.154998 | +0.050000 | +0.400000 | -0.900000 |
| INC0 | -0.156441 | -0.160068 | -0.125000 | -0.151000 | +0.200000 | -0.600000 |

| Dataset | Delta I2T R@1/5/10 vs INC0(pp) | Delta T2I R@1/5/10 vs INC0(pp) |
|---|---|---|
| COCO | -0.140000 / -0.240000 / +0.180000 | +0.076000 / +0.088000 / +0.028000 |
| Urban-1k | +0.199997 / -0.400001 / -0.099999 | -0.599998 / +0.000000 / -0.200003 |
| Flickr30k-test1k | -0.200000 / +0.000000 / +0.100000 | -0.340000 / -0.120000 / -0.040000 |
| DOCCI | -0.060000 / +0.060000 / +0.040000 | -0.040000 / -0.060000 / -0.040000 |
| Long-DCI | -0.171008 / -0.342015 / -0.210471 | -0.289398 / -0.105235 / -0.236780 |

| Hierarchy | INC0 | INC0.5 | Anchor |
|---|---:|---:|---:|
| Dall_F_hard_violation | 0.027397 | 0.020256 | 0.010362 |
| D3_Dall_hard_violation | 0.041259 | 0.030301 | 0.016811 |
| F_Dall_mask_iou | 0.920955 | 0.928012 | 0.965864 |
| Dall_D3_mask_iou | 0.849885 | 0.871171 | 0.900053 |

Hierarchy position: `{'violation_between_INC0_and_Anchor': {'Dall_F_hard_violation': True, 'D3_Dall_hard_violation': True}, 'IoU_between_INC0_and_Anchor': {'F_Dall_mask_iou': True, 'Dall_D3_mask_iou': True}, 'all_four_between': True, 'clearly_stronger_than_INC0': True, 'mean_coverage_ordered': True}`; hard violation deltas(pp): `{'Anchor': {'Dall_F_hard_violation': 0.9893480055034161, 'D3_Dall_hard_violation': 1.3489808589220047}, 'INC0': {'Dall_F_hard_violation': -0.7140930667519567, 'D3_Dall_hard_violation': -1.0958497002720835}}`.
Raw/weighted gradient changes vs references: `{'Anchor': {'weighted_D3_Dall_ratio': 0.0005994031714006143, 'raw_norms': {'F': 1.3907750844955444, 'Dall': 0.9945173263549805, 'D3': 2.3009889125823975}}, 'INC0': {'weighted_D3_Dall_ratio': -0.05609233534228064, 'raw_norms': {'F': 0.5031522512435913, 'Dall': 0.24917268753051758, 'D3': -3.2369892597198486}}}`.
Applied/raw inclusion audit: `{'passed': True, 'optimizer_updates': 500, 'inclusion_max': 0.5, 'all500_weights_exactly_half_Anchor': True, 'formula_and_detached_child_unchanged': True, 'ramp_completed_updates': 200, 'weight_by_update': {'1': 0.0, '5': 0.01, '100': 0.2475, '200': 0.4975, '201': 0.5, '500': 0.5}, 'last50_raw_inclusion': 0.013497757408767939, 'last50_applied_inclusion': 0.006748878704383969, 'optimizer_definition_groups_counters_and_scheduler_exact': True, 'optimizer_moments_and_model_values_cross_run_bitwise_required': False}`.
Operational thresholds frozen in SEARCH_PLAN.json before launch. Mean coverage ordering is distinct from sample-level coordinate inclusion.
Checkpoint,bare,images and raw large logs stay local; paths/sizes/SHA/time windows in RUNTIME_STATS.json.
Stopped at exactly500; no full,0.25/0.75,schedule,sparsity,K,alignment or other experiment.
