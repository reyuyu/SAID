# INC0: isolated Nested D3 local500 search arm

Completed exactly500 fresh-common0 optimizer updates, horizon4868. Only changed axis:`inclusion`.
Alignment:[1.35, 1.35, 0.3]; absolute sparsity coefficients:[1.0, 2.0, 2.0] (mass5.0); mode:nested_detail_d3.
Other construction,optimizer/LR,workers8,batch256/rank,preprocess and native protocol frozen. Inclusion max:0.0; zero bypasses the schedule and inclusion autograd graph. No combination/full run.
Local-only:`/root/said_s02_stage500/ShareGPT4V`; missing/symlink/escape fails, NFS fallback forbidden. /root is disposable overlay; NFS originals retained.

| Dataset | I2T R@1 / R@5 / R@10 (%) | T2I R@1 / R@5 / R@10 (%) |
|---|---|---|
| COCO | 60.680000 / 82.940000 / 89.240000 | 41.512000 / 67.328000 / 76.916000 |
| Urban-1k | 91.100007 / 98.800004 / 99.700004 | 89.400005 / 98.700005 / 99.500006 |
| Flickr30k-test1k | 87.400000 / 97.600000 / 99.100000 | 72.120000 / 91.500000 / 95.240000 |
| DOCCI | 77.860000 / 95.400000 / 98.040000 | 77.620000 / 95.220000 / 97.860000 |
| Long-DCI | 57.129703 / 75.743225 / 81.925809 | 56.761379 / 76.072086 / 82.044199 |

| Metric | Arm | Delta vs Anchor(pp) |
|---|---:|---:|
| Score5 | 71.158309 | +0.094710 |
| J_long3 | 74.978516 | +0.023849 |
| J_long | 83.995003 | -0.029998 |
| Short4 | 65.428000 | +0.201000 |
| Urban_I2T | 91.100000 | +0.200000 |
| Urban_T2I | 89.400000 | -0.300000 |

| Dataset | Delta I2T R@1/5/10(pp) | Delta T2I R@1/5/10(pp) |
|---|---|---|
| COCO | +0.280000 / +0.320000 / -0.020000 | -0.076000 / +0.052000 / -0.016000 |
| Urban-1k | +0.200003 / +0.199997 / +0.099999 | -0.299996 / +0.199997 / +0.000000 |
| Flickr30k-test1k | +0.400000 / -0.200000 / -0.100000 | +0.200000 / +0.120000 / +0.120000 |
| DOCCI | +0.040000 / +0.060000 / +0.040000 | -0.060000 / -0.040000 / +0.140000 |
| Long-DCI | +0.105235 / +0.210471 / +0.276243 | +0.157853 / -0.013154 / +0.105235 |

| View | Combined CE | Weighted CE | Alignment share | Keep ratio |
|---|---:|---:|---:|---:|
| F | 0.116096 | 0.522432 | 18.130% | 0.798756 |
| Dall | 0.209241 | 0.941585 | 32.676% | 0.789056 |
| D3 | 1.417579 | 1.417579 | 49.194% | 0.746862 |

Raw gradient norms:`{'F': 11.614385843276978, 'Dall': 15.069649457931519, 'D3': 38.084611654281616}`.
Weighted gradient norms:`{'F': 52.264736294746406, 'Dall': 67.81342256069185, 'D3': 38.084611654281616}`; weighted D3/Dall ratio:`0.561609`.
Frozen gradient protocol:8 first seed0 epoch0 global1024 batches at fixed step500; production backbone group; raw directional-summed CE,all-reduce/4; no optimizer updates. Norm magnitudes are not a signed gradient decomposition.
Mask hierarchy:`{'inc': 0.016670161969959737, 'inc_weight': 0.0, 'F_Dall_mask_iou': 0.920955491065979, 'Dall_D3_mask_iou': 0.8498847889900207, 'Dall_F_hard_violation': 0.02739653244614601, 'D3_Dall_hard_violation': 0.041259484589099886}`. Soft detached-child inclusion does not impose zero hard-mask violations.
Sampling:`{'records': 512000, 'valid_records': 511936, 'K_histogram_all': {'2': 3924, '3': 507507, '1': 505, '0': 64}, 'K_histogram_valid': {'2': 3924, '3': 507507, '1': 505}, 'K_histogram_by_m': {'8': {'3': 77266}, '5': {'3': 92074}, '6': {'3': 92953}, '9': {'3': 68110}, '4': {'3': 37167}, '10': {'3': 41151}, '12': {'3': 4210}, '7': {'3': 76923}, '11': {'3': 16257}, '3': {'2': 3924}, '13': {'3': 922}, '14': {'3': 232}, '1': {'1': 173}, '2': {'1': 332}, '24': {'3': 4}, '17': {'3': 18}, '15': {'3': 84}, '26': {'3': 5}, '16': {'3': 45}, '95': {'3': 1}, '23': {'3': 3}, '20': {'3': 4}, '21': {'3': 7}, '47': {'3': 2}, '18': {'3': 11}, '29': {'3': 1}, '41': {'3': 1}, '25': {'3': 4}, '51': {'3': 1}, '33': {'3': 2}, '34': {'3': 2}, '22': {'3': 4}, '19': {'3': 12}, '31': {'3': 1}, '30': {'3': 3}, '28': {'3': 3}, '39': {'3': 1}, '101': {'3': 1}, '92': {'3': 2}, '100': {'3': 1}, '32': {'3': 1}, '27': {'3': 3}, '82': {'3': 1}, '105': {'3': 2}, '44': {'3': 1}, '54': {'3': 1}, '79': {'3': 1}, '86': {'3': 1}, '89': {'3': 1}, '56': {'3': 1}, '40': {'3': 1}, '37': {'3': 1}, '99': {'3': 1}, '87': {'3': 1}, '77': {'3': 1}, '74': {'3': 1}}, 'm_histogram': {'3': 3924, '4': 37167, '5': 92074, '6': 92953, '7': 76923, '8': 77266, '9': 68110, '10': 41151, '11': 16257, '12': 4210, '13': 922, '14': 232, '1': 173, '2': 332, '24': 4, '17': 18, '15': 84, '0': 64, '26': 5, '16': 45, '95': 1, '23': 3, '20': 4, '21': 7, '47': 2, '18': 11, '29': 1, '41': 1, '25': 4, '51': 1, '33': 2, '34': 2, '22': 4, '19': 12, '31': 1, '30': 3, '28': 3, '39': 1, '101': 1, '92': 2, '100': 1, '32': 1, '27': 3, '82': 1, '105': 2, '44': 1, '54': 1, '79': 1, '86': 1, '89': 1, '56': 1, '40': 1, '37': 1, '99': 1, '87': 1, '77': 1, '74': 1}, 'mean_K_valid': 2.9903620765095638, 'mean_K_all': 2.98998828125, 'Dall_mean_sentences': 7.0531414083010375, 'strict_subset_ratio': 0.999537109375, 'mean_effective_tokens': {'F': 171.46915239404925, 'Dall': 153.89054100512564, 'D3': 66.38841183272909}, 'mean_content_tokens': {'F': 169.46915239404925, 'Dall': 151.89054100512564, 'D3': 64.38841183272909}, 'mean_lowest_Dall_content_token_coverage': 0.45601314895918665, 'pooled_lowest_Dall_content_token_coverage': 0.42391324309363193, 'coverage_definition': 'Actual EOT-delimited content tokens excluding SOT/EOT on identical valid samples'}`.
Full cycle(s):`{'count': 500, 'median': 2.1261794567108154, 'p95': 2.2681718826293946, 'p99': 2.343686690330505, 'max': 27.205604553222656}`; slowest-rank data_wait(s):`{'count': 500, 'median': 0.0007134638726711273, 'p95': 0.0009244579821825024, 'p99': 0.0012596236914396285, 'max': 23.104105792939663}`.
OOM kills:0; true training I/O errors:0; Pod anomaly:false. GPU peak:`{'0': 27.759406089782715, '1': 27.759406089782715, '2': 27.759406089782715, '3': 27.759406089782715}` GiB.
First5 gate passed before update6; all512000 actual IDs,F/Dall token hashes and selected-index digests checked. Arms with frozen lowest-view sampling also match its text/token/K/index trajectory exactly to the declared reference.
Strict bare/native inference only:normalized native image embedding @ normalized native full-caption text embedding.T. No masks/gates/local-view inference/rerank/ensemble/TTA.
Common0 SHA256:`54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`.
Checkpoint SHA256:`937039b2342ac6ad581f2b1a08a55130ba85b4b063b2b4d72878848b6964fb2c`.
Bare SHA256:`ebedbfc489ab72ebd17bf9024909d1564999657a5deef76475d314cdf610f67c`.
Checkpoints/bare/raw logs stay local. RUNTIME_STATS.json records raw paths,bytes,SHA256 and UTC containment windows.
This one arm is classified after complete native evaluation/review. No arm-specific tuning or early metric stop.

Actual sampling: median valid K=3.0; mean K/m=0.4596849174826309; pooled K/m=0.42397591419195474.

## Inclusion ablation comparison

Classification: `TRADEOFF`.
Only inclusion_max changes1 ->0. INC0 excludes the objective term, bypasses its ramp, and computes raw inclusion violation without autograd. Raw inc telemetry is not an applied loss.
The first5 gate and complete512000-record proof match Anchor IDs,F/Dall/D3 strings/tokens,K,selected indices and LR exactly.

| Model | Inclusion | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I |
|---|---|---:|---:|---:|---:|---:|---:|
| Anchor | ramp200/max1 | 71.063600 | 74.954666 | 84.025001 | 65.227000 | 90.900000 | 89.700000 |
| INC0 | off | 71.158309 | 74.978516 | 83.995003 | 65.428000 | 91.100000 | 89.400000 |

Observed natural mean coverage hierarchy F>Dall>D3: True.
Keep ratios: `{'F': 0.7987556457519531, 'Dall': 0.7890561425685882, 'D3': 0.7468616938591004}`; deltas vs Anchor: `{'F': -0.05391109466552735, 'Dall': -0.05523339748382572, 'D3': -0.045201323032379115}`.
Hard-violation deltas(pp): `{'Dall_F_hard_violation': 1.7034410722553728, 'D3_Dall_hard_violation': 2.444830559194088}`; IoU/telemetry changes: `{'inc': 0.008562093023210765, 'inc_weight': -1.0, 'F_Dall_mask_iou': -0.04490838289260868, 'Dall_D3_mask_iou': -0.05016792297363282, 'Dall_F_hard_violation': 0.01703441072255373, 'D3_Dall_hard_violation': 0.024448305591940882}`.
Retrieval deltas(pp): `{'Score5': 0.09470959088035613, 'J_long3': 0.023849318133926545, 'J_long': -0.02999818801879428, 'Short4': 0.2009999999999934, 'Urban_I2T': 0.19999999999998863, 'Urban_T2I': -0.29999999999999716}`.
Raw gradient changes: `{'weighted_D3_Dall_ratio': 0.056691738513681256, 'raw_norms': {'F': 0.8876228332519531, 'Dall': 0.7453446388244629, 'D3': 5.537978172302246}}`.
Coverage ordering of means does not guarantee per-sample nested masks; hard violation separately measures outside-parent coordinates.
If retrieval improves with looser hierarchy, that is consistent with inclusion constraining specialization; this500-step single-seed ablation does not by itself establish that mechanism.
Classification thresholds were frozen in SEARCH_PLAN.json before launch; no schedule/weight/other-arm tuning.
Every five-set directional R@1/5/10 delta is above and in RESULTS.json. Checkpoint/bare/raw logs stay local with path/size/SHA/time ranges in RUNTIME_STATS.json.
Stopped at exactly500. No full,inc0.5,schedule,sparsity,K,alignment or other experiment launched.
