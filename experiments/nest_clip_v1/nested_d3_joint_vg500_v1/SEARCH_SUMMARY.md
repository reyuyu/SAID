# Joint-VG: isolated Nested D3 local500 search arm

Completed exactly500 fresh-common0 optimizer updates, horizon4868. Only changed axis:`joint_vg`.
Alignment:[1.35, 1.35, 0.3]; Joint-VG regularizer:Omega_F/3 + two V/G edges; old child coefficients unused; mode:nested_detail_d3.
Other construction,optimizer/LR,workers8,batch256/rank,preprocess and native protocol frozen. Old inclusion/ramp disabled; legacy config inclusion_max1 is unused. Joint V/G gradients train both endpoints. No combination/full run.
Local-only:`/root/said_s02_stage500/ShareGPT4V`; missing/symlink/escape fails, NFS fallback forbidden. /root is disposable overlay; NFS originals retained.

| Dataset | I2T R@1 / R@5 / R@10 (%) | T2I R@1 / R@5 / R@10 (%) |
|---|---|---|
| COCO | 60.140000 / 82.340000 / 89.000000 | 41.268000 / 67.100000 / 76.732000 |
| Urban-1k | 90.600002 / 98.500007 / 99.800003 | 88.900006 / 98.800004 / 99.400002 |
| Flickr30k-test1k | 86.700000 / 97.400000 / 98.900000 | 71.760000 / 91.320000 / 95.200000 |
| DOCCI | 77.840000 / 95.160000 / 98.040000 | 77.540000 / 95.180000 / 97.760000 |
| Long-DCI | 56.800842 / 75.861615 / 81.820574 | 56.656143 / 75.808998 / 81.886346 |

| Metric | Arm | Delta vs Anchor(pp) |
|---|---:|---:|
| Score5 | 70.820499 | -0.243101 |
| J_long3 | 74.722832 | -0.231834 |
| J_long | 83.720002 | -0.304999 |
| Short4 | 64.967000 | -0.260000 |
| Urban_I2T | 90.600000 | -0.300000 |
| Urban_T2I | 88.900000 | -0.800000 |

| Dataset | Delta I2T R@1/5/10(pp) | Delta T2I R@1/5/10(pp) |
|---|---|---|
| COCO | -0.260000 / -0.280000 / -0.260000 | -0.320000 / -0.176000 / -0.200000 |
| Urban-1k | -0.300002 / -0.099999 / +0.199997 | -0.799996 / +0.299996 / -0.100005 |
| Flickr30k-test1k | -0.300000 / -0.400000 / -0.300000 | -0.160000 / -0.060000 / +0.080000 |
| DOCCI | +0.020000 / -0.180000 / +0.040000 | -0.140000 / -0.080000 / +0.040000 |
| Long-DCI | -0.223625 / +0.328861 / +0.171008 | +0.052618 / -0.276243 / -0.052618 |

| View | Combined CE | Weighted CE | Alignment share | Keep ratio |
|---|---:|---:|---:|---:|
| F | 0.110819 | 0.498684 | 18.082% | 0.896721 |
| Dall | 0.197639 | 0.889373 | 32.248% | 0.913170 |
| D3 | 1.369820 | 1.369820 | 49.669% | 0.896346 |

Raw gradient norms:`{'F': 12.664927959442139, 'Dall': 16.7771018743515, 'D3': 41.59274387359619}`.
Weighted gradient norms:`{'F': 56.99217581748964, 'Dall': 75.49695843458177, 'D3': 41.59274387359619}`; weighted D3/Dall ratio:`0.550919`.
Frozen gradient protocol:8 first seed0 epoch0 global1024 batches at fixed step500; production backbone group; raw directional-summed CE,all-reduce/4; no optimizer updates. Norm magnitudes are not a signed gradient decomposition.
Mask hierarchy:`{'inc': 0.02334210166707635, 'inc_weight': 0.0, 'F_Dall_mask_iou': 0.9591544640064239, 'Dall_D3_mask_iou': 0.9428885972499848, 'Dall_F_hard_violation': 0.02693588115274906, 'D3_Dall_hard_violation': 0.018206856735050677}`. V/G jointly optimize both endpoints; hard violations remain telemetry.
Sampling:`{'records': 512000, 'valid_records': 511936, 'K_histogram_all': {'2': 3924, '3': 507507, '1': 505, '0': 64}, 'K_histogram_valid': {'2': 3924, '3': 507507, '1': 505}, 'K_histogram_by_m': {'8': {'3': 77266}, '5': {'3': 92074}, '6': {'3': 92953}, '9': {'3': 68110}, '4': {'3': 37167}, '10': {'3': 41151}, '12': {'3': 4210}, '7': {'3': 76923}, '11': {'3': 16257}, '3': {'2': 3924}, '13': {'3': 922}, '14': {'3': 232}, '1': {'1': 173}, '2': {'1': 332}, '24': {'3': 4}, '17': {'3': 18}, '15': {'3': 84}, '26': {'3': 5}, '16': {'3': 45}, '95': {'3': 1}, '23': {'3': 3}, '20': {'3': 4}, '21': {'3': 7}, '47': {'3': 2}, '18': {'3': 11}, '29': {'3': 1}, '41': {'3': 1}, '25': {'3': 4}, '51': {'3': 1}, '33': {'3': 2}, '34': {'3': 2}, '22': {'3': 4}, '19': {'3': 12}, '31': {'3': 1}, '30': {'3': 3}, '28': {'3': 3}, '39': {'3': 1}, '101': {'3': 1}, '92': {'3': 2}, '100': {'3': 1}, '32': {'3': 1}, '27': {'3': 3}, '82': {'3': 1}, '105': {'3': 2}, '44': {'3': 1}, '54': {'3': 1}, '79': {'3': 1}, '86': {'3': 1}, '89': {'3': 1}, '56': {'3': 1}, '40': {'3': 1}, '37': {'3': 1}, '99': {'3': 1}, '87': {'3': 1}, '77': {'3': 1}, '74': {'3': 1}}, 'm_histogram': {'3': 3924, '4': 37167, '5': 92074, '6': 92953, '7': 76923, '8': 77266, '9': 68110, '10': 41151, '11': 16257, '12': 4210, '13': 922, '14': 232, '1': 173, '2': 332, '24': 4, '17': 18, '15': 84, '0': 64, '26': 5, '16': 45, '95': 1, '23': 3, '20': 4, '21': 7, '47': 2, '18': 11, '29': 1, '41': 1, '25': 4, '51': 1, '33': 2, '34': 2, '22': 4, '19': 12, '31': 1, '30': 3, '28': 3, '39': 1, '101': 1, '92': 2, '100': 1, '32': 1, '27': 3, '82': 1, '105': 2, '44': 1, '54': 1, '79': 1, '86': 1, '89': 1, '56': 1, '40': 1, '37': 1, '99': 1, '87': 1, '77': 1, '74': 1}, 'mean_K_valid': 2.9903620765095638, 'mean_K_all': 2.98998828125, 'Dall_mean_sentences': 7.0531414083010375, 'strict_subset_ratio': 0.999537109375, 'mean_effective_tokens': {'F': 171.46915239404925, 'Dall': 153.89054100512564, 'D3': 66.38841183272909}, 'mean_content_tokens': {'F': 169.46915239404925, 'Dall': 151.89054100512564, 'D3': 64.38841183272909}, 'mean_lowest_Dall_content_token_coverage': 0.45601314895918665, 'pooled_lowest_Dall_content_token_coverage': 0.42391324309363193, 'coverage_definition': 'Actual EOT-delimited content tokens excluding SOT/EOT on identical valid samples'}`.
Full cycle(s):`{'count': 500, 'median': 2.1295167207717896, 'p95': 2.266711413860321, 'p99': 2.339231219291687, 'max': 25.418014526367188}`; slowest-rank data_wait(s):`{'count': 500, 'median': 0.0006997585296630859, 'p95': 0.0008880708366632436, 'p99': 0.0014804589748382564, 'max': 21.132680766284466}`.
OOM kills:0; true training I/O errors:0; Pod anomaly:false. GPU peak:`{'0': 27.759894371032715, '1': 27.759894371032715, '2': 27.759894371032715, '3': 27.759894371032715}` GiB.
First5 gate passed before update6; all512000 actual IDs,F/Dall token hashes and selected-index digests checked. Arms with frozen lowest-view sampling also match its text/token/K/index trajectory exactly to the declared reference.
Strict bare/native inference only:normalized native image embedding @ normalized native full-caption text embedding.T. No masks/gates/local-view inference/rerank/ensemble/TTA.
Common0 SHA256:`54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`.
Checkpoint SHA256:`c23f82d3ea16ab11689d4bc0f48f691204f83cf06c6316b334269b43e2a9d875`.
Bare SHA256:`cc0291bbdd9d4886730118e3f56c06ef898e3fe898f6c339d08eb3cfdd38da56`.
Checkpoints/bare/raw logs stay local. RUNTIME_STATS.json records raw paths,bytes,SHA256 and UTC containment windows.
One frozen Joint-VG arm; classification after complete native evaluation and feasible-region review. No arm-specific tuning or early metric stop.

Actual sampling: median valid K=3.0; mean K/m=0.4596849174826309; pooled K/m=0.42397591419195474.

## Joint V/G frozen feasible-region comparison

Classification: `JOINT_VG_NEGATIVE`.
Loss: alignment + Omega_F/3 + edge(Dall,F) + edge(D3,Dall). No stop-gradient,child global sparsity,old inclusion,ramp,beta or online region changes.
Frozen regions: `{'Dall_F': {'eps_v': 0.013686737418174746, 'gamma_low': 0.011261347867548467, 'gamma_high': 0.024891537427902226}, 'D3_Dall': {'eps_v': 0.020522184297442438, 'gamma_low': 0.04266522899270058, 'gamma_high': 0.08650477081537247}}`; eps_v=P80(V),gamma_low=P20(G),gamma_high=P80(G) from fixed16384 Anchor training samples.

| Model | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I |
|---|---:|---:|---:|---:|---:|---:|
| Anchor | 71.063600 | 74.954666 | 84.025001 | 65.227000 | 90.900000 | 89.700000 |
| INC0 | 71.158309 | 74.978516 | 83.995003 | 65.428000 | 91.100000 | 89.400000 |
| Coupled v1 | 71.072953 | 74.906255 | 84.005002 | 65.323000 | 91.300000 | 89.500000 |
| BBNS | 70.964966 | 74.766276 | 83.920001 | 65.263000 | 91.000000 | 89.700000 |
| Joint-VG | 70.820499 | 74.722832 | 83.720002 | 64.967000 | 90.600000 | 88.900000 |

| Reference | Delta Score5 | Delta J_long3 | Delta J_long | Delta Short4 | Delta Urban I2T | Delta Urban T2I |
|---|---:|---:|---:|---:|---:|---:|
| Anchor | -0.243101 | -0.231834 | -0.304999 | -0.260000 | -0.300000 | -0.800000 |
| INC0 | -0.337810 | -0.255684 | -0.275001 | -0.461000 | -0.500000 | -0.500000 |
| Coupled v1 | -0.252454 | -0.183423 | -0.285000 | -0.356000 | -0.700000 | -0.600000 |
| BBNS | -0.144466 | -0.043444 | -0.199999 | -0.296000 | -0.400000 | -0.800000 |

All native recalls vs Anchor (pp):

| Dataset | Delta I2T R@1/5/10 | Delta T2I R@1/5/10 |
|---|---|---|
| COCO | -0.260000 / -0.280000 / -0.260000 | -0.320000 / -0.176000 / -0.200000 |
| Urban-1k | -0.300002 / -0.099999 / +0.199997 | -0.799996 / +0.299996 / -0.100005 |
| Flickr30k-test1k | -0.300000 / -0.400000 / -0.300000 | -0.160000 / -0.060000 / +0.080000 |
| DOCCI | +0.020000 / -0.180000 / +0.040000 | -0.140000 / -0.080000 / +0.040000 |
| Long-DCI | -0.223625 / +0.328861 / +0.171008 | +0.052618 / -0.276243 / -0.052618 |

All native recalls vs INC0 (pp):

| Dataset | Delta I2T R@1/5/10 | Delta T2I R@1/5/10 |
|---|---|---|
| COCO | -0.540000 / -0.600000 / -0.240000 | -0.244000 / -0.228000 / -0.184000 |
| Urban-1k | -0.500005 / -0.299996 / +0.099999 | -0.500000 / +0.099999 / -0.100005 |
| Flickr30k-test1k | -0.700000 / -0.200000 / -0.200000 | -0.360000 / -0.180000 / -0.040000 |
| DOCCI | -0.020000 / -0.240000 / +0.000000 | -0.080000 / -0.040000 / -0.100000 |
| Long-DCI | -0.328861 / +0.118390 / -0.105235 | -0.105235 / -0.263089 / -0.157853 |

All native recalls vs Coupled v1 (pp):

| Dataset | Delta I2T R@1/5/10 | Delta T2I R@1/5/10 |
|---|---|---|
| COCO | -0.480000 / -0.340000 / -0.360000 | -0.204000 / -0.188000 / -0.164000 |
| Urban-1k | -0.700003 / -0.199997 / +0.199997 | -0.599998 / +0.199997 / -0.100005 |
| Flickr30k-test1k | -0.700000 / +0.100000 / -0.200000 | -0.040000 / -0.060000 / +0.100000 |
| DOCCI | +0.080000 / -0.400000 / -0.080000 | +0.080000 / +0.040000 / +0.040000 |
| Long-DCI | +0.013154 / +0.513023 / +0.263089 | +0.026309 / -0.171008 / +0.131544 |

All native recalls vs BBNS (pp):

| Dataset | Delta I2T R@1/5/10 | Delta T2I R@1/5/10 |
|---|---|---|
| COCO | -0.600000 / -0.140000 / -0.100000 | -0.144000 / -0.144000 / -0.128000 |
| Urban-1k | -0.400001 / -0.199997 / +0.099999 | -0.799996 / +0.099999 / +0.000000 |
| Flickr30k-test1k | -0.400000 / -0.200000 / -0.200000 | -0.040000 / +0.040000 / +0.080000 |
| DOCCI | +0.320000 / -0.220000 / -0.020000 | +0.080000 / +0.100000 / +0.100000 |
| Long-DCI | +0.236780 / +0.394633 / +0.302552 | +0.302552 / -0.013154 / +0.144699 |

Last50 V/G telemetry: `{'Dall_F_V': 0.0376049292832613, 'Dall_F_G': 0.011752701969817281, 'Dall_F_V_penalty': 0.001100475874745399, 'Dall_F_G_lower_penalty': 1.0113034618370875e-05, 'Dall_F_G_upper_penalty': 7.626843527412941e-06, 'Dall_F_V_legal': 0.13451456353068353, 'Dall_F_G_legal': 0.3501157170534134, 'Dall_F_zero_edge': 0.02291187461465597, 'Dall_F_edge_loss': 0.0011182157621078658, 'D3_Dall_V': 0.022446436770260335, 'D3_Dall_G': 0.046257085781544444, 'D3_Dall_V_penalty': 0.00015089550491485858, 'D3_Dall_G_lower_penalty': 0.00024764381756540386, 'D3_Dall_G_upper_penalty': 0.00017238691048191869, 'D3_Dall_V_legal': 0.5531381165981293, 'D3_Dall_G_legal': 0.3328406226634979, 'D3_Dall_zero_edge': 0.16198928594589235, 'D3_Dall_edge_loss': 0.0005709262331947684, 'F_mean_soft_support': 0.7762813866138458, 'Dall_mean_soft_support': 0.7967270398139954, 'D3_mean_soft_support': 0.7773218166828155, 'Omega_F': 0.8967208099365235, 'total_vg_edges': 0.0016891419794410466, 'total_nested_regularizer': 0.3005960911512375}`.
Feasible-region behavior:False; collapse:False; all-near-one:False; D3-over-sparse:False.
Keep/IoU/violation changes: `{'Anchor': {'last50': {'inc': 0.015234032720327377, 'inc_weight': -1.0, 'F_Dall_mask_iou': -0.006709409952163736, 'Dall_D3_mask_iou': 0.0428358852863312, 'Dall_F_hard_violation': 0.01657375942915678, 'D3_Dall_hard_violation': 0.0013956777378916728}, 'keep_ratio': {'F': 0.044054069519043004, 'Dall': 0.0688800609111786, 'D3': 0.10428269028663628}}, 'INC0': {'last50': {'inc': 0.006671939697116613, 'inc_weight': 0.0, 'F_Dall_mask_iou': 0.03819897294044494, 'Dall_D3_mask_iou': 0.09300380825996402, 'Dall_F_hard_violation': -0.0004606512933969485, 'D3_Dall_hard_violation': -0.02305262785404921}, 'keep_ratio': {'F': 0.09796516418457035, 'Dall': 0.12411345839500432, 'D3': 0.1494840133190154}}, 'Coupled v1': {'last50': {'inc': 0.011375294160097837, 'inc_weight': -1.0, 'F_Dall_mask_iou': 0.034675803184509246, 'Dall_D3_mask_iou': 0.1048627245426178, 'Dall_F_hard_violation': 0.008855693265795706, 'D3_Dall_hard_violation': -0.008276730328798294}, 'keep_ratio': {'F': 0.06764163970947268, 'Dall': 0.11119424819946289, 'D3': 0.17598203539848323}}, 'BBNS': {'last50': {'inc': 0.004828358869999647, 'inc_weight': 0.0, 'F_Dall_mask_iou': -0.0012006998062134011, 'Dall_D3_mask_iou': 0.02032317042350773, 'Dall_F_hard_violation': 0.007229752205312252, 'D3_Dall_hard_violation': -0.0047154553607106214}, 'keep_ratio': {'F': 0.021545524597167964, 'Dall': 0.03397520065307624, 'D3': 0.04118175864219664}}}`.
Weighted D3/Dall gradient ratio:0.5509194639892198; deltas:`{'Anchor': 0.04600244989648972, 'INC0': -0.010689288617191539, 'Coupled v1': 0.027199810871689656, 'BBNS': 0.05580937127365021}`.
At exact equality PyTorch ReLU has zero subgradient: lower penalty is positive, but its endpoint gradients are zero. Positive gamma_low discourages collapse away from this kink; it does not mathematically guarantee escape from exact collapse.
Diagnostic steps1/100/200/500 and last50 are in VG_AUDIT.json and TRAINING_DIAGNOSTICS.json.
All512000 sample/text/token/K/index/LR trajectories match Anchor. Final500 full checkpoint and strict native bare immutable.
Stopped500; no full,quantile/lambda/beta/K/weight search,SG version or other experiment. Binary/raw paths,size,SHA,time ranges in RUNTIME_STATS.json.
