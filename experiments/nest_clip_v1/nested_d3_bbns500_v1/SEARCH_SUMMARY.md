# BBNS: isolated Nested D3 local500 search arm

Completed exactly500 fresh-common0 optimizer updates, horizon4868. Only changed axis:`support_band`.
Alignment:[1.35, 1.35, 0.3]; BBNS regularizer:Omega_F/3 + two support-band edges; old child coefficients unused; mode:nested_detail_d3.
Other construction,optimizer/LR,workers8,batch256/rank,preprocess and native protocol frozen. Old inclusion is disabled in BBNS; the legacy config value1 is unused. No old ramp/autograd loss. No combination/full run.
Local-only:`/root/said_s02_stage500/ShareGPT4V`; missing/symlink/escape fails, NFS fallback forbidden. /root is disposable overlay; NFS originals retained.

| Dataset | I2T R@1 / R@5 / R@10 (%) | T2I R@1 / R@5 / R@10 (%) |
|---|---|---|
| COCO | 60.740000 / 82.480000 / 89.100000 | 41.412000 / 67.244000 / 76.860000 |
| Urban-1k | 91.000003 / 98.700005 / 99.700004 | 89.700001 / 98.700005 / 99.400002 |
| Flickr30k-test1k | 87.100000 / 97.600000 / 99.100000 | 71.800000 / 91.280000 / 95.120000 |
| DOCCI | 77.520000 / 95.380000 / 98.060000 | 77.460000 / 95.080000 / 97.660000 |
| Long-DCI | 56.564062 / 75.466982 / 81.518022 | 56.353591 / 75.822152 / 81.741647 |

| Metric | Arm | Delta vs Anchor(pp) |
|---|---:|---:|
| Score5 | 70.964966 | -0.098634 |
| J_long3 | 74.766276 | -0.188390 |
| J_long | 83.920001 | -0.105000 |
| Short4 | 65.263000 | +0.036000 |
| Urban_I2T | 91.000000 | +0.100000 |
| Urban_T2I | 89.700000 | +0.000000 |

| Dataset | Delta I2T R@1/5/10(pp) | Delta T2I R@1/5/10(pp) |
|---|---|---|
| COCO | +0.340000 / -0.140000 / -0.160000 | -0.176000 / -0.032000 / -0.072000 |
| Urban-1k | +0.099999 / +0.099999 / +0.099999 | +0.000000 / +0.199997 / -0.100005 |
| Flickr30k-test1k | +0.100000 / -0.200000 / -0.100000 | -0.120000 / -0.100000 / +0.000000 |
| DOCCI | -0.300000 / +0.040000 / +0.060000 | -0.220000 / -0.180000 / -0.060000 |
| Long-DCI | -0.460405 / -0.065772 / -0.131544 | -0.249934 / -0.263089 / -0.197316 |

| View | Combined CE | Weighted CE | Alignment share | Keep ratio |
|---|---:|---:|---:|---:|
| F | 0.112449 | 0.506019 | 18.129% | 0.875175 |
| Dall | 0.204103 | 0.918462 | 32.905% | 0.879194 |
| D3 | 1.366764 | 1.366764 | 48.966% | 0.855164 |

Raw gradient norms:`{'F': 11.278983235359192, 'Dall': 15.159113764762878, 'D3': 33.77443599700928}`.
Weighted gradient norms:`{'F': 50.75542455911637, 'Dall': 68.21601194143297, 'D3': 33.77443599700928}`; weighted D3/Dall ratio:`0.495110`.
Frozen gradient protocol:8 first seed0 epoch0 global1024 batches at fixed step500; production backbone group; raw directional-summed CE,all-reduce/4; no optimizer updates. Norm magnitudes are not a signed gradient decomposition.
Mask hierarchy:`{'inc': 0.018513742797076703, 'inc_weight': 0.0, 'F_Dall_mask_iou': 0.9603551638126373, 'Dall_D3_mask_iou': 0.922565426826477, 'Dall_F_hard_violation': 0.01970612894743681, 'D3_Dall_hard_violation': 0.0229223120957613}`. BBNS uses detached child coverage and detached parent refinement; hard violations remain telemetry.
Sampling:`{'records': 512000, 'valid_records': 511936, 'K_histogram_all': {'2': 3924, '3': 507507, '1': 505, '0': 64}, 'K_histogram_valid': {'2': 3924, '3': 507507, '1': 505}, 'K_histogram_by_m': {'8': {'3': 77266}, '5': {'3': 92074}, '6': {'3': 92953}, '9': {'3': 68110}, '4': {'3': 37167}, '10': {'3': 41151}, '12': {'3': 4210}, '7': {'3': 76923}, '11': {'3': 16257}, '3': {'2': 3924}, '13': {'3': 922}, '14': {'3': 232}, '1': {'1': 173}, '2': {'1': 332}, '24': {'3': 4}, '17': {'3': 18}, '15': {'3': 84}, '26': {'3': 5}, '16': {'3': 45}, '95': {'3': 1}, '23': {'3': 3}, '20': {'3': 4}, '21': {'3': 7}, '47': {'3': 2}, '18': {'3': 11}, '29': {'3': 1}, '41': {'3': 1}, '25': {'3': 4}, '51': {'3': 1}, '33': {'3': 2}, '34': {'3': 2}, '22': {'3': 4}, '19': {'3': 12}, '31': {'3': 1}, '30': {'3': 3}, '28': {'3': 3}, '39': {'3': 1}, '101': {'3': 1}, '92': {'3': 2}, '100': {'3': 1}, '32': {'3': 1}, '27': {'3': 3}, '82': {'3': 1}, '105': {'3': 2}, '44': {'3': 1}, '54': {'3': 1}, '79': {'3': 1}, '86': {'3': 1}, '89': {'3': 1}, '56': {'3': 1}, '40': {'3': 1}, '37': {'3': 1}, '99': {'3': 1}, '87': {'3': 1}, '77': {'3': 1}, '74': {'3': 1}}, 'm_histogram': {'3': 3924, '4': 37167, '5': 92074, '6': 92953, '7': 76923, '8': 77266, '9': 68110, '10': 41151, '11': 16257, '12': 4210, '13': 922, '14': 232, '1': 173, '2': 332, '24': 4, '17': 18, '15': 84, '0': 64, '26': 5, '16': 45, '95': 1, '23': 3, '20': 4, '21': 7, '47': 2, '18': 11, '29': 1, '41': 1, '25': 4, '51': 1, '33': 2, '34': 2, '22': 4, '19': 12, '31': 1, '30': 3, '28': 3, '39': 1, '101': 1, '92': 2, '100': 1, '32': 1, '27': 3, '82': 1, '105': 2, '44': 1, '54': 1, '79': 1, '86': 1, '89': 1, '56': 1, '40': 1, '37': 1, '99': 1, '87': 1, '77': 1, '74': 1}, 'mean_K_valid': 2.9903620765095638, 'mean_K_all': 2.98998828125, 'Dall_mean_sentences': 7.0531414083010375, 'strict_subset_ratio': 0.999537109375, 'mean_effective_tokens': {'F': 171.46915239404925, 'Dall': 153.89054100512564, 'D3': 66.38841183272909}, 'mean_content_tokens': {'F': 169.46915239404925, 'Dall': 151.89054100512564, 'D3': 64.38841183272909}, 'mean_lowest_Dall_content_token_coverage': 0.45601314895918665, 'pooled_lowest_Dall_content_token_coverage': 0.42391324309363193, 'coverage_definition': 'Actual EOT-delimited content tokens excluding SOT/EOT on identical valid samples'}`.
Full cycle(s):`{'count': 500, 'median': 2.1173900365829468, 'p95': 2.2842223048210144, 'p99': 2.3719730234146117, 'max': 26.48662567138672}`; slowest-rank data_wait(s):`{'count': 500, 'median': 0.0006784498691558838, 'p95': 0.0009129699319601059, 'p99': 0.0016035408526659002, 'max': 22.855799332261086}`.
OOM kills:0; true training I/O errors:0; Pod anomaly:false. GPU peak:`{'0': 27.759894371032715, '1': 27.759894371032715, '2': 27.759894371032715, '3': 27.759894371032715}` GiB.
First5 gate passed before update6; all512000 actual IDs,F/Dall token hashes and selected-index digests checked. Arms with frozen lowest-view sampling also match its text/token/K/index trajectory exactly to the declared reference.
Strict bare/native inference only:normalized native image embedding @ normalized native full-caption text embedding.T. No masks/gates/local-view inference/rerank/ensemble/TTA.
Common0 SHA256:`54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`.
Checkpoint SHA256:`d94823d1b88ec70085fbf6ea634accdaa13b4e5130c5dda866a6f594f4af951e`.
Bare SHA256:`49c6d0c23166ba4f9533a98ccd6ee52db63742bbecad0faace1d9e3cfa7562f5`.
Checkpoints/bare/raw logs stay local. RUNTIME_STATS.json records raw paths,bytes,SHA256 and UTC containment windows.
One frozen BBNS arm, classified after complete native evaluation and support review. No arm-specific tuning or early metric stop.

Actual sampling: median valid K=3.0; mean K/m=0.4596849174826309; pooled K/m=0.42397591419195474.

## BBNS frozen support band comparison

Classification: `BBNS_TRADEOFF`.
Formula: alignment + Omega_F/3 + edge(Dall,F) + edge(D3,Dall); each edge cover+lower+upper squared hinges. No child global sparse,old inclusion,ramp,beta or online band adjustment.
Before training, independently per edge: `{'Dall_F': {'kappa': 0.6466294646263122, 'tau_low': 0.6404582858085632, 'tau_high': 0.6976943612098694}, 'D3_Dall': {'kappa': 0.6413809061050415, 'tau_low': 0.6085701584815979, 'tau_high': 0.6629944801330566}}`; kappa=P20(C),tau_low=P20(R),tau_high=P80(R) from fixed16384 successful Anchor samples only.

| Model | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I |
|---|---:|---:|---:|---:|---:|---:|
| Anchor | 71.063600 | 74.954666 | 84.025001 | 65.227000 | 90.900000 | 89.700000 |
| INC0 | 71.158309 | 74.978516 | 83.995003 | 65.428000 | 91.100000 | 89.400000 |
| Coupled v1 | 71.072953 | 74.906255 | 84.005002 | 65.323000 | 91.300000 | 89.500000 |
| BBNS | 70.964966 | 74.766276 | 83.920001 | 65.263000 | 91.000000 | 89.700000 |

| Reference | Delta Score5 | Delta J_long3 | Delta J_long | Delta Short4 | Delta Urban I2T | Delta Urban T2I |
|---|---:|---:|---:|---:|---:|---:|
| Anchor | -0.098634 | -0.188390 | -0.105000 | +0.036000 | +0.100000 | +0.000000 |
| INC0 | -0.193344 | -0.212239 | -0.075002 | -0.165000 | -0.100000 | +0.300000 |
| Coupled v1 | -0.107987 | -0.139979 | -0.085001 | -0.060000 | -0.300000 | +0.200000 |

All native recalls vs Anchor (pp):

| Dataset | Delta I2T R@1/5/10 | Delta T2I R@1/5/10 |
|---|---|---|
| COCO | +0.340000 / -0.140000 / -0.160000 | -0.176000 / -0.032000 / -0.072000 |
| Urban-1k | +0.099999 / +0.099999 / +0.099999 | +0.000000 / +0.199997 / -0.100005 |
| Flickr30k-test1k | +0.100000 / -0.200000 / -0.100000 | -0.120000 / -0.100000 / +0.000000 |
| DOCCI | -0.300000 / +0.040000 / +0.060000 | -0.220000 / -0.180000 / -0.060000 |
| Long-DCI | -0.460405 / -0.065772 / -0.131544 | -0.249934 / -0.263089 / -0.197316 |

All native recalls vs INC0 (pp):

| Dataset | Delta I2T R@1/5/10 | Delta T2I R@1/5/10 |
|---|---|---|
| COCO | +0.060000 / -0.460000 / -0.140000 | -0.100000 / -0.084000 / -0.056000 |
| Urban-1k | -0.100005 / -0.099999 / +0.000000 | +0.299996 / +0.000000 / -0.100005 |
| Flickr30k-test1k | -0.300000 / +0.000000 / +0.000000 | -0.320000 / -0.220000 / -0.120000 |
| DOCCI | -0.340000 / -0.020000 / +0.020000 | -0.160000 / -0.140000 / -0.200000 |
| Long-DCI | -0.565641 / -0.276243 / -0.407787 | -0.407787 / -0.249934 / -0.302552 |

All native recalls vs Coupled v1 (pp):

| Dataset | Delta I2T R@1/5/10 | Delta T2I R@1/5/10 |
|---|---|---|
| COCO | +0.120000 / -0.200000 / -0.260000 | -0.060000 / -0.044000 / -0.036000 |
| Urban-1k | -0.300002 / +0.000000 / +0.099999 | +0.199997 / +0.099999 / -0.100005 |
| Flickr30k-test1k | -0.300000 / +0.300000 / +0.000000 | +0.000000 / -0.100000 / +0.020000 |
| DOCCI | -0.240000 / -0.180000 / -0.060000 | +0.000000 / -0.060000 / -0.060000 |
| Long-DCI | -0.223625 / +0.118390 / -0.039463 | -0.276243 / -0.157853 / -0.013154 |

Last50 support/hit/loss telemetry: `{'Dall_F_coverage': 0.7728584778308868, 'Dall_F_relative_support': 0.7780459678173065, 'Dall_F_cover_loss': 3.5999354963678343e-09, 'Dall_F_lower_band_loss': 0.0, 'Dall_F_upper_band_loss': 0.007469153087586164, 'Dall_F_zero_refine_band': 0.005761814210563898, 'Dall_F_zero_edge_band': 0.005742282960563898, 'Dall_F_percent_inside_zero_loss_band': 0.5742282956838608, 'Dall_F_percent_inside_refinement_band': 0.5761814206838608, 'Dall_F_edge_loss': 0.007469156691804528, 'D3_Dall_coverage': 0.7754539096355438, 'D3_Dall_relative_support': 0.7531013834476471, 'D3_Dall_cover_loss': 0.0, 'D3_Dall_lower_band_loss': 0.0, 'D3_Dall_upper_band_loss': 0.009267418403178454, 'D3_Dall_zero_refine_band': 0.002753944434225559, 'D3_Dall_zero_edge_band': 0.002753944434225559, 'D3_Dall_percent_inside_zero_loss_band': 0.2753944432735443, 'D3_Dall_percent_inside_refinement_band': 0.2753944432735443, 'D3_Dall_edge_loss': 0.009267418403178454, 'Omega_F': 0.8751752853393555, 'total_band_edges': 0.01673657501116395, 'total_nested_regularizer': 0.3084616780281067}`.
Mean support inside bounds:False; many samples zero band:False; expected behavior:False.
Keep/IoU/violation changes vs references: `{'Anchor': {'last50': {'inc': 0.01040567385032773, 'inc_weight': -1.0, 'F_Dall_mask_iou': -0.005508710145950335, 'Dall_D3_mask_iou': 0.02251271486282347, 'Dall_F_hard_violation': 0.009344007223844528, 'D3_Dall_hard_violation': 0.006111133098602294}, 'keep_ratio': {'F': 0.02250854492187504, 'Dall': 0.034904860258102355, 'D3': 0.06310093164443964}}, 'INC0': {'last50': {'inc': 0.0018435808271169661, 'inc_weight': 0.0, 'F_Dall_mask_iou': 0.03939967274665834, 'Dall_D3_mask_iou': 0.07268063783645629, 'Dall_F_hard_violation': -0.007690403498709201, 'D3_Dall_hard_violation': -0.018337172493338588}, 'keep_ratio': {'F': 0.07641963958740239, 'Dall': 0.09013825774192807, 'D3': 0.10830225467681875}}, 'Coupled v1': {'last50': {'inc': 0.0065469352900981905, 'inc_weight': -1.0, 'F_Dall_mask_iou': 0.03587650299072265, 'Dall_D3_mask_iou': 0.08453955411911007, 'Dall_F_hard_violation': 0.0016259410604834539, 'D3_Dall_hard_violation': -0.003561274968087672}, 'keep_ratio': {'F': 0.046096115112304714, 'Dall': 0.07721904754638664, 'D3': 0.13480027675628659}}}`.
Weighted D3/Dall gradient ratio:0.49511009271556955; deltas:`{'Anchor': -0.009806921377160494, 'INC0': -0.06649865989084175, 'Coupled v1': -0.028609560401960554}`.
Gradient routing manual tensor audit passed; GRADIENT_ROUTING_AUDIT.json contains separate loss/gradient cases.
Anchor audit stats and exact pretraining thresholds in ANCHOR_SUPPORT_AUDIT.md/JSON; fixed sample IDs in ANCHOR_SUPPORT_COHORT.json.
Operational classification thresholds were declared before training in SEARCH_PLAN.json, never selected against retrieval.
All512000 sample/text/token/K/index/LR trajectories match Anchor. Final500 full checkpoint and strict native bare immutable.
Stopped500; no full,new bands/beta/K/weights/other experiment. Binary/raw paths,size,SHA,time ranges in RUNTIME_STATS.json.
