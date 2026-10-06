# Nested D3 alignment-weight-only experiment: 1.35/1.35/0.30

Status:`D3_BALANCED_STRONG_POSITIVE`; updates500/500, horizon4868; no automatic follow-up.

The balanced weights pass every declared strong-positive gate: Score5 71.063600, J_long3 74.954666, and Urban T2I 89.700. All 512000 records match the equal-weight D3 sample/text/token/selected-index trajectory, with a fresh common step0 and no NFS training-image fallback. The final independent review passed; 91 related CPU tests passed before launch.

The last50 D3 alignment share falls from 78.355% to 49.674% (-28.681pp). Raw D3 CE rises from 1.176043 to 1.411348, and raw gradient norm remains high at 32.547; the reduced weight brings its actual scaled norm below both parent views. Weighted D3/Dall is 0.504917 versus 2.104920 for equal weights and 0.458758 for the low-dose Ds reference. No fixed diagnostic batch has weighted D3 exceeding weighted F+Dall (0/8). These are per-view norm comparisons, not a signed decomposition of the total gradient.

Relative to low-dose Nested Ds, Score5 improves 0.313281pp, J_long3 0.626801pp, and Urban T2I 1.200pp, while Short4 declines 0.157pp. Long-DCI I2T improves 1.328598pp but T2I declines 0.407787pp; the improvement is not uniform across every direction. The frozen soft inclusion remains approximate: last50 hard-mask violations are 1.036% for Dall/F and 1.681% for D3/Dall, both lower than equal-weight D3.

Training stops at exactly500. Steady full-cycle median/p95/p99/max is 2.1212/2.2949/2.3710/2.4245s; overall >3s and >10s counts are both2, including startup/gate overhead. No training I/O error, actual oom_kill, or Pod/supervisor anomaly was observed. Peak cgroup usage includes file cache and must not be interpreted as process RSS. Checkpoint, bare export, local images, raw sampling evidence and raw logs remain local; RUNTIME_STATS.json records paths, sizes, SHA256 and time ranges for traceability.

Only method change vs D3 equal:alignment weights[1,1,1] ->[1.35,1.35,0.30], sum3. All dataset/text/sampler/model/CE/temperature/optimizer/LR/batch/workers/preprocess/inclusion/sparsity frozen.
Fresh common0 SHA256:`54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`; no prior Nested checkpoint resumed. Local-only:`/root/said_s02_stage500/ShareGPT4V`; missing/symlink/escape fails, no NFS fallback. Disposable cache; NFS originals retained.
D3 exactly reuses ordered uniform stateless sampling, K_eff=min(3,m-1) for m>=2 and original m<=1 fallback. Added indices digest is detached CPU telemetry, no RNG draws.
Alignment10/3*(1.35*L_F+1.35*L_Dall+0.30*L_D3), each I2T+T2I CE. Chain Dall->F,D3->Dall only, detached-child,ramp200,max1. Sparsity(Omega_F+2*Omega_Dall+2*Omega_D3)/3.

| Dataset | I2T R@1 / R@5 / R@10 (%) | T2I R@1 / R@5 / R@10 (%) |
|---|---|---|
| COCO | 60.400000 / 82.620000 / 89.260000 | 41.588000 / 67.276000 / 76.932000 |
| Urban-1k | 90.900004 / 98.600006 / 99.600005 | 89.700001 / 98.500007 / 99.500006 |
| Flickr30k-test1k | 87.000000 / 97.800000 / 99.200000 | 71.920000 / 91.380000 / 95.120000 |
| DOCCI | 77.820000 / 95.340000 / 98.000000 | 77.680000 / 95.260000 / 97.720000 |
| Long-DCI | 57.024467 / 75.532755 / 81.649566 | 56.603525 / 76.085241 / 81.938963 |

| Model | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I |
|---|---:|---:|---:|---:|---:|---:|
| RandomDetail | 70.397054 | 73.838424 | 82.765002 | 65.235000 | 89.800 | 88.200 |
| AllDetail | 70.684209 | 74.267014 | 83.375002 | 65.310000 | 91.000 | 88.100 |
| Nested_Ds_low | 70.750319 | 74.327865 | 83.315002 | 65.384000 | 90.900 | 88.500 |
| Nested_D3_equal | 70.713901 | 74.456501 | 83.235002 | 65.100000 | 90.500 | 88.500 |
| Nested D3 balanced | 71.063600 | 74.954666 | 84.025001 | 65.227000 | 90.900 | 89.700 |

| Baseline | ΔScore5 | ΔJ_long3 | ΔJ_long | ΔShort4 | ΔUrban I2T | ΔUrban T2I |
|---|---:|---:|---:|---:|---:|---:|
| RandomDetail | +0.666545 | +1.116242 | +1.259999 | -0.008000 | +1.100 | +1.500 |
| AllDetail | +0.379391 | +0.687652 | +0.650000 | -0.083000 | -0.100 | +1.600 |
| Nested_Ds_low | +0.313281 | +0.626801 | +0.709999 | -0.157000 | +0.000 | +1.200 |
| Nested_D3_equal | +0.349699 | +0.498165 | +0.789999 | +0.127000 | +0.400 | +1.200 |

## All recall deltas vs RandomDetail

| Dataset | ΔI2T R@1 / R@5 / R@10 (pp) | ΔT2I R@1 / R@5 / R@10 (pp) |
|---|---|---|
| COCO | -0.280000 / -0.080000 / -0.220000 | -0.312000 / -0.012000 / -0.220000 |
| Urban-1k | +1.099998 / +0.500000 / +0.000000 | +1.499999 / +0.200003 / +0.200003 |
| Flickr30k-test1k | +0.200000 / +0.100000 / +0.000000 | +0.360000 / +0.000000 / -0.300000 |
| DOCCI | +1.520000 / +0.480000 / +0.420000 | +0.920000 / +0.440000 / +0.200000 |
| Long-DCI | +2.104709 / +0.723494 / +0.591949 | -0.447251 / -0.420942 / -0.065772 |

## All recall deltas vs AllDetail

| Dataset | ΔI2T R@1 / R@5 / R@10 (pp) | ΔT2I R@1 / R@5 / R@10 (pp) |
|---|---|---|
| COCO | -0.380000 / -0.020000 / +0.140000 | +0.008000 / +0.120000 / -0.024000 |
| Urban-1k | -0.099999 / +0.099999 / -0.199997 | +1.599997 / +0.000000 / +0.100005 |
| Flickr30k-test1k | -0.700000 / +0.300000 / -0.100000 | +0.740000 / +0.000000 / +0.120000 |
| DOCCI | +0.440000 / +0.060000 / -0.160000 | +0.660000 / +0.440000 / -0.060000 |
| Long-DCI | +0.565641 / +0.670876 / +0.710339 | +0.960274 / +0.894501 / +0.868193 |

## All recall deltas vs Nested_Ds_low

| Dataset | ΔI2T R@1 / R@5 / R@10 (pp) | ΔT2I R@1 / R@5 / R@10 (pp) |
|---|---|---|
| COCO | +0.160000 / +0.020000 / +0.220000 | -0.028000 / +0.020000 / +0.108000 |
| Urban-1k | +0.000000 / -0.099999 / +0.000000 | +1.199996 / +0.100005 / +0.100005 |
| Flickr30k-test1k | -1.000000 / +0.200000 / +0.000000 | +0.240000 / -0.120000 / -0.020000 |
| DOCCI | +0.980000 / +0.180000 / -0.060000 | +0.660000 / +0.340000 / +0.020000 |
| Long-DCI | +1.328598 / +1.183899 / +0.907656 | -0.407787 / -0.184162 / -0.157853 |

## All recall deltas vs Nested_D3_equal

| Dataset | ΔI2T R@1 / R@5 / R@10 (pp) | ΔT2I R@1 / R@5 / R@10 (pp) |
|---|---|---|
| COCO | +0.340000 / +0.220000 / -0.020000 | +0.048000 / -0.028000 / +0.032000 |
| Urban-1k | +0.400001 / +0.099999 / +0.000000 | +1.199996 / -0.099999 / +0.300002 |
| Flickr30k-test1k | +0.300000 / +0.400000 / +0.000000 | -0.180000 / -0.200000 / -0.180000 |
| DOCCI | +0.840000 / +0.260000 / +0.140000 | +0.720000 / +0.520000 / +0.220000 |
| Long-DCI | +0.407787 / +0.157853 / +0.000000 | -0.578795 / -0.197316 / -0.171008 |

| View | I2T CE | T2I CE | Combined CE | Weighted CE | Alignment share | Keep ratio |
|---|---:|---:|---:|---:|---:|---:|
| F | 0.050556 | 0.062310 | 0.112866 | 0.507898 | 17.876% | 0.852667 |
| Dall | 0.093815 | 0.111072 | 0.204888 | 0.921994 | 32.450% | 0.844290 |
| D3 | 0.685827 | 0.725521 | 1.411348 | 1.411348 | 49.674% | 0.792063 |

Shares exclude sparsity/inclusion. D3 equal comparison:`{"gradient_pressure": {"mean_norms": {"F": 10.726763010025024, "Dall": 14.324304819107056, "D3": 32.54663348197937}, "ratios": {"D3_Dall": 2.272126563417285, "D3_F": 3.0341523767759124}, "weighted_mean_norms": {"F": 48.27043354511261, "Dall": 64.45937168598175, "D3": 32.54663348197937}, "coefficients": {"F": 4.5, "Dall": 4.5, "D3": 1.0}, "weighted_ratios": {"D3_Dall": 0.50491701409273, "D3_F": 0.6742560837279806}, "D3_larger_than_F_plus_Dall_batches": 8, "weighted_D3_larger_than_F_plus_Dall_batches": 0, "dominant": false, "no_longer_dominant": true, "rule": "Strong non-dominance:weighted mean lowest norm<=both parent norms and at most1/8 weighted sum-dominance batches", "scope": "Per-view norm magnitudes, not a signed decomposition of total gradient"}, "gradient_dominant": false, "no_longer_dominant": true, "gradient_balance_improved": true, "equal_baseline_weighted_ratios": {"D3_Dall": 2.1049202546242003, "D3_F": 2.5688743917960246}, "weighted_ratio_reduction": {"D3_Dall": 0.7601253477496349, "D3_F": 0.7375285899998498}, "alignment_share_drop_pp": 28.681320398678505, "CE_ratio_vs_D3_equal": 1.2000817429260209, "low_Ds_weighted_Ds_Dall_ratio": 0.45875808366704734, "weighted_D3_Dall_distance_to_low_Ds": 0.0461589304256827, "reference_note": "Observed low-dose Ds ratio, not an additional pass threshold or tuned target"}`.
Raw and actual10/3*weight gradient norms:`{"mean_norms": {"F": 10.726763010025024, "Dall": 14.324304819107056, "D3": 32.54663348197937}, "ratios": {"D3_Dall": 2.272126563417285, "D3_F": 3.0341523767759124}, "weighted_mean_norms": {"F": 48.27043354511261, "Dall": 64.45937168598175, "D3": 32.54663348197937}, "coefficients": {"F": 4.5, "Dall": 4.5, "D3": 1.0}, "weighted_ratios": {"D3_Dall": 0.50491701409273, "D3_F": 0.6742560837279806}, "D3_larger_than_F_plus_Dall_batches": 8, "weighted_D3_larger_than_F_plus_Dall_batches": 0, "dominant": false, "no_longer_dominant": true, "rule": "Strong non-dominance:weighted mean lowest norm<=both parent norms and at most1/8 weighted sum-dominance batches", "scope": "Per-view norm magnitudes, not a signed decomposition of total gradient"}`.
Last50 masks:`{"inc": 0.008108068946748972, "inc_weight": 1.0, "F_Dall_mask_iou": 0.9658638739585876, "Dall_D3_mask_iou": 0.9000527119636536, "Dall_F_hard_violation": 0.010362121723592282, "D3_Dall_hard_violation": 0.016811178997159004}`.
Mask deltas:`{"Nested_D3_low": {"inc": -0.009847795292735101, "inc_weight": 0.0, "F_Dall_mask_iou": -0.0015140676498412997, "Dall_D3_mask_iou": 0.11537079691886898, "Dall_F_hard_violation": -0.0008067380450665941, "D3_Dall_hard_violation": -0.02361300654709339}, "Nested_D3_equal": {"inc": -0.006921937074512243, "inc_weight": 0.0, "F_Dall_mask_iou": 0.00035365343093873847, "Dall_D3_mask_iou": -0.03282575607299809, "Dall_F_hard_violation": -0.004046938680112362, "D3_Dall_hard_violation": -0.006369330026209354}, "Nested_Ds_low": {"inc": -0.009847795292735101, "inc_weight": 0.0, "F_Dall_mask_iou": -0.0015140676498412997, "Dall_D3_mask_iou": 0.11537079691886898, "Dall_F_hard_violation": -0.0008067380450665941, "D3_Dall_hard_violation": -0.02361300654709339}}`.
Keep deltas:`{"Nested_D3_low": {"F": -0.02283718109130861, "Dall": -0.024752724170684748, "D3": 0.03784934282302854}, "Nested_D3_equal": {"F": -0.018899421691894513, "Dall": -0.025628517866134626, "D3": -0.06417956113815304}, "Nested_Ds_low": {"F": -0.02283718109130861, "Dall": -0.024752724170684748, "D3": 0.03784934282302854}}`.
Full500 sampling:`{"valid_records": 511936, "Dall_mean_sentences": 7.0531414083010375, "mean_effective_tokens": {"F": 171.46915239404925, "Dall": 153.89054100512564, "D3": 66.38841183272909}, "mean_content_tokens": {"F": 169.46915239404925, "Dall": 151.89054100512564, "D3": 64.38841183272909}, "mean_per_sample_content_token_coverage": {"Dall_F": 0.8883068352970271, "D3_F": 0.4004027686653213, "D3_Dall": 0.45601314895918665}, "pooled_content_token_coverage": {"Dall_F": 0.8962725006846681, "D3_F": 0.3799417824608771, "D3_Dall": 0.42391324309363193}, "coverage_definition": "EOT-delimited content lengths excluding SOT/EOT, same valid records", "total_records": 512000, "D3_mean_sentences": 2.9903620765095638, "K_eff_histogram": {"2": 3924, "3": 507507, "1": 505, "0": 64}, "m_histogram": {"3": 3924, "4": 37167, "5": 92074, "6": 92953, "7": 76923, "8": 77266, "9": 68110, "10": 41151, "11": 16257, "12": 4210, "13": 922, "14": 232, "1": 173, "2": 332, "24": 4, "17": 18, "15": 84, "0": 64, "26": 5, "16": 45, "95": 1, "23": 3, "20": 4, "21": 7, "47": 2, "18": 11, "29": 1, "41": 1, "25": 4, "51": 1, "33": 2, "34": 2, "22": 4, "19": 12, "31": 1, "30": 3, "28": 3, "39": 1, "101": 1, "92": 2, "100": 1, "32": 1, "27": 3, "82": 1, "105": 2, "44": 1, "54": 1, "79": 1, "86": 1, "89": 1, "56": 1, "40": 1, "37": 1, "99": 1, "87": 1, "77": 1, "74": 1}, "degenerate_samples": 237, "degenerate_ratio": 0.000462890625, "strict_subset_samples": 511763, "D3_selected_sentence_position_histogram_1based": {"2": 235636, "3": 235099, "4": 234810, "5": 232704, "6": 204600, "7": 148861, "8": 102768, "9": 69611, "10": 40863, "11": 18241, "12": 5763, "13": 1353, "14": 308, "15": 77, "23": 9, "16": 40, "17": 19, "92": 1, "24": 4, "18": 16, "42": 2, "45": 2, "26": 6, "22": 4, "36": 1, "25": 6, "43": 1, "49": 1, "20": 7, "34": 3, "30": 2, "38": 3, "87": 1, "96": 1, "32": 3, "33": 3, "19": 3, "37": 2, "52": 1, "93": 2, "21": 6, "101": 1, "31": 2, "35": 3, "59": 1, "62": 2, "46": 1, "48": 2, "69": 1, "54": 1, "41": 2, "55": 2, "66": 1, "86": 2, "27": 2, "39": 2, "28": 2, "100": 1, "73": 1, "85": 1, "72": 1}}`.
Soft chain remains unchanged; learned hard masks can have nonzero violations. No projection or extra constraints.

Decision:`{"status": "D3_BALANCED_STRONG_POSITIVE", "Urban_R1_percent": {"I2T": 90.9, "T2I": 89.7}, "strong_gate": true, "positive_gate": true, "gradient_not_dominant": true, "gradient_balance_improved": true, "thresholds": {"strong_Score5_min": 70.750319, "positive_Score5_min": 70.7, "J_long3_min": 74.4, "Urban_T2I_min": 88.5, "weighted_ratio_reduction_min": 0.25, "tradeoff_max_Score5_or_Short4_decline_pp": 0.2}, "interpretation": "Long-tradeoff compared with D3 equal; no new long advantage means neither J_long3 nor J_long improves over low-dose Ds. Boundary outcomes retained as inconclusive.", "automatic_continuation": false, "automatic_new_experiments": false}`.
Strong weighted non-dominance:mean weighted D3<=both parents and at most1/8 D3>F+Dall batches. Improved balance:both weighted ratios decrease>=25% vs D3 equal, without prior severe systemic-dominance criterion. Long-tradeoff significant decline:>0.2pp Score5 or Short4 vs D3 equal while J_long3 increases. Inconclusive fallback covers undefined boundary outcomes.
Gradient protocol:exact same8 fixed seed0 epoch0 global1024 batches, step500 native-backbone optimizer group, raw directional-summed view CE and native F-reference gradients. No optimizer updates. Weighted composite cosine uses linear combination of separately computed view gradients.
Native inference:normalized image @ normalized full-caption text transpose; no mask/gate/detail/rerank/ensemble.
Strict export:`{"passed": true, "strict_load": true, "optimizer_steps": [500], "image_max_abs": 0.0, "text_max_abs": 0.0, "checkpoint_sha256": "24792ba9e6e3c34f733820f489cb3a8953278246ae58f2f688652894be871e4b", "bare_sha256": "27779a0c403431ecf55364c09e487a171313a2bea8e24e82cea9fe9b72dadfbe"}`.
Full512000 stream and indices proof:`{"passed": true, "records": 512000, "all500_sample_ids_F_Dall_D3_strings_tokens_indices_exact": true, "exact_LR": true, "baseline_steps_path": "/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/nested-detail-d3-equal500-20261007-r1/step500/steps.jsonl", "baseline_steps_sha256": "40b5abce8bbc027303be7fdd4d95a0f36e1dad8f97717bd603dce53c93731d7e", "selection_proof": "Actual current indices digests match immutable baseline sampler replay from actual baseline n/IDs, epoch0/seed0", "first5_gate": "BEFORE_UPDATE6"}`.
Full checkpoint/RNG/cursor, immutable SHA, evaluation manifests/protocols, launch Git source archive and CPU tests checked in VALIDATION.json/RESULTS.json.
Local raw log inventory:path/bytes/SHA256/time ranges in RUNTIME_STATS.json. Checkpoints/bare/cache/raw audit remain local. Memory includes page cache, not RSS or proof of OOM.
GitHub:experiment/nested-detail-d3-balanced500; only reviewed small code/config/tests/reports. No full4868 or other K/weight/sparsity/inclusion experiments.
