# Nested D3 equal-weight matched500 experiment

Status: `D3_TRADEOFF`; 500/500 updates, horizon4868; no automatic follow-up.
Relative to atomic equal-weight: only lowest view Ds -> D3 changes. F/Dall packing, tokens, sample order, all model/objective/optimizer/LR/loader settings frozen.
D3 uses ordered uniform sampling without replacement: K=min(3,m-1) for m>=2; m=1 retains its single detail; m=0 retains invalid local-view masking. No fabricated/duplicated/split sentences.
Private RNG: SHA256(seed:epoch:sample_id:nested_detail_d3_v1), Random.sample, sorted selected indices. No global RNG advancement.
Fresh common0 SHA256: `54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`. Local-only `/root/said_s02_stage500/ShareGPT4V`; missing/symlink/escape fails; no fallback. Ephemeral cache; NFS originals retained.
Alignment10/3*(L_F+L_Dall+L_D3), each directional-summed CE. Detached-child chain Dall->F,D3->Dall only,200-step ramp,max1. Sparsity(Omega_F+2*Omega_Dall+2*Omega_D3)/3.
Frozen model logs use O/E and legacy Ds names internally; reports map lowest slot to D3 without changing the graph.
An initial five-update attempt stopped because the admission checker rejected trainer-added runtime metadata. All five sample/F/Dall checks passed. Checker fixed and regression-tested; the formal r1 run restarted from common0, never resumed that checkpoint. Prior artifacts retained locally in launch_provenance.prior_aborted_attempt.

| Dataset | I2T R@1 / R@5 / R@10 (%) | T2I R@1 / R@5 / R@10 (%) |
|---|---|---|
| COCO | 60.060000 / 82.400000 / 89.280000 | 41.540000 / 67.304000 / 76.900000 |
| Urban-1k | 90.500003 / 98.500007 / 99.600005 | 88.500005 / 98.600006 / 99.200004 |
| Flickr30k-test1k | 86.700000 / 97.400000 / 99.200000 | 72.100000 / 91.580000 / 95.300000 |
| DOCCI | 76.980000 / 95.080000 / 97.860000 | 76.960000 / 94.740000 / 97.500000 |
| Long-DCI | 56.616680 / 75.374901 / 81.649566 | 57.182320 / 76.282557 / 82.109971 |

| Model | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I |
|---|---:|---:|---:|---:|---:|---:|
| RandomDetail | 70.397054 | 73.838424 | 82.765002 | 65.235000 | 89.800 | 88.200 |
| AllDetail | 70.684209 | 74.267014 | 83.375002 | 65.310000 | 91.000 | 88.100 |
| Nested_Ds_low | 70.750319 | 74.327865 | 83.315002 | 65.384000 | 90.900 | 88.500 |
| Nested_Ds_equal | 69.835511 | 73.329185 | 81.840002 | 64.595000 | 88.600 | 87.700 |
| Nested D3 equal | 70.713901 | 74.456501 | 83.235002 | 65.100000 | 90.500 | 88.500 |

| Baseline | ΔScore5 | ΔJ_long3 | ΔJ_long | ΔShort4 | ΔUrban I2T | ΔUrban T2I |
|---|---:|---:|---:|---:|---:|---:|
| RandomDetail | +0.316847 | +0.618078 | +0.470000 | -0.135000 | +0.700 | +0.300 |
| AllDetail | +0.029692 | +0.189487 | -0.140000 | -0.210000 | -0.500 | +0.400 |
| Nested_Ds_low | -0.036418 | +0.128636 | -0.080000 | -0.284000 | -0.400 | +0.000 |
| Nested_Ds_equal | +0.878390 | +1.127317 | +1.395000 | +0.505000 | +1.900 | +0.800 |

## All recall deltas vs RandomDetail

| Dataset | ΔI2T R@1 / R@5 / R@10 (pp) | ΔT2I R@1 / R@5 / R@10 (pp) |
|---|---|---|
| COCO | -0.620000 / -0.300000 / -0.200000 | -0.360000 / +0.016000 / -0.252000 |
| Urban-1k | +0.699997 / +0.400001 / +0.000000 | +0.300002 / +0.300002 / -0.099999 |
| Flickr30k-test1k | -0.100000 / -0.300000 / +0.000000 | +0.540000 / +0.200000 / -0.120000 |
| DOCCI | +0.680000 / +0.220000 / +0.280000 | +0.200000 / -0.080000 / -0.020000 |
| Long-DCI | +1.696922 / +0.565641 / +0.591949 | +0.131544 / -0.223625 / +0.105235 |

## All recall deltas vs AllDetail

| Dataset | ΔI2T R@1 / R@5 / R@10 (pp) | ΔT2I R@1 / R@5 / R@10 (pp) |
|---|---|---|
| COCO | -0.720000 / -0.240000 / +0.160000 | -0.040000 / +0.148000 / -0.056000 |
| Urban-1k | -0.500000 / +0.000000 / -0.199997 | +0.400001 / +0.099999 / -0.199997 |
| Flickr30k-test1k | -1.000000 / -0.100000 / -0.100000 | +0.920000 / +0.200000 / +0.300000 |
| DOCCI | -0.400000 / -0.200000 / -0.300000 | -0.060000 / -0.080000 / -0.280000 |
| Long-DCI | +0.157853 / +0.513023 / +0.710339 | +1.539069 / +1.091818 / +1.039200 |

## All recall deltas vs Nested_Ds_low

| Dataset | ΔI2T R@1 / R@5 / R@10 (pp) | ΔT2I R@1 / R@5 / R@10 (pp) |
|---|---|---|
| COCO | -0.180000 / -0.200000 / +0.240000 | -0.076000 / +0.048000 / +0.076000 |
| Urban-1k | -0.400001 / -0.199997 / +0.000000 | +0.000000 / +0.200003 / -0.199997 |
| Flickr30k-test1k | -1.300000 / -0.200000 / +0.000000 | +0.420000 / +0.080000 / +0.160000 |
| DOCCI | +0.140000 / -0.080000 / -0.200000 | -0.060000 / -0.180000 / -0.200000 |
| Long-DCI | +0.920810 / +1.026046 / +0.907656 | +0.171008 / +0.013154 / +0.013154 |

## All recall deltas vs Nested_Ds_equal

| Dataset | ΔI2T R@1 / R@5 / R@10 (pp) | ΔT2I R@1 / R@5 / R@10 (pp) |
|---|---|---|
| COCO | +0.960000 / +0.120000 / +0.800000 | +0.260000 / +0.668000 / +0.356000 |
| Urban-1k | +1.899999 / +0.400001 / +0.300002 | +0.800002 / +0.500000 / +0.099999 |
| Flickr30k-test1k | -0.300000 / +0.000000 / +0.100000 | +1.100000 / +0.440000 / +0.180000 |
| DOCCI | +1.760000 / +0.620000 / +0.220000 | +1.120000 / +0.160000 / +0.120000 |
| Long-DCI | +0.894501 / +0.736648 / +0.368324 | +0.289398 / -0.171008 / +0.118390 |

| View | I2T CE | T2I CE | Combined CE | Alignment share | Keep ratio |
|---|---:|---:|---:|---:|---:|
| F | 0.055585 | 0.064649 | 0.120234 | 8.011% | 0.871566 |
| Dall | 0.095429 | 0.109210 | 0.204639 | 13.634% | 0.869918 |
| D3 | 0.568050 | 0.607993 | 1.176043 | 78.355% | 0.856243 |

CE_D3/CE_Dall: 5.74691322701368. Shares exclude inclusion/sparsity.
Atomic equal-weight comparison: `{"CE_ratio_vs_atomic_equal": 0.2602705154411625, "alignment_share_drop_pp": 14.520457395486853, "gradient_dominant": true, "both_gradient_ratio_improvement": false, "gradient_pressure": {"mean_norms": {"F": 10.947014331817627, "Dall": 13.359890818595886, "D3": 28.12150478363037}, "ratios": {"D3_Dall": 2.1049202546242003, "D3_F": 2.5688743917960246}, "D3_larger_than_F_plus_Dall_batches": 8, "dominant": true, "rule": "Both mean ratios>2 and lowest-view norm>F+Dall norms in at least6/8 same frozen batches", "scope": "Per-view native-backbone norm magnitudes, not a signed decomposition of total gradient"}}`.
Full500 sampling: `{"valid_records": 511936, "Dall_mean_sentences": 7.0531414083010375, "mean_effective_tokens": {"F": 171.46915239404925, "Dall": 153.89054100512564, "D3": 66.38841183272909}, "mean_content_tokens": {"F": 169.46915239404925, "Dall": 151.89054100512564, "D3": 64.38841183272909}, "mean_per_sample_content_token_coverage": {"Dall_F": 0.8883068352970271, "D3_F": 0.4004027686653213, "D3_Dall": 0.45601314895918665}, "pooled_content_token_coverage": {"Dall_F": 0.8962725006846681, "D3_F": 0.3799417824608771, "D3_Dall": 0.42391324309363193}, "coverage_definition": "EOT-delimited content lengths excluding SOT/EOT, same valid records", "total_records": 512000, "D3_mean_sentences": 2.9903620765095638, "K_eff_histogram": {"2": 3924, "3": 507507, "1": 505, "0": 64}, "m_histogram": {"3": 3924, "4": 37167, "5": 92074, "6": 92953, "7": 76923, "8": 77266, "9": 68110, "10": 41151, "11": 16257, "12": 4210, "13": 922, "14": 232, "1": 173, "2": 332, "24": 4, "17": 18, "15": 84, "0": 64, "26": 5, "16": 45, "95": 1, "23": 3, "20": 4, "21": 7, "47": 2, "18": 11, "29": 1, "41": 1, "25": 4, "51": 1, "33": 2, "34": 2, "22": 4, "19": 12, "31": 1, "30": 3, "28": 3, "39": 1, "101": 1, "92": 2, "100": 1, "32": 1, "27": 3, "82": 1, "105": 2, "44": 1, "54": 1, "79": 1, "86": 1, "89": 1, "56": 1, "40": 1, "37": 1, "99": 1, "87": 1, "77": 1, "74": 1}, "degenerate_samples": 237, "degenerate_ratio": 0.000462890625, "strict_subset_samples": 511763, "D3_selected_sentence_position_histogram_1based": {"2": 235636, "3": 235099, "4": 234810, "5": 232704, "6": 204600, "7": 148861, "8": 102768, "9": 69611, "10": 40863, "11": 18241, "12": 5763, "13": 1353, "14": 308, "15": 77, "23": 9, "16": 40, "17": 19, "92": 1, "24": 4, "18": 16, "42": 2, "45": 2, "26": 6, "22": 4, "36": 1, "25": 6, "43": 1, "49": 1, "20": 7, "34": 3, "30": 2, "38": 3, "87": 1, "96": 1, "32": 3, "33": 3, "19": 3, "37": 2, "52": 1, "93": 2, "21": 6, "101": 1, "31": 2, "35": 3, "59": 1, "62": 2, "46": 1, "48": 2, "69": 1, "54": 1, "41": 2, "55": 2, "66": 1, "86": 2, "27": 2, "39": 2, "28": 2, "100": 1, "73": 1, "85": 1, "72": 1}}`.
Last50 masks: `{"inc": 0.015030006021261215, "inc_weight": 1.0, "F_Dall_mask_iou": 0.9655102205276489, "Dall_D3_mask_iou": 0.9328784680366516, "Dall_F_hard_violation": 0.014409060403704644, "D3_Dall_hard_violation": 0.023180509023368358}`.
Mask deltas: `{"Nested_D3_equal": {"inc": -0.005973011665046215, "inc_weight": 0.0, "F_Dall_mask_iou": -0.009707431793212873, "Dall_D3_mask_iou": 0.04635416030883788, "Dall_F_hard_violation": 0.003753587752580643, "D3_Dall_hard_violation": -0.015985224805772302}, "Nested_D3_low": {"inc": -0.002925858218222858, "inc_weight": 0.0, "F_Dall_mask_iou": -0.0018677210807800382, "Dall_D3_mask_iou": 0.14819655299186707, "Dall_F_hard_violation": 0.003240200635045768, "D3_Dall_hard_violation": -0.017243676520884037}, "Nested_Ds_equal": {"inc": -0.005973011665046215, "inc_weight": 0.0, "F_Dall_mask_iou": -0.009707431793212873, "Dall_D3_mask_iou": 0.04635416030883788, "Dall_F_hard_violation": 0.003753587752580643, "D3_Dall_hard_violation": -0.015985224805772302}, "Nested_Ds_low": {"inc": -0.002925858218222858, "inc_weight": 0.0, "F_Dall_mask_iou": -0.0018677210807800382, "Dall_D3_mask_iou": 0.14819655299186707, "Dall_F_hard_violation": 0.003240200635045768, "D3_Dall_hard_violation": -0.017243676520884037}}`.
Keep deltas: `{"Nested_D3_equal": {"F": -0.03915966033935547, "Dall": -0.03931149363517761, "D3": -0.02376019120216366}, "Nested_D3_low": {"F": -0.003937759399414098, "Dall": 0.000875793695449878, "D3": 0.10202890396118158}, "Nested_Ds_equal": {"F": -0.03915966033935547, "Dall": -0.03931149363517761, "D3": -0.02376019120216366}, "Nested_Ds_low": {"F": -0.003937759399414098, "Dall": 0.000875793695449878, "D3": 0.10202890396118158}}`.
Soft chain objective unchanged; nonzero hard-mask violations are measured, never projected away.
Gradient protocol: same8 fixed seed0 epoch0 global batches, step500, exact native-backbone optimizer group; raw per-view gradients and native full-caption reference. No optimizer updates.

Decision: `{"status": "D3_TRADEOFF", "Urban_R1_percent": {"I2T": 90.5, "T2I": 88.5}, "retrieval_near_low_Ds": true, "CE_significantly_lower": true, "alignment_contribution_improved": true, "gradient_balance_improved": false, "gradient_not_dominant": false, "thresholds": {"Score5_strong_min": 70.750319, "J_long3_strong_min": 74.327865, "Urban_T2I_strong_min": 88.5, "near_max_aggregate_decline_pp": 0.2, "near_max_Urban_T2I_decline_pp": 0.3, "CE_reduction_min": 0.25, "alignment_share_drop_min_pp": 10, "gradient_ratio_reduction_min": 0.25}, "tradeoff_scope": "Mixed retrieval or optimization/retrieval results outside the positive/negative gates", "automatic_continuation": false, "automatic_new_experiments": false}`.
Significant CE reduction>=25%; contribution decrease>=10pp; gradient ratio reduction>=25%; same prior dominance criterion. Retrieval near guard:Score5/J_long3 down<=0.2pp, Urban T2I down<=0.3pp vs low-Ds.
Native inference only normalized image @ normalized full-caption text transpose. No mask/gate/detail/rerank/ensemble.
Strict export: `{"passed": true, "strict_load": true, "optimizer_steps": [500], "image_max_abs": 0.0, "text_max_abs": 0.0, "checkpoint_sha256": "5ae825a15eb1ab60160b70115d04dd735da2108e6855457737d8b5156e0a6764", "bare_sha256": "92bc2ed2f4e72cb3aa59740ce5c4fdec760b9165919e0666dc190732ef20e7c6"}`.
Full stream proof: `{"passed": true, "records": 512000, "all500_sample_ids_F_Dall_strings_tokens_exact": true, "all500_D3_ordered_strict_subset_or_original_fallback": true, "exact_LR": true, "baseline_steps_path": "/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/nested-detail-equal-weight500-20261007/step500/steps.jsonl", "baseline_steps_sha256": "17c438ce2079c40b69a10a79d511ed5385ba57d7ec5088f210483983bb6e115f", "first5_gate": "BEFORE_UPDATE6"}`.
Checkpoint completeness, immutable SHA, evaluation manifests/protocols and launch Git source snapshot checked in VALIDATION.json/RESULTS.json.
Raw logs/checkpoints/bare/cache/1000-record raw audit remain local; path/bytes/SHA256/time inventories in RUNTIME_STATS.json/SAMPLING_AUDIT.json. Cgroup memory includes page cache, not RSS or OOM evidence.
GitHub:experiment/nested-detail-d3-equal500, reviewed small code/config/tests/reports only. No full4868, K/weight/sparsity/sampling/inclusion sweeps.

## Interpretation of the four experiment questions

1. Lowest-view CE falls substantially: D3 last50 combined CE1.176043 versus atomic equal-weight Ds4.518543, a73.973% reduction; D3/Dall CE ratio remains5.7469. Atomic low-dose Ds CE was5.472811. Broader detail context substantially reduces the measured atomic CE pressure.
2. Equal-weight optimization remains concentrated in the lowest view. Its alignment share falls92.875% ->78.355% (14.520pp), but per-view native-backbone gradient ratios only fall Ds/Dall2.3229 ->D3/Dall2.1049 and Ds/F2.8078 ->D3/F2.5689. All8/8 fixed batches still satisfy lowest-view norm > F+Dall norm magnitudes. The unchanged dominance criterion remains true; this is partial relief, not elimination of gradient pressure.
3. Retrieval stays near low-dose Nested overall: Score5 delta-0.036418pp, J_long3+0.128636pp, Urban90.5/88.5 versus90.9/88.5. Long-DCI improves+0.920810/+0.171008pp atR1, while Flickr I2T falls1.3pp and Short4 falls0.284pp. Versus atomic equal-weight, Score5 recovers0.878390pp and J_long3 recovers1.127317pp. These mixed changes and persistent gradient dominance support D3_TRADEOFF; neither positive gate is met.
4. Mean keep coverage has the requested ordering, F0.871566 >= Dall0.869918 >= D3 0.856243, but gaps are modest (0.165pp and1.368pp). IoUs0.965510/0.932878 show substantial mask overlap. Masks are not identical, and nesting is approximate: hard violations1.441%/2.318%, inclusion loss0.015030. This supports a weak coverage hierarchy rather than a sharply separated coarse-to-fine hierarchy. No extra constraint was applied.

Final evidence review passed:81 related CPU tests; all512000 sample IDs/F/Dall strings/tokens matched; full resumable500 checkpoint and optimizer/RNG/cursor verified; strict export embeddings exact; checkpoint SHA unchanged by gradient diagnosis/evaluation; native manifests/protocols and launch Git blobs verified. The task stops here, without any new experiment or full run.
