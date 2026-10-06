# Nested Detail alignment-weight-only ablation: 1/1/1

Status: `ATOMIC_DETAIL_OVERWEIGHTED`; updates500/500, horizon4868; no automatic continuation.
Only method change: alignment weights[1.4,1.4,0.2] -> [1,1,1]. Common step0, model/adapter initialization, sample order, all F/Dall/Ds strings/tokens/draws, CE/temperature, optimizer/LR, batch/workers, mask, detached-child chain inclusion/ramp and sparsity frozen.
Common step0 SHA256: `54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`. Independent fresh start; prior nested checkpoint never used for resume.
Local-only root: `/root/said_s02_stage500/ShareGPT4V`. Missing/symlink/escape fails; no NFS fallback. No staging/full audit/new experiment.
Alignment:10/3*(L_F+L_Dall+L_Ds), each view directional CE summed. Inclusion:Dall->F, Ds->Dall only. Sparsity:(Omega_F+2*Omega_Dall+2*Omega_Ds)/3.

| Dataset | I2T R@1 / R@5 / R@10 (%) | T2I R@1 / R@5 / R@10 (%) |
|---|---|---|
| COCO | 59.100000 / 82.280000 / 88.480000 | 41.280000 / 66.636000 / 76.544000 |
| Urban-1k | 88.600004 / 98.100007 / 99.300003 | 87.700003 / 98.100007 / 99.100006 |
| Flickr30k-test1k | 87.000000 / 97.400000 / 99.100000 | 71.000000 / 91.140000 / 95.120000 |
| DOCCI | 75.220000 / 94.460000 / 97.640000 | 75.840000 / 94.580000 / 97.380000 |
| Long-DCI | 55.722178 / 74.638253 / 81.281242 | 56.892923 / 76.453565 / 81.991581 |

| Model | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I |
|---|---:|---:|---:|---:|---:|---:|
| S0.2 RandomDetail | 70.397054 | 73.838424 | 82.765002 | 65.235 | 89.800 | 88.200 |
| S0.2 AllDetail | 70.684209 | 74.267014 | 83.375002 | 65.310 | 91.000 | 88.100 |
| Nested1.4/1.4/0.2 | 70.750319 | 74.327865 | 83.315002 | 65.384 | 90.900 | 88.500 |
| Nested1/1/1 | 69.835511 | 73.329185 | 81.840002 | 64.595000 | 88.600 | 87.700 |

| Baseline | ΔScore5 | ΔJ_long3 | ΔJ_long | ΔShort4 | ΔUrban I2T | ΔUrban T2I |
|---|---:|---:|---:|---:|---:|---:|
| RandomDetail | -0.561543 | -0.509239 | -0.925000 | -0.640000 | -1.200 | -0.500 |
| AllDetail | -0.848698 | -0.937829 | -1.535000 | -0.715000 | -2.400 | -0.400 |
| Nested_low_Ds | -0.914808 | -0.998680 | -1.475000 | -0.789000 | -2.300 | -0.800 |

| Dataset | ΔI2T R@1 / R@5 / R@10 vs nested low-Ds (pp) | ΔT2I R@1 / R@5 / R@10 (pp) |
|---|---|---|
| COCO | -1.140000 / -0.320000 / -0.560000 | -0.336000 / -0.620000 / -0.280000 |
| Urban-1k | -2.300000 / -0.599998 / -0.300002 | -0.800002 / -0.299996 / -0.299996 |
| Flickr30k-test1k | -1.000000 / -0.200000 / -0.100000 | -0.680000 / -0.360000 / -0.020000 |
| DOCCI | -1.620000 / -0.700000 / -0.420000 | -1.180000 / -0.340000 / -0.320000 |
| Long-DCI | +0.026309 / +0.289398 / +0.539332 | -0.118390 / +0.184162 / -0.105235 |

| View | I2T CE | T2I CE | Combined CE | Weighted CE | Alignment-loss share | Keep ratio |
|---|---:|---:|---:|---:|---:|---:|
| F | 0.059288 | 0.066454 | 0.125742 | 0.419141 | 2.585% | 0.910726 |
| Dall | 0.106322 | 0.114557 | 0.220879 | 0.736263 | 4.540% | 0.909230 |
| Ds | 2.253197 | 2.265345 | 4.518543 | 15.061809 | 92.875% | 0.880003 |

Weighted-loss shares use last50 means and exclude sparsity/inclusion. Sampling/gate/selected-step telemetry:TRAINING_DIAGNOSTICS.json.
Last50 mask hierarchy: `{"inc": 0.02100301768630743, "inc_weight": 1.0, "F_Dall_mask_iou": 0.9752176523208618, "Dall_Ds_mask_iou": 0.8865243077278138, "Dall_F_hard_violation": 0.010655472651124001, "Ds_Dall_hard_violation": 0.03916573382914066}`.
Mask hierarchy delta vs prior nested: `{"inc": 0.0030471534468233565, "inc_weight": 0.0, "F_Dall_mask_iou": 0.007839710712432835, "Dall_Ds_mask_iou": 0.10184239268302919, "Dall_F_hard_violation": -0.0005133871175348748, "Ds_Dall_hard_violation": -0.0012584517151117344}`.
Gradient pressure: `{"mean_norms": {"F": 11.331637144088745, "Dall": 13.697024822235107, "Ds": 31.816879987716675}, "ratios": {"Ds_Dall": 2.3229044555768525, "Ds_F": 2.8077919883195563}, "Ds_fraction_of_sum_of_view_norms": 0.5597075671024571, "Ds_larger_than_F_plus_Dall_batches": 8, "dominant": true, "rule": "Both mean norm ratios >2 and Ds norm > F+Dall norm in at least6 of8 fixed batches", "scope": "Native-backbone per-view CE gradients with equal coefficients; not a signed decomposition of total norm"}`.
Soft detached-child inclusion is unchanged; learned hard masks may still violate nesting. No projection or additional loss was added.

Decision: `{"status": "ATOMIC_DETAIL_OVERWEIGHTED", "Urban_R1_percent": {"I2T": 88.6, "T2I": 87.7}, "positive_guard": false, "Ds_gradient_dominant": true, "thresholds": {"strong_Urban_T2I_min": 88.7, "strong_Score5_min": 70.750319, "strong_J_long3_min": 74.327865, "positive_Urban_T2I_gt": 88.5, "positive_max_decline_pp": 0.2}, "precedence": "Strong, positive, Urban/global tradeoff, gradient-dominated regression, no clear gain", "automatic_continuation": false, "automatic_new_experiments": false}`.
Positive “no material decline” tolerance was fixed before training at0.2pp for each of Score5/J_long3. Strong gate uses exact user thresholds; Urban uses canonical1000-hit rounding. Explicit no-clear-gain fallback handles outcomes outside the named gates.
Gradient protocol: same8 seed0 epoch0 global1024 batches, fixed step500, same native-backbone optimizer group, raw per-view CE and native F-reference gradient. No optimizer updates; all first8 batch-ID hashes verified against prior run.
Strict native inference uses normalized image/native full-caption embeddings and plain inner product; no mask/gate/detail/rerank/ensemble.
Checkpoint/bare hashes and immutable evaluation: `{"passed": true, "strict_load": true, "optimizer_steps": [500], "image_max_abs": 0.0, "text_max_abs": 0.0, "checkpoint_sha256": "67df75f4b6c917155ac8639b0d11468a0a618fd832f226755fa348da63fd9508", "bare_sha256": "12b5bd36e889f70d8327a45716f95796bf99b3c94be0b89f87f7b94d0519483e"}`.
Exact all512000 sample/text/token stream proof: `{"passed": true, "records": 512000, "all500_sample_ids_F_Dall_Ds_strings_tokens_exact": true, "exact_sentence_draws": true, "exact_LR": true, "baseline_steps_path": "/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/nested-detail500-20261006/step500/steps.jsonl", "baseline_steps_sha256": "9abd02ba57df6669cef7d71882ce9d7390ef6cfc384b4d63b98cd90b84cfed6d", "first5_gate": "BEFORE_UPDATE6"}`.
Raw logs/checkpoints/bare/local mirror/full1000-record audit stay local; path/size/SHA256/time-range inventory:RUNTIME_STATS.json and SAMPLING_AUDIT.json.
Ephemeral Docker overlay cache, persistent NFS originals retained. Checkpoints remain on persistent project runtime. GitHub only receives reviewed small code/config/tests/reports; branch experiment/nested-detail-equal-weight500.
No automatic full4868, weight sweep, sampling/shuffle/offset/sparsity/inclusion changes.
