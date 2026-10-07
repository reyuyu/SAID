# Nested D3 Balanced full4868

Status: `FULL_STRONG_POSITIVE`. Fresh common step0, exactly4868 optimizer updates; no further experiment.

Score5 improved by 0.571902pp and J_long3 by 1.031836pp versus S0.2; Urban improved by 0.900pp in both directions. Short4 declined by 0.118pp, within the predeclared 0.2pp material-decline guard. COCO R@1 declined by 0.540/0.192pp; the gains are concentrated in Urban, DOCCI and Long rather than uniform across all datasets.

Input config identical to reviewed500 version; only CLI stop500->4868. Frozen F/Dall/D3 [1.35,1.35,.30], native summed directional CE, alignment10/3, unchanged detached-child chain/ramp200/max1 and sparsity(F+2Dall+2D3)/3.
ViT-B/16 Balanced-Stack-Patch, pair-conditioned Hard-ST mask/shared pool/balanced gate,4 A10080GB,256/rank,accum1,seed0,workers8,horizon4868; native optimizer/LR/AMP/preprocess unchanged. Original epoch tail180/rank retained.
Local-only training root `/root/said_s02_stage500/ShareGPT4V`; missing/symlink/escape fails, no NFS fallback. Ephemeral Docker overlay; NFS originals retained. No copy/full decode audit started.

| Dataset | I2T R@1 / R@5 / R@10 (%) | T2I R@1 / R@5 / R@10 (%) |
|---|---|---|
| COCO | 61.620000 / 83.780000 / 89.860000 | 43.132000 / 68.564000 / 78.108000 |
| Urban-1k | 93.700004 / 99.100006 / 99.700004 | 92.700005 / 99.100006 / 99.600005 |
| Flickr30k-test1k | 90.000000 / 98.400000 / 99.300000 | 74.040000 / 92.120000 / 95.800000 |
| DOCCI | 80.200000 / 96.400000 / 98.520000 | 81.240000 / 96.080000 / 98.380000 |
| Long-DCI | 60.181531 / 78.032097 / 84.017364 | 60.997106 / 78.650355 / 83.714812 |

| Model | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I |
|---|---:|---:|---:|---:|---:|---:|
| S02_full | 73.209163 | 77.137938 | 85.945002 | 67.316000 | 92.800 | 91.800 |
| DeBias | 73.226838 | 77.430063 | 86.495002 | 66.922000 | 93.300 | 93.100 |
| D3 Balanced full | 73.781065 | 78.169774 | 86.960002 | 67.198000 | 93.700 | 92.700 |

## Deltas vs S02_full

Aggregate deltas(pp): `{"J_long3": 1.031836470114925, "J_long": 1.015000166893003, "Score5": 0.571901882068957, "Short4": -0.117999999999995}`.

| Dataset | Baseline I2T/T2I R@1 | D3 Balanced I2T/T2I R@1 | Delta I2T/T2I(pp) |
|---|---|---|---|
| COCO | 62.160000 / 43.324000 | 61.620000 / 43.132000 | -0.540000 / -0.192000 |
| Urban-1k | 92.800003 / 91.800004 | 93.700004 / 92.700005 | +0.900000 / +0.900000 |
| Flickr30k-test1k | 89.800000 / 73.980000 | 90.000000 / 74.040000 | +0.200000 / +0.060000 |
| DOCCI | 79.200000 / 79.980000 | 80.200000 / 81.240000 | +1.000000 / +1.260000 |
| Long-DCI | 58.510918 / 60.536701 | 60.181531 / 60.997106 | +1.670613 / +0.460405 |

All available R@1/5/10 deltas are in RESULTS.json. Reference limitations: None.

## Deltas vs DeBias

Aggregate deltas(pp): `{"Score5": 0.5542265402335715, "J_long3": 0.7397112337226162, "J_long": 0.4650000503997802, "Short4": 0.2759995000000117}`.

| Dataset | Baseline I2T/T2I R@1 | D3 Balanced I2T/T2I R@1 | Delta I2T/T2I(pp) |
|---|---|---|---|
| COCO | missing / missing | 61.620000 / 43.132000 | missing / missing |
| Urban-1k | 93.300000 / 93.100000 | 93.700004 / 92.700005 | +0.400004 / -0.399995 |
| Flickr30k-test1k | missing / missing | 90.000000 / 74.040000 | missing / missing |
| DOCCI | missing / missing | 80.200000 / 81.240000 | missing / missing |
| Long-DCI | missing / missing | 60.181531 / 60.997106 | missing / missing |

All available R@1/5/10 deltas are in RESULTS.json. User supplied three measured aggregate metrics and Urban R@1. Short4 and Long bidirectional mean are explicitly derived from rounded aggregates under the shared frozen definitions. Individual Long I2T/T2I, other dataset directions and R@5/10 remain missing; none are guessed. No complete DeBias reference JSON was available at report completion.

The DeBias Short4 value in the comparison table is derived, not separately measured. Its Long bidirectional R@1 mean is also derived: `3*77.430063-2*86.495002=59.300185%`, assuming the shared frozen definitions (rounding bound 0.0000025pp). D3 Balanced's measured Long mean is `60.589319%`, delta `+1.289134pp`. Individual DeBias Long directional deltas remain unavailable. DeBias Urban changes are +0.400/-0.400pp; this run does not improve both directions versus DeBias.

## Final last50 diagnostics

| View | I2T CE | T2I CE | Combined CE | Weighted CE | Alignment share | Keep ratio |
|---|---:|---:|---:|---:|---:|---:|
| F | 0.015406 | 0.019576 | 0.034982 | 0.157419 | 12.422% | 0.739587 |
| Dall | 0.031464 | 0.036671 | 0.068135 | 0.306606 | 24.194% | 0.688520 |
| D3 | 0.387658 | 0.415576 | 0.803234 | 0.803234 | 63.384% | 0.624368 |

Mask hierarchy: `{"inc": 0.009977074079215527, "inc_weight": 1.0, "F_Dall_mask_iou": 0.8531972336769104, "Dall_D3_mask_iou": 0.7575385642051696, "Dall_F_hard_violation": 0.030487017072737217, "D3_Dall_hard_violation": 0.05950697794556618}`.
Mask delta vs500: `{"inc": 0.001869005132466555, "inc_weight": 0.0, "F_Dall_mask_iou": -0.11266664028167728, "Dall_D3_mask_iou": -0.14251414775848392, "Dall_F_hard_violation": 0.020124895349144935, "D3_Dall_hard_violation": 0.04269579894840717}`. Soft inclusion does not guarantee zero hard violations; no projection added.

Final hard violations are 3.049% (Dall outside F) and 5.951% (D3 outside Dall), increases of 2.012/4.270pp versus step500. D3's alignment-loss share increased from 49.674% at step500 to 63.384%; loss share alone is not evidence of gradient dominance. No new full-run gradient spot-check was requested or performed.
Four-epoch scalar curves, gate mean/variance/saturation, LR and inclusion ramp:TRAINING_DIAGNOSTICS.json.

Full-cycle(s): `{"count": 4868, "median": 2.1244181394577026, "p95": 2.270459258556366, "p99": 2.3351109576225277, "max": 33.47019958496094}`.
Slowest-rank data_wait(s): `{"count": 4868, "median": 0.0007021352648735046, "p95": 0.0009491208940744399, "p99": 0.001381881013512611, "max": 28.7423033490777}`.
Steps >3s:7; steps >10s:6; I/O errors:0; oom_kill:0; Pod/supervisor anomaly:False.
GPU peak GiB/rank: `{"0": 27.760382652282715, "1": 27.760382652282715, "2": 27.760382652282715, "3": 27.760382652282715}`.
Peak cgroup:500.000GiB; file cache:470.399GiB; anon:28.434GiB. Cache is not process RSS/OOM proof. PSI is host-scoped.

Training wall time:10719.255s (2h58m39s). All four original epoch tail batches retain 180 samples/rank, matching the frozen production loader. Stream audit covered 4,983,616 records; D3 strict-subset rate 99.951280%, with unchanged degenerate fallbacks.

Strict native inference:normalized native image embedding @ normalized native full-caption text embedding.T. No mask/gate/Dall/D3/rerank/ensemble/TTA.
Common0 SHA256:`54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`.
Final full checkpoint SHA256:`d159163e3e127e20ea17f8932454545e2648f2d2f052c60ac3667ac15b79542f`.
Bare SHA256:`2b0150feb3588063da3c93b3f93ac1b4bb2b3df223e06ed1e5ba458a4e3f2487`.
Exact export embeddings/checkpoint immutability:EXPORT_AUDIT.json; persistent checkpoints500/1217/2434/3651/4868 and raw log inventories:RUNTIME_STATS.json.

Raw inventories were finalized after supervisor exit:each local file's recorded size/SHA256 was checked against its final bytes. Time ranges are command intervals or conservative containment windows as documented in RUNTIME_STATS.json; prelaunch audits precede training, and review logs/telemetry extend through review completion.
Sample audit:SAMPLING_AUDIT.json,4000 offline strings/tokens/indices across4 epochs; all runtime IDs/indices/LR checked and first500 text stream matched independently.
Decision and thresholds declared before training: `{"status": "FULL_STRONG_POSITIVE", "strong": true, "positive": true, "tradeoff": false, "negative": false, "scores_delta_pp": {"Score5": 0.571901882068957, "J_long3": 1.031836470114925, "J_long": 1.015000166893003, "Short4": -0.117999999999995}, "Urban_delta_pp": {"I2T": 0.9000000000000057, "T2I": 0.9000000000000057}, "declared_before_training": true, "clear_gain_pp": 0.1, "material_decline_pp": 0.2, "automatic_new_experiments": false, "automatic_continuation": false}`.
Git branch:experiment/nested-detail-d3-balanced-full-v1. Checkpoints/bare/datasets/local mirror/cache/raw logs not uploaded. No automatic new experiment.
