# D3 Balanced vs HNS-v1 epoch replay

Pure retrospective evaluation. No training, optimizer update, HNS-Half continuation, model/loss change or new experiment.
Eight existing checkpoints sequentially replayed with the validated four-GPU scheduler. Both finals reproduce every original directional R@1/5/10 and all four aggregates exactly.
Existing native evaluators and aggregation functions are reused unchanged. No mask, auxiliary inference, rerank, ensemble, TTA or checkpoint averaging.

| Model / Score5 | E1 / 1217 | E2 / 2434 | E3 / 3651 | E4 / 4868 |
|---|---:|---:|---:|---:|
| D3 Balanced | 72.812657 | 73.598216 | 73.830182 | 73.781065 |
| HNS-v1 | 72.836442 | 73.475830 | 73.774418 | 73.638084 |
| HNS minus Balanced | +0.023785 | -0.122386 | -0.055764 | -0.142980 |

## Score5 (%)

| Epoch | Balanced | HNS | HNS minus Balanced(pp) |
|---|---:|---:|---:|
| 1 | 72.812657 | 72.836442 | +0.023785 |
| 2 | 73.598216 | 73.475830 | -0.122386 |
| 3 | 73.830182 | 73.774418 | -0.055764 |
| 4 | 73.781065 | 73.638084 | -0.142980 |

## J_long3 (%)

| Epoch | Balanced | HNS | HNS minus Balanced(pp) |
|---|---:|---:|---:|
| 1 | 77.001762 | 76.971404 | -0.030358 |
| 2 | 77.947693 | 77.824383 | -0.123311 |
| 3 | 78.205636 | 78.188029 | -0.017607 |
| 4 | 78.169774 | 78.014807 | -0.154967 |

## J_long (%)

| Epoch | Balanced | HNS | HNS minus Balanced(pp) |
|---|---:|---:|---:|
| 1 | 86.040002 | 85.955001 | -0.085001 |
| 2 | 86.765002 | 86.705003 | -0.059999 |
| 3 | 86.925002 | 87.040003 | +0.115000 |
| 4 | 86.960002 | 86.915002 | -0.045000 |

## Short4 (%)

| Epoch | Balanced | HNS | HNS minus Balanced(pp) |
|---|---:|---:|---:|
| 1 | 66.529000 | 66.634000 | +0.105000 |
| 2 | 67.074000 | 66.953000 | -0.121000 |
| 3 | 67.267000 | 67.154000 | -0.113000 |
| 4 | 67.198000 | 67.073000 | -0.125000 |

## Epoch-to-epoch gains

| Segment | Metric | Balanced gain(pp) | HNS gain(pp) | Relative HNS growth(pp) |
|---|---|---:|---:|---:|
| E1->E2 | Score5 | +0.785559 | +0.639387 | -0.146171 |
| E1->E2 | J_long3 | +0.945931 | +0.852979 | -0.092952 |
| E1->E2 | J_long | +0.725000 | +0.750002 | +0.025002 |
| E1->E2 | Short4 | +0.545000 | +0.319000 | -0.226000 |
| E2->E3 | Score5 | +0.231966 | +0.298588 | +0.066622 |
| E2->E3 | J_long3 | +0.257943 | +0.363647 | +0.105704 |
| E2->E3 | J_long | +0.160000 | +0.334999 | +0.174999 |
| E2->E3 | Short4 | +0.193000 | +0.201000 | +0.008000 |
| E3->E4 | Score5 | -0.049117 | -0.136334 | -0.087216 |
| E3->E4 | J_long3 | -0.035862 | -0.173223 | -0.137361 |
| E3->E4 | J_long | +0.035000 | -0.125001 | -0.160001 |
| E3->E4 | Short4 | -0.069000 | -0.081000 | -0.012000 |

## Directional R@1 trajectory

| Dataset / direction | E1 B/H | E2 B/H | E3 B/H | E4 B/H | Final HNS minus Balanced(pp) |
|---|---|---|---|---|---:|
| COCO/I2T | 62.020000/61.800000 | 61.740000/62.060000 | 62.080000/61.900000 | 61.620000/61.600000 | -0.020000 |
| COCO/T2I | 42.496000/42.476000 | 42.836000/42.792000 | 43.128000/43.036000 | 43.132000/42.972000 | -0.160000 |
| Urban-1k/I2T | 93.600005/93.400002 | 93.500006/94.000006 | 93.700004/94.100004 | 93.700004/93.800002 | +0.099999 |
| Urban-1k/T2I | 91.900003/91.500002 | 92.400002/92.500007 | 92.600006/92.600006 | 92.700005/92.700005 | +0.000000 |
| Flickr30k-test1k/I2T | 88.400000/89.100000 | 89.800000/89.200000 | 89.900000/89.700000 | 90.000000/89.800000 | -0.200000 |
| Flickr30k-test1k/T2I | 73.200000/73.160000 | 73.920000/73.760000 | 73.960000/73.980000 | 74.040000/73.920000 | -0.120000 |
| DOCCI/I2T | 79.160000/79.320000 | 80.080000/79.780000 | 80.120000/80.300000 | 80.200000/80.160000 | -0.040000 |
| DOCCI/T2I | 79.500000/79.600000 | 81.080000/80.540000 | 81.280000/81.160000 | 81.240000/81.000000 | -0.240000 |
| Long-DCI/I2T | 58.339911/58.458300 | 59.865825/59.576427 | 60.536701/60.365693 | 60.181531/60.010524 | -0.171008 |
| Long-DCI/T2I | 59.510655/59.550118 | 60.760326/60.549855 | 60.997106/60.602473 | 60.997106/60.418311 | -0.578795 |

## Matched training diagnostics

Extracted from immutable existing scalar training logs. CE, weighted CE, shares and keep are epoch-last50 means. Hierarchy telemetry retains its actual observation count and step list; partial observations are labeled selected-step, never treated as 50 full observations.

| Model | Epoch | F keep | Dall keep | D3 keep | F-Dall IoU | Dall-D3 IoU | Dall outside F(%) | D3 outside Dall(%) | D3 alignment share(%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D3_Balanced | 1 | 0.806952 | 0.776244 | 0.699513 | 0.910296 | 0.819298 | 2.151423 | 3.539928 | 54.837698 |
| D3_Balanced | 2 | 0.775733 | 0.734226 | 0.665410 | 0.886453 | 0.800191 | 2.427117 | 4.409860 | 57.215390 |
| D3_Balanced | 3 | 0.748237 | 0.703475 | 0.637671 | 0.869278 | 0.774226 | 2.789231 | 5.334879 | 61.616959 |
| D3_Balanced | 4 | 0.739587 | 0.688520 | 0.624368 | 0.853197 | 0.757539 | 3.048702 | 5.950698 | 63.383543 |
| HNS_v1 | 1 | 0.745086 | 0.729030 | 0.680372 | 0.886534 | 0.805559 | 3.572095 | 5.228874 | 54.330763 |
| HNS_v1 | 2 | 0.692517 | 0.671504 | 0.635937 | 0.856866 | 0.778075 | 4.136377 | 6.467222 | 56.384755 |
| HNS_v1 | 3 | 0.658292 | 0.631118 | 0.602780 | 0.835094 | 0.750137 | 4.370582 | 7.493145 | 60.889332 |
| HNS_v1 | 4 | 0.631543 | 0.603681 | 0.587563 | 0.816821 | 0.733756 | 4.769415 | 8.459074 | 61.760906 |

## Interpretation

Observed Score5 crossover intervals: `[{"between_epochs": [1, 2], "delta_before": 0.023784900036034173, "delta_after": -0.12238642595026761}]`. No exact crossing update is inferred.
First epoch with negative HNS-minus-Balanced gap by aggregate: `{"Score5": 2, "J_long3": 1, "J_long": 1, "Short4": 2}`.
First segment with slower relative HNS Score5 growth: `E1->E2`.
Final directional gaps(pp): `{"COCO/I2T": -0.01999999999999602, "COCO/T2I": -0.1599999999999966, "Urban-1k/I2T": 0.09999871253967285, "Urban-1k/T2I": 0.0, "Flickr30k-test1k/I2T": -0.20000000000000284, "Flickr30k-test1k/T2I": -0.11999999999999034, "DOCCI/I2T": -0.04000000000000625, "DOCCI/T2I": -0.23999999999999488, "Long-DCI/I2T": -0.17100762957116444, "Long-DCI/T2I": -0.5787950539331632}`.
`HALF_CONTINUE_TO_2434 = YES`. This is a recommendation only; no Half training starts.
Keep/violation changes and retrieval are observational at discrete epoch points; temporal correspondence alone establishes no causal mechanism. CE share does not establish gradient dominance.

Total replay wall time: 4246.927s; sum of five-set evaluation wall times: 3954.178s.
GPU0 COCO; GPU1 DOCCI; GPU2 Long-DCI; GPU3 Flickr then Urban. No simultaneous checkpoints.
Checkpoint/bare weights, data and raw logs remain local. CHECKPOINT_INVENTORY.json and EXPORT_AUDIT_SUMMARY.json preserve exact paths, hashes, sizes, update/optimizer/cursor/config provenance and strict embedding equality.
All 30 directional recalls per checkpoint are in RESULTS.json and evaluations/. EVAL_COMMANDS.json and evidence/ preserve exact subprocess commands, GPU mapping, UTC, return codes and durations. Optional step500 was not replayed, so it cannot obstruct epoch1-4.

## Direct answers to the scientific questions

Q1. First observed Score5 ranking reversal: `{"between_epochs": [1, 2], "direction": "HNS_to_Balanced", "observed_delta_before_pp": 0.023784900036034173, "observed_delta_after_pp": -0.12238642595026761}`. A crossing interval is reported, never an exact update.
Q2. First negative HNS-minus-Balanced epoch: `{"Score5": 2, "J_long3": 1, "J_long": 1, "Short4": 2}`. A deficit already present at E1 predates the observed epoch window; it is not assigned a made-up reversal step.

Q3. Directional sources (sorted by final deficit):

| Direction | First negative epoch | Final HNS-minus-Balanced(pp) | Gap change across Score5 crossover(pp) |
|---|---:|---:|---:|
| Long-DCI/T2I | 2 | -0.578795 | -0.249934 |
| DOCCI/T2I | 2 | -0.240000 | -0.640000 |
| Flickr30k-test1k/I2T | 2 | -0.200000 | -1.300000 |
| Long-DCI/I2T | 2 | -0.171008 | -0.407787 |
| COCO/T2I | 1 | -0.160000 | -0.024000 |
| Flickr30k-test1k/T2I | 1 | -0.120000 | -0.120000 |
| DOCCI/I2T | 2 | -0.040000 | -0.460000 |
| COCO/I2T | 1 | -0.020000 | +0.540000 |
| Urban-1k/T2I | 1 | +0.000000 | +0.500005 |
| Urban-1k/I2T | 1 | +0.099999 | +0.700003 |

Q4. Does HNS coverage fall faster? Matched epoch-last50, E1 to E4:

| View | Balanced keep change | HNS keep change | HNS falls faster |
|---|---:|---:|---|
| F | -0.067365 | -0.113542 | True |
| Dall | -0.087724 | -0.125349 | True |
| D3 | -0.075145 | -0.092809 | True |

Q5. Temporal correspondence: `{"F": "lower HNS coverage observed before crossover interval", "Dall": "lower HNS coverage observed before crossover interval", "D3": "lower HNS coverage observed before crossover interval"}`. This is observational; no causal claim.
Q6. Relative slow-growth intervals: `["E1->E2", "E3->E4"]`.
Actual negative directional epoch-to-epoch gains: `[{"model": "Balanced", "direction": "COCO/I2T", "between_epochs": [1, 2], "R1_change_pp": -0.28000000000000114}, {"model": "HNS", "direction": "COCO/I2T", "between_epochs": [2, 3], "R1_change_pp": -0.1600000000000037}, {"model": "Balanced", "direction": "COCO/I2T", "between_epochs": [3, 4], "R1_change_pp": -0.46000000000000085}, {"model": "HNS", "direction": "COCO/I2T", "between_epochs": [3, 4], "R1_change_pp": -0.29999999999999716}, {"model": "HNS", "direction": "COCO/T2I", "between_epochs": [3, 4], "R1_change_pp": -0.06400000000000006}, {"model": "Balanced", "direction": "Urban-1k/I2T", "between_epochs": [1, 2], "R1_change_pp": -0.09999871253967285}, {"model": "HNS", "direction": "Urban-1k/I2T", "between_epochs": [3, 4], "R1_change_pp": -0.3000020980834961}, {"model": "HNS", "direction": "Flickr30k-test1k/T2I", "between_epochs": [3, 4], "R1_change_pp": -0.060000000000002274}, {"model": "HNS", "direction": "DOCCI/I2T", "between_epochs": [3, 4], "R1_change_pp": -0.14000000000001478}, {"model": "Balanced", "direction": "DOCCI/T2I", "between_epochs": [3, 4], "R1_change_pp": -0.04000000000000625}, {"model": "HNS", "direction": "DOCCI/T2I", "between_epochs": [3, 4], "R1_change_pp": -0.1599999999999966}, {"model": "Balanced", "direction": "Long-DCI/I2T", "between_epochs": [3, 4], "R1_change_pp": -0.3551696921862728}, {"model": "HNS", "direction": "Long-DCI/I2T", "between_epochs": [3, 4], "R1_change_pp": -0.3551696921862728}, {"model": "HNS", "direction": "Long-DCI/T2I", "between_epochs": [3, 4], "R1_change_pp": -0.18416206261510126}]`. Distinguish these from Balanced simply improving faster.
`HALF_CONTINUE_TO_2434 = YES`. This recommendation combines the original curve with the existing weak Half@1217 signal; no continuation is run.
Matched keep, hard violations, CE/weighted CE and D3 alignment shares are fully preserved in TRAINING_DIAGNOSTICS_COMPARISON.json and SCIENTIFIC_INTERPRETATION.json. No new gradient audit is run.
Four discrete epoch observations from one seed. No exact crossover update, within-interval temporal ordering, causality or statistical significance is inferred. CE share is not a gradient norm.
