# Dual-Branch Mask Fusion: 500-Update Exploration

Generated UTC: 2026-09-30T10:26:23.484832+00:00

## Outcome

Retain TI-fast as the current reference: no new feasible candidate improves the native long-text summary.

All rankings use step500, seed0. Resource-excluded candidates have no inferred retrieval scores. One seed and previously examined benchmarks do not establish significance or a final best model.

| Candidate | Visual slots | Fusion | Total / added parameters | Formal updates | Resource | Full step mean/P95/max s | Peak allocated GiB | Urban I2T/T2I R1 | DOCCI I2T/T2I R1 | J_long | vs TI pp |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| S-CLS | 1 | stack_pool | 156533762/3545600 | 500 | passed | 1.999/2.116/2.217 | 22.51 | 88.80/86.70 | 75.82/75.48 | 81.700 | -0.580 |
| C-CLS | 1 | crossscore_flat | 156726786/3738624 | 500 | passed | 1.999/2.123/2.300 | 23.39 | 84.00/82.00 | 75.40/75.66 | 79.265 | -3.015 |
| S-PATCH | 196 | stack_pool | 156533762/3545600 | 500 | passed | 2.088/2.214/2.357 | 24.76 | 88.50/87.40 | 76.52/76.22 | 82.160 | -0.120 |
| C-PATCH | 196 | crossscore_flat | 181487106/28498944 | 0 | resource_infeasible | not run | not run | not run | not run | not run | not run |

TI-fast@500 J_long: 82.280%.

## Native Retrieval

Recall is shown as percent; differences are percentage points.

| Dataset | Direction | Metric | TI-fast | T-fast | VCP background | S-CLS | C-CLS | S-PATCH | C-PATCH |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| COCO | I2T | R@1 | 60.04 | 59.00 | 58.58 | 58.78 | 57.70 | 59.60 | not run |
| COCO | I2T | R@5 | 82.12 | 81.48 | 81.18 | 81.46 | 80.84 | 82.20 | not run |
| COCO | I2T | R@10 | 88.76 | 88.42 | 88.34 | 88.34 | 87.86 | 88.58 | not run |
| COCO | T2I | R@1 | 41.15 | 40.44 | 40.52 | 40.41 | 40.01 | 40.89 | not run |
| COCO | T2I | R@5 | 66.68 | 66.23 | 66.15 | 66.05 | 65.60 | 66.47 | not run |
| COCO | T2I | R@10 | 76.57 | 75.81 | 75.89 | 75.86 | 75.36 | 76.14 | not run |
| Urban-1k | I2T | R@1 | 89.20 | 89.30 | 89.40 | 88.80 | 84.00 | 88.50 | not run |
| Urban-1k | I2T | R@5 | 98.10 | 97.80 | 97.60 | 97.60 | 95.70 | 98.20 | not run |
| Urban-1k | I2T | R@10 | 99.20 | 99.20 | 99.30 | 99.10 | 98.20 | 99.20 | not run |
| Urban-1k | T2I | R@1 | 87.10 | 86.50 | 86.10 | 86.70 | 82.00 | 87.40 | not run |
| Urban-1k | T2I | R@5 | 97.90 | 97.70 | 97.70 | 97.90 | 95.40 | 98.00 | not run |
| Urban-1k | T2I | R@10 | 99.00 | 98.80 | 98.70 | 98.80 | 97.10 | 99.00 | not run |
| Flickr30k-test1k | I2T | R@1 | 86.80 | 84.90 | 85.70 | 85.10 | 84.30 | 87.10 | not run |
| Flickr30k-test1k | I2T | R@5 | 97.20 | 97.20 | 97.00 | 97.10 | 96.70 | 97.50 | not run |
| Flickr30k-test1k | I2T | R@10 | 98.90 | 98.70 | 98.80 | 98.70 | 98.30 | 99.10 | not run |
| Flickr30k-test1k | T2I | R@1 | 70.70 | 70.10 | 70.08 | 69.74 | 68.70 | 70.14 | not run |
| Flickr30k-test1k | T2I | R@5 | 90.70 | 90.18 | 90.18 | 90.24 | 89.80 | 90.56 | not run |
| Flickr30k-test1k | T2I | R@10 | 94.82 | 94.38 | 94.32 | 94.32 | 94.34 | 94.76 | not run |
| DOCCI | I2T | R@1 | 76.22 | 75.92 | 75.82 | 75.82 | 75.40 | 76.52 | not run |
| DOCCI | I2T | R@5 | 94.98 | 94.50 | 94.46 | 94.48 | 94.48 | 94.78 | not run |
| DOCCI | I2T | R@10 | 97.40 | 97.44 | 97.42 | 97.38 | 97.42 | 97.52 | not run |
| DOCCI | T2I | R@1 | 76.60 | 75.56 | 75.58 | 75.48 | 75.66 | 76.22 | not run |
| DOCCI | T2I | R@5 | 94.92 | 94.14 | 94.20 | 94.36 | 94.28 | 94.36 | not run |
| DOCCI | T2I | R@10 | 97.48 | 97.16 | 97.32 | 97.26 | 97.28 | 97.52 | not run |

## Factor Comparisons

| Difference | J_long delta pp |
|---|---:|
| C-CLS_minus_S-CLS | -2.435 |
| C-PATCH_minus_S-PATCH | not available |
| S-PATCH_minus_S-CLS | +0.460 |
| C-PATCH_minus_C-CLS | not available |
| S-CLS_minus_TI-fast | -0.580 |
| C-CLS_minus_TI-fast | -3.015 |
| S-PATCH_minus_TI-fast | -0.120 |

## Resource Gates and Exclusions

Each successful gate has 5 warmup and 30 consecutive real-DataLoader full updates. Reported time is the maximum across all four ranks. A failed measured update stops that gate; no slow entries are dropped and no formal training starts for failures.

- S-CLS: passed; gate `{'warmup_steps': 5, 'measured_steps': 30, 'threshold_seconds': 3.0, 'mean_seconds': 1.9807360212008158, 'median_seconds': 1.9684473276138306, 'p95_seconds': 2.127161180973053, 'max_seconds': 2.147155523300171, 'all_steps_at_most_3s': True, 'allocated_limit_gib': 65.0, 'every_rank_peak_allocated_at_most_65gib': True}`; reason `None`.
- C-CLS: passed; gate `{'warmup_steps': 5, 'measured_steps': 30, 'threshold_seconds': 3.0, 'mean_seconds': 2.0234450618426005, 'median_seconds': 1.994078278541565, 'p95_seconds': 2.1720908761024473, 'max_seconds': 2.190122604370117, 'all_steps_at_most_3s': True, 'allocated_limit_gib': 65.0, 'every_rank_peak_allocated_at_most_65gib': True}`; reason `None`.
- S-PATCH: passed; gate `{'warmup_steps': 5, 'measured_steps': 30, 'threshold_seconds': 3.0, 'mean_seconds': 2.071880650520325, 'median_seconds': 2.063446521759033, 'p95_seconds': 2.1682000875473024, 'max_seconds': 2.1734838485717773, 'all_steps_at_most_3s': True, 'allocated_limit_gib': 65.0, 'every_rank_peak_allocated_at_most_65gib': True}`; reason `None`.
- C-PATCH: resource_infeasible; gate `{'warmup_steps': 5, 'measured_steps': 1, 'threshold_seconds': 3.0, 'mean_seconds': 12.620274543762207, 'median_seconds': 12.620274543762207, 'p95_seconds': 12.620274543762207, 'max_seconds': 12.620274543762207, 'all_steps_at_most_3s': False, 'allocated_limit_gib': 65.0, 'every_rank_peak_allocated_at_most_65gib': True}`; reason `measured full update exceeded 3 seconds`.

## Mechanism and Integrity

| Group | Common loss | F/P/R positive keep | F/P/R negative keep | Hard violation | P/R IoU |
|---|---:|---:|---:|---:|---:|
| S-CLS | 9.4848 | 0.9198/0.8916/0.8893 | 0.9198/0.8916/0.8893 | 0.0249 | 0.8626 |
| C-CLS | 10.2478 | 0.5977/0.5977/0.5977 | 0.5977/0.5977/0.5977 | 0.0000 | 1.0000 |
| S-PATCH | 9.1941 | 0.8522/0.8522/0.8522 | 0.8522/0.8522/0.8522 | 0.0000 | 1.0000 |

Initial and steps100/200/300/400/500 logits/saturation, all-open/all-closed ratios, Stack modality pooling mass, CrossScore raw QK moments and replaced-image mask switches are preserved in RESULTS.json. These are mechanism diagnostics, not evidence of semantic grounding.

For completed groups, all 500 updates and every rank/sample/F/P/R/K stream digest match the TI-fast@500 reference. Shared CLIP, B_T, B_V and A_V initial state digests are identical. CrossScore Q/K initial tensors also match between CLS and patch. Optimizer assignment is unique. Strict exports load at optimizer step500 with native image/text embedding error 0.

The new blocks are training-only. Stack uses a scalar mix of modality summaries, not additional token cross-attention. CrossScore uses no softmax, value aggregation or old logits residual. Patch readout remains dense at 24,887,296 weights plus 512 bias parameters.

## Validation Caveats

45 unit/regression cases and 20 two-rank cases passed. Named gradient checks include None states. The existing VCP gradient tolerance is retained (absolute 1.5e-3, relative 1e-4). AdamW differences above 2e-6 are accepted only for gradients below 2e-5 where the exact first-update AdamW formula explains the amplification within FP32 parameter rounding. No gradients or parameters are removed. Theoretical K bias and single-CLS visual Q/K null directions are explicitly distinguished.

Early 8-channel synthetic patch fixtures could yield all-closed masks and very large 1/eps gradients. Those failure logs are retained. A separately labeled nonzero-bias state passed, and final reference tests use 32 channels with the original zero-bias initialization. All four production groups still use 512 channels and the frozen initialization rules.

## Artifacts

Source, configs, reproduction commands, actual training commits, initialization and checkpoint hashes, resource/smoke/strict-export evidence, raw native JSON and execution exit codes are preserved here. Large checkpoints, full token logs, images and caches stay server-local. DCI and Long-DCI are excluded. This pipeline stops after the authorized 500-update exploration; it launches no full epochs, extra seeds, shuffle controls or combined candidates.
