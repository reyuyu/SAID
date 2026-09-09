# Phase 2.5 measured results

All grounding values below use identity coordinates, primary tau=0.07. See the protocol document for denominators and fixed gates.

## Implementation checks

```json
{
  "encode_image_legacy_max_diff": 0.0,
  "encode_image_with_patches_legacy_max_diff": 0.0,
  "legacy_forward_train_max_diff": 0.0,
  "diagnostic_max_diff": 0.0,
  "global_max_diff": 0.0,
  "parameter_count": 153513474,
  "old_checkpoint_strict_load": true
}
```

## Matched A/B setup validation

Each of the four ranks has matching streams across the two arms: True. Initial parameters bitwise identical: True.

Dataset JSON SHA-256: `8770e784c9c646b27c4a4d45852795b981c638df9fa89a1bb545c4902fdf2127`.

Saved small maps checked: 38160; max saved-map/logit reconstruction error: 3.696e-09.

The separate native-CLIP/LongCLIP initialization comparison retains inherited text-position precision differences; see the [protocol numerical audit](phase25_local_evidence_router.md#numerical-audit-detail). Same-model production/diagnostic equivalence above is exact.

| Initial direct source | native/production max logit difference | max attention difference | changed peaks |
|---|---:|---:|---:|
| residual | 3.336370e-05 | 1.905596e-06 | 0 |
| attention_delta | 2.333522e-05 | 1.001591e-06 | 0 |

## residual training

| Completed updates | loss_global | loss_route | loss_evidence | loss_said | route_top1_acc | evidence_top1_acc | route_margin | evidence_margin | said_attention_entropy | said_effective_patch_count | said_attention_max | router_input_feature_norm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.330202 | 4.539154 | 0.360588 | 2.449871 | 0.078125 | 0.910156 | 1.647120 | 13.130558 | 4.924033 | 139.582977 | 0.029440 | 7.043737 |
| 200 | 0.196898 | 0.279769 | 0.185735 | 0.232752 | 0.933594 | 0.945312 | 16.564323 | 24.510324 | 4.439131 | 88.109818 | 0.057944 | 6.585340 |
| 400 | 0.144074 | 0.108656 | 0.161325 | 0.134991 | 0.972656 | 0.949219 | 19.941895 | 26.825979 | 4.392760 | 84.158127 | 0.055861 | 6.528618 |
| 659 | 0.128848 | 0.087778 | 0.064135 | 0.075957 | 0.980469 | 0.988281 | 20.368202 | 27.151733 | 4.384935 | 83.592361 | 0.056956 | 6.502772 |

Training elapsed: 397.46 seconds.

## attention_delta training

| Completed updates | loss_global | loss_route | loss_evidence | loss_said | route_top1_acc | evidence_top1_acc | route_margin | evidence_margin | said_attention_entropy | said_effective_patch_count | said_attention_max | router_input_feature_norm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.321205 | 4.732574 | 0.341366 | 2.536970 | 0.140625 | 0.906250 | 1.362338 | 19.128124 | 5.230784 | 187.165649 | 0.009706 | 8.428905 |
| 200 | 0.199351 | 0.644773 | 0.279081 | 0.461927 | 0.855469 | 0.925781 | 15.881435 | 23.843603 | 4.976000 | 147.569046 | 0.019902 | 9.008992 |
| 400 | 0.145141 | 0.190207 | 0.202483 | 0.196345 | 0.941406 | 0.945312 | 20.813059 | 26.167027 | 4.954477 | 144.200424 | 0.020798 | 8.731550 |
| 659 | 0.130245 | 0.175231 | 0.115022 | 0.145127 | 0.945312 | 0.972656 | 21.506258 | 26.416950 | 4.990521 | 149.194458 | 0.020080 | 8.653007 |

Training elapsed: 394.25 seconds.

## Fixed small grounding trajectory

| Candidate | pointing | gt_mass | mass_gain | semantic_mass_excess | switch_margin | switch_positive_fraction | semantic_pointing_excess | target_gt_distractor | localization_margin | orientation semantic gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| residual_initial_direct | 0.247904 | 0.315033 | -0.018680 | -0.014978 | -0.014978 | 0.164557 | -0.006329 | 0.497017 | -0.052645 | -0.019365 |
| residual_initial_router | 0.314990 | 0.325276 | -0.008437 | 0.000181 | 0.000181 | 0.497890 | 0.014768 | 0.505967 | -0.038475 | 0.001921 |
| attention_delta_initial_direct | 0.641509 | 0.357991 | 0.024277 | 0.018517 | 0.018517 | 0.873418 | 0.310127 | 0.525060 | -0.008076 | 0.023186 |
| attention_delta_initial_router | 0.415618 | 0.335310 | 0.001597 | -0.000310 | -0.000310 | 0.493671 | 0.010549 | 0.504773 | -0.034340 | -0.000367 |
| residual_step100_direct | 0.262055 | 0.313346 | -0.020368 | -0.016569 | -0.016569 | 0.168776 | -0.008439 | 0.494033 | -0.054871 | -0.021450 |
| residual_step100_router | 0.190252 | 0.299337 | -0.034376 | -0.003929 | -0.003929 | 0.392405 | -0.010549 | 0.491050 | -0.049845 | -0.003518 |
| attention_delta_step100_direct | 0.655660 | 0.360888 | 0.027175 | 0.020649 | 0.020649 | 0.877637 | 0.345992 | 0.527446 | -0.005205 | 0.025793 |
| attention_delta_step100_router | 0.426101 | 0.336705 | 0.002992 | 0.003321 | 0.003321 | 0.670886 | 0.016878 | 0.508950 | -0.030623 | 0.004328 |
| residual_step200_direct | 0.242662 | 0.303682 | -0.030031 | -0.018606 | -0.018606 | 0.185654 | -0.018987 | 0.489260 | -0.061032 | -0.023612 |
| residual_step200_router | 0.290356 | 0.306650 | -0.027063 | -0.024672 | -0.024672 | 0.189873 | -0.059072 | 0.477327 | -0.066248 | -0.028377 |
| attention_delta_step200_direct | 0.702306 | 0.407245 | 0.073532 | 0.034847 | 0.034847 | 0.852321 | 0.333333 | 0.552506 | 0.023647 | 0.040005 |
| attention_delta_step200_router | 0.473795 | 0.344125 | 0.010412 | 0.026741 | 0.026741 | 0.831224 | 0.099156 | 0.518496 | -0.004523 | 0.026979 |
| residual_step400_direct | 0.248952 | 0.300867 | -0.032847 | -0.019545 | -0.019545 | 0.172996 | 0.000000 | 0.488067 | -0.062972 | -0.024428 |
| residual_step400_router | 0.289832 | 0.301503 | -0.032210 | -0.027683 | -0.027683 | 0.164557 | -0.097046 | 0.479714 | -0.070678 | -0.030763 |
| attention_delta_step400_direct | 0.724843 | 0.409129 | 0.075416 | 0.036573 | 0.036573 | 0.864979 | 0.367089 | 0.552506 | 0.026044 | 0.042287 |
| attention_delta_step400_router | 0.421908 | 0.337328 | 0.003614 | 0.029382 | 0.029382 | 0.831224 | 0.164557 | 0.523270 | -0.005605 | 0.029840 |
| residual_final_direct | 0.255241 | 0.300392 | -0.033321 | -0.019745 | -0.019745 | 0.164557 | 0.004219 | 0.488067 | -0.063453 | -0.024616 |
| residual_final_router | 0.288784 | 0.300613 | -0.033100 | -0.028177 | -0.028177 | 0.168776 | -0.099156 | 0.479714 | -0.071229 | -0.031361 |
| attention_delta_final_direct | 0.726939 | 0.408832 | 0.075119 | 0.036780 | 0.036780 | 0.869198 | 0.373418 | 0.554296 | 0.025921 | 0.042650 |
| attention_delta_final_router | 0.394130 | 0.332046 | -0.001668 | 0.028151 | 0.028151 | 0.843882 | 0.103376 | 0.520883 | -0.008825 | 0.028186 |

## Router versus direct geometry (5,000 phrases per final arm)

| Arm | Spearman | top1 | top5 | top10 | top20 | rho versus mass gain Pearson | rho versus mass gain Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| residual_final | 0.388937 | 0.025600 | 0.105080 | 0.212220 | 0.320320 | -0.318819 | -0.319403 |
| attention_delta_final | -0.179125 | 0.033600 | 0.075160 | 0.096700 | 0.126150 | 0.532418 | 0.528367 |

Quartiles are sorted by per-phrase router/direct Spearman.

| Arm | Quartile | Mean Spearman | Mean router mass gain | n |
|---|---:|---:|---:|---:|
| residual_final | 1 | 0.112419 | -0.012204 | 1250 |
| residual_final | 2 | 0.334014 | -0.023304 | 1250 |
| residual_final | 3 | 0.474621 | -0.035354 | 1250 |
| residual_final | 4 | 0.634693 | -0.058968 | 1250 |
| attention_delta_final | 1 | -0.630011 | -0.031980 | 1250 |
| attention_delta_final | 2 | -0.384804 | -0.020457 | 1250 |
| attention_delta_final | 3 | -0.120448 | 0.003458 | 1250 |
| attention_delta_final | 4 | 0.418765 | 0.050243 | 1250 |

## Full Flickr gate

Triggered: False.

The predefined gate did not pass; no full-val grounding benchmark was run. The separate 5,000-phrase geometry diagnostic is not a full-val replacement.

## Standard COCO retrieval

| Arm | I2T R@1 | R@5 | R@10 | T2I R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|---:|---:|
| residual | 58.54% | 81.36% | 88.50% | 40.17% | 65.90% | 75.53% |
| attention_delta | 58.22% | 81.12% | 88.16% | 39.52% | 65.56% | 75.42% |

## Tests and dashboard

91 passed, 1 skipped, 3 warnings, no errors. The real 4-GPU batch256 smoke passed. Production and pre-merge comparisons have zero observed numerical difference in fp32; all state and gradient checks passed.

The Local-Evidence Router page reads precomputed small artifacts only. It shows four evidence maps and GT with shared scale, phrase metrics, patch-rank overlap and the full small trajectory. Run `streamlit run tools/said_dashboard/app.py --server.address 127.0.0.1 --server.port 8501` and select the page; artifact root is `outputs/local_evidence_router/small`.
