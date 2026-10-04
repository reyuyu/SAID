# No-training gradient composition audit

## RandomK@500

On native backbone total, largest gradient: **R**; most native aligned: **F**; least native aligned: **R**; lowest I2T/T2I cosine: **F**. Lowest cosine denotes relative conflict, not necessarily a negative dot product.

| View | Gradient norm mean | Native cosine mean | Native projection mean | Native negative fraction |
|---|---:|---:|---:|---:|
| F | 12.322644 | 0.802224 | 0.977479 | 0.000000 |
| P | 19.128829 | 0.438777 | 0.851937 | 0.000000 |
| R | 31.830605 | 0.201433 | 0.596778 | 0.000000 |

## Arm B@500

On native backbone total, largest gradient: **S**; most native aligned: **F**; least native aligned: **S**; lowest I2T/T2I cosine: **F**. Lowest cosine denotes relative conflict, not necessarily a negative dot product.

| View | Gradient norm mean | Native cosine mean | Native projection mean | Native negative fraction |
|---|---:|---:|---:|---:|
| F | 10.269918 | 0.796098 | 0.981110 | 0.000000 |
| S | 29.714849 | 0.175815 | 0.626599 | 0.000000 |
| D | 26.149284 | 0.206628 | 0.642748 | 0.000000 |

> Based on gradient evidence, which view should be downweighted or strengthened next? Prefer Full_anchor on random_k with weights [1.5, 0.75, 0.75]. RandomK remainder R aligns worse than prefix P; reducing R is better supported than reducing P. Do not prioritize RandomK prefix downweight. Arm B Full-anchor weights[1.5,0.4,1.1] have additional fixed-state gradient support.

This is a diagnostic recommendation only. No optimizer/scaler step, scheduler update, EMA, checkpoint overwrite or new training occurred. All3×32 main batches use the exact first32 official seed0 epoch0 global training batches; all4 ranks hash unchanged model state before/after every batch. Native Full CE is a diagnostic reference only.

The largest raw gradient routes above do not imply the largest AdamW-preconditioned parameter displacement. Nor do the combined native cosine means imply zero conflicts in every parameter group: text-only negative fractions are reported separately. Within-view native-backbone I2T/T2I cosine is lowest for F in both trained checkpoints; Summary is not the most internally conflicted view on that aggregate. This does not rule out text-specific ambiguity, but it does not support treating Summary T2I conflict as the sole mechanism.

## Scope and correctness

Every objective is independently forward/backwarded through real DDP,4×256, with full1024 candidates. Six CE tensors come directly from production fusion_view_terms locals; production forward supplies combined views, alignment, sparsity, inclusion and total. BF16 encoders and FP32 loss/mask match the training implementation. Python profile callbacks only observe function-return locals. Production sources are unchanged.

Twenty active parameter gradient tensors per objective are identical across4 ranks after every backward. First and last batches repeat F I2T gradients bitwise. Native matrix1024×1024 uses scale100 and diagonal positives; its top1 equals direct ranking of the same embedding matrix. Matrix entries match within FP32 tolerance because local/global GEMM tiling changes rounding.

Native auxiliary gradients are None. N/A is used for cosine/projection in auxiliary-only groups. Native-backbone-total and total-trainable-native-comparable both include only G1+G2; full G8 additionally includes auxiliary gradient norms. Fixed-scale logit_scale has no gradient.

Norm shares below are the share of separately measured weighted norms, not exact linear contribution percentages. Common outer10/3 is omitted only for relative norm/share comparisons. Counterfactual weights use fixed-state gradient linear combinations, not trained candidate models.

Main inclusion_weight=1 represents the mature objective at every checkpoint; the0.5 appendix scales only inclusion. Stage0 does not use initialization-time weight0, to avoid confounding the objective comparison.

## Gradient strength and dominance

| Group | Trainable parameter count |
|---|---|
| G1_vision_backbone | 86192640 |
| G2_text_backbone | 63515649 |
| G3_text_mask_blocks | 3152384 |
| G4_shared_attention_pool | 513 |
| G5_visual_mask_blocks | 3152384 |
| G6_visual_adapter | 393216 |
| G7_balanced_gate | 524288 |

G8 is the sum of these seven disjoint trainable groups. Native backbone total=G1+G2. G3 excludes Shared AttentionPool (G4). Group definitions and parameter-name ownership are encoded in stats.parameter_group.

| Stage | Group | ‖gF‖ | ‖gP/S‖ | ‖gR/D‖ | P/S over F | R/D over F |
|---|---|---|---|---|---|---|
| 0 | G1_vision_backbone | 81.872661 | 83.885098 | 105.130718 | 1.028746 | 1.284209 |
| 0 | G2_text_backbone | 416.928248 | 404.840315 | 379.462392 | 0.973350 | 0.910992 |
| 0 | G3_text_mask_blocks | 13.449855 | 26.182403 | 36.036474 | 1.954139 | 2.686628 |
| 0 | G4_shared_attention_pool | 1.124705 | 1.763174 | 2.718359 | 1.578550 | 2.432006 |
| 0 | G5_visual_mask_blocks | 9.422533 | 17.633083 | 24.409038 | 1.878162 | 2.596762 |
| 0 | G6_visual_adapter | 6.960923 | 12.887785 | 17.705188 | 1.858328 | 2.549806 |
| 0 | G7_balanced_gate | 3.327849 | 7.760871 | 11.138698 | 2.373254 | 3.404654 |
| 0 | G8_total_trainable | 425.329719 | 415.019244 | 397.018018 | 0.978060 | 0.934335 |
| 0 | native_backbone_total | 424.938618 | 413.527783 | 394.041651 | 0.975441 | 0.928166 |
| 0 | total_trainable_native_comparable | 424.938618 | 413.527783 | 394.041651 | 0.975441 | 0.928166 |
| R | G1_vision_backbone | 10.335184 | 16.344878 | 25.714953 | 1.712392 | 2.860804 |
| R | G2_text_backbone | 6.276936 | 9.667363 | 18.437076 | 1.555327 | 2.972363 |
| R | G3_text_mask_blocks | 0.170612 | 0.294066 | 0.427051 | 1.880669 | 2.695357 |
| R | G4_shared_attention_pool | 0.014387 | 0.036398 | 0.055520 | 2.552393 | 3.933476 |
| R | G5_visual_mask_blocks | 0.072627 | 0.141564 | 0.249250 | 1.964562 | 3.468541 |
| R | G6_visual_adapter | 0.076662 | 0.148887 | 0.249693 | 1.958906 | 3.296364 |
| R | G7_balanced_gate | 0.107424 | 0.222951 | 0.425964 | 2.127451 | 4.112814 |
| R | G8_total_trainable | 12.325110 | 19.133830 | 31.838653 | 1.643741 | 2.854598 |
| R | native_backbone_total | 12.322644 | 19.128829 | 31.830605 | 1.643666 | 2.854540 |
| R | total_trainable_native_comparable | 12.322644 | 19.128829 | 31.830605 | 1.643666 | 2.854540 |
| B | G1_vision_backbone | 8.117877 | 25.091069 | 20.435142 | 3.154852 | 2.548787 |
| B | G2_text_backbone | 6.256085 | 15.840297 | 16.211844 | 2.549929 | 2.611173 |
| B | G3_text_mask_blocks | 0.197625 | 0.729870 | 0.591523 | 4.243968 | 3.408707 |
| B | G4_shared_attention_pool | 0.022778 | 0.069913 | 0.050826 | 3.551221 | 2.621723 |
| B | G5_visual_mask_blocks | 0.073943 | 0.301729 | 0.215356 | 4.131169 | 2.954500 |
| B | G6_visual_adapter | 0.086765 | 0.401413 | 0.268085 | 4.773652 | 3.206477 |
| B | G7_balanced_gate | 0.177023 | 0.838224 | 0.566112 | 5.249739 | 3.582299 |
| B | G8_total_trainable | 10.274508 | 29.742386 | 26.166036 | 2.927650 | 2.566916 |
| B | native_backbone_total | 10.269918 | 29.714849 | 26.149284 | 2.926249 | 2.566358 |
| B | total_trainable_native_comparable | 10.269918 | 29.714849 | 26.149284 | 2.926249 | 2.566358 |

| Stage | Group | Raw ratio | mean | median | p10 | p25 | p75 | p90 | max |
|---|---|---|---|---|---|---|---|---|---|
| 0 | G1_vision_backbone | O_over_F | 1.028746 | 1.013019 | 0.930006 | 0.959573 | 1.101661 | 1.138346 | 1.178206 |
| 0 | G1_vision_backbone | E_over_F | 1.284209 | 1.246777 | 1.126487 | 1.167407 | 1.400785 | 1.451953 | 1.699191 |
| 0 | G1_vision_backbone | E_over_O | 1.255707 | 1.227220 | 1.068813 | 1.152359 | 1.343645 | 1.440861 | 1.692329 |
| 0 | G1_vision_backbone | partials_over_F | 2.312956 | 2.312544 | 2.101731 | 2.181744 | 2.412529 | 2.540746 | 2.715744 |
| 0 | G2_text_backbone | O_over_F | 0.973350 | 0.965757 | 0.825973 | 0.920094 | 1.055674 | 1.110261 | 1.182904 |
| 0 | G2_text_backbone | E_over_F | 0.910992 | 0.890698 | 0.819130 | 0.858312 | 0.957543 | 1.028033 | 1.087536 |
| 0 | G2_text_backbone | E_over_O | 0.944260 | 0.936132 | 0.819648 | 0.866172 | 1.017079 | 1.083175 | 1.210777 |
| 0 | G2_text_backbone | partials_over_F | 1.884342 | 1.895389 | 1.706531 | 1.803268 | 1.971023 | 2.066275 | 2.146043 |
| 0 | native_backbone_total | O_over_F | 0.975441 | 0.966113 | 0.828791 | 0.922760 | 1.050446 | 1.109481 | 1.179427 |
| 0 | native_backbone_total | E_over_F | 0.928166 | 0.914562 | 0.843595 | 0.880131 | 0.965834 | 1.045415 | 1.099147 |
| 0 | native_backbone_total | E_over_O | 0.959547 | 0.951162 | 0.832601 | 0.880508 | 1.040421 | 1.094416 | 1.216530 |
| 0 | native_backbone_total | partials_over_F | 1.903607 | 1.913630 | 1.725747 | 1.827367 | 1.983849 | 2.080011 | 2.164170 |
| R | G1_vision_backbone | O_over_F | 1.712392 | 1.758151 | 1.375754 | 1.587375 | 1.845568 | 2.021159 | 2.201984 |
| R | G1_vision_backbone | E_over_F | 2.860804 | 2.721743 | 2.150220 | 2.467588 | 2.933055 | 3.853110 | 7.042191 |
| R | G1_vision_backbone | E_over_O | 1.659384 | 1.605122 | 1.257072 | 1.454897 | 1.749149 | 1.931967 | 3.819663 |
| R | G1_vision_backbone | partials_over_F | 4.573196 | 4.421153 | 3.597951 | 4.150702 | 4.811823 | 6.029978 | 8.885859 |
| R | G2_text_backbone | O_over_F | 1.555327 | 1.556257 | 1.363472 | 1.434289 | 1.651487 | 1.751321 | 2.043587 |
| R | G2_text_backbone | E_over_F | 2.972363 | 2.945730 | 2.525055 | 2.814154 | 3.198018 | 3.316570 | 3.849002 |
| R | G2_text_backbone | E_over_O | 1.917724 | 1.910967 | 1.671581 | 1.801491 | 2.031182 | 2.076447 | 2.395582 |
| R | G2_text_backbone | partials_over_F | 4.527690 | 4.581600 | 3.982143 | 4.243420 | 4.776951 | 4.985053 | 5.892590 |
| R | native_backbone_total | O_over_F | 1.643666 | 1.682202 | 1.373317 | 1.514656 | 1.750597 | 1.928719 | 2.033150 |
| R | native_backbone_total | E_over_F | 2.854540 | 2.840573 | 2.341818 | 2.580892 | 2.900437 | 3.629662 | 5.705578 |
| R | native_backbone_total | E_over_O | 1.729016 | 1.690144 | 1.433097 | 1.548025 | 1.812998 | 1.905682 | 3.368468 |
| R | native_backbone_total | partials_over_F | 4.498205 | 4.487513 | 3.839696 | 4.138984 | 4.664065 | 5.655383 | 7.399397 |
| B | G1_vision_backbone | O_over_F | 3.154852 | 2.992276 | 2.592697 | 2.793473 | 3.436079 | 3.866757 | 5.942400 |
| B | G1_vision_backbone | E_over_F | 2.548787 | 2.474445 | 2.113536 | 2.271244 | 2.793524 | 3.010905 | 4.247231 |
| B | G1_vision_backbone | E_over_O | 0.828010 | 0.826309 | 0.685153 | 0.745273 | 0.880476 | 0.945725 | 1.635873 |
| B | G1_vision_backbone | partials_over_F | 5.703639 | 5.444974 | 4.752369 | 5.145172 | 6.245274 | 6.680421 | 8.824366 |
| B | G2_text_backbone | O_over_F | 2.549929 | 2.496072 | 2.274958 | 2.417469 | 2.699643 | 2.894876 | 3.209915 |
| B | G2_text_backbone | E_over_F | 2.611173 | 2.619024 | 2.357242 | 2.399793 | 2.758678 | 2.977318 | 3.310974 |
| B | G2_text_backbone | E_over_O | 1.025182 | 1.035047 | 0.936113 | 0.986997 | 1.064099 | 1.105409 | 1.181839 |
| B | G2_text_backbone | partials_over_F | 5.161102 | 5.076675 | 4.630242 | 4.825737 | 5.449072 | 5.785982 | 6.520889 |
| B | native_backbone_total | O_over_F | 2.926249 | 2.793771 | 2.570027 | 2.637301 | 3.189592 | 3.449378 | 4.647981 |
| B | native_backbone_total | E_over_F | 2.566358 | 2.483084 | 2.239670 | 2.316802 | 2.693699 | 3.011213 | 3.801197 |
| B | native_backbone_total | E_over_O | 0.889164 | 0.895568 | 0.766550 | 0.806771 | 0.942105 | 0.964285 | 1.479169 |
| B | native_backbone_total | partials_over_F | 5.492607 | 5.237631 | 4.881954 | 5.065313 | 5.863487 | 6.277000 | 7.317800 |

| Stage | Group | F effective norm share | P/S share | R/D share |
|---|---|---|---|---|
| 0 | G1_vision_backbone | 0.302691 | 0.310548 | 0.386761 |
| 0 | G2_text_backbone | 0.347472 | 0.336860 | 0.315669 |
| 0 | native_backbone_total | 0.345119 | 0.335379 | 0.319502 |
| R | G1_vision_backbone | 0.188261 | 0.312705 | 0.499034 |
| R | G2_text_backbone | 0.182153 | 0.281209 | 0.536638 |
| R | native_backbone_total | 0.188278 | 0.302140 | 0.509582 |
| B | G1_vision_backbone | 0.197904 | 0.306545 | 0.495551 |
| B | G2_text_backbone | 0.205762 | 0.260813 | 0.533424 |
| B | native_backbone_total | 0.200459 | 0.290350 | 0.509191 |

## View conflict

| Stage | Group | cos(F,P/S) | cos(F,R/D) | cos(P/S,R/D) |
|---|---|---|---|---|
| 0 | G1_vision_backbone | 0.703151 | 0.631958 | 0.620916 |
| 0 | G2_text_backbone | 0.752376 | 0.831644 | 0.832606 |
| 0 | G3_text_mask_blocks | 0.938250 | 0.915001 | 0.980685 |
| 0 | G4_shared_attention_pool | 0.844516 | 0.716205 | 0.936935 |
| 0 | G5_visual_mask_blocks | 0.972286 | 0.959097 | 0.992050 |
| 0 | G6_visual_adapter | 0.972016 | 0.959492 | 0.991963 |
| 0 | G7_balanced_gate | 0.435887 | 0.428347 | 0.943276 |
| 0 | G8_total_trainable | 0.750111 | 0.816400 | 0.819959 |
| 0 | native_backbone_total | 0.750144 | 0.818549 | 0.818891 |
| 0 | total_trainable_native_comparable | 0.750144 | 0.818549 | 0.818891 |
| R | G1_vision_backbone | 0.511245 | 0.252227 | 0.133170 |
| R | G2_text_backbone | 0.379835 | 0.105358 | 0.024500 |
| R | G3_text_mask_blocks | 0.132735 | 0.079842 | 0.080838 |
| R | G4_shared_attention_pool | 0.276343 | 0.176800 | 0.086240 |
| R | G5_visual_mask_blocks | 0.427025 | 0.194590 | 0.071255 |
| R | G6_visual_adapter | 0.436258 | 0.226661 | 0.105329 |
| R | G7_balanced_gate | 0.270790 | 0.068793 | 0.012696 |
| R | G8_total_trainable | 0.480576 | 0.207642 | 0.105036 |
| R | native_backbone_total | 0.480689 | 0.207680 | 0.105053 |
| R | total_trainable_native_comparable | 0.480689 | 0.207680 | 0.105053 |
| B | G1_vision_backbone | 0.226214 | 0.281780 | 0.122710 |
| B | G2_text_backbone | 0.094043 | 0.153290 | 0.044090 |
| B | G3_text_mask_blocks | 0.222256 | 0.314886 | 0.302465 |
| B | G4_shared_attention_pool | 0.170616 | 0.268879 | 0.197890 |
| B | G5_visual_mask_blocks | 0.183313 | 0.224526 | 0.133278 |
| B | G6_visual_adapter | 0.256297 | 0.271793 | 0.251644 |
| B | G7_balanced_gate | 0.181413 | 0.300675 | 0.285660 |
| B | G8_total_trainable | 0.183156 | 0.232660 | 0.095457 |
| B | native_backbone_total | 0.183088 | 0.232568 | 0.095146 |
| B | total_trainable_native_comparable | 0.183088 | 0.232568 | 0.095146 |

## Native Full alignment

| Stage | Group | cosine F→native | cosine P/S→native | cosine R/D→native |
|---|---|---|---|---|
| 0 | G1_vision_backbone | 0.426899 | 0.249842 | 0.215205 |
| 0 | G2_text_backbone | 0.739893 | 0.450657 | 0.477893 |
| 0 | native_backbone_total | 0.719235 | 0.436822 | 0.457132 |
| R | G1_vision_backbone | 0.820591 | 0.479112 | 0.257134 |
| R | G2_text_backbone | 0.754718 | 0.321459 | 0.079708 |
| R | native_backbone_total | 0.802224 | 0.438777 | 0.201433 |
| B | G1_vision_backbone | 0.821515 | 0.222139 | 0.275884 |
| B | G2_text_backbone | 0.755738 | 0.085209 | 0.101814 |
| B | native_backbone_total | 0.796098 | 0.175815 | 0.206628 |

| Stage | Group | projection_onto_second F→native | projection_onto_second P/S→native | projection_onto_second R/D→native |
|---|---|---|---|---|
| 0 | G1_vision_backbone | 0.975447 | 0.588698 | 0.614759 |
| 0 | G2_text_backbone | 2.434211 | 1.445623 | 1.429872 |
| 0 | native_backbone_total | 2.313336 | 1.373552 | 1.361583 |
| R | G1_vision_backbone | 1.022126 | 0.990685 | 0.774435 |
| R | G2_text_backbone | 0.886545 | 0.584055 | 0.276973 |
| R | native_backbone_total | 0.977479 | 0.851937 | 0.596778 |
| B | G1_vision_backbone | 1.027299 | 0.856701 | 0.854814 |
| B | G2_text_backbone | 0.907425 | 0.261592 | 0.317876 |
| B | native_backbone_total | 0.981110 | 0.626599 | 0.642748 |

| Stage | Group | actual alignment cosine native | actual full objective cosine native |
|---|---|---|---|
| 0 | G1_vision_backbone | 0.330083 | 0.330083 |
| 0 | G2_text_backbone | 0.600110 | 0.600110 |
| 0 | native_backbone_total | 0.581795 | 0.581795 |
| R | G1_vision_backbone | 0.576316 | 0.576316 |
| R | G2_text_backbone | 0.395184 | 0.395184 |
| R | native_backbone_total | 0.523997 | 0.523997 |
| B | G1_vision_backbone | 0.515133 | 0.515133 |
| B | G2_text_backbone | 0.345194 | 0.345194 |
| B | native_backbone_total | 0.452895 | 0.452895 |

## Directional composition

| Stage | View | Group | cos(I2T,T2I) | I2T norm | T2I norm |
|---|---|---|---|---|---|
| 0 | F | G1_vision_backbone | 0.368105 | 46.059605 | 52.929433 |
| 0 | P | G1_vision_backbone | 0.522292 | 52.731508 | 43.356208 |
| 0 | R | G1_vision_backbone | 0.427362 | 69.330874 | 54.815842 |
| 0 | native | G1_vision_backbone | 0.251664 | 16.665857 | 29.061418 |
| 0 | F | G2_text_backbone | 0.830563 | 297.091449 | 135.750172 |
| 0 | P | G2_text_backbone | 0.953168 | 250.345114 | 159.128962 |
| 0 | R | G2_text_backbone | 0.909548 | 248.689060 | 138.987254 |
| 0 | native | G2_text_backbone | 0.390914 | 102.001095 | 46.174049 |
| 0 | F | native_backbone_total | 0.783847 | 300.654831 | 145.888674 |
| 0 | P | native_backbone_total | 0.926920 | 255.937541 | 165.026017 |
| 0 | R | native_backbone_total | 0.854772 | 258.298681 | 149.855792 |
| 0 | native | native_backbone_total | 0.348555 | 103.374874 | 54.837449 |
| R | F | G1_vision_backbone | 0.403088 | 5.622932 | 6.394781 |
| R | P | G1_vision_backbone | 0.496685 | 8.982871 | 9.718496 |
| R | R | G1_vision_backbone | 0.658791 | 14.326847 | 13.773799 |
| R | native | G1_vision_backbone | 0.425713 | 4.381665 | 5.256113 |
| R | F | G2_text_backbone | 0.287425 | 3.739951 | 4.066326 |
| R | P | G2_text_backbone | 0.366944 | 5.920293 | 5.767519 |
| R | R | G2_text_backbone | 0.544986 | 11.041175 | 9.922098 |
| R | native | G2_text_backbone | 0.352581 | 2.943301 | 3.538100 |
| R | F | native_backbone_total | 0.375093 | 6.873453 | 7.688855 |
| R | P | native_backbone_total | 0.463056 | 10.833969 | 11.363541 |
| R | R | native_backbone_total | 0.621734 | 18.200633 | 17.039729 |
| R | native | native_backbone_total | 0.414444 | 5.366546 | 6.441075 |
| B | F | G1_vision_backbone | 0.344762 | 4.604649 | 5.262910 |
| B | S | G1_vision_backbone | 0.571021 | 13.299871 | 14.938629 |
| B | D | G1_vision_backbone | 0.557618 | 11.341589 | 11.777298 |
| B | native | G1_vision_backbone | 0.387369 | 3.526864 | 4.265871 |
| B | F | G2_text_backbone | 0.288313 | 3.777731 | 4.002042 |
| B | S | G2_text_backbone | 0.469683 | 8.993537 | 9.483988 |
| B | D | G2_text_backbone | 0.468677 | 9.857674 | 9.050906 |
| B | native | G2_text_backbone | 0.322360 | 2.953460 | 3.454855 |
| B | F | native_backbone_total | 0.323724 | 5.968177 | 6.625951 |
| B | S | native_backbone_total | 0.539976 | 16.102073 | 17.704573 |
| B | D | native_backbone_total | 0.521735 | 15.071774 | 14.878170 |
| B | native | native_backbone_total | 0.360474 | 4.604234 | 5.498244 |

## Native alignment and conflict distributions

| Stage | Group | View | mean | median | p10 | p25 | p75 | p90 | cos<0 fraction | proj<0 fraction |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | G1_vision_backbone | F | 0.426899 | 0.447830 | 0.290602 | 0.376320 | 0.492339 | 0.524834 | 0.000000 | 0.000000 |
| 0 | G1_vision_backbone | P | 0.249842 | 0.296974 | 0.084227 | 0.162368 | 0.323956 | 0.360022 | 0.031250 | 0.031250 |
| 0 | G1_vision_backbone | R | 0.215205 | 0.199511 | 0.114473 | 0.160629 | 0.269388 | 0.304546 | 0.000000 | 0.000000 |
| 0 | G2_text_backbone | F | 0.739893 | 0.745455 | 0.712750 | 0.722722 | 0.752449 | 0.772789 | 0.000000 | 0.000000 |
| 0 | G2_text_backbone | P | 0.450657 | 0.453627 | 0.417767 | 0.438867 | 0.468303 | 0.477668 | 0.000000 | 0.000000 |
| 0 | G2_text_backbone | R | 0.477893 | 0.478305 | 0.422023 | 0.451206 | 0.503438 | 0.528783 | 0.000000 | 0.000000 |
| 0 | native_backbone_total | F | 0.719235 | 0.721326 | 0.684656 | 0.705686 | 0.735399 | 0.750726 | 0.000000 | 0.000000 |
| 0 | native_backbone_total | P | 0.436822 | 0.439843 | 0.404973 | 0.421328 | 0.453654 | 0.469074 | 0.000000 | 0.000000 |
| 0 | native_backbone_total | R | 0.457132 | 0.460114 | 0.404506 | 0.430015 | 0.481957 | 0.505832 | 0.000000 | 0.000000 |
| R | G1_vision_backbone | F | 0.820591 | 0.817836 | 0.743318 | 0.772010 | 0.857273 | 0.894893 | 0.000000 | 0.000000 |
| R | G1_vision_backbone | P | 0.479112 | 0.460320 | 0.301957 | 0.393522 | 0.528529 | 0.666424 | 0.000000 | 0.000000 |
| R | G1_vision_backbone | R | 0.257134 | 0.211715 | 0.113096 | 0.173275 | 0.287903 | 0.475055 | 0.000000 | 0.000000 |
| R | G2_text_backbone | F | 0.754718 | 0.769492 | 0.701277 | 0.736561 | 0.779029 | 0.791325 | 0.000000 | 0.000000 |
| R | G2_text_backbone | P | 0.321459 | 0.323389 | 0.245389 | 0.293809 | 0.357559 | 0.395650 | 0.000000 | 0.000000 |
| R | G2_text_backbone | R | 0.079708 | 0.075949 | 0.033566 | 0.048917 | 0.119205 | 0.132376 | 0.062500 | 0.062500 |
| R | native_backbone_total | F | 0.802224 | 0.801725 | 0.726950 | 0.757443 | 0.824241 | 0.856037 | 0.000000 | 0.000000 |
| R | native_backbone_total | P | 0.438777 | 0.416826 | 0.301239 | 0.372934 | 0.470154 | 0.588529 | 0.000000 | 0.000000 |
| R | native_backbone_total | R | 0.201433 | 0.162290 | 0.093662 | 0.126701 | 0.209255 | 0.373156 | 0.000000 | 0.000000 |
| B | G1_vision_backbone | F | 0.821515 | 0.828506 | 0.768920 | 0.778545 | 0.857055 | 0.869517 | 0.000000 | 0.000000 |
| B | G1_vision_backbone | S | 0.222139 | 0.223277 | 0.087444 | 0.143496 | 0.277281 | 0.365203 | 0.000000 | 0.000000 |
| B | G1_vision_backbone | D | 0.275884 | 0.281288 | 0.200384 | 0.216542 | 0.317191 | 0.401357 | 0.000000 | 0.000000 |
| B | G2_text_backbone | F | 0.755738 | 0.758646 | 0.707343 | 0.736482 | 0.780815 | 0.791819 | 0.000000 | 0.000000 |
| B | G2_text_backbone | S | 0.085209 | 0.084036 | 0.050704 | 0.065611 | 0.104429 | 0.118905 | 0.000000 | 0.000000 |
| B | G2_text_backbone | D | 0.101814 | 0.096312 | 0.054413 | 0.078272 | 0.125829 | 0.139847 | 0.000000 | 0.000000 |
| B | native_backbone_total | F | 0.796098 | 0.803210 | 0.754171 | 0.763789 | 0.823633 | 0.834924 | 0.000000 | 0.000000 |
| B | native_backbone_total | S | 0.175815 | 0.162338 | 0.094967 | 0.124622 | 0.224812 | 0.275302 | 0.000000 | 0.000000 |
| B | native_backbone_total | D | 0.206628 | 0.210363 | 0.141972 | 0.168879 | 0.236581 | 0.286040 | 0.000000 | 0.000000 |

All other cosine, dot, norm and projection distributions (including every view-pair and within-view direction pair) are in GRADIENT_SUMMARY.json, VIEW_CONFLICT.json, DIRECTION_CONFLICT.json and NATIVE_ALIGNMENT.json. Every one of the96 main batches retains its full small Gram matrix and statistics in RAW_BATCH_GRADIENT_STATS.json.gz.

## Auxiliary gradients

| Stage | Group | sparse norm | inc norm | cos(sparse,align) | cos(inc,align) |
|---|---|---|---|---|---|
| 0 | G1_vision_backbone | 0.000000 | 0.000000 | N/A | N/A |
| 0 | G2_text_backbone | 0.000000 | 0.000000 | N/A | N/A |
| 0 | G3_text_mask_blocks | 0.166323 | 0.087589 | -0.107874 | 0.183907 |
| 0 | G4_shared_attention_pool | 0.078506 | 0.011963 | -0.486731 | 0.775909 |
| 0 | G5_visual_mask_blocks | 0.131881 | 0.061715 | -0.091618 | 0.186120 |
| 0 | G6_visual_adapter | 0.105326 | 0.044206 | -0.095088 | 0.195277 |
| 0 | G7_balanced_gate | 0.086034 | 0.045379 | -0.101718 | 0.166947 |
| 0 | native_backbone_total | 0.000000 | 0.000000 | N/A | N/A |
| R | G1_vision_backbone | 0.000000 | 0.000000 | N/A | N/A |
| R | G2_text_backbone | 0.000000 | 0.000000 | N/A | N/A |
| R | G3_text_mask_blocks | 0.244814 | 0.071452 | -0.078069 | 0.060378 |
| R | G4_shared_attention_pool | 0.081605 | 0.020970 | -0.292644 | 0.292531 |
| R | G5_visual_mask_blocks | 0.339407 | 0.097974 | -0.244966 | 0.244769 |
| R | G6_visual_adapter | 0.465420 | 0.133596 | -0.322979 | 0.320194 |
| R | G7_balanced_gate | 0.081079 | 0.032553 | -0.036274 | 0.039493 |
| R | native_backbone_total | 0.000000 | 0.000000 | N/A | N/A |
| B | G1_vision_backbone | 0.000000 | 0.000000 | N/A | N/A |
| B | G2_text_backbone | 0.000000 | 0.000000 | N/A | N/A |
| B | G3_text_mask_blocks | 0.295519 | 0.052832 | 0.098442 | -0.019870 |
| B | G4_shared_attention_pool | 0.073591 | 0.014482 | 0.078246 | -0.034665 |
| B | G5_visual_mask_blocks | 0.225422 | 0.046302 | -0.014140 | 0.048161 |
| B | G6_visual_adapter | 0.340216 | 0.069202 | 0.089731 | -0.008570 |
| B | G7_balanced_gate | 0.074393 | 0.025290 | -0.033867 | 0.106236 |
| B | native_backbone_total | 0.000000 | 0.000000 | N/A | N/A |

## CE values alongside actual gradient norms

| Stage | Objective | CE mean | native backbone norm | total trainable norm |
|---|---|---|---|---|
| 0 | F_i2t | 2.169403 | 300.654831 | 300.905429 |
| 0 | F_t2i | 1.898499 | 145.888674 | 146.052947 |
| 0 | O_i2t | 2.929572 | 255.937541 | 256.727105 |
| 0 | O_t2i | 2.470518 | 165.026017 | 165.757925 |
| 0 | E_i2t | 5.334885 | 258.298681 | 259.772437 |
| 0 | E_t2i | 4.508635 | 149.855792 | 151.450792 |
| 0 | native_i2t | 0.468928 | 103.374874 | 103.374874 |
| 0 | native_t2i | 0.510069 | 54.837449 | 54.837449 |
| R | F_i2t | 0.056276 | 6.873453 | 6.876183 |
| R | F_t2i | 0.065777 | 7.688855 | 7.689820 |
| R | O_i2t | 0.195999 | 10.833969 | 10.838603 |
| R | O_t2i | 0.254190 | 11.363541 | 11.365697 |
| R | E_i2t | 0.874410 | 18.200633 | 18.208016 |
| R | E_t2i | 0.948272 | 17.039729 | 17.043018 |
| R | native_i2t | 0.049621 | 5.366546 | 5.366546 |
| R | native_t2i | 0.061691 | 6.441075 | 6.441075 |
| B | F_i2t | 0.053951 | 5.968177 | 5.972618 |
| B | F_t2i | 0.061679 | 6.625951 | 6.627493 |
| B | O_i2t | 0.697703 | 16.102073 | 16.122033 |
| B | O_t2i | 0.851287 | 17.704573 | 17.716134 |
| B | E_i2t | 0.457046 | 15.071774 | 15.085846 |
| B | E_t2i | 0.513219 | 14.878170 | 14.883811 |
| B | native_i2t | 0.050431 | 4.604234 | 4.604234 |
| B | native_t2i | 0.062379 | 5.498244 | 5.498244 |

## Actual weighted alignment gradient reconstruction

| Stage | BF16 relative L2 mean | BF16 relative L2 max | BF16 max absolute error | FP32 control relative L2 | FP32 control max abs |
|---|---|---|---|---|---|
| 0 | 0.002600 | 0.003900 | 1.770180 | 0.000001 | 0.000137 |
| R | 0.013875 | 0.025226 | 0.573242 | 0.000003 | 0.000021 |
| B | 0.012162 | 0.017675 | 0.178272 | 0.000003 | 0.000024 |

BF16 casts quantize each independent backward; therefore a linear combination of separately backwarded view gradients cannot be bitwise identical to a single combined backward. Actual errors are retained, not hidden or zeroed. FP32 mask groups pass strict reassembly tolerance on every batch; the real full-FP32 first-batch controls validate linearity for all parameter groups. The main norm/cosine statistics use actual DDP-backwarded gradients, including an independent actual alignment and actual total backward.

## Existing ideas and recommendations

**Idea1_RandomK_Prefix_downweight: NO**. Fixed-state cosine changes on vision/text/native-total: G1_vision_backbone=-0.032814, G2_text_backbone=-0.032108, native_backbone_total=-0.033473.

**Idea2_ArmB_Full_anchor_1.5_0.4_1.1: YES**. Fixed-state cosine changes on vision/text/native-total: G1_vision_backbone=+0.058004, G2_text_backbone=+0.069029, native_backbone_total=+0.061354.

**Idea3_ArmB_1.3_0.4_1.3: YES**. Fixed-state cosine changes on vision/text/native-total: G1_vision_backbone=+0.010176, G2_text_backbone=+0.007903, native_backbone_total=+0.008067.

### Candidate 1

Sampling: random_k; F/P/R or F/S/D weights: **[1.5, 0.75, 0.75]**. Relative to registered weights, local native-backbone alignment cosine changes +0.094907; resulting cosine 0.618826, native projection 8.509186.

Measured raw view norms: {'F': 12.3226437274366, 'P': 19.1288290851351, 'R': 31.83060509291307}; native cosines: {'F': 0.802223797924534, 'P': 0.4387765124982078, 'R': 0.2014327710817999}; native projections: {'F': 0.9774794920984874, 'P': 0.8519372351081185, 'R': 0.5967781390112332}.

At fixed checkpoint, this weighting increases mean alignment-only native-gradient cosine by more than0.01. Actual training may differ; no optimizer update is authorized or launched.

### Candidate 2

Sampling: random_k; F/P/R or F/S/D weights: **[1.2, 1.2, 0.6]**. Relative to registered weights, local native-backbone alignment cosine changes +0.091742; resulting cosine 0.615661, native projection 8.511223.

Measured raw view norms: {'F': 12.3226437274366, 'P': 19.1288290851351, 'R': 31.83060509291307}; native cosines: {'F': 0.802223797924534, 'P': 0.4387765124982078, 'R': 0.2014327710817999}; native projections: {'F': 0.9774794920984874, 'P': 0.8519372351081185, 'R': 0.5967781390112332}.

At fixed checkpoint, this weighting increases mean alignment-only native-gradient cosine by more than0.01. Actual training may differ; no optimizer update is authorized or launched.

### Candidate 3

Sampling: summary_random_detail; F/P/R or F/S/D weights: **[1.5, 0.4, 1.1]**. Relative to registered weights, local native-backbone alignment cosine changes +0.061354; resulting cosine 0.514408, native projection 8.097758.

Measured raw view norms: {'F': 10.269917604214482, 'S': 29.714849396250173, 'D': 26.149283871439145}; native cosines: {'F': 0.7960983256104202, 'S': 0.17581533710232894, 'D': 0.20662804237471588}; native projections: {'F': 0.9811097067470098, 'S': 0.6265988441042295, 'D': 0.6427483428682472}.

At fixed checkpoint, this weighting increases mean alignment-only native-gradient cosine by more than0.01. Actual training may differ; no optimizer update is authorized or launched.

## Why Summary downweighting can help, and its limits

Against the previous equal-weight Summary+RandomDetail run, Arm B improves Score5 by0.355103pp and J_long3 by0.579172pp while preserving Short4 (+0.019pp). Against RandomK it improves Score5 by only0.010927pp and lowers J_long3 by0.471122pp. These are different comparisons; B did not fix the deficit against RandomK.

| B fixed-state group | equal weights native cosine | registered weights native cosine | cosine delta | equal native projection | registered native projection |
|---|---|---|---|---|---|
| G1_vision_backbone | 0.468665 | 0.515449 | 0.046784 | 9.129380 | 9.241855 |
| G2_text_backbone | 0.308500 | 0.345194 | 0.036694 | 4.956311 | 5.424389 |
| native_backbone_total | 0.411917 | 0.453054 | 0.041137 | 7.501523 | 7.748630 |

This same-checkpoint comparison isolates the immediate gradient composition of downweighting S and redistributing strength to F/D. A positive cosine change is compatible with reducing poorly aligned high-norm Summary pressure; it is not proof that this mechanism caused downstream retrieval gains. Projection magnitude can also change, so cosine alone cannot determine the best optimization scale.

Actual total B minus RandomK native-backbone cosine: -0.071102. This compares different learned checkpoints and view constructions, so it does not isolate a causal effect of Summary downweighting. The matched500 results remain: B Score5=69.911321%,J_long3=73.132202%; RandomK69.900394%,73.603324%. A local gradient alignment improvement cannot substitute for downstream long-text validation.

At the same B checkpoint, equal versus registered weights and the proposed anchors are additionally evaluated by fixed-state component-vector combinations. These counterfactuals isolate immediate composition but remain limited by BF16 reassembly rounding and are not optimizer/AdamW update predictions. This audit measures ordinary gradients, not AdamW-preconditioned parameter displacement.

If masked F has poor native alignment, pure view reweighting is insufficient; prioritize train/inference gradient mismatch research. If partial views align positively but dominate norm, a conservative Full anchor is supported as a diagnostic candidate. If total B aligns better while long retrieval stays lower, a simple first-order explanation is incomplete. No gradient-derived weight is claimed to improve retrieval without a separately authorized experiment.

No optimizer/scaler step, EMA, schedule update, new training, full gradient dumps or checkpoint overwrite. All checkpoint SHA256 identities match before and after.

In these32 batches, none of the three main backbone groups has a negative within-view I2T/T2I cosine for F/P/R or F/S/D. RandomK text-backbone R/native cosine is negative on2/32 batches (6.25%); all native-backbone-total view/native cosines are nonnegative. Thus the strongest evidence is poor alignment and partial norm dominance, rather than pervasive directional opposition.
