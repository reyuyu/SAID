# Balanced Stack / Cosine CrossScore: 500-Step Results

Date: 2026-09-30T13:25:51.742180+00:00

This fixed experiment repairs separate modality competition and uncontrolled QK scale. Both new arms retain A3-RandomK, full candidates, losses, optimizer groups, seed and horizon3651. All rankings compare formal step500 only. TI-fast is reused, not retrained.

## Main Summary

| Model | J_long % | Change from original pp | Change from TI pp |
|---|---:|---:|---:|
| TI-fast | 82.280 | reference | +0.000 |
| S-PATCH | 82.160 | reference | -0.120 |
| C-CLS | 79.265 | reference | -3.015 |
| Balanced-Stack-Patch | 82.295 | +0.135 | +0.015 |
| Cosine-CrossScore-CLS | 81.250 | +1.985 | -1.030 |

## Native Retrieval

All recalls are percentages; differences use percentage points. Scores are normalized bare-student image/text embedding inner products, without mask reranking.

| Dataset | Direction | Metric | TI-fast | Original Stack-Patch | Balanced | Original CrossScore-CLS | Cosine |
|---|---|---:|---:|---:|---:|---:|---:|
| COCO | I2T | R@1 | 60.04 | 59.60 | 59.60 | 57.70 | 58.94 |
| COCO | I2T | R@5 | 82.12 | 82.20 | 81.88 | 80.84 | 81.76 |
| COCO | I2T | R@10 | 88.76 | 88.58 | 88.80 | 87.86 | 88.40 |
| COCO | T2I | R@1 | 41.15 | 40.89 | 40.93 | 40.01 | 40.47 |
| COCO | T2I | R@5 | 66.68 | 66.47 | 66.64 | 65.60 | 65.84 |
| COCO | T2I | R@10 | 76.57 | 76.14 | 76.32 | 75.36 | 75.57 |
| Urban-1k | I2T | R@1 | 89.20 | 88.50 | 89.70 | 84.00 | 87.80 |
| Urban-1k | I2T | R@5 | 98.10 | 98.20 | 98.40 | 95.70 | 97.90 |
| Urban-1k | I2T | R@10 | 99.20 | 99.20 | 99.50 | 98.20 | 99.00 |
| Urban-1k | T2I | R@1 | 87.10 | 87.40 | 87.40 | 82.00 | 86.20 |
| Urban-1k | T2I | R@5 | 97.90 | 98.00 | 98.20 | 95.40 | 97.30 |
| Urban-1k | T2I | R@10 | 99.00 | 99.00 | 99.10 | 97.10 | 98.60 |
| Flickr30k-test1k | I2T | R@1 | 86.80 | 87.10 | 86.10 | 84.30 | 85.80 |
| Flickr30k-test1k | I2T | R@5 | 97.20 | 97.50 | 97.50 | 96.70 | 97.10 |
| Flickr30k-test1k | I2T | R@10 | 98.90 | 99.10 | 98.80 | 98.30 | 99.00 |
| Flickr30k-test1k | T2I | R@1 | 70.70 | 70.14 | 70.34 | 68.70 | 69.92 |
| Flickr30k-test1k | T2I | R@5 | 90.70 | 90.56 | 90.66 | 89.80 | 90.30 |
| Flickr30k-test1k | T2I | R@10 | 94.82 | 94.76 | 94.68 | 94.34 | 94.62 |
| DOCCI | I2T | R@1 | 76.22 | 76.52 | 76.18 | 75.40 | 75.44 |
| DOCCI | I2T | R@5 | 94.98 | 94.78 | 94.56 | 94.48 | 94.36 |
| DOCCI | I2T | R@10 | 97.40 | 97.52 | 97.56 | 97.42 | 97.38 |
| DOCCI | T2I | R@1 | 76.60 | 76.22 | 75.90 | 75.66 | 75.56 |
| DOCCI | T2I | R@5 | 94.92 | 94.36 | 94.72 | 94.28 | 94.04 |
| DOCCI | T2I | R@10 | 97.48 | 97.52 | 97.54 | 97.28 | 97.32 |

## Mechanism

Gate diagnostics refer to global valid positive examples. Quantiles and cross-view differences are sampled at updates 1/100/200/300/400/500; logits/saturation and cosine QK include all valid candidate pairs. Image replacement uses one fixed rank-order roll of conditions, without changing scoring labels.

### Balanced-Stack-Patch

| Metric | Step1 | Step500 |
|---|---:|---:|
| F_sigmoid_saturation | 0.001948 | 0.006102 |
| O_sigmoid_saturation | 0.001680 | 0.004899 |
| E_sigmoid_saturation | 0.001467 | 0.015145 |
| F_logit_mean | -0.045948 | 1.103005 |
| F_logit_variance | 0.629170 | 1.041348 |
| F_keep_ratio | 0.466455 | 0.894470 |
| O_keep_ratio | 0.465872 | 0.865427 |
| E_keep_ratio | 0.466316 | 0.865730 |
| hard_inclusion_violation | 0.054660 | 0.021644 |
| oe_iou | 0.726818 | 0.873330 |
| F_replaced_image_mask_switch | 0.136396 | 0.127522 |
| O_replaced_image_mask_switch | 0.130342 | 0.147154 |
| E_replaced_image_mask_switch | 0.131115 | 0.139790 |
| F_g_mean | 0.500000 | 0.373497 |
| F_g_variance | 0.000000 | 0.025758 |
| F_g_q05 | 0.500000 | 0.146968 |
| F_g_q25 | 0.500000 | 0.261627 |
| F_g_q50 | 0.500000 | 0.356944 |
| F_g_q75 | 0.500000 | 0.456083 |
| F_g_q95 | 0.500000 | 0.696295 |
| O_g_mean | 0.500000 | 0.381293 |
| E_g_mean | 0.500000 | 0.384233 |
| g_F_P_abs_difference | 0.000000 | 0.028169 |
| g_F_R_abs_difference | 0.000000 | 0.046812 |
| g_P_R_abs_difference | 0.000000 | 0.054928 |

Balanced changes J_long by +0.135 pp from original Stack-Patch. The gate/condition-switch results above determine whether the modal collapse was repaired; retrieval differences alone do not identify a single causal mechanism.

### Cosine-CrossScore-CLS

| Metric | Step1 | Step500 |
|---|---:|---:|
| F_sigmoid_saturation | 0.000000 | 0.000000 |
| O_sigmoid_saturation | 0.000000 | 0.000000 |
| E_sigmoid_saturation | 0.000000 | 0.000000 |
| F_logit_mean | 0.000043 | 0.397004 |
| F_logit_variance | 0.006177 | 0.216386 |
| F_keep_ratio | 0.499109 | 0.911066 |
| O_keep_ratio | 0.505686 | 0.910238 |
| E_keep_ratio | 0.505590 | 0.909908 |
| hard_inclusion_violation | 0.100903 | 0.012259 |
| oe_iou | 0.643768 | 0.973982 |
| F_replaced_image_mask_switch | 0.341419 | 0.013041 |
| O_replaced_image_mask_switch | 0.328629 | 0.012688 |
| E_replaced_image_mask_switch | 0.326664 | 0.013065 |
| F_qk_mean | 0.011458 | -0.605996 |
| F_qk_variance | 0.013415 | 0.000038 |
| F_qk_abs_max | 0.603035 | 0.644106 |
| O_qk_abs_max | 0.603599 | 0.647653 |
| E_qk_abs_max | 0.572733 | 0.647375 |

Cosine changes J_long by +1.985 pp from original CrossScore-CLS. The bounded QK and saturation measurements test the numerical mechanism directly. This arm also uses the requested standard Linear readout initialization, including a uniform bias instead of the original zero bias, so improvement is not a pure normalization-only ablation.

## Cost and Verification

| Arm | Full-update mean/median/P95/max s | Peak allocated/reserved GiB | Training-loop minutes |
|---|---:|---:|---:|
| Balanced-Stack-Patch | 2.103/2.093/2.203/2.284 | 27.76/28.45 | 17.97 |
| Cosine-CrossScore-CLS | 2.010/1.992/2.121/2.211 | 23.44/23.79 | 17.20 |

The real-DataLoader gate runs 5 warmup + 30 consecutive measured updates, each <=3s. Each successful arm has separate 5-step four-rank smoke, formal step0/100/200/300/400/500 checkpoints and uninterrupted logs. Every rank/sample/F/P/R/K stream matches TI-fast@500. CLIP/B_T/B_V/A_V initialization digests match original counterparts; cosine Q/K match C-CLS. Final rank parameter difference is zero; losses and gradients are finite. Strict native export loads at optimizer step500 and has image/text embedding maximum absolute error zero.

## Limitations and Delivery

This is a single-seed exploration on previously inspected benchmarks, not evidence of statistical significance or a final best architecture. Restoring condition sensitivity does not itself prove semantic grounding. No temperature, learning rate, rank or mechanism sweep was performed. No mixed Balanced/Cosine architecture or 3651-step continuation was launched.

Raw JSON, runtime configs, initialization/checkpoint hashes, commands, exit codes, smoke/gate acceptance and sampled mechanism data are in RESULTS.json and evidence/. Large model weights, data and full token logs remain on the server. COCO uses 5000/25000 candidates and similarity chunk512; Urban uses1000 pairs; Flickr test1K uses1000/5000; DOCCI uses5000 pairs. Image batch64. DCI and Long-DCI are excluded.
