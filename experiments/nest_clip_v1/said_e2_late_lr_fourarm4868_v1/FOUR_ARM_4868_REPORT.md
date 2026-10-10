# E2-Uniform late LR four-arm epoch4 experiment

All four multipliers were fixed before training. Every arm independently resumes the same E2@3651.
Results are exploratory: public benchmarks were repeatedly observed; no trustworthy independent, deduplicated validation protocol was available.

| Model | BB | Text mask | Visual mask | Adapter | Score5 | J_long3 | J_long | Short4 | Urban I2T/T2I | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| E2-Uniform | 1.0 | 1.0 | 1.0 | 1.0 | 73.812187 | 78.193644 | 87.055002 | 67.240000 | 93.900 / 92.900 | 93.400 |
| B1-BB085 | 0.85 | 1.0 | 1.0 | 1.0 | 73.791848 | 78.185747 | 87.030002 | 67.201000 | 93.900 / 92.900 | 93.400 |
| B2-BB115 | 1.15 | 1.0 | 1.0 | 1.0 | 73.769663 | 78.162772 | 87.035002 | 67.180000 | 93.900 / 92.900 | 93.400 |
| B3-MASK085 | 1.0 | 0.85 | 0.85 | 1.0 | 73.758809 | 78.169349 | 87.035002 | 67.143000 | 93.900 / 92.900 | 93.400 |
| B4-MASK115 | 1.0 | 1.15 | 1.15 | 1.0 | 73.807387 | 78.186978 | 87.045002 | 67.238000 | 93.900 / 92.900 | 93.400 |

## B1-BB085

Delta vs E2 (pp): `{"Score5": -0.020338226782428137, "J_long3": -0.007897044637388717, "J_long": -0.025000000000005684, "Short4": -0.03900000000000148, "Urban_I2T": 0.0, "Urban_T2I": 0.0}`
Checkpoint SHA256: `f925f3858e12f9f00f7db06f0ce66f06907f6988fb3a8d39feee699df1f94bbd`.

| Dataset | I2T R@1/5/10 (%) | T2I R@1/5/10 (%) | Δ I2T R@1/5/10 (pp) | Δ T2I R@1/5/10 (pp) |
|---|---|---|---|---|
| COCO | 61.880000 / 83.740000 / 89.760000 | 42.984000 / 68.724000 / 78.192000 | +0.020000 / +0.080000 / +0.060000 | -0.016000 / -0.016000 / +0.032000 |
| Urban-1k | 93.900007 / 99.200004 / 99.600005 | 92.900002 / 99.100006 / 99.400002 | +0.000000 / +0.099999 / +0.000000 | +0.000000 / +0.000000 / +0.000000 |
| Flickr30k-test1k | 89.900000 / 98.500000 / 99.300000 | 74.040000 / 92.100000 / 95.780000 | -0.200000 / +0.000000 / +0.000000 | +0.040000 / +0.020000 / -0.040000 |
| DOCCI | 80.140000 / 96.420000 / 98.580000 | 81.180000 / 96.160000 / 98.420000 | -0.060000 / +0.000000 / +0.020000 | -0.040000 / -0.020000 / +0.000000 |
| Long-DCI | 60.286767 / 78.150487 / 83.938437 | 60.707708 / 78.729282 / 83.714812 | +0.013154 / -0.039463 / +0.013154 | +0.039463 / +0.026309 / +0.013154 |

## B2-BB115

Delta vs E2 (pp): `{"Score5": -0.042523546435148774, "J_long3": -0.030872577391917844, "J_long": -0.020000000000010232, "Short4": -0.05999999999998806, "Urban_I2T": 0.0, "Urban_T2I": 0.0}`
Checkpoint SHA256: `30122a42a0e5102ab6b105ab1c7091cd082350c65e30bb003f7f5920b4313041`.

| Dataset | I2T R@1/5/10 (%) | T2I R@1/5/10 (%) | Δ I2T R@1/5/10 (pp) | Δ T2I R@1/5/10 (pp) |
|---|---|---|---|---|
| COCO | 61.860000 / 83.680000 / 89.840000 | 43.000000 / 68.708000 / 78.152000 | +0.000000 / +0.020000 / +0.140000 | +0.000000 / -0.032000 / -0.008000 |
| Urban-1k | 93.900007 / 99.200004 / 99.600005 | 92.900002 / 99.100006 / 99.400002 | +0.000000 / +0.099999 / +0.000000 | +0.000000 / +0.000000 / +0.000000 |
| Flickr30k-test1k | 89.900000 / 98.500000 / 99.300000 | 73.960000 / 92.040000 / 95.840000 | -0.200000 / +0.000000 / +0.000000 | -0.040000 / -0.040000 / +0.020000 |
| DOCCI | 80.160000 / 96.380000 / 98.560000 | 81.180000 / 96.140000 / 98.420000 | -0.040000 / -0.040000 / +0.000000 | -0.040000 / -0.040000 / +0.000000 |
| Long-DCI | 60.220994 / 78.203104 / 83.925283 | 60.615627 / 78.676664 / 83.675349 | -0.052618 / +0.013154 / +0.000000 | -0.052618 / -0.026309 / -0.026309 |

## B3-MASK085

Delta vs E2 (pp): `{"Score5": -0.05337721652196592, "J_long3": -0.02429536086994233, "J_long": -0.020000000000010232, "Short4": -0.0969999999999942, "Urban_I2T": 0.0, "Urban_T2I": 0.0}`
Checkpoint SHA256: `f74d902bb537a0c6084887171e463b5962fc59352d2d8a2f49e6c3101b0bc4d4`.

| Dataset | I2T R@1/5/10 (%) | T2I R@1/5/10 (%) | Δ I2T R@1/5/10 (pp) | Δ T2I R@1/5/10 (pp) |
|---|---|---|---|---|
| COCO | 61.740000 / 83.580000 / 89.760000 | 43.012000 / 68.732000 / 78.160000 | -0.120000 / -0.080000 / +0.060000 | +0.012000 / -0.008000 / +0.000000 |
| Urban-1k | 93.900007 / 99.200004 / 99.600005 | 92.900002 / 99.100006 / 99.400002 | +0.000000 / +0.099999 / +0.000000 | +0.000000 / +0.000000 / +0.000000 |
| Flickr30k-test1k | 89.800000 / 98.500000 / 99.300000 | 74.020000 / 92.080000 / 95.800000 | -0.300000 / +0.000000 / +0.000000 | +0.020000 / +0.000000 / -0.020000 |
| DOCCI | 80.180000 / 96.460000 / 98.560000 | 81.160000 / 96.200000 / 98.380000 | -0.020000 / +0.040000 / +0.000000 | -0.060000 / +0.020000 / -0.040000 |
| Long-DCI | 60.194686 / 78.163641 / 83.938437 | 60.681400 / 78.676664 / 83.727966 | -0.078927 / -0.026309 / +0.013154 | +0.013154 / -0.026309 / +0.026309 |

## B4-MASK115

Delta vs E2 (pp): `{"Score5": -0.004800000000003024, "J_long3": -0.006666666666660603, "J_long": -0.010000000000005116, "Short4": -0.001999999999995339, "Urban_I2T": 0.0, "Urban_T2I": 0.0}`
Checkpoint SHA256: `79b18294b6c66d3f5b8842b64f086fdad902bc2c04b8e15bcfdb65855fab7f34`.

| Dataset | I2T R@1/5/10 (%) | T2I R@1/5/10 (%) | Δ I2T R@1/5/10 (pp) | Δ T2I R@1/5/10 (pp) |
|---|---|---|---|---|
| COCO | 61.820000 / 83.680000 / 89.820000 | 43.032000 / 68.712000 / 78.180000 | -0.040000 / +0.020000 / +0.120000 | +0.032000 / -0.028000 / +0.020000 |
| Urban-1k | 93.900007 / 99.100006 / 99.600005 | 92.900002 / 99.100006 / 99.400002 | +0.000000 / +0.000000 / +0.000000 | +0.000000 / +0.000000 / +0.000000 |
| Flickr30k-test1k | 90.100000 / 98.600000 / 99.300000 | 74.000000 / 92.080000 / 95.820000 | +0.000000 / +0.100000 / +0.000000 | +0.000000 / +0.000000 / +0.000000 |
| DOCCI | 80.160000 / 96.480000 / 98.560000 | 81.220000 / 96.220000 / 98.420000 | -0.040000 / +0.060000 / +0.000000 | +0.000000 / +0.040000 / +0.000000 |
| Long-DCI | 60.234149 / 78.203104 / 83.938437 | 60.707708 / 78.663510 / 83.662194 | -0.039463 / +0.013154 / +0.013154 | +0.039463 / -0.039463 / -0.039463 |

Urban 0.1pp is one query; single-seed differences are not established statistical gains.
Lower mask density/violation alone does not prove better evidence selection.
No fifth arm, new seed, new multiplier, checkpoint fusion or epoch5 is authorized.


## Original E2@4868 complete recall reference

| Dataset | I2T R@1/5/10 (%) | T2I R@1/5/10 (%) |
|---|---|---|
| COCO | 61.860000 / 83.660000 / 89.700000 | 43.000000 / 68.740000 / 78.160000 |
| Urban-1k | 93.900007 / 99.100006 / 99.600005 | 92.900002 / 99.100006 / 99.400002 |
| Flickr30k-test1k | 90.100000 / 98.500000 / 99.300000 | 74.000000 / 92.080000 / 95.820000 |
| DOCCI | 80.200000 / 96.420000 / 98.560000 | 81.220000 / 96.180000 / 98.420000 |
| Long-DCI | 60.273612 / 78.189950 / 83.925283 | 60.668245 / 78.702973 / 83.701657 |

## Final scientific and engineering review

This experiment reports all four arms without selecting a new formal model. The held-out candidate screen found one caption duplicated in training; image-content and semantic overlap remain unverified. The public five-set results cannot establish unbiased model selection or a new SOTA.

| Arm | Δ Score5 | Δ J_long3 | Δ J_long | Δ Short4 | Δ Urban I2T | Δ Urban T2I | Δ Urban Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1-BB085 | -0.020338 | -0.007897 | -0.025000 | -0.039000 | +0.000000 | +0.000000 | +0.000000 |
| B2-BB115 | -0.042524 | -0.030873 | -0.020000 | -0.060000 | +0.000000 | +0.000000 | +0.000000 |
| B3-MASK085 | -0.053377 | -0.024295 | -0.020000 | -0.097000 | +0.000000 | +0.000000 | +0.000000 |
| B4-MASK115 | -0.004800 | -0.006667 | -0.010000 | -0.002000 | +0.000000 | +0.000000 | +0.000000 |

All deltas are percentage points. Urban is reported to three decimal places in the absolute table; 0.100pp is one correct query.

### Last50 structure compared with original E2

| Model | F keep | Dall keep | D3 keep | DF violation | 3D violation | DF IoU | 3D IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| E2-Uniform | 0.557932 | 0.583648 | 0.577811 | 0.069411 | 0.087991 | 0.818489 | 0.732320 |
| B1-BB085 | 0.558511 | 0.584010 | 0.578658 | 0.069019 | 0.088211 | 0.819394 | 0.732598 |
| B2-BB115 | 0.557329 | 0.583875 | 0.577352 | 0.070019 | 0.087625 | 0.817915 | 0.732378 |
| B3-MASK085 | 0.557729 | 0.584264 | 0.578986 | 0.070005 | 0.087802 | 0.818005 | 0.733847 |
| B4-MASK115 | 0.557481 | 0.583135 | 0.577020 | 0.069140 | 0.087820 | 0.819007 | 0.732162 |

Original E2 density-order reversal updates: 1217/1217. Interpret arm reversals relative to this existing structure, rather than treating the ordering alone as an engineering failure.

### B1-BB085

Both Urban directions improve: False. Score5 and Urban Mean both improve: False.
R@1 regressions (dataset, direction, pp): [["COCO", "T2I", -0.015999999999999348], ["Flickr30k-test1k", "I2T", -0.20000000000000018], ["DOCCI", "I2T", -0.060000000000004494], ["DOCCI", "T2I", -0.0400000000000067]].
Last50 F/Dall/D3 keep: 0.558511 / 0.584010 / 0.578658.
Last50 DF/3D hard violations and IoU: 0.069019 / 0.088211 / 0.819394 / 0.732598.
Density order maintained in last50 mean: False; reversal updates: 1217/1217.
Gradient diagnostics (weighted norms and hierarchy/sparsity cosine):

| Group | Alignment | Sparsity | Hierarchy | Total | H/S cosine |
|---|---:|---:|---:|---:|---:|
| native_text_backbone | 40.776676 | 0.000000 | 0.000000 | 40.776676 | N/A |
| native_visual_backbone | 60.339031 | 0.000000 | 0.000000 | 60.339031 | N/A |
| text_mask_shared_pool | 0.799457 | 0.206066 | 0.010505 | 0.747106 | 0.250202 |
| visual_mask | 0.557827 | 0.169607 | 0.001631 | 0.539505 | -0.356535 |
| fusion_adapter_gate | 1.648766 | 0.264127 | 0.005157 | 1.646571 | -0.075548 |

Resource summary: {"updates": 1217, "full_cycle_seconds": {"min": 1.6910215616226196, "median": 2.131960391998291, "mean": 2.181735290987333, "max": 28.904903411865234}, "actual_first_update_preoptimization": {"loss_difference": 0.0, "max_group_gradient_norm_difference": 0.0, "loss_and_group_gradient_norms_exact": true, "note": "Norm equality supplements the saved full-vector preupdate probe; it does not establish vector equality alone."}, "GPU_peak_allocated_GiB": {"0": 27.758377075195312, "1": 27.758377075195312, "2": 27.758222579956055, "3": 27.758377075195312}, "errors": [], "nonfinite_updates": 0, "mask_all_closed_fraction_max": {"F": 0.0, "Dall": 0.0, "D3": 0.0}, "mask_all_open_fraction_max": {"F": 0.0, "Dall": 0.0, "D3": 0.0}, "density_order_reversal_updates": 1217}.

### B2-BB115

Both Urban directions improve: False. Score5 and Urban Mean both improve: False.
R@1 regressions (dataset, direction, pp): [["Flickr30k-test1k", "I2T", -0.20000000000000018], ["Flickr30k-test1k", "T2I", -0.039999999999995595], ["DOCCI", "I2T", -0.0400000000000067], ["DOCCI", "T2I", -0.0400000000000067], ["Long-DCI", "I2T", -0.052617732175741505], ["Long-DCI", "T2I", -0.052617732175741505]].
Last50 F/Dall/D3 keep: 0.557329 / 0.583875 / 0.577352.
Last50 DF/3D hard violations and IoU: 0.070019 / 0.087625 / 0.817915 / 0.732378.
Density order maintained in last50 mean: False; reversal updates: 1217/1217.
Gradient diagnostics (weighted norms and hierarchy/sparsity cosine):

| Group | Alignment | Sparsity | Hierarchy | Total | H/S cosine |
|---|---:|---:|---:|---:|---:|
| native_text_backbone | 39.450787 | 0.000000 | 0.000000 | 39.450787 | N/A |
| native_visual_backbone | 59.827496 | 0.000000 | 0.000000 | 59.827496 | N/A |
| text_mask_shared_pool | 0.720947 | 0.210119 | 0.010541 | 0.665857 | 0.233906 |
| visual_mask | 0.515469 | 0.171435 | 0.001619 | 0.498082 | -0.345755 |
| fusion_adapter_gate | 1.509684 | 0.264827 | 0.005094 | 1.509064 | -0.074971 |

Resource summary: {"updates": 1217, "full_cycle_seconds": {"min": 1.7049504518508911, "median": 2.1280479431152344, "mean": 2.1761720103128286, "max": 25.679576873779297}, "actual_first_update_preoptimization": {"loss_difference": 0.0, "max_group_gradient_norm_difference": 0.0, "loss_and_group_gradient_norms_exact": true, "note": "Norm equality supplements the saved full-vector preupdate probe; it does not establish vector equality alone."}, "GPU_peak_allocated_GiB": {"0": 27.758377075195312, "1": 27.758377075195312, "2": 27.758222579956055, "3": 27.758377075195312}, "errors": [], "nonfinite_updates": 0, "mask_all_closed_fraction_max": {"F": 0.0, "Dall": 0.0, "D3": 0.0}, "mask_all_open_fraction_max": {"F": 0.0, "Dall": 0.0, "D3": 0.0}, "density_order_reversal_updates": 1217}.

### B3-MASK085

Both Urban directions improve: False. Score5 and Urban Mean both improve: False.
R@1 regressions (dataset, direction, pp): [["COCO", "I2T", -0.12000000000000899], ["Flickr30k-test1k", "I2T", -0.30000000000000027], ["DOCCI", "I2T", -0.0200000000000089], ["DOCCI", "T2I", -0.060000000000004494], ["Long-DCI", "I2T", -0.07892659826361781]].
Last50 F/Dall/D3 keep: 0.557729 / 0.584264 / 0.578986.
Last50 DF/3D hard violations and IoU: 0.070005 / 0.087802 / 0.818005 / 0.733847.
Density order maintained in last50 mean: False; reversal updates: 1217/1217.
Gradient diagnostics (weighted norms and hierarchy/sparsity cosine):

| Group | Alignment | Sparsity | Hierarchy | Total | H/S cosine |
|---|---:|---:|---:|---:|---:|
| native_text_backbone | 41.354443 | 0.000000 | 0.000000 | 41.354443 | N/A |
| native_visual_backbone | 61.986294 | 0.000000 | 0.000000 | 61.986294 | N/A |
| text_mask_shared_pool | 0.858508 | 0.209915 | 0.010481 | 0.801401 | 0.225865 |
| visual_mask | 0.591795 | 0.172042 | 0.001625 | 0.572360 | -0.353011 |
| fusion_adapter_gate | 1.865561 | 0.266271 | 0.005119 | 1.863000 | -0.074642 |

Resource summary: {"updates": 1217, "full_cycle_seconds": {"min": 1.6967253684997559, "median": 2.1277916431427, "mean": 2.178219188699589, "max": 26.279817581176758}, "actual_first_update_preoptimization": {"loss_difference": 0.0, "max_group_gradient_norm_difference": 0.0, "loss_and_group_gradient_norms_exact": true, "note": "Norm equality supplements the saved full-vector preupdate probe; it does not establish vector equality alone."}, "GPU_peak_allocated_GiB": {"0": 27.758377075195312, "1": 27.758377075195312, "2": 27.758377075195312, "3": 27.758377075195312}, "errors": [], "nonfinite_updates": 0, "mask_all_closed_fraction_max": {"F": 0.0, "Dall": 0.0, "D3": 0.0}, "mask_all_open_fraction_max": {"F": 0.0, "Dall": 0.0, "D3": 0.0}, "density_order_reversal_updates": 1217}.

### B4-MASK115

Both Urban directions improve: False. Score5 and Urban Mean both improve: False.
R@1 regressions (dataset, direction, pp): [["COCO", "I2T", -0.0400000000000067], ["DOCCI", "I2T", -0.0400000000000067], ["Long-DCI", "I2T", -0.03946329913180335]].
Last50 F/Dall/D3 keep: 0.557481 / 0.583135 / 0.577020.
Last50 DF/3D hard violations and IoU: 0.069140 / 0.087820 / 0.819007 / 0.732162.
Density order maintained in last50 mean: False; reversal updates: 1217/1217.
Gradient diagnostics (weighted norms and hierarchy/sparsity cosine):

| Group | Alignment | Sparsity | Hierarchy | Total | H/S cosine |
|---|---:|---:|---:|---:|---:|
| native_text_backbone | 40.303486 | 0.000000 | 0.000000 | 40.303486 | N/A |
| native_visual_backbone | 57.174328 | 0.000000 | 0.000000 | 57.174328 | N/A |
| text_mask_shared_pool | 0.809291 | 0.202121 | 0.010535 | 0.756259 | 0.253291 |
| visual_mask | 0.579834 | 0.167655 | 0.001598 | 0.560119 | -0.345241 |
| fusion_adapter_gate | 1.736442 | 0.260942 | 0.005136 | 1.733501 | -0.071759 |

Resource summary: {"updates": 1217, "full_cycle_seconds": {"min": 1.5939847230911255, "median": 2.127291679382324, "mean": 2.183196260680306, "max": 28.979555130004883}, "actual_first_update_preoptimization": {"loss_difference": 0.0, "max_group_gradient_norm_difference": 0.0, "loss_and_group_gradient_norms_exact": true, "note": "Norm equality supplements the saved full-vector preupdate probe; it does not establish vector equality alone."}, "GPU_peak_allocated_GiB": {"0": 27.758377075195312, "1": 27.758377075195312, "2": 27.758377075195312, "3": 27.758222579956055}, "errors": [], "nonfinite_updates": 0, "mask_all_closed_fraction_max": {"F": 0.0, "Dall": 0.0, "D3": 0.0}, "mask_all_open_fraction_max": {"F": 0.0, "Dall": 0.0, "D3": 0.0}, "density_order_reversal_updates": 1217}.

### LR-axis trends and limits

backbone: {"Score5": {"lower": 73.79184841503438, "baseline": 73.81218664181681, "higher": 73.76966309538166, "higher_minus_lower_pp": -0.022185319652720636, "monotonic_increase": false, "monotonic_decrease": false, "observed_pattern": "BASELINE_LOCAL_PEAK"}, "J_long3": {"lower": 78.18574735839063, "baseline": 78.19364440302802, "higher": 78.1627718256361, "higher_minus_lower_pp": -0.022975532754529127, "monotonic_increase": false, "monotonic_decrease": false, "observed_pattern": "BASELINE_LOCAL_PEAK"}, "J_long": {"lower": 87.03000225305557, "baseline": 87.05500225305558, "higher": 87.03500225305557, "higher_minus_lower_pp": 0.0049999999999954525, "monotonic_increase": false, "monotonic_decrease": false, "observed_pattern": "BASELINE_LOCAL_PEAK"}, "Short4": {"lower": 67.201, "baseline": 67.24, "higher": 67.18, "higher_minus_lower_pp": -0.020999999999986585, "monotonic_increase": false, "monotonic_decrease": false, "observed_pattern": "BASELINE_LOCAL_PEAK"}, "Urban_I2T": {"lower": 93.90000700950623, "baseline": 93.90000700950623, "higher": 93.90000700950623, "higher_minus_lower_pp": 0.0, "monotonic_increase": true, "monotonic_decrease": true, "observed_pattern": "UNCHANGED"}, "Urban_T2I": {"lower": 92.90000200271606, "baseline": 92.90000200271606, "higher": 92.90000200271606, "higher_minus_lower_pp": 0.0, "monotonic_increase": true, "monotonic_decrease": true, "observed_pattern": "UNCHANGED"}, "Urban_Mean": {"lower": 93.40000450611115, "baseline": 93.40000450611115, "higher": 93.40000450611115, "higher_minus_lower_pp": 0.0, "monotonic_increase": true, "monotonic_decrease": true, "observed_pattern": "UNCHANGED"}}.
mask: {"Score5": {"lower": 73.75880942529484, "baseline": 73.81218664181681, "higher": 73.8073866418168, "higher_minus_lower_pp": 0.048577216521962896, "monotonic_increase": false, "monotonic_decrease": false, "observed_pattern": "BASELINE_LOCAL_PEAK"}, "J_long3": {"lower": 78.16934904215807, "baseline": 78.19364440302802, "higher": 78.18697773636136, "higher_minus_lower_pp": 0.017628694203281725, "monotonic_increase": false, "monotonic_decrease": false, "observed_pattern": "BASELINE_LOCAL_PEAK"}, "J_long": {"lower": 87.03500225305557, "baseline": 87.05500225305558, "higher": 87.04500225305557, "higher_minus_lower_pp": 0.010000000000005116, "monotonic_increase": false, "monotonic_decrease": false, "observed_pattern": "BASELINE_LOCAL_PEAK"}, "Short4": {"lower": 67.143, "baseline": 67.24, "higher": 67.238, "higher_minus_lower_pp": 0.09499999999999886, "monotonic_increase": false, "monotonic_decrease": false, "observed_pattern": "BASELINE_LOCAL_PEAK"}, "Urban_I2T": {"lower": 93.90000700950623, "baseline": 93.90000700950623, "higher": 93.90000700950623, "higher_minus_lower_pp": 0.0, "monotonic_increase": true, "monotonic_decrease": true, "observed_pattern": "UNCHANGED"}, "Urban_T2I": {"lower": 92.90000200271606, "baseline": 92.90000200271606, "higher": 92.90000200271606, "higher_minus_lower_pp": 0.0, "monotonic_increase": true, "monotonic_decrease": true, "observed_pattern": "UNCHANGED"}, "Urban_Mean": {"lower": 93.40000450611115, "baseline": 93.40000450611115, "higher": 93.40000450611115, "higher_minus_lower_pp": 0.0, "monotonic_increase": true, "monotonic_decrease": true, "observed_pattern": "UNCHANGED"}}.

### Experimental outcome

All four arms leave Urban I2T/T2I R@1 and Urban Mean unchanged. Every arm has lower Score5 than original E2. No arm meets the proposed joint improvement objective; no new formal model is selected.
B4 is closest to original E2: Score5 -0.004800pp, J_long3 -0.006667pp, J_long -0.010000pp, Short4 -0.002000pp. B1/B2/B3 Score5 deltas are -0.020338/-0.042524/-0.053377pp. These single-seed observations do not establish statistically reliable gains or losses.
Among the two perturbations, increasing backbone LR reduces Score5 relative to decreasing it; increasing mask LR performs better than decreasing it. Original LR remains the local peak for both axes, and neither axis changes Urban R@1. There is no common beneficial trend.
No nonfinite update or complete closed mask was observed. Last50 mask structure remains close to original E2, including its existing F<Dall density order. The frozen-batch hierarchy/sparsity cosine is negative in visual masks (-0.357 to -0.345) and positive in text masks/shared pool (0.226 to 0.253). This is descriptive mechanism evidence, not evidence of improved semantic selection.
CPU validation: 32 tests passed; see FINAL_CPU_TESTS.json. Frozen full-batch preupdate validation, four complete stream/LR proofs, strict exports, and final source/checkpoint immutability checks all passed. Full weights and raw logs remain local.

Gradient probes are frozen-checkpoint measurements on the first seed0 epoch0 batch; they describe mechanism, not held-out retrieval quality. Density order reversals are structural observations rather than failures. No semantic-evidence improvement follows from lower density or violation alone.

All original parent and baseline checkpoint/bare/results/evaluation hashes remain unchanged. All four full-stage sample/token/index summaries match the independent original E2 epoch4 run. Production manifest and evaluator mathematics remain unchanged. GPU processes have exited. No new training is scheduled.
