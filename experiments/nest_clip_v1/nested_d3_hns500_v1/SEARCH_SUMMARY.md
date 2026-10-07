# Hard Nested Sparsity: INC0-matched local500

Classification: `NEGATIVE`.
Base remote INC0 commit `e201975982809961cf4f078522e3137203c52726`. Fresh common0 smoke5 and independent formal500; exactly500 updates, horizon4868, local-only.
Original alignment and global sparsity preserved; only adjacent illegal-support ReLU surcharge added. No mask-to-mask stop-gradient. Actual masks have512 dimensions.

| Model | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I |
|---|---:|---:|---:|---:|---:|---:|
| INC0 | 71.158309 | 74.978516 | 83.995003 | 65.428000 | 91.100007 | 89.400005 |
| Anchor | 71.063600 | 74.954666 | 84.025001 | 65.227000 | 90.900004 | 89.700001 |
| KR234 | 71.107901 | 75.046502 | 84.120002 | 65.200000 | 91.600007 | 89.200002 |
| Joint-VG | 70.820499 | 74.722832 | 83.720002 | 64.967000 | 90.600002 | 88.900006 |
| HNS | 71.202056 | 75.152094 | 84.160001 | 65.277000 | 91.500002 | 89.700001 |

## Five frozen native benchmarks

| Dataset | I2T R1/R5/R10 (%) | T2I R1/R5/R10 (%) |
|---|---|---|
| COCO | 60.420000 / 82.920000 / 89.340000 | 41.548000 / 67.412000 / 76.988000 |
| Urban-1k | 91.500002 / 98.600006 / 99.700004 | 89.700001 / 98.800004 / 99.500006 |
| Flickr30k-test1k | 87.200000 / 97.800000 / 99.300000 | 71.940000 / 91.540000 / 95.140000 |
| DOCCI | 77.940000 / 95.440000 / 98.080000 | 77.500000 / 95.140000 / 97.900000 |
| Long-DCI | 57.208629 / 75.940542 / 82.070508 | 57.063931 / 76.098395 / 82.175743 |

## All baseline deltas

### vs INC0

Quality delta(pp): `{'Score5': 0.04374695460518385, 'J_long3': 0.17357825767529977, 'J_long': 0.1649977469444286, 'Short4': -0.15100000000000113, 'Urban_I2T': 0.3999948501586914, 'Urban_T2I': 0.29999613761901855}`.

All dataset/direction/R1/R5/R10 delta(pp): `{"COCO": {"I2T": {"R@1": -0.26000000000000467, "R@10": 0.10000000000000009, "R@5": -0.019999999999997797}, "T2I": {"R@1": 0.036000000000002697, "R@10": 0.07200000000000539, "R@5": 0.08400000000000629}}, "DOCCI": {"I2T": {"R@1": 0.08000000000000229, "R@10": 0.039999999999995595, "R@5": 0.0400000000000067}, "T2I": {"R@1": -0.11999999999999789, "R@10": 0.039999999999995595, "R@5": -0.08000000000000229}}, "Flickr30k-test1k": {"I2T": {"R@1": -0.20000000000000018, "R@10": 0.20000000000000018, "R@5": 0.20000000000000018}, "T2I": {"R@1": -0.17999999999999128, "R@10": -0.10000000000000009, "R@5": 0.039999999999995595}}, "Long-DCI": {"I2T": {"R@1": 0.07892659826361781, "R@10": 0.14469876348329747, "R@5": 0.19731649565902787}, "T2I": {"R@1": 0.3025519600105331, "R@10": 0.1315443304393593, "R@5": 0.026308866087876304}}, "Urban-1k": {"I2T": {"R@1": 0.3999948501586914, "R@10": 0.0, "R@5": -0.1999974250793457}, "T2I": {"R@1": 0.29999613761901855, "R@10": 0.0, "R@5": 0.09999871253967285}}}`.

### vs Anchor

Quality delta(pp): `{'Score5': 0.13845654548554354, 'J_long3': 0.197427575809217, 'J_long': 0.13499955892563031, 'Short4': 0.04999999999999449, 'Urban_I2T': 0.5999982357025146, 'Urban_T2I': 0.0}`.

All dataset/direction/R1/R5/R10 delta(pp): `{"COCO": {"I2T": {"R@1": 0.019999999999997797, "R@10": 0.08000000000000229, "R@5": 0.30000000000000027}, "T2I": {"R@1": -0.040000000000001146, "R@10": 0.056000000000000494, "R@5": 0.13600000000000279}}, "DOCCI": {"I2T": {"R@1": 0.11999999999999789, "R@10": 0.08000000000000229, "R@5": 0.10000000000000009}, "T2I": {"R@1": -0.18000000000000238, "R@10": 0.18000000000000238, "R@5": -0.11999999999999789}}, "Flickr30k-test1k": {"I2T": {"R@1": 0.20000000000000018, "R@10": 0.10000000000000009, "R@5": 0.0}, "T2I": {"R@1": 0.0200000000000089, "R@10": 0.019999999999997797, "R@5": 0.16000000000000458}}, "Long-DCI": {"I2T": {"R@1": 0.18416206261510082, "R@10": 0.42094185740594314, "R@5": 0.407787424362005}, "T2I": {"R@1": 0.4604051565377576, "R@10": 0.23677979479084232, "R@5": 0.013154433043938152}}, "Urban-1k": {"I2T": {"R@1": 0.5999982357025146, "R@10": 0.09999871253967285, "R@5": 0.0}, "T2I": {"R@1": 0.0, "R@10": 0.0, "R@5": 0.29999613761901855}}}`.

### vs KR234

Quality delta(pp): `{'Score5': 0.09415544397404219, 'J_long3': 0.10559240662338931, 'J_long': 0.03999871253967946, 'Short4': 0.07699999999999374, 'Urban_I2T': -0.10000467300415039, 'Urban_T2I': 0.4999995231628418}`.

All dataset/direction/R1/R5/R10 delta(pp): `{"COCO": {"I2T": {"R@1": -0.16000000000000458, "R@10": 0.13999999999999568, "R@5": 0.26000000000000467}, "T2I": {"R@1": 0.10800000000000254, "R@10": 0.048000000000003595, "R@5": 0.17200000000000548}}, "DOCCI": {"I2T": {"R@1": 0.18000000000000238, "R@10": -0.10000000000000009, "R@5": -0.08000000000000229}, "T2I": {"R@1": -0.41999999999999815, "R@10": 0.039999999999995595, "R@5": 0.14000000000000679}}, "Flickr30k-test1k": {"I2T": {"R@1": 0.0, "R@10": 0.10000000000000009, "R@5": 0.20000000000000018}, "T2I": {"R@1": 0.36000000000000476, "R@10": -0.15999999999999348, "R@5": 0.13999999999999568}}, "Long-DCI": {"I2T": {"R@1": 0.3814785582741398, "R@10": 0.3814785582741398, "R@5": 0.605103920021044}, "T2I": {"R@1": 0.09208103130755596, "R@10": 0.013154433043938152, "R@5": -0.14469876348329747}}, "Urban-1k": {"I2T": {"R@1": -0.10000467300415039, "R@10": -0.09999871253967285, "R@5": -0.09999871253967285}, "T2I": {"R@1": 0.4999995231628418, "R@10": 0.0, "R@5": 0.1999974250793457}}}`.

### vs Joint-VG

Quality delta(pp): `{'Score5': 0.3815570843291871, 'J_long3': 0.4292618072153087, 'J_long': 0.4399989986419772, 'Short4': 0.30999999999999917, 'Urban_I2T': 0.9000003337860107, 'Urban_T2I': 0.7999956607818604}`.

All dataset/direction/R1/R5/R10 delta(pp): `{"COCO": {"I2T": {"R@1": 0.27999999999999137, "R@10": 0.33999999999999586, "R@5": 0.5800000000000027}, "T2I": {"R@1": 0.28000000000000247, "R@10": 0.25600000000000067, "R@5": 0.31200000000000117}}, "DOCCI": {"I2T": {"R@1": 0.10000000000000009, "R@10": 0.039999999999995595, "R@5": 0.28000000000000247}, "T2I": {"R@1": -0.039999999999995595, "R@10": 0.13999999999999568, "R@5": -0.039999999999995595}}, "Flickr30k-test1k": {"I2T": {"R@1": 0.5000000000000004, "R@10": 0.40000000000000036, "R@5": 0.40000000000000036}, "T2I": {"R@1": 0.18000000000000238, "R@10": -0.05999999999999339, "R@5": 0.21999999999999797}}, "Long-DCI": {"I2T": {"R@1": 0.407787424362005, "R@10": 0.24993422783478048, "R@5": 0.0789265982636067}, "T2I": {"R@1": 0.4077874243620161, "R@10": 0.28939752696658383, "R@5": 0.28939752696658383}}, "Urban-1k": {"I2T": {"R@1": 0.9000003337860107, "R@10": -0.09999871253967285, "R@5": 0.09999871253967285}, "T2I": {"R@1": 0.7999956607818604, "R@10": 0.10000467300415039, "R@5": 0.0}}}`.

## Scientific questions

A. Relative to INC0: `{'Score5': 0.04374695460518385, 'J_long3': 0.17357825767529977, 'J_long': 0.1649977469444286, 'Short4': -0.15100000000000113, 'Urban_I2T': 0.3999948501586914, 'Urban_T2I': 0.29999613761901855}`; violation/keep changes: `{'HNS_F_keep': 0.005156513452529876, 'HNS_Dall_keep': 0.00685052871704106, 'HNS_D3_keep': -0.0026906538009644487, 'HNS_DF_hard_violation_ratio': -0.0014650729298591597, 'HNS_3D_hard_violation_ratio': -0.005392710492014889, 'HNS_gap_F_D': -0.001694015264511118, 'HNS_gap_D_D3': 0.009541182518005432}`. Decision uses retrieval, both violations and collapse evidence together.
B. Actual endpoint gradient descent directions: `{"V_3D": {"HardST_output": {"child": "D3", "child_negative_count": 0, "child_norm": 0.0002494327782187611, "child_positive_count": 17102, "gradient_descent_child_contraction_verified": true, "gradient_descent_parent_expansion_verified": true, "parent": "Dall", "parent_child_norm_ratio": 1.0, "parent_negative_count": 17102, "parent_norm": 0.0002494327782187611, "parent_positive_count": 0}, "soft_probability": {"child": "D3", "child_negative_count": 0, "child_norm": 0.0002494327782187611, "child_positive_count": 17102, "gradient_descent_child_contraction_verified": true, "gradient_descent_parent_expansion_verified": true, "parent": "Dall", "parent_child_norm_ratio": 1.0, "parent_negative_count": 17102, "parent_norm": 0.0002494327782187611, "parent_positive_count": 0}}, "V_DF": {"HardST_output": {"child": "Dall", "child_negative_count": 0, "child_norm": 0.0002064523141598329, "child_positive_count": 11716, "gradient_descent_child_contraction_verified": true, "gradient_descent_parent_expansion_verified": true, "parent": "F", "parent_child_norm_ratio": 1.0, "parent_negative_count": 11716, "parent_norm": 0.0002064523141598329, "parent_positive_count": 0}, "soft_probability": {"child": "Dall", "child_negative_count": 0, "child_norm": 0.0002064523141598329, "child_positive_count": 11716, "gradient_descent_child_contraction_verified": true, "gradient_descent_parent_expansion_verified": true, "parent": "F", "parent_child_norm_ratio": 1.0, "parent_negative_count": 11716, "parent_norm": 0.0002064523141598329, "parent_positive_count": 0}}}`. Shared mask parameter groups are reported in GRADIENT_AUDIT.json; there are no independent per-view mask branches. Endpoint directions do not establish the cause of density changes across training.
C. Joint inflation/equality evidence: `{'classification': 'NEGATIVE', 'both_violations_down_at_least25percent': False, 'common_keep_increase_at_least5pp': False, 'matched_exact_equality_increase_at_least15pp': False, 'both_mean_gaps_at_most_point002': False, 'collapse_warning': False, 'native_tradeoff_acceptable': True, 'automatic_full': False, 'automatic_other_arms': False, 'wait_for_human': True}`. Exact pair/triple equality uses whole512-coordinate masks and a matched1024 cohort. Last50 population is separate.
D. Compared with Anchor soft inclusion: `{'Score5': 0.13845654548554354, 'J_long3': 0.197427575809217, 'J_long': 0.13499955892563031, 'Short4': 0.04999999999999449, 'Urban_I2T': 0.5999982357025146, 'Urban_T2I': 0.0}`. This one seed at500 updates supports only this tested formulation, not a general ranking of hard versus soft inclusion.
E. Dataset/direction R1 changes vs INC0 sorted from greatest loss to greatest gain(pp): `[(-0.26000000000000467, 'COCO', 'I2T'), (-0.20000000000000018, 'Flickr30k-test1k', 'I2T'), (-0.17999999999999128, 'Flickr30k-test1k', 'T2I'), (-0.11999999999999789, 'DOCCI', 'T2I'), (0.036000000000002697, 'COCO', 'T2I'), (0.07892659826361781, 'Long-DCI', 'I2T'), (0.08000000000000229, 'DOCCI', 'I2T'), (0.29999613761901855, 'Urban-1k', 'T2I'), (0.3025519600105331, 'Long-DCI', 'T2I'), (0.3999948501586914, 'Urban-1k', 'I2T')]`. All R5/R10 deltas are above. This identifies where the observed aggregate differences arise.
F. Mechanistic evidence: `{"both_violation_improvement": false, "child_shrink": -0.0026906538009644487, "hierarchy_gradient_conflicts": {"fusion_shared_module": {"cosine_hierarchy_alignment": 0.03612527251243591, "cosine_hierarchy_original_sparsity": -0.2566054165363312, "hierarchy_norm": 0.0038157973904162645}, "shared_text_backbone": {"cosine_hierarchy_alignment": null, "cosine_hierarchy_original_sparsity": null, "hierarchy_norm": 0.0}, "shared_text_mask_and_pool": {"cosine_hierarchy_alignment": 0.015192897990345955, "cosine_hierarchy_original_sparsity": -0.009580991230905056, "hierarchy_norm": 0.00462031364440918}, "shared_visual_backbone": {"cosine_hierarchy_alignment": null, "cosine_hierarchy_original_sparsity": null, "hierarchy_norm": 0.0}, "shared_visual_mask": {"cosine_hierarchy_alignment": 0.1676579862833023, "cosine_hierarchy_original_sparsity": -0.8389728665351868, "hierarchy_norm": 0.0007174641359597445}}, "mask_homogenization_warning": false, "parent_inflation": {"Dall": 0.00685052871704106, "F": 0.005156513452529876}}`. Shared-group cosines measure gradient conflicts, not causation. Native encoder hidden detach is inherited unchanged: regularizer gradients to native visual/text backbone are zero; alignment updates them. Density/equality alone cannot establish over-constraint or over-shrink. No automatic continuation/variant; human review decides next steps.

Actual ramp(step1/100/200/300/400/500): `{'1': 0.0, '100': 0.495, '200': 0.995, '300': 1.0, '400': 1.0, '500': 1.0}`.
Selected-step and last50 counts/IoU/exact equality/coordinate equality/keep/gaps: TRAINING_DIAGNOSTICS.json. Full raw scalar curves, checkpoint, bare and logs remain server-local with size/SHA/time ranges in RUNTIME_STATS.json.
Correctness,1000 real matched preflight,independent smoke,five-step hard gate,512000 stream match,optimizer/RNG/cursor acceptance,strict export and all five raw evaluator JSONs are included. No training images,weights,cache or credentials uploaded.
