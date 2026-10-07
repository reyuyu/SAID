# Final HNS local500 comparison

| Model | Beta DF/3D | Score5 | J_long3 | J_long | Short4 | Urban I2T/T2I |
|---|---|---:|---:|---:|---:|---|
| INC0 | [0, 0] | 71.158309 | 74.978516 | 83.995003 | 65.428000 | 91.100007 / 89.400005 |
| HNS-v1 | [2, 2] | 71.202056 | 75.152094 | 84.160001 | 65.277000 | 91.500002 / 89.700001 |
| HNS-Weak | [1.0, 1.0] | 71.148277 | 74.863795 | 83.915003 | 65.575000 | 91.100007 / 89.300007 |
| HNS-InnerFocus | [1.0, 2.0] | 71.150660 | 75.016434 | 84.190002 | 65.352000 | 91.500002 / 89.500004 |

| Model | COCO I/T | Urban I/T | Flickr I/T | DOCCI I/T | Long I/T |
|---|---|---|---|---|---|
| INC0 | 60.680000 / 41.512000 | 91.100007 / 89.400005 | 87.400000 / 72.120000 | 77.860000 / 77.620000 | 57.129703 / 56.761379 |
| HNS-v1 | 60.420000 / 41.548000 | 91.500002 / 89.700001 | 87.200000 / 71.940000 | 77.940000 / 77.500000 | 57.208629 / 57.063931 |
| HNS-Weak | 60.620000 / 41.520000 | 91.100007 / 89.300007 | 88.300000 / 71.860000 | 77.700000 / 77.560000 | 56.761379 / 56.761379 |
| HNS-InnerFocus | 60.360000 / 41.588000 | 91.500002 / 89.500004 | 87.400000 / 72.060000 | 77.980000 / 77.780000 | 56.853460 / 56.485135 |

| Model | F keep | Dall keep | D3 keep | DF violation | 3D violation | F=D | D=D3 | all equal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HNS-v1 | 0.80391216 | 0.79590667 | 0.74417104 | 0.02593146 | 0.03586677 | 0.00000000 | 0.00023437 | 0.00000000 |
| HNS-Weak | 0.80643966 | 0.79664406 | 0.74782400 | 0.02466265 | 0.03707396 | 0.00000000 | 0.00023437 | 0.00000000 |
| HNS-InnerFocus | 0.80278655 | 0.78966128 | 0.74315150 | 0.03282375 | 0.04068283 | 0.00000000 | 0.00023437 | 0.00000000 |
| INC0 | 0.79875565 | 0.78905614 | 0.74686169 | 0.02739653 | 0.04125948 | 0.00000000 | 0.00000000 | 0.00000000 |

## Scientific contrasts Q1–Q6

INC0 equality uses a matched1024 audit because historical last50 whole-mask equality was not recorded. INC0 F keep includes all samples; new HNS keep is valid-only. Comparable matched1024 diagnostics are in each gradient audit.

Q1: `{"contrast": "Weak vs v1; uniform pressure reduction", "quality_delta_pp": {"Score5": -0.05377923838217891, "J_long3": -0.28829873063696, "J_long": -0.24499742507935185, "Short4": 0.29800000000000937, "Urban_I2T": -0.3999948501586914, "Urban_T2I": -0.3999948501586914}}`.

Q2: `{"contrast": "Inner vs Weak (inner only) and Inner vs v1 (outer only)", "Inner_minus_Weak_pp": {"Score5": 0.0023830212622977243, "J_long3": 0.15263836877050796, "J_long": 0.2749980688095066, "Short4": -0.22300000000000653, "Urban_I2T": 0.3999948501586914, "Urban_T2I": 0.1999974250793457}, "Inner_minus_v1_pp": {"Score5": -0.05139621711988118, "J_long3": -0.13566036186645203, "J_long": 0.030000643730154763, "Short4": 0.07500000000000284, "Urban_I2T": 0.0, "Urban_T2I": -0.1999974250793457}}`.

Q3: `{"contrast": "Inner minus Weak isolates inner coefficient1->2; inspect J_long3,J_long and all three long datasets", "delta_pp": {"Score5": 0.0023830212622977243, "J_long3": 0.15263836877050796, "J_long": 0.2749980688095066, "Short4": -0.22300000000000653, "Urban_I2T": 0.3999948501586914, "Urban_T2I": 0.1999974250793457}}`.

Q4: `{"contrast": "Inner minus v1 isolates outer2->1; inspect Short4 and individual COCO/Flickr alongside long datasets", "delta_pp": {"Score5": -0.05139621711988118, "J_long3": -0.13566036186645203, "J_long": 0.030000643730154763, "Short4": 0.07500000000000284, "Urban_I2T": 0.0, "Urban_T2I": -0.1999974250793457}}`.

Q5: `{"HNS-Weak": {"keep_delta_vs_v1": {"F": 0.002527501583099445, "Dall": 0.0007373893260955722, "D3": 0.003652955293655391}, "exact_equality": {"HNS_DF_exact_equality_ratio": 0.0, "HNS_3D_exact_equality_ratio": 0.000234375, "HNS_triple_exact_equality_ratio": 0.0}}, "HNS-InnerFocus": {"keep_delta_vs_v1": {"F": -0.0011256122589110573, "Dall": -0.00624538779258732, "D3": -0.0010195422172546431}, "exact_equality": {"HNS_DF_exact_equality_ratio": 0.0, "HNS_3D_exact_equality_ratio": 0.000234375, "HNS_triple_exact_equality_ratio": 0.0}}}`.

Q6: `{"uniform_contrast_pp": {"Score5": -0.05377923838217891, "J_long3": -0.28829873063696, "J_long": -0.24499742507935185, "Short4": 0.29800000000000937, "Urban_I2T": -0.3999948501586914, "Urban_T2I": -0.3999948501586914}, "inner_granularity_contrast_pp": {"Score5": 0.0023830212622977243, "J_long3": 0.15263836877050796, "J_long": 0.2749980688095066, "Short4": -0.22300000000000653, "Urban_I2T": 0.3999948501586914, "Urban_T2I": 0.1999974250793457}, "limitation": "One seed,500 updates; these measured contrasts suggest local effects only, not general causation or full-training ranking."}`.

## Selection

{"BEST_SCORE5_CANDIDATE": "HNS-v1", "BEST_LONG_CANDIDATE": "HNS-v1", "raw_Score5_maximum": "HNS-v1", "approximate_tie_cohort": ["INC0", "HNS-v1"], "tie_threshold_raw": 0.0005, "automatic_full": false, "automatic_third_arm": false}
Highest raw Score5 primary; candidates within strictly<.05pp of raw maximum form a tie cohort, then J_long3,J_long,UrbanT2I,Short4,lower summed violations. Long candidate: J_long3 then J_long. Structure does not override a clear Score5 advantage.
Recommend BEST_SCORE5_CANDIDATE for human consideration of full training. No full/third beta/second seed/other variant started.
