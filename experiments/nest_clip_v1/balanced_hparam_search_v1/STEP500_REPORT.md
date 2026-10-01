# Balanced Search: Complete 500-Step Results

Snapshot: 2026-10-02 01:42:18 Asia/Shanghai (2026-10-01T17:42:18.018Z).

All ten new trials completed 500 updates and five native evaluations; B0 is reused.
The final experiment is still running. This report is a fixed snapshot; final results will be in REPORT.md and RESULTS.json.

## Progress

Screening Top-1 has completed 3651 updates and all five evaluations. Screening Top-2 is at 1371/3651, with recent four-rank updates averaging 2.100 seconds. Rough remaining training: 80 minutes, plus approximately 16 minutes for export and evaluations.

Global Top-2 resume directly from their own step500 to3651. No intermediate1217 screening is scheduled.

## Complete 500-Step Parameter Leaderboard

Scores are percentages; deltas are percentage points versus B0@500. All five parameter values are explicit.

| Trial | fusion_lr | visual scale | F/P/R weights | sparsity | inclusion max | Score5_R1 | Delta B0 | J_long3 | J_long |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 2ea4a3ae907a | 0.0002 | 1 | [1,1,1] | 1 | 1.5 | 70.265202 | +0.406908 | 73.775337 | 82.575002 |
| 6d44ae8d5c34 | 0.0002 | 1 | [1,1,1] | 1 | 1 | 69.990027 | +0.131732 | 73.734711 | 82.445002 |
| d6542aae963d | 0.0002 | 1 | [1,1,1] | 1 | 0.5 | 69.906756 | +0.048462 | 73.610593 | 82.295001 |
| a72ea0fb10cb | 0.0002 | 1 | [1,1,1] | 1.25 | 1 | 69.888746 | +0.030451 | 73.447909 | 82.245003 |
| 50cd96066da8 | 0.0002 | 1 | [2,1,1] | 1 | 1 | 69.887529 | +0.029235 | 73.567215 | 82.470002 |
| a009f49f814a (B0) | 0.0001 | 1 | [1,1,1] | 1 | 1 | 69.858294 | +0.000000 | 73.601824 | 82.295001 |
| 61afe6117bae | 0.0002 | 0.5 | [1,1,1] | 1 | 1 | 69.814630 | -0.043664 | 73.395716 | 82.170002 |
| e4724e65e6a4 | 0.0002 | 1 | [1,1,1] | 0.75 | 1 | 69.744885 | -0.113409 | 73.462809 | 82.195003 |
| d16d3af78e46 | 0.00005 | 1 | [1,1,1] | 1 | 1 | 69.698257 | -0.160038 | 73.312428 | 82.295003 |
| dcdf2119cb11 | 0.0002 | 1 | [1,1,1.5] | 1 | 1 | 69.550168 | -0.308126 | 73.256947 | 81.975002 |
| ffb3402cbb33 | 0.0002 | 2 | [1,1,1] | 1 | 1 | 69.332616 | -0.525678 | 72.919026 | 81.820002 |

## Effects Within Each Coordinate Round

Compare each candidate to that round's reference. A gain versus B0 does not establish a gain from the parameter changed in later rounds. Rounds2-5 use fusion_lr=2e-4 with all other coefficients default as their reference.

| Parameter | Candidate | Score5_R1 | Delta round reference (pp) | Round winner |
|---|---|---:|---:|---|
| fusion_lr | 0.00005 | 69.698257 | -0.160038 | 0.0002 |
| fusion_lr | 0.0002 | 69.990027 | +0.131732 | 0.0002 |
| visual_mask_lr_scale | 0.5 | 69.814630 | -0.175397 | 1 |
| visual_mask_lr_scale | 2 | 69.332616 | -0.657411 | 1 |
| view_weights | [2,1,1] | 69.887529 | -0.102497 | [1,1,1] |
| view_weights | [1,1,1.5] | 69.550168 | -0.439858 | [1,1,1] |
| sparsity_scale | 0.75 | 69.744885 | -0.245141 | 1 |
| sparsity_scale | 1.25 | 69.888746 | -0.101281 | 1 |
| inclusion_max | 0.5 | 69.906756 | -0.083271 | 1.5 |
| inclusion_max | 1.5 | 70.265202 | +0.275176 | 1.5 |

## Interpretation

- fusion_lr=2e-4 raises Score5_R1 by 0.131732 pp versus B0 at 500; 5e-5 lowers it by 0.160038 pp. The higher fusion LR is the useful tested direction at the screening budget. Its isolated 3651 result is pending.
- inclusion_max=1.5 adds 0.275176 pp over fusion_lr=2e-4 alone, for a combined 0.406908 pp versus B0 at 500. inclusion_max=0.5 loses 0.083271 pp versus that round's reference.
- visual_mask_lr_scale=0.5/2, view_weights=[2,1,1]/[1,1,1.5], and sparsity_scale=0.75/1.25 all lose against their respective round references; defaults remain best among tested values.
- In particular, view_weights=[2,1,1] and sparsity_scale=1.25 marginally beat B0 but lose to fusion_lr=2e-4 alone. Their scores do not support a beneficial effect from either change.
- The combined fusion_lr=2e-4/inclusion_max=1.5 configuration improves Score5_R1 by only 0.040408 pp versus B0 at 3651. The 500-step advantage largely shrinks after full training. The effect of inclusion at 3651 requires comparison with the still-running fusion-only candidate.
- These are descriptive effects in the fixed seed0 coordinate search. Five benchmarks participate in tuning; there is no multi-seed significance claim or claim of global hyperparameter optimality.

fusion_lr controls the visual projection A_V and fusion gate W_g. inclusion_max is the maximum weight of the existing inclusion loss, reached through the frozen 200-update ramp. visual_mask_lr_scale controls B_V only; view_weights redistributes alignment strength; sparsity_scale multiplies the existing sparsity term.

## Completed Final-Budget Comparison

| Configuration | Updates | Score5_R1 | J_long3 | J_long | Delta same-budget B0 (pp) |
|---|---:|---:|---:|---:|---:|
| Balanced B0 baseline | 500 | 69.858294 | 73.601824 | 82.295001 | +0.000000 |
| Balanced B0 reference | 3651 | 72.444202 | 76.657670 | 85.195002 | +0.000000 |
| Screening Top-1 | 3651 | 72.484610 | 76.689683 | 85.220003 | +0.040408 |

## Complete 500-Step Native Recall

Five datasets, two directions, R@1/R@5/R@10. All values below are percentages.

### 2ea4a3ae907a

Parameters: `{"fusion_lr":0.0002,"visual_mask_lr_scale":1,"view_weights":[1,1,1],"sparsity_scale":1,"inclusion_max":1.5}`.
Checkpoint SHA256: `a576c5ae2f28411113055fd298c03bc2dc5afa40ff176bf92085fb1d7c7b723b`.

| Dataset | Direction | R@1 | R@5 | R@10 |
|---|---|---:|---:|---:|
| COCO | I2T | 60.040000 | 82.340000 | 89.100000 |
| COCO | T2I | 41.440000 | 66.804000 | 76.712000 |
| Urban-1k | I2T | 89.200002 | 98.400003 | 99.200004 |
| Urban-1k | T2I | 87.500006 | 98.300004 | 99.100006 |
| Flickr30k-test1k | I2T | 87.300000 | 97.200000 | 99.000000 |
| Flickr30k-test1k | T2I | 71.220000 | 90.860000 | 94.940000 |
| DOCCI | I2T | 76.940000 | 94.980000 | 97.480000 |
| DOCCI | T2I | 76.660000 | 94.820000 | 97.660000 |
| Long-DCI | I2T | 55.603788 | 75.256511 | 81.689029 |
| Long-DCI | T2I | 56.748224 | 76.308866 | 81.965272 |

### 6d44ae8d5c34

Parameters: `{"fusion_lr":0.0002,"visual_mask_lr_scale":1,"view_weights":[1,1,1],"sparsity_scale":1,"inclusion_max":1}`.
Checkpoint SHA256: `8bd6517c4b8d04b31e40bb4e12825cea08ac390027f44f331036739ef456831c`.

| Dataset | Direction | R@1 | R@5 | R@10 |
|---|---|---:|---:|---:|
| COCO | I2T | 59.780000 | 82.140000 | 89.100000 |
| COCO | T2I | 41.112000 | 66.728000 | 76.368000 |
| Urban-1k | I2T | 89.300007 | 98.300004 | 99.400002 |
| Urban-1k | T2I | 88.200003 | 98.400003 | 99.200004 |
| Flickr30k-test1k | I2T | 86.100000 | 97.200000 | 98.800000 |
| Flickr30k-test1k | T2I | 70.500000 | 90.700000 | 94.660000 |
| DOCCI | I2T | 76.220000 | 94.680000 | 97.620000 |
| DOCCI | T2I | 76.060000 | 94.680000 | 97.600000 |
| Long-DCI | I2T | 55.551171 | 75.111813 | 81.268087 |
| Long-DCI | T2I | 57.077085 | 76.335175 | 81.965272 |

### d6542aae963d

Parameters: `{"fusion_lr":0.0002,"visual_mask_lr_scale":1,"view_weights":[1,1,1],"sparsity_scale":1,"inclusion_max":0.5}`.
Checkpoint SHA256: `1a549417ff287ccd4e5fafb4872574fc2a03079db2b78449a5e16c810181fbc9`.

| Dataset | Direction | R@1 | R@5 | R@10 |
|---|---|---:|---:|---:|
| COCO | I2T | 59.880000 | 81.860000 | 88.900000 |
| COCO | T2I | 41.024000 | 66.624000 | 76.256000 |
| Urban-1k | I2T | 89.700001 | 98.100007 | 99.400002 |
| Urban-1k | T2I | 87.400001 | 98.200005 | 99.100006 |
| Flickr30k-test1k | I2T | 86.000000 | 97.400000 | 98.800000 |
| Flickr30k-test1k | T2I | 70.500000 | 90.740000 | 94.640000 |
| DOCCI | I2T | 76.320000 | 94.680000 | 97.600000 |
| DOCCI | T2I | 75.760000 | 94.520000 | 97.480000 |
| Long-DCI | I2T | 55.380163 | 74.848724 | 81.004999 |
| Long-DCI | T2I | 57.103394 | 76.479874 | 81.860037 |

### a72ea0fb10cb

Parameters: `{"fusion_lr":0.0002,"visual_mask_lr_scale":1,"view_weights":[1,1,1],"sparsity_scale":1.25,"inclusion_max":1}`.
Checkpoint SHA256: `69607e7d67a052a89ca35f435c916f6e8fb1cd7605b7713f8820723c27b98bbb`.

| Dataset | Direction | R@1 | R@5 | R@10 |
|---|---|---:|---:|---:|
| COCO | I2T | 59.840000 | 81.980000 | 88.980000 |
| COCO | T2I | 41.060000 | 66.604000 | 76.240000 |
| Urban-1k | I2T | 89.300007 | 98.100007 | 99.500006 |
| Urban-1k | T2I | 87.700003 | 98.400003 | 99.100006 |
| Flickr30k-test1k | I2T | 86.700000 | 97.200000 | 98.900000 |
| Flickr30k-test1k | T2I | 70.600000 | 90.680000 | 94.640000 |
| DOCCI | I2T | 76.100000 | 94.640000 | 97.560000 |
| DOCCI | T2I | 75.880000 | 94.600000 | 97.400000 |
| Long-DCI | I2T | 54.775059 | 74.375164 | 80.847145 |
| Long-DCI | T2I | 56.932386 | 76.243094 | 81.846882 |

### 50cd96066da8

Parameters: `{"fusion_lr":0.0002,"visual_mask_lr_scale":1,"view_weights":[2,1,1],"sparsity_scale":1,"inclusion_max":1}`.
Checkpoint SHA256: `65024b4265e71c23e004cec97364ec7dc02723e0b066506436cfc7c378f58508`.

| Dataset | Direction | R@1 | R@5 | R@10 |
|---|---|---:|---:|---:|
| COCO | I2T | 59.740000 | 81.920000 | 88.800000 |
| COCO | T2I | 41.112000 | 66.652000 | 76.328000 |
| Urban-1k | I2T | 90.000004 | 98.100007 | 99.400002 |
| Urban-1k | T2I | 87.700003 | 98.100007 | 99.000007 |
| Flickr30k-test1k | I2T | 86.100000 | 97.500000 | 98.800000 |
| Flickr30k-test1k | T2I | 70.520000 | 90.760000 | 94.620000 |
| DOCCI | I2T | 76.480000 | 94.620000 | 97.700000 |
| DOCCI | T2I | 75.700000 | 94.560000 | 97.480000 |
| Long-DCI | I2T | 54.814522 | 74.467245 | 80.755064 |
| Long-DCI | T2I | 56.708761 | 76.085241 | 81.820574 |

### a009f49f814a (B0 baseline)

Parameters: `{"fusion_lr":0.0001,"visual_mask_lr_scale":1,"view_weights":[1,1,1],"sparsity_scale":1,"inclusion_max":1}`.
Checkpoint SHA256: `c09e13d6d636c69491914643db81d8270b90d1c85ef0729513d240c4b108293d`.

| Dataset | Direction | R@1 | R@5 | R@10 |
|---|---|---:|---:|---:|
| COCO | I2T | 59.600000 | 81.880000 | 88.800000 |
| COCO | T2I | 40.932000 | 66.636000 | 76.320000 |
| Urban-1k | I2T | 89.700001 | 98.400003 | 99.500006 |
| Urban-1k | T2I | 87.400001 | 98.200005 | 99.100006 |
| Flickr30k-test1k | I2T | 86.100000 | 97.500000 | 98.800000 |
| Flickr30k-test1k | T2I | 70.340000 | 90.660000 | 94.680000 |
| DOCCI | I2T | 76.180000 | 94.560000 | 97.560000 |
| DOCCI | T2I | 75.900000 | 94.720000 | 97.540000 |
| Long-DCI | I2T | 55.209155 | 74.611944 | 81.044462 |
| Long-DCI | T2I | 57.221784 | 76.400947 | 82.057353 |

### 61afe6117bae

Parameters: `{"fusion_lr":0.0002,"visual_mask_lr_scale":0.5,"view_weights":[1,1,1],"sparsity_scale":1,"inclusion_max":1}`.
Checkpoint SHA256: `1e5ea96ee7395b0f1b6a01f7dcbb9ed78f1f4230e68f3d3a09927b196c529396`.

| Dataset | Direction | R@1 | R@5 | R@10 |
|---|---|---:|---:|---:|
| COCO | I2T | 59.780000 | 82.000000 | 88.960000 |
| COCO | T2I | 40.912000 | 66.532000 | 76.244000 |
| Urban-1k | I2T | 89.500004 | 98.000002 | 99.300003 |
| Urban-1k | T2I | 87.200004 | 98.500007 | 99.100006 |
| Flickr30k-test1k | I2T | 86.500000 | 97.200000 | 98.700000 |
| Flickr30k-test1k | T2I | 70.580000 | 90.680000 | 94.720000 |
| DOCCI | I2T | 76.040000 | 94.640000 | 97.600000 |
| DOCCI | T2I | 75.940000 | 94.360000 | 97.440000 |
| Long-DCI | I2T | 55.011839 | 74.388319 | 80.715601 |
| Long-DCI | T2I | 56.682452 | 76.098395 | 81.925809 |

### e4724e65e6a4

Parameters: `{"fusion_lr":0.0002,"visual_mask_lr_scale":1,"view_weights":[1,1,1],"sparsity_scale":0.75,"inclusion_max":1}`.
Checkpoint SHA256: `61c3ab5b58fb2d34904eeafa295f37990b49b0b738e2438b3c2cbb898963d6bb`.

| Dataset | Direction | R@1 | R@5 | R@10 |
|---|---|---:|---:|---:|
| COCO | I2T | 59.520000 | 81.580000 | 88.680000 |
| COCO | T2I | 40.772000 | 66.500000 | 76.092000 |
| Urban-1k | I2T | 89.400005 | 98.100007 | 99.300003 |
| Urban-1k | T2I | 87.500006 | 98.100007 | 99.100006 |
| Flickr30k-test1k | I2T | 86.000000 | 97.200000 | 98.800000 |
| Flickr30k-test1k | T2I | 70.380000 | 90.640000 | 94.640000 |
| DOCCI | I2T | 76.080000 | 94.480000 | 97.560000 |
| DOCCI | T2I | 75.800000 | 94.420000 | 97.440000 |
| Long-DCI | I2T | 55.090766 | 74.730334 | 80.833991 |
| Long-DCI | T2I | 56.906077 | 76.085241 | 81.925809 |

### d16d3af78e46

Parameters: `{"fusion_lr":0.00005,"visual_mask_lr_scale":1,"view_weights":[1,1,1],"sparsity_scale":1,"inclusion_max":1}`.
Checkpoint SHA256: `cc236352c39e4c4e9760cba735a4c847c9b10eb3a59842132b7677aa6c4db66b`.

| Dataset | Direction | R@1 | R@5 | R@10 |
|---|---|---:|---:|---:|
| COCO | I2T | 59.820000 | 81.980000 | 88.780000 |
| COCO | T2I | 40.768000 | 66.416000 | 76.176000 |
| Urban-1k | I2T | 89.300007 | 98.300004 | 99.300003 |
| Urban-1k | T2I | 87.100005 | 97.900003 | 98.900002 |
| Flickr30k-test1k | I2T | 86.200000 | 97.400000 | 98.800000 |
| Flickr30k-test1k | T2I | 70.320000 | 90.580000 | 94.740000 |
| DOCCI | I2T | 76.700000 | 94.800000 | 97.520000 |
| DOCCI | T2I | 76.080000 | 94.620000 | 97.480000 |
| Long-DCI | I2T | 54.446198 | 74.191002 | 80.741910 |
| Long-DCI | T2I | 56.248356 | 75.703762 | 81.662720 |

### dcdf2119cb11

Parameters: `{"fusion_lr":0.0002,"visual_mask_lr_scale":1,"view_weights":[1,1,1.5],"sparsity_scale":1,"inclusion_max":1}`.
Checkpoint SHA256: `e5d455ea5bbc69f22df7cf2460e8d942fc388aa83f38f1c952ccbeb491b0d005`.

| Dataset | Direction | R@1 | R@5 | R@10 |
|---|---|---:|---:|---:|
| COCO | I2T | 59.300000 | 81.940000 | 88.680000 |
| COCO | T2I | 40.780000 | 66.448000 | 75.936000 |
| Urban-1k | I2T | 89.100003 | 97.900003 | 99.400002 |
| Urban-1k | T2I | 87.000006 | 98.200005 | 99.100006 |
| Flickr30k-test1k | I2T | 85.700000 | 97.100000 | 98.700000 |
| Flickr30k-test1k | T2I | 70.180000 | 90.380000 | 94.380000 |
| DOCCI | I2T | 75.920000 | 94.520000 | 97.480000 |
| DOCCI | T2I | 75.880000 | 94.400000 | 97.420000 |
| Long-DCI | I2T | 54.853986 | 74.388319 | 80.899763 |
| Long-DCI | T2I | 56.787687 | 76.019469 | 81.846882 |

### ffb3402cbb33

Parameters: `{"fusion_lr":0.0002,"visual_mask_lr_scale":2,"view_weights":[1,1,1],"sparsity_scale":1,"inclusion_max":1}`.
Checkpoint SHA256: `613a7051c05f3f68e43375c44da34b7c9b8cd55cc36e35716023e372ac242c01`.

| Dataset | Direction | R@1 | R@5 | R@10 |
|---|---|---:|---:|---:|
| COCO | I2T | 59.140000 | 81.380000 | 88.220000 |
| COCO | T2I | 40.692000 | 66.204000 | 76.052000 |
| Urban-1k | I2T | 89.400005 | 97.700006 | 99.300003 |
| Urban-1k | T2I | 86.800003 | 98.000002 | 98.900002 |
| Flickr30k-test1k | I2T | 85.600000 | 97.200000 | 98.700000 |
| Flickr30k-test1k | T2I | 70.380000 | 90.460000 | 94.460000 |
| DOCCI | I2T | 75.620000 | 94.320000 | 97.400000 |
| DOCCI | T2I | 75.460000 | 94.300000 | 97.360000 |
| Long-DCI | I2T | 53.920021 | 73.467509 | 80.202578 |
| Long-DCI | T2I | 56.314128 | 75.585372 | 81.570639 |

## Evidence and Verification

leaderboard_500.csv preserves original score floats and full parameter values. SEARCH_STATE.json includes metrics, checkpoint identities, rounds and promotion lineage. configs/ contains trial JSON; evidence/<trial-prefix>/step500/ contains all five raw evaluator JSON files plus export and acceptance checks.

All11 sets of Score5_R1/J_long3/J_long were independently recomputed from per-direction recalls. All500-step acceptance/export checks passed; no recorded stage failures were found. Small evidence includes the completed Top-1@3651 result. Large checkpoints and data stay server-local.

The final winner remains undecided until Top-2@3651 completes. The existing supervisor will produce the final report, commit the remaining evidence and push the experiment branch.
