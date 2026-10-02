# Three Balanced Followups

Status: `queued`. Stage: `waiting-for-first-two-followups`.

Experiments1/2 use the latest instruction: full3651 updates, with evaluations at500 and3651. Both keep fusion_lr=2e-4. Experiment3 is confirmed as a fresh four-epoch run from shared step0, with horizon4868 from the first update and evaluations at3651/4868. It waits for both preceding experiments.

Experiment3 selected parent: `6d44ae8d5c34438e672287b09f3e6930af01473a4aa4905cd90c02c259b2a4a8`; parameters `{"fusion_lr": 0.0002, "inclusion_max": 1.0, "sparsity_scale": 1.0, "view_weights": [1.0, 1.0, 1.0], "visual_mask_lr_scale": 1.0}`.

## Scores

| Configuration | Updates | Horizon | Score5_R1 % | J_long3 % | J_long % |
|---|---:|---:|---:|---:|---:|
| Balanced B0 (a009f49f814a@500) | 500 | 3651 | 69.858294 | 73.601824 | 82.295001 |
| Balanced B0 (a009f49f814a@3651) | 3651 | 3651 | 72.444202 | 76.657670 | 85.195002 |
| Fusion-only C (6d44ae8d5c34@500) | 500 | 3651 | 69.990027 | 73.734711 | 82.445002 |
| Fusion-only C (6d44ae8d5c34@3651) | 3651 | 3651 | 72.531265 | 76.723441 | 85.195001 |
| Previous inclusion1.5 (2ea4a3ae907a@500) | 500 | 3651 | 70.265202 | 73.775337 | 82.575002 |
| Previous inclusion1.5 (2ea4a3ae907a@3651) | 3651 | 3651 | 72.484610 | 76.689683 | 85.220003 |
| Inclusion++ | 500 | 3651 | 69.678630 | 73.329050 | 82.070002 |
| Inclusion++ | 3651 | 3651 | 72.492140 | 76.608900 | 85.135002 |

## Experiment Status

### Inclusion++

Status: `completed`. Parameters: `{"fusion_lr": 0.0002, "inclusion_max": 2.0, "sparsity_scale": 1.0, "view_weights": [1.0, 1.0, 1.0], "visual_mask_lr_scale": 1.0}`.

At500, Score5_R1 change versus the inclusion1.5 parent: -0.586572 pp.

At500: full checkpoint SHA256 `d6364a13fa6f93ab31efa4e64a4ab6188cd9e5451c9fe2f5091c95526715cbd4`; bare student SHA256 `9dbe58e255dd6d46ddf8b80b8944d22dd42d16ce0af6bd09bc8559ce81123d1f`.

Export/acceptance: `True` / `True`.

Commands and exit codes are recorded in SEARCH_STATE.json and evidence/execution/. Diagnostic summaries are in SEARCH_STATE.json; full step/token logs remain server-local.

At3651, Score5_R1 change versus the inclusion1.5 parent: +0.007530 pp.

At3651: full checkpoint SHA256 `651b8687a1a90b564e9e68d454c1220ebeb518eb420817c3d6a5cd7c9e390b51`; bare student SHA256 `b53e1dfa65acc0b5117bddbd8b33750e5aa37825e20800ea8dc60df57edc65cc`.

Export/acceptance: `True` / `True`.

Commands and exit codes are recorded in SEARCH_STATE.json and evidence/execution/. Diagnostic summaries are in SEARCH_STATE.json; full step/token logs remain server-local.

### Remainder++

Status: `running`. Parameters: `{"fusion_lr": 0.0002, "inclusion_max": 1.5, "sparsity_scale": 1.0, "view_weights": [1.0, 1.0, 2.0], "visual_mask_lr_scale": 1.0}`.

### Four-epoch fusion-only

Status: `pending`. Parameters: `{"fusion_lr": 0.0002, "inclusion_max": 1.0, "sparsity_scale": 1.0, "view_weights": [1.0, 1.0, 1.0], "visual_mask_lr_scale": 1.0}`.

## Complete Native Recall

### Balanced B0 (a009f49f814a@500) @500

| Dataset | Direction | R@1 % | R@5 % | R@10 % |
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

### Balanced B0 (a009f49f814a@3651) @3651

| Dataset | Direction | R@1 % | R@5 % | R@10 % |
|---|---|---:|---:|---:|
| COCO | I2T | 61.400000 | 84.100000 | 90.080000 |
| COCO | T2I | 42.116000 | 67.844000 | 77.580000 |
| Urban-1k | I2T | 92.200005 | 98.300004 | 99.400002 |
| Urban-1k | T2I | 90.900004 | 98.500007 | 99.200004 |
| Flickr30k-test1k | I2T | 89.200000 | 98.200000 | 99.400000 |
| Flickr30k-test1k | T2I | 71.780000 | 91.480000 | 95.340000 |
| DOCCI | I2T | 78.160000 | 95.660000 | 98.240000 |
| DOCCI | T2I | 79.520000 | 95.640000 | 98.060000 |
| Long-DCI | I2T | 58.958169 | 77.374375 | 83.438569 |
| Long-DCI | T2I | 60.207840 | 78.203104 | 83.372797 |

### Fusion-only C (6d44ae8d5c34@500) @500

| Dataset | Direction | R@1 % | R@5 % | R@10 % |
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

### Fusion-only C (6d44ae8d5c34@3651) @3651

| Dataset | Direction | R@1 % | R@5 % | R@10 % |
|---|---|---:|---:|---:|
| COCO | I2T | 61.840000 | 83.940000 | 90.180000 |
| COCO | T2I | 42.092000 | 67.788000 | 77.628000 |
| Urban-1k | I2T | 91.900003 | 98.400003 | 99.300003 |
| Urban-1k | T2I | 91.000003 | 98.600006 | 99.100006 |
| Flickr30k-test1k | I2T | 89.000000 | 98.100000 | 99.500000 |
| Flickr30k-test1k | T2I | 72.040000 | 91.400000 | 95.420000 |
| DOCCI | I2T | 78.300000 | 95.740000 | 98.220000 |
| DOCCI | T2I | 79.580000 | 95.820000 | 98.120000 |
| Long-DCI | I2T | 59.129177 | 77.729545 | 83.635885 |
| Long-DCI | T2I | 60.431465 | 78.400421 | 83.622731 |

### Previous inclusion1.5 (2ea4a3ae907a@500) @500

| Dataset | Direction | R@1 % | R@5 % | R@10 % |
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

### Previous inclusion1.5 (2ea4a3ae907a@3651) @3651

| Dataset | Direction | R@1 % | R@5 % | R@10 % |
|---|---|---:|---:|---:|
| COCO | I2T | 61.660000 | 83.780000 | 90.120000 |
| COCO | T2I | 42.168000 | 68.020000 | 77.500000 |
| Urban-1k | I2T | 92.100006 | 98.600006 | 99.400002 |
| Urban-1k | T2I | 90.900004 | 98.600006 | 99.200004 |
| Flickr30k-test1k | I2T | 88.600000 | 98.100000 | 99.500000 |
| Flickr30k-test1k | T2I | 72.280000 | 91.500000 | 95.340000 |
| DOCCI | I2T | 78.260000 | 95.640000 | 98.180000 |
| DOCCI | T2I | 79.620000 | 95.740000 | 98.260000 |
| Long-DCI | I2T | 58.787161 | 77.558537 | 83.346488 |
| Long-DCI | T2I | 60.470929 | 78.505656 | 83.859511 |

### Inclusion++ @500

| Dataset | Direction | R@1 % | R@5 % | R@10 % |
|---|---|---:|---:|---:|
| COCO | I2T | 59.580000 | 82.040000 | 88.780000 |
| COCO | T2I | 41.052000 | 66.540000 | 76.188000 |
| Urban-1k | I2T | 89.100003 | 98.200005 | 99.400002 |
| Urban-1k | T2I | 87.200004 | 98.100007 | 99.100006 |
| Flickr30k-test1k | I2T | 85.800000 | 97.400000 | 98.900000 |
| Flickr30k-test1k | T2I | 70.380000 | 90.620000 | 94.580000 |
| DOCCI | I2T | 76.140000 | 94.620000 | 97.500000 |
| DOCCI | T2I | 75.840000 | 94.420000 | 97.460000 |
| Long-DCI | I2T | 54.840831 | 74.401473 | 80.939227 |
| Long-DCI | T2I | 56.853460 | 76.006314 | 81.860037 |

### Inclusion++ @3651

| Dataset | Direction | R@1 % | R@5 % | R@10 % |
|---|---|---:|---:|---:|
| COCO | I2T | 61.560000 | 84.020000 | 90.000000 |
| COCO | T2I | 42.208000 | 67.900000 | 77.620000 |
| Urban-1k | I2T | 92.000002 | 98.500007 | 99.400002 |
| Urban-1k | T2I | 90.700006 | 98.500007 | 99.200004 |
| Flickr30k-test1k | I2T | 89.400000 | 98.200000 | 99.500000 |
| Flickr30k-test1k | T2I | 72.100000 | 91.540000 | 95.400000 |
| DOCCI | I2T | 78.240000 | 95.700000 | 98.200000 |
| DOCCI | T2I | 79.600000 | 95.700000 | 98.180000 |
| Long-DCI | I2T | 58.734544 | 77.308603 | 83.425414 |
| Long-DCI | T2I | 60.378848 | 78.374112 | 83.635885 |

## Current Conclusion

Highest completed three-epoch/horizon3651 Score5_R1: Fusion-only C (6d44ae8d5c34@3651), 72.531265%.

Four-epoch evaluation at3651 has the same number of updates as the original run but a different LR trajectory. Evaluation at4868 is the complete four-epoch result. Neither resets the LR or continues an old horizon3651 checkpoint.

Unfinished experiments cannot be ranked at3651. Five datasets participate in selection; these are fixed-seed tuning results, without a significance or global optimality claim.

## Reproduction

```bash
cd /root/lk_projects/SAID-balanced-three-followup-v1
/root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.three_followup_v1.run --prepare
/root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.three_followup_v1.run --launch
# In the isolated SAID-balanced-four-epoch-v1 worktree:
cd /root/lk_projects/SAID-balanced-four-epoch-v1
/root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.three_followup_v1.four_epoch --prepare
/root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.three_followup_v1.four_epoch --launch
```

Each successful evaluation preserves exact train/export/verify/evaluator commands and runtime commit in evidence/execution/. Failed stages are not automatically retried. No fourth experiment or additional parameter values are scheduled. Large checkpoints, data and caches are not uploaded.
