# Balanced-Stack-Patch: Final Three-Epoch Results

Generated UTC: 2026-09-30T16:08:27.262704+00:00

Training commit: `8e65ad077fefd9fea8be5fe6c1fa01b0079f365f`. Evaluation commit: `913841fbc6fcb12c83dfdaa0647b5bc4557fb085`.

The audited step500 formal state continued to3651, restoring the model, channel gate, optimizer and four rank RNG states. The original three-epoch schedule and method were unchanged. Only the final full training checkpoint was saved in the continuation.

## J_long

| Model | J_long % | Balanced3651 minus model pp |
|---|---:|---:|
| Balanced@3651 | 85.195 | +0.000 |
| Balanced@500 | 82.295 | +2.900 |
| TI-fast@3651 | 84.765 | +0.430 |
| VCP@3651 | 84.745 | +0.450 |

J_long is the frozen Urban/DOCCI bidirectional R@1 average. Long-DCI does not change its definition. Balanced@500 shows the budget extension, while TI-fast@3651 and VCP@3651 provide matched-budget method comparisons. One seed does not establish statistical significance.

## Native Retrieval

All recall values are percentages. Bare-student normalized image/text embeddings use plain inner product without mask, fusion, reranking or ensemble. Differences in JSON are percentage points.

| Dataset | Direction | Recall | Balanced@500 | Balanced@3651 | TI-fast@3651 | VCP@3651 | Balanced-TI pp |
|---|---|---:|---:|---:|---:|---:|---:|
| COCO | I2T | R@1 | 59.60 | 61.40 | 61.66 | 61.42 | -0.26 |
| COCO | I2T | R@5 | 81.88 | 84.10 | 83.44 | 83.66 | +0.66 |
| COCO | I2T | R@10 | 88.80 | 90.08 | 90.14 | 89.86 | -0.06 |
| COCO | T2I | R@1 | 40.93 | 42.12 | 41.94 | 41.90 | +0.17 |
| COCO | T2I | R@5 | 66.64 | 67.84 | 67.61 | 67.75 | +0.24 |
| COCO | T2I | R@10 | 76.32 | 77.58 | 77.36 | 77.38 | +0.22 |
| Urban-1k | I2T | R@1 | 89.70 | 92.20 | 91.50 | 91.80 | +0.70 |
| Urban-1k | I2T | R@5 | 98.40 | 98.30 | 98.50 | 98.50 | -0.20 |
| Urban-1k | I2T | R@10 | 99.50 | 99.40 | 99.30 | 99.30 | +0.10 |
| Urban-1k | T2I | R@1 | 87.40 | 90.90 | 89.80 | 89.80 | +1.10 |
| Urban-1k | T2I | R@5 | 98.20 | 98.50 | 98.20 | 98.20 | +0.30 |
| Urban-1k | T2I | R@10 | 99.10 | 99.20 | 98.90 | 98.80 | +0.30 |
| Flickr30k-test1k | I2T | R@1 | 86.10 | 89.20 | 88.00 | 88.50 | +1.20 |
| Flickr30k-test1k | I2T | R@5 | 97.50 | 98.20 | 97.90 | 98.10 | +0.30 |
| Flickr30k-test1k | I2T | R@10 | 98.80 | 99.40 | 99.50 | 99.50 | -0.10 |
| Flickr30k-test1k | T2I | R@1 | 70.34 | 71.78 | 71.96 | 71.92 | -0.18 |
| Flickr30k-test1k | T2I | R@5 | 90.66 | 91.48 | 91.22 | 91.30 | +0.26 |
| Flickr30k-test1k | T2I | R@10 | 94.68 | 95.34 | 95.16 | 95.16 | +0.18 |
| DOCCI | I2T | R@1 | 76.18 | 78.16 | 78.48 | 78.10 | -0.32 |
| DOCCI | I2T | R@5 | 94.56 | 95.66 | 95.64 | 95.46 | +0.02 |
| DOCCI | I2T | R@10 | 97.56 | 98.24 | 98.16 | 98.08 | +0.08 |
| DOCCI | T2I | R@1 | 75.90 | 79.52 | 79.28 | 79.28 | +0.24 |
| DOCCI | T2I | R@5 | 94.72 | 95.64 | 95.56 | 95.70 | +0.08 |
| DOCCI | T2I | R@10 | 97.54 | 98.06 | 98.08 | 98.02 | -0.02 |
| Long-DCI | I2T | R@1 | 55.21 | 58.96 | 58.67 | not evaluated | +0.29 |
| Long-DCI | I2T | R@5 | 74.61 | 77.37 | 77.45 | not evaluated | -0.08 |
| Long-DCI | I2T | R@10 | 81.04 | 83.44 | 83.35 | not evaluated | +0.09 |
| Long-DCI | T2I | R@1 | 57.22 | 60.21 | 60.17 | not evaluated | +0.04 |
| Long-DCI | T2I | R@5 | 76.40 | 78.20 | 78.20 | not evaluated | +0.00 |
| Long-DCI | T2I | R@10 | 82.06 | 83.37 | 83.28 | not evaluated | +0.09 |

## Cost and Integrity

Training loop time: 1.920 hours. Real DataLoader full-cycle mean/median/P95/max: 2.109/2.089/2.206/12.059s. First resumed update: 7.044s. Slow regular updates are preserved in JSON.

All four ranks completed3151 continuation updates, with consecutive logs501-3651, finite losses and gradients, final parameter max difference0 and successful NCCL all-reduce. Export strictly loads at optimizer step3651 with native image/text embedding error0.

| Rank | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|
| 0 | 27.758 | 29.166 |
| 1 | 27.758 | 29.166 |
| 2 | 27.758 | 29.168 |
| 3 | 27.758 | 29.166 |

## Mechanism

The values below average the last50 updates. O/E are P/R in the implementation.

| Metric | Balanced@500 | Balanced@3651 |
|---|---:|---:|
| common_loss | 9.477809 | 6.211138 |
| F_i2t | 0.055817 | 0.020619 |
| F_t2i | 0.063871 | 0.023753 |
| O_i2t | 0.195840 | 0.111423 |
| O_t2i | 0.258488 | 0.160085 |
| E_i2t | 0.875230 | 0.536526 |
| E_t2i | 0.957907 | 0.633294 |
| F_keep_ratio | 0.894205 | 0.815760 |
| O_keep_ratio | 0.867089 | 0.722608 |
| E_keep_ratio | 0.866760 | 0.757717 |
| inc | 0.021616 | 0.021091 |
| inc_weight | 1.000000 | 1.000000 |
| hard_inclusion_violation | 0.022643 | 0.037417 |
| oe_iou | 0.872802 | 0.727491 |

Gate/logit diagnostics were preset only through update500, so final gate quantiles, logit moments and replacement-image switching are not claimed here. No additional mechanism benchmark was run.

## Protocol and Artifacts

COCO canonical5000 images/25000 texts (similarity chunk512), Urban1000 pairs, Flickr test1K1000 images/5000 texts, DOCCI5000 pairs, Long-DCI7602 pairs. Image batch64 on cuda:0. No DCI Full evaluation was run. VCP has no Long-DCI measurement in this comparison.

Training checkpoint SHA256: `ff986c5ca8cf94e718f1f5b28eea54760e3c92416c9e8de63aaf98917fec9234`.

Bare student SHA256: `0f8db00860f119d43a72fdb1a09efab5c570a31ddba3ac23c93c5c243708d086`.

Long-DCI manifest SHA256: `8890a2be15e2b64c142f9a1e39224d34b6ce92fc17ffb6099e14942cc8161c4b`.

Raw evaluator JSON, actual code commits, commands, stage exit codes, timing, hash and strict-export evidence are preserved under evidence/. Weights, data and full training/token logs remain server-local. All training and evaluation stop at the requested final model; no new seeds or parameter scans are started.
