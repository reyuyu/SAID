# VCP-Mask: Full Three-Epoch Native Results

Training commit: `7236101ad2370df8cf9d12d495d2476352fd6f7a`. Evaluation commit: `82239ef4884522807fb9f7f4e9b66ef08b229a48`.

The audited VCP@500 formal state was continued to update 3651 with the unchanged three-epoch cosine schedule. Only the final training checkpoint was saved in this continuation. All four ranks completed 3151 new updates.

J_long: VCP@500 81.725%, VCP@3651 84.745%, TI-fast@3651 84.765%. VCP change from step500: +3.020 pp; difference from TI-fast@3651: -0.020 pp.

Against TI-fast at the same budget, VCP has 8 wins, 7 ties and 9 losses across the 24 recall values. This single-seed experiment does not establish statistical significance.

## Native Retrieval

All recall values are percentages; differences are percentage points. Bare-student normalized image/text vectors are scored by inner product, without masks, adapters, fusion or reranking.

| Dataset | Direction | Metric | VCP@500 | VCP@3651 | TI-fast@3651 | VCP3651-500 | VCP-TI3651 |
|---|---|---:|---:|---:|---:|---:|---:|
| COCO | I2T | R@1 | 58.58 | 61.42 | 61.66 | +2.84 | -0.24 |
| COCO | I2T | R@5 | 81.18 | 83.66 | 83.44 | +2.48 | +0.22 |
| COCO | I2T | R@10 | 88.34 | 89.86 | 90.14 | +1.52 | -0.28 |
| COCO | T2I | R@1 | 40.52 | 41.90 | 41.94 | +1.39 | -0.04 |
| COCO | T2I | R@5 | 66.15 | 67.75 | 67.61 | +1.60 | +0.14 |
| COCO | T2I | R@10 | 75.89 | 77.38 | 77.36 | +1.50 | +0.02 |
| Urban-1k | I2T | R@1 | 89.40 | 91.80 | 91.50 | +2.40 | +0.30 |
| Urban-1k | I2T | R@5 | 97.60 | 98.50 | 98.50 | +0.90 | +0.00 |
| Urban-1k | I2T | R@10 | 99.30 | 99.30 | 99.30 | +0.00 | +0.00 |
| Urban-1k | T2I | R@1 | 86.10 | 89.80 | 89.80 | +3.70 | +0.00 |
| Urban-1k | T2I | R@5 | 97.70 | 98.20 | 98.20 | +0.50 | +0.00 |
| Urban-1k | T2I | R@10 | 98.70 | 98.80 | 98.90 | +0.10 | -0.10 |
| Flickr30k-test1k | I2T | R@1 | 85.70 | 88.50 | 88.00 | +2.80 | +0.50 |
| Flickr30k-test1k | I2T | R@5 | 97.00 | 98.10 | 97.90 | +1.10 | +0.20 |
| Flickr30k-test1k | I2T | R@10 | 98.80 | 99.50 | 99.50 | +0.70 | +0.00 |
| Flickr30k-test1k | T2I | R@1 | 70.08 | 71.92 | 71.96 | +1.84 | -0.04 |
| Flickr30k-test1k | T2I | R@5 | 90.18 | 91.30 | 91.22 | +1.12 | +0.08 |
| Flickr30k-test1k | T2I | R@10 | 94.32 | 95.16 | 95.16 | +0.84 | +0.00 |
| DOCCI | I2T | R@1 | 75.82 | 78.10 | 78.48 | +2.28 | -0.38 |
| DOCCI | I2T | R@5 | 94.46 | 95.46 | 95.64 | +1.00 | -0.18 |
| DOCCI | I2T | R@10 | 97.42 | 98.08 | 98.16 | +0.66 | -0.08 |
| DOCCI | T2I | R@1 | 75.58 | 79.28 | 79.28 | +3.70 | +0.00 |
| DOCCI | T2I | R@5 | 94.20 | 95.70 | 95.56 | +1.50 | +0.14 |
| DOCCI | T2I | R@10 | 97.32 | 98.02 | 98.08 | +0.70 | -0.06 |

## Training and Export

Training-loop wall time: 1.907 hours. Logged regular step mean/median/P95/max: 2.014/1.994/2.109/3.186 seconds. These per-rank update diagnostics exclude DataLoader waiting and disk writes; the complete loop time includes them.

All 3151 continuation records are consecutive. Losses and gradients are finite; final rank parameter difference is 0 and NCCL all-reduce passes. Strict export loads at optimizer step 3651 and native image/text embedding maximum absolute errors are 0.

| Rank | Peak allocated GiB | Peak reserved GiB | Final parameter difference |
|---|---:|---:|---:|
| 0 | 23.421 | 25.141 | 0.0 |
| 1 | 23.421 | 25.141 | 0.0 |
| 2 | 23.421 | 25.141 | 0.0 |
| 3 | 23.421 | 25.141 | 0.0 |

Regular logged updates over 3s: `{'1218': 3.086760923266411, '2435': 3.186002030968666}`. First resumed update: 6.482s. All slow regular entries are retained.

## Mechanism

The following values average the last 50 updates. O/E are the implementation names for P/R.

| Quantity | VCP@500 | VCP@3651 |
|---|---:|---:|
| common_loss | 9.383970 | 6.312041 |
| F_i2t | 0.057781 | 0.021013 |
| F_t2i | 0.061920 | 0.023622 |
| O_i2t | 0.195797 | 0.115923 |
| O_t2i | 0.253600 | 0.160806 |
| E_i2t | 0.878031 | 0.547108 |
| E_t2i | 0.924270 | 0.634769 |
| F_keep_ratio | 0.913194 | 0.855970 |
| O_keep_ratio | 0.883958 | 0.740038 |
| E_keep_ratio | 0.878402 | 0.783834 |
| inc | 0.027046 | 0.027143 |
| inc_weight | 1.000000 | 1.000000 |
| hard_inclusion_violation | 0.026032 | 0.044357 |
| oe_iou | 0.848975 | 0.678362 |
| q_minus_w_norm | 0.433955 | 0.527443 |
| w_norm | 0.694807 | 0.544228 |
| F_hard_mask_switch_fraction | 0.031407 | 0.098630 |
| O_hard_mask_switch_fraction | 0.040933 | 0.123533 |
| E_hard_mask_switch_fraction | 0.034444 | 0.110745 |

## Protocol and Evidence

COCO canonical: 5000 images/25000 captions, similarity chunk 512. Urban-1k: 1000 pairs. Flickr30k test1K: 1000 images/5000 captions. DOCCI test5K: 5000 pairs. Image batch 64 on cuda:0. DCI and Long-DCI were not run.

Final full-checkpoint SHA256: `4fbbf8d6239a851552d0ab63a7b82304bf68d667fea4732624a7b3adbab2afa0`.

Bare-student SHA256: `cedc2a210b40f7d5d39a722790a562a976b41564c50d8e366ecf73e0381b7656`.

Original evaluator JSON, commands, evaluator commit, stage timings, exit codes and export/acceptance evidence are under `evidence/`. Full training checkpoints, student weights, data and large logs remain on the server. Results use one seed on previously examined benchmarks. The 3651-update model is not ranked against unrelated 500-update experiments.
