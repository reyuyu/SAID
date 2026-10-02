# Balanced Four-Epoch Results

All three directed experiments completed. Final evaluation finished at 2026-10-02 15:09:34 Asia/Shanghai.

Final parameters: `{"fusion_lr":0.0002,"visual_mask_lr_scale":1,"view_weights":[1,1,1],"sparsity_scale":1,"inclusion_max":1}`. Four epochs, 1217 updates per epoch, horizon4868 from shared step0.

Final Score5_R1: **72.768147%**, +0.236882 pp versus the original three-epoch fusion-only best, and +0.103432 pp versus this run at3651.

## Score Comparison

| Configuration | Updates | Horizon | Score5_R1 % | J_long3 % | J_long % |
|---|---:|---:|---:|---:|---:|
| Balanced B0 baseline | 500 | 3651 | 69.858294 | 73.601824 | 82.295001 |
| Balanced B0 reference | 3651 | 3651 | 72.444202 | 76.657670 | 85.195002 |
| Original fusion-only best | 3651 | 3651 | 72.531265 | 76.723441 | 85.195001 |
| Inclusion++ | 3651 | 3651 | 72.492140 | 76.608900 | 85.135002 |
| Remainder++ | 3651 | 3651 | 72.169885 | 76.485141 | 85.025002 |
| Four-epoch run at3651 | 3651 | 4868 | 72.664715 | 76.972525 | 85.345001 |
| Four-epoch run at4868 | 4868 | 4868 | 72.768147 | 77.070244 | 85.485003 |

## Final Per-Dataset Recall

All Recall values are percentages. Deltas are R@1 percentage points. Raw floats are preserved in RESULTS.json and evaluator evidence.

| Dataset | Direction | Original3epoch R1 | New run@3651 R1 | Final4epoch R1 | Final R5 | Final R10 | Delta original R1 | Delta new3651 R1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| COCO | I2T | 61.840000 | 61.720000 | 61.700000 | 84.100000 | 90.300000 | -0.140000 | -0.020000 |
| COCO | T2I | 42.092000 | 42.252000 | 42.220000 | 68.032000 | 77.720000 | +0.128000 | -0.032000 |
| Urban-1k | I2T | 91.900003 | 91.900003 | 92.100006 | 98.400003 | 99.300003 | +0.200003 | +0.200003 |
| Urban-1k | T2I | 91.000003 | 91.000003 | 91.100007 | 98.700005 | 99.200004 | +0.100005 | +0.100005 |
| Flickr30k-test1k | I2T | 89.000000 | 88.500000 | 88.900000 | 98.300000 | 99.400000 | -0.100000 | +0.400000 |
| Flickr30k-test1k | T2I | 72.040000 | 72.340000 | 72.440000 | 91.540000 | 95.500000 | +0.400000 | +0.100000 |
| DOCCI | I2T | 78.300000 | 78.820000 | 78.760000 | 95.720000 | 98.260000 | +0.460000 | -0.060000 |
| DOCCI | T2I | 79.580000 | 79.660000 | 79.980000 | 95.880000 | 98.200000 | +0.400000 | +0.320000 |
| Long-DCI | I2T | 59.129177 | 59.484346 | 59.668508 | 78.295185 | 83.793738 | +0.539332 | +0.184162 |
| Long-DCI | T2I | 60.431465 | 60.970797 | 60.812944 | 78.624046 | 84.109445 | +0.381479 | -0.157853 |

## Interpretation

The longer4868-step cosine already adds +0.133450 pp at3651 compared with the original horizon3651 run. Both have3651 updates, but their LR trajectories differ. The extra fourth epoch then adds +0.103432 pp along the same4868-step schedule, giving a combined +0.236882 pp.

The final model improves on the original fusion-only run in Urban, DOCCI, Long-DCI and Flickr T2I R@1. COCO I2T and Flickr I2T R@1 decline slightly. The last epoch also has mixed directional effects, including a decline in Long-DCI T2I relative to the intermediate3651 checkpoint; its mean Score5_R1 still improves.

Inclusion++ with max2 does not beat the original fusion-only best. Remainder++ with weights[1,1,2] and max1.5 lowers the final mean score. The four-epoch fusion-only configuration has the highest measured Score5_R1 among these requested runs. All comparisons are seed0 tuning results on the five examined datasets, without a significance or global-optimality claim.

## Checkpoints and Validation

Intermediate3651 full checkpoint SHA256: `1600bd0fa33c120351ffd8557ca2d8c1183254f58239360b95bcbf6e10f04ff4`.
Final4868 full checkpoint SHA256: `42901d24ae3b0e0147213fd45f5d3f663d9193d17b690536957625901e75500c`.
Final bare student SHA256: `f36438934947ed7f1523fe87e877a0e62dd807dc8a7c89c751eac537e0ab64bb`.
Final server-local checkpoint: `/root/lk_projects/SAID-nest-clip-v1/three_followup_v1/four_epoch/trials/6d44ae8d5c34438e672287b09f3e6930af01473a4aa4905cd90c02c259b2a4a8/step4868/step004868.pt`.

Both checkpoints passed training acceptance and strict export: four-rank parameter differences are zero and image/text embedding export differences are zero. All60 raw Recall values across the two five-dataset evaluations were checked against recorded metrics and bare-student hashes; Score5_R1/J_long3/J_long were independently recomputed. Long-DCI uses the frozen7602-pair reconstructed manifest. There are no recorded failed stages.

The final segment restores the intermediate checkpoint from the same horizon4868 run, preserving optimizer, RNG and the epoch3/batch0 data cursor. The original horizon3651 checkpoint is not resumed.

## Evidence

Automatic full-report commit before this focused report: `4f9f24d697d5e39e16bd922071f2ef37678b2b10`.

THREE_FOLLOWUP_REPORT.md and RESULTS.json contain the unified results. SEARCH_STATE.json and evidence/execution/ preserve exact training/export/verification/evaluation commands, commits, exit codes, diagnostic summaries and checkpoint identities. Native normalized bare-student embeddings only; no masked reranking or DCI Full. Large weights, data and caches remain server-local. No additional experiments are scheduled.
