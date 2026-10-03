# R-SentenceDrop: Four-Epoch Confirmation

Status: `running`; stage: `6d44ae8d5c34-formal-4868`.

User-authorized continuation of the completed R-SentenceDrop500 experiment. Restore its full checkpoint, AdamW state, all four rank RNG states and loader cursor at update500; continue to4868 with the original horizon4868. The complete trajectory starts at the same shared step0 as the compact old-R baseline.

ViT-B/16,224,context248,seed0,sampling_seed0,4x256,global1024,accumulation1,workers8/rank, BF16 encoders and FP32 parameters/mask/gate/scoring/loss. Frozen coefficients fusion_lr=2e-4,visual_mask_lr_scale=1,view_weights=[1,1,1],sparsity_scale=1,inclusion_max=1. Encoder checkpoint ON,pair OFF,image/text chunks128. Training/data/model source files are identical to the verified500 run. Only the authorized stopping point changes.

R remains an ordered subset of complete suffix sentences, with uniform q in1..m and independent salted SHA256 sampling. No additional tuning, seed, view or loss.

## Main Scores

| Model | Score5_R1 % | J_long3 % | J_long % |
|---|---:|---:|---:|
| Matched old-R@4868 | 72.768147 | 77.070244 | 85.485003 |
| R-SentenceDrop@4868 | pending | pending | pending |

## Reused Old-R Baseline: All Recall

| Dataset | Direction | R@1 % | R@5 % | R@10 % |
|---|---|---:|---:|---:|
| COCO | I2T | 61.700000 | 84.100000 | 90.300000 |
| COCO | T2I | 42.220000 | 68.032000 | 77.720000 |
| Urban-1k | I2T | 92.100006 | 98.400003 | 99.300003 |
| Urban-1k | T2I | 91.100007 | 98.700005 | 99.200004 |
| Flickr30k-test1k | I2T | 88.900000 | 98.300000 | 99.400000 |
| Flickr30k-test1k | T2I | 72.440000 | 91.540000 | 95.500000 |
| DOCCI | I2T | 78.760000 | 95.720000 | 98.260000 |
| DOCCI | T2I | 79.980000 | 95.880000 | 98.200000 |
| Long-DCI | I2T | 59.668508 | 78.295185 | 83.793738 |
| Long-DCI | T2I | 60.812944 | 78.624046 | 84.109445 |

## Provenance

Parent RDrop500 full SHA256: `d55dbc0975718f1b69158d9c8ffa27356e5ba182578a51a6567bd5095c2af113`.
Baseline4868 full SHA256: `42901d24ae3b0e0147213fd45f5d3f663d9193d17b690536957625901e75500c`.
Baseline4868 bare SHA256: `f36438934947ed7f1523fe87e877a0e62dd807dc8a7c89c751eac537e0ab64bb`.

Preflight, raw five-dataset JSON, strict export verification, exact commands/commits and acceptance evidence are in evidence/. Baseline is reused without retraining. Native normalized image/text inner product only; reconstructed Long-DCI7602, no DCI Full. Full weights, data and per-update logs remain local.

## Reproduction

```bash
cd /root/lk_projects/SAID-balanced-rdrop-4epoch-v1
CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.balanced_rdrop_4epoch_v1.verify
/root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.balanced_rdrop_4epoch_v1.run --launch
```

