# Balanced View-Relation500

Status: `running`; stage: `6d44ae8d5c34-formal-500`.

Single variable: add0.5*[ReLU(c_RP-c_PP)+ReLU(c_PR-c_RR)] to the original inclusion term, using unscaled FP32 cosine with normalize eps1e-6. Only cross masks detach; correct masks retain Hard-ST gradients and z/t keep gradients. Reuse positive masks and cached existing P/R encoder outputs. No extra encoder, mask generation, model module, fourth view, transpose constraints, margin or CE denominator change.

Frozen compact old-R Balanced recipe: B16,224,context248,seed0,sampling_seed0,4x256/global1024, accum1,workers8,encoder checkpoint ON,pair OFF,chunks128x128. Original audited runtime; no cancelled runtime optimizations. fusion2e-4,visual scale1,view1:1:1,sparse1,inclusion_max1. The original min(1,completed/200) scales inclusion+sibling together. H4868, stop500.

Run a5-step smoke, then independently start formal500 from the shared step0, never smoke or an old500 checkpoint. Reuse the previously completed matched baseline.

## Main Scores

| Model | Score5_R1 % | J_long3 % | J_long % |
|---|---:|---:|---:|
| Matched Balanced old-R@500 | 69.900394 | 73.603324 | 82.340003 |
| View-Relation@500 | pending | pending | pending |

## Matched Baseline Complete Recall

| Dataset | Direction | R@1 % | R@5 % | R@10 % |
|---|---|---:|---:|---:|
| COCO | I2T | 59.960000 | 81.820000 | 88.680000 |
| COCO | T2I | 40.924000 | 66.660000 | 76.268000 |
| Urban-1k | I2T | 89.000005 | 98.200005 | 99.400002 |
| Urban-1k | T2I | 87.900007 | 98.200005 | 99.100006 |
| Flickr30k-test1k | I2T | 86.200000 | 97.200000 | 98.800000 |
| Flickr30k-test1k | T2I | 70.300000 | 90.520000 | 94.660000 |
| DOCCI | I2T | 76.320000 | 94.540000 | 97.560000 |
| DOCCI | T2I | 76.140000 | 94.480000 | 97.520000 |
| Long-DCI | I2T | 55.393318 | 74.875033 | 81.057616 |
| Long-DCI | T2I | 56.866614 | 76.229939 | 82.044199 |

## Correctness, Guard and Provenance

Unit tests cover no violation, isolated P/R mask gradient directions, equality at zero margin, coefficient0 bitwise loss/all gradients/AdamW recovery, no extra encoder calls and F-only fallback. A four-rank explicit full-global reference checks normal, sparse, zero-valid, V1 and tail cases. Global means use sum_valid then the original world/valid_count DDP scaling; no repeated averaging.

Frozen collapse guard:25-update running means; after8 consecutive bad windows stop if P/R IoU < max(0.15,0.25*matched-baseline-window-IoU), or any positive keep ratio outside(0.02,0.995). Guard only stops and records, never retunes loss, coefficient, masks or valid samples.

Reused baseline full SHA256: `f44e3ab0566a12e189514a92fedac3416299f87ca129b649a2c76a63b222b3f8`.
Reused baseline bare SHA256: `90e60b028a7d8c038b41512c295bcede21ccc5cb57d7b31c3b3bd42af75dcf6e`.

Strict bare export and all five frozen native protocols use normalized image/full-text embeddings and plain inner product only. Reconstructed Long-DCI7602, no DCI Full, training mask, gate, sibling score, reranking or ensemble in inference. Raw JSON and small evidence are here; large checkpoints/data/cache remain on the server.

## Reproduction

```bash
cd /root/lk_projects/SAID-balanced-view-relation-500-v1
CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.balanced_view_relation_500_v1.verify
/root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.balanced_view_relation_500_v1.run --launch
```
