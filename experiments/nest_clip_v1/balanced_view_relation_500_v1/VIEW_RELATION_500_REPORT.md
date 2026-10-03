# Balanced View-Relation500

Status: `completed`; stage: `view-relation-single500-complete`.

Single variable: add0.5*[ReLU(c_RP-c_PP)+ReLU(c_PR-c_RR)] to the original inclusion term, using unscaled FP32 cosine with normalize eps1e-6. Only cross masks detach; correct masks retain Hard-ST gradients and z/t keep gradients. Reuse positive masks and cached existing P/R encoder outputs. No extra encoder, mask generation, model module, fourth view, transpose constraints, margin or CE denominator change.

Frozen compact old-R Balanced recipe: B16,224,context248,seed0,sampling_seed0,4x256/global1024, accum1,workers8,encoder checkpoint ON,pair OFF,chunks128x128. Original audited runtime; no cancelled runtime optimizations. fusion2e-4,visual scale1,view1:1:1,sparse1,inclusion_max1. The original min(1,completed/200) scales inclusion+sibling together. H4868, stop500.

Run a5-step smoke, then independently start formal500 from the shared step0, never smoke or an old500 checkpoint. Reuse the previously completed matched baseline.

## Main Scores

| Model | Score5_R1 % | J_long3 % | J_long % |
|---|---:|---:|---:|
| Matched Balanced old-R@500 | 69.900394 | 73.603324 | 82.340003 |
| View-Relation@500 | 69.842827 | 73.561378 | 82.185003 |
| Delta pp | -0.057568 | -0.041946 | -0.155000 |

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

## View-Relation Complete Recall

| Dataset | Direction | R@1 % | R@5 % | R@10 % |
|---|---|---:|---:|---:|
| COCO | I2T | 59.640000 | 81.760000 | 88.840000 |
| COCO | T2I | 40.940000 | 66.512000 | 76.164000 |
| Urban-1k | I2T | 89.300007 | 98.300004 | 99.300003 |
| Urban-1k | T2I | 87.200004 | 98.100007 | 99.100006 |
| Flickr30k-test1k | I2T | 86.000000 | 97.200000 | 98.800000 |
| Flickr30k-test1k | T2I | 70.480000 | 90.760000 | 94.840000 |
| DOCCI | I2T | 76.240000 | 94.720000 | 97.540000 |
| DOCCI | T2I | 76.000000 | 94.540000 | 97.480000 |
| Long-DCI | I2T | 55.511707 | 74.822415 | 81.110234 |
| Long-DCI | T2I | 57.116548 | 76.348329 | 82.070508 |

## R@1 Changes

| Dataset | Direction | Delta pp |
|---|---|---:|
| COCO | I2T | -0.320000 |
| COCO | T2I | +0.016000 |
| Urban-1k | I2T | +0.300002 |
| Urban-1k | T2I | -0.700003 |
| Flickr30k-test1k | I2T | -0.200000 |
| Flickr30k-test1k | T2I | +0.180000 |
| DOCCI | I2T | -0.080000 |
| DOCCI | T2I | -0.140000 |
| Long-DCI | I2T | +0.118390 |
| Long-DCI | T2I | +0.249934 |

## Relation Diagnostics

| Metric | step1 | step100 | step200 | step500 | last50 |
|---|---:|---:|---:|---:|---:|
| inc | 0.02392925 | 0.00731635 | 0.01236009 | 0.01871979 | 0.01991879 |
| sib_loss | 0.01172170 | 0.00309060 | 0.00293754 | 0.00334233 | 0.00371801 |
| relation_loss | 0.03565095 | 0.01040695 | 0.01529763 | 0.02206213 | 0.02363680 |
| P_violation_rate | 0.49902344 | 0.36132812 | 0.36621094 | 0.27539062 | 0.28425537 |
| R_violation_rate | 0.44628906 | 0.43750000 | 0.45214844 | 0.38769531 | 0.39105786 |
| P_violation_magnitude | 0.01258764 | 0.00263848 | 0.00256199 | 0.00242626 | 0.00282582 |
| R_violation_magnitude | 0.01085575 | 0.00354273 | 0.00331309 | 0.00425841 | 0.00461021 |
| c_PP_mean | 0.19165553 | 0.33937338 | 0.34302357 | 0.33282280 | 0.33561274 |
| c_RP_mean | 0.19150062 | 0.33457839 | 0.33838081 | 0.32205474 | 0.32484410 |
| c_RR_mean | 0.16123809 | 0.31251401 | 0.31759295 | 0.30637029 | 0.30755377 |
| c_PR_mean | 0.15222891 | 0.31059733 | 0.31587097 | 0.29700142 | 0.29929960 |
| P_cosine_preference | 0.00015491 | 0.00479499 | 0.00464275 | 0.01076806 | 0.01076865 |
| R_cosine_preference | 0.00900918 | 0.00191668 | 0.00172198 | 0.00936887 | 0.00825417 |
| oe_iou | 0.72681761 | 0.91325414 | 0.90474999 | 0.89176905 | 0.88381430 |
| F_keep_ratio | 0.46645546 | 0.92259789 | 0.91948509 | 0.89197540 | 0.88438332 |
| O_keep_ratio | 0.46587181 | 0.89991570 | 0.90022087 | 0.87442970 | 0.86733778 |
| E_keep_ratio | 0.46631622 | 0.88298988 | 0.88267136 | 0.86785889 | 0.86195320 |

## Resource Summary

```json
{
  "mean_seconds": 2.102127424875895,
  "median_seconds": 2.0823862552642822,
  "p95_seconds": 2.202843427658081,
  "max_seconds": 2.3484864234924316,
  "regular_updates": 495,
  "peak_allocated_gib": 27.76184606552124,
  "peak_reserved_gib": 28.453125
}
```

## Conclusion

View-Relation@500 does not improve the matched baseline.
No statistical significance claim. Violation reduction alone does not qualify a better model. Stop after the five500-step native evaluations; no full4868 run, coefficient/margin/constraint variants or seeds.

## Correctness, Guard and Provenance

Unit tests cover no violation, isolated P/R mask gradient directions, equality at zero margin, coefficient0 bitwise loss/all gradients/AdamW recovery, no extra encoder calls and F-only fallback. A four-rank explicit full-global reference checks normal, sparse, zero-valid, V1 and tail cases. Global means use sum_valid then the original world/valid_count DDP scaling; no repeated averaging.

Frozen collapse guard:25-update running means; after8 consecutive bad windows stop if P/R IoU < max(0.15,0.25*matched-baseline-window-IoU), or any positive keep ratio outside(0.02,0.995). Guard only stops and records, never retunes loss, coefficient, masks or valid samples.

Reused baseline full SHA256: `f44e3ab0566a12e189514a92fedac3416299f87ca129b649a2c76a63b222b3f8`.
Reused baseline bare SHA256: `90e60b028a7d8c038b41512c295bcede21ccc5cb57d7b31c3b3bd42af75dcf6e`.
New full SHA256: `cf175c13eeef5d6efccdbe0dc9932539f0abce4274e5f7a5d5c0a2da62fbbfb7`.
New bare SHA256: `354de7080f942a4c82e792d12270495cb4955327163eefe388395c7ed2845685`.

Strict bare export and all five frozen native protocols use normalized image/full-text embeddings and plain inner product only. Reconstructed Long-DCI7602, no DCI Full, training mask, gate, sibling score, reranking or ensemble in inference. Raw JSON and small evidence are here; large checkpoints/data/cache remain on the server.

## Reproduction

```bash
cd /root/lk_projects/SAID-balanced-view-relation-500-v1
CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.balanced_view_relation_500_v1.verify
/root/miniconda3/envs/said-repro/bin/python -m experiments.nest_clip_v1.balanced_view_relation_500_v1.run --launch
```
