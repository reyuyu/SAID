# Shared 180-degree signal: root-cause audit

**Decision D: no implementation coordinate bug found; the 180-degree relation exists in live model patch semantics.** All requested pipeline checks passed. This supersedes the earlier report calling the shared D4 signal a confirmed global spatial mapping issue. No heatmap orientation, model, training algorithm or formal benchmark result was changed.

## Denominators

The full audit has **14,334 phrase mentions / 1,000 images**. The spatially separated control has **1,837 unordered pairs / 3,674 directed query-target observations**, involving **2,053 unique phrase IDs / 791 images**. A phrase can recur in several pairs; its repeated occurrences remain in the paired averages.

`all_phrase_pointing` uses all 14,334 phrases. `pair_true_pointing`, `pair_shuffled_pointing`, `pair_semantic_pointing_excess`, `pair_true_mass`, `pair_shuffled_mass` and `pair_semantic_mass_excess` all use the same 3,674 pair observations. Semantic excess is computed within that subset only. The clarified JSON renames fields without changing any metric values.

| Model | Transform | all_phrase_pointing | pair_true_pointing | pair_shuffled_pointing | pair_semantic_pointing_excess |
|---|---|---:|---:|---:|---:|
| A Initial direct | identity | 0.233710 | 0.127654 | 0.149156 | -0.021502 |
| A Initial direct | rotate180 | 0.321473 | 0.189984 | 0.169842 | 0.020142 |
| B Phase 2.1 Router | identity | 0.256662 | 0.150517 | 0.152967 | -0.002450 |
| B Phase 2.1 Router | rotate180 | 0.316450 | 0.198421 | 0.181002 | 0.017420 |
| C Phase 2.2 Router | identity | 0.296498 | 0.156505 | 0.208492 | -0.051987 |
| C Phase 2.2 Router | rotate180 | 0.302358 | 0.177191 | 0.171203 | 0.005988 |
| D Phase 2.2 direct | identity | 0.232385 | 0.131192 | 0.148340 | -0.017148 |
| D Phase 2.2 direct | rotate180 | 0.307521 | 0.181818 | 0.176919 | 0.004899 |

| Model | Transform | pair_true_mass | pair_shuffled_mass | pair_semantic_mass_excess |
|---|---|---:|---:|---:|
| A Initial direct | identity | 0.171515 | 0.185736 | -0.014220 |
| A Initial direct | rotate180 | 0.189378 | 0.184945 | 0.004433 |
| B Phase 2.1 Router | identity | 0.170984 | 0.174213 | -0.003229 |
| B Phase 2.1 Router | rotate180 | 0.187639 | 0.185264 | 0.002376 |
| C Phase 2.2 Router | identity | 0.165517 | 0.192032 | -0.026515 |
| C Phase 2.2 Router | rotate180 | 0.186619 | 0.180464 | 0.006155 |
| D Phase 2.2 direct | identity | 0.163277 | 0.183303 | -0.020026 |
| D Phase 2.2 direct | rotate180 | 0.189215 | 0.184108 | 0.005107 |

## Saved artifacts versus live checkpoints

A deterministic seed-231 sample selects 100 phrase IDs from 100 distinct images, covering every official category. All four variants evaluate exactly the same sample (400 variant-query evaluations). Checkpoints B/C/D are verified against the original audit SHA-256 hashes; A uses the official OpenAI ViT-B/16 weights. Inputs are the original JPEGs with the evaluator's actual preprocess and phrase tokenizer.

Independent forward hooks observe the real visual model at the final transformer block; live patches are projected with the existing ln_post/proj. Direct cosine logits and router linear projections, normalization and matrix product are recomputed explicitly. This does not read saved maps until comparison. FP32, TF32 disabled, inference mode; pass threshold max absolute difference <= 1e-6. The original audit used phrase batches, while the recomputation uses single phrases, allowing normal FP32 reduction/GEMM noise.

| Variant | max_abs_diff | mean_abs_diff | Worst phrase ID | Status |
|---|---:|---:|---|---|
| A Initial direct | 2.14204192e-08 | 1.73726407e-09 | `199412869_s0_p2` | PASS |
| B Phase 2.1 Router | 2.16066837e-07 | 2.16626195e-09 | `6357655519_s2_p2` | PASS |
| C Phase 2.2 Router | 7.4505806e-08 | 2.30756179e-09 | `3639005388_s1_p2` | PASS |
| D Phase 2.2 direct | 3.7252903e-08 | 2.20499009e-09 | `3232994074_s2_p1` | PASS |

## Actual preprocess, crop and EXIF

All 100 selected original images were processed with the actual CLIP transform. Inverting normalization, clipping and rounding to uint8 reproduces the saved PNG **pixel-exactly**: max difference 0, mean difference 0. Before uint8 rounding, the maximum error is 0.00003052 on the 0..255 pixel scale. An independent manual bicubic short-side Resize(224)/CenterCrop(224) also reproduces the PNG exactly. Resize dimensions and crop origin match on all 100 images. Independently rescaled/clipped original boxes match saved transformed boxes with max coordinate difference 0.

GT boxes were rendered directly on recovered model-input RGB for all 100 images. Twenty deterministically selected overlays were visually inspected (IDs saved below): people, body parts, clothes, animals, vehicles, small objects and scene regions align with visible content; partial boxes at crop boundaries follow the clipping convention. No global reflection, rotation or offset was observed. This visual sample cannot exclude every individual annotation error.

`2708744743`, `2103104748`, `2394919002`, `7863806816`, `4903277818`, `2987593918`, `3418113028`, `3051754615`, `2726301121`, `2201480583`, `4910373302`, `4589066890`, `40464580`, `241347204`, `3220151692`, `29339871`, `4781889179`, `296430084`, `3227140905`, `4841972083`.

EXIF Orientation distribution over all 1,000 validation JPEGs: `1=0, 3=0, 6=0, 8=0, missing=1000, other=0`. No non-1 images need special handling. PIL opens the stored raster; the actual preprocess and saved crop use that same raster without EXIF transposition. XML dimensions were checked against the raster in dataset loading; the overlays support the same annotation orientation. No EXIF transposition was introduced.

## Patch and position order

A 224x224 synthetic RGB input assigns a unique scalar code to every 16x16 cell. For all 196 positions, including all four corners, full-image conv1 output is compared with independently convolving the isolated patch. Flattened token `row*14+col` matches that raster cell. The max difference is 2.145767e-6 for both backbones, from different convolution batch shapes (tolerance 1e-5).

A pre-hook checks the actual ln_pre input against `[CLS, raster patches] + positional_embedding`: max difference **0**. The actual transformer output is compared with explicitly applying each block's attention residual and tokenwise MLP residual to the same slots: max difference **0**. Finally, the native/custom patch extraction output is compared with projecting the corresponding final sequence slots: max difference **0**. A regression test deliberately flips transformer output slots and confirms this probe fails. Learned positional embedding content is not assumed to be a coordinate label.

## Identity router

A disposable evaluation-only copy of SaidRouter uses identity q/k modules. On the same Phase 2.2 features, normalized text and tau=0.07, **1,908 phrases** give max difference **4.09781933e-08**, below 1e-6: **PASS**. No checkpoint or production router is modified.

## Where the 180-degree signal first appears

Mean Pearson correlation over the same 100 sampled phrases, excluding zero-variance coverage if any:

| Model | corr(l, W) | corr(rotate180(l), W) | corr(l, rotate180(W)) |
|---|---:|---:|---:|
| A Initial direct | -0.196786 | -0.001236 | -0.001236 |
| B Phase 2.1 Router | -0.031443 | -0.022872 | -0.022872 |
| C Phase 2.2 Router | -0.071093 | -0.023868 | -0.023868 |
| D Phase 2.2 direct | -0.232855 | -0.001563 | -0.001563 |

The rotation advantage is already present in **live pre-softmax logits** for all four variants. Saved attention matches live attention, so serialization and dashboard rendering do not introduce it. The two rotated-correlation expressions agree to numerical precision, as expected from applying the same permutation to the dot product. Rotated correlations remain near zero or negative; this is a reduction in anti-localization, not strong positive semantic grounding.

## Layerwise diagnostic

Official fixed small audit: **128 images / 1,908 phrases**. Indices **3,6,9,11 are zero-based** (fourth, seventh, tenth and final transformer blocks). Every layer uses the existing final ln_post/proj as a **diagnostic projection** and direct phrase cosine attention at tau 0.07. Intermediate layers are not assumed to inhabit a naturally text-aligned CLIP space. No fitting or training occurs.

| Backbone | Block index | Pointing identity / rotate180 | Mass gain identity / rotate180 | Logit correlation identity / rotate180 |
|---|---:|---:|---:|---:|
| initial | 3 | 30.92% / 29.45% | -0.008677 / -0.003977 | -0.088171 / -0.043064 |
| initial | 6 | 20.23% / 31.34% | -0.021411 / -0.003147 | -0.214777 / -0.032907 |
| initial | 9 | 20.34% / 33.28% | -0.030667 / 0.001839 | -0.291742 / 0.006285 |
| initial | 11 | 24.79% / 33.49% | -0.018680 / 0.001906 | -0.210749 / 0.015389 |
| phase22 | 3 | 27.62% / 29.45% | -0.014350 / -0.004902 | -0.128696 / -0.048724 |
| phase22 | 6 | 20.60% / 31.97% | -0.027022 / -0.002787 | -0.244896 / -0.030298 |
| phase22 | 9 | 20.49% / 29.72% | -0.039255 / -0.000386 | -0.318649 / -0.006649 |
| phase22 | 11 | 24.16% / 32.97% | -0.035063 / 0.000195 | -0.261410 / 0.004814 |

The effect is not restricted to the final block: mass/correlation improvement is visible at sampled block 3, with a strong pointing difference by block 6. Initial block 3 is an exception for pointing (identity is better), so it is not a universal literal 180-degree encoding. These diagnostics locate the signal upstream in live patch scores, but do not identify the mechanistic cause of the learned spatial semantics.

## Reproduction and validation

Run from the repository root with the existing said-smartclip environment and a supplied image root:

```bash
python -m eval.salu.grounding_root_cause --mode live --image_root "$FLICKR30K_ROOT"
# Continue only if all four saved_vs_live entries pass:
python -m eval.salu.grounding_root_cause --mode later --image_root "$FLICKR30K_ROOT"
```

Artifacts: `outputs/grounding_root_cause/{selection,saved_vs_live,crop_exif,patch_order,identity_router,layerwise_summary}.json`, per-phrase layerwise JSON, clarified spatial denominator JSON, and 100 PNG overlays. Artifacts and checkpoints are untracked. The later command checks live prerequisites, crop, slot order and identity equality before layerwise evaluation.

Tests add deterministic stratified sampling, real hook extraction, explicit router formula agreement, rotation-correlation equivalence, and detection of an injected sequence permutation. Full suite: **71 passed, 1 skipped, 3 warnings**. No model/train files changed.

**Root cause category D.** Within the tested pipeline, the shared 180-degree advantage is a property of live model patch scores and their relation to annotations, not a coordinate implementation defect. It does not authorize rotating official heatmaps. No prompt sweep, checkpoint trajectory, three-epoch training, Prototype or Unsaid work was performed. Stop for Review.
