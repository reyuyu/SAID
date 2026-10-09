# Visual Patch Gradient @500: NEGATIVE

| Model | Score5 | J_long3 | J_long | Short4 | Urban I2T/T2I | Urban Mean |
|---|---:|---:|---:|---:|---|---:|
| E2-Uniform baseline | 71.314371 | 75.120618 | 84.195003 | 65.605000 | 91.100 / 90.000 | 90.550 |
| E2-VisualGrad | 71.180994 | 74.996323 | 84.025003 | 65.458000 | 90.700 / 89.800 | 90.250 |

Deltas vs E2-Uniform@500 (percentage points): `{"J_long": -0.16999955892562468, "J_long3": -0.12429506682033775, "Score5": -0.1333770400922134, "Short4": -0.14700000000000557, "Urban_I2T": -0.40000081062316895, "Urban_T2I": -0.1999974250793457}`.

| Dataset | I2T R@1/R@5/R@10 (%) | T2I R@1/R@5/R@10 (%) | Δ R@1 I2T/T2I (pp) |
|---|---|---|---|
| COCO | 60.680000 / 82.940000 / 89.300000 | 41.532000 / 67.312000 / 76.976000 | +0.100000 / -0.028000 |
| Urban-1k | 90.700006 / 98.500007 / 99.600005 | 89.800006 / 98.700005 / 99.500006 | -0.400001 / -0.199997 |
| Flickr30k-test1k | 87.800000 / 97.700000 / 99.300000 | 71.820000 / 91.520000 / 95.140000 | -0.300000 / -0.360000 |
| DOCCI | 77.860000 / 95.460000 / 98.100000 | 77.740000 / 95.180000 / 97.880000 | -0.160000 / +0.080000 |
| Long-DCI | 57.024467 / 75.545909 / 81.886346 | 56.853460 / 76.124704 / 82.070508 | -0.131544 / +0.065772 |

The sole production change is the visual Patch `hidden.detach().float()` removal. The old independent-component additivity failure is retained as evidence; the formal gate here is one-shot total-loss A/B gradients.
Global1024 visual-backbone A/B: A=87.152776, B=87.785554, Δ/A=0.198597, cosine(A,B)=0.980448.
Patch-boundary B total norm=0.038614117; alignment component norm=0.038703529; sparsity=0.000809177; hierarchy=0.000003854.
Training completed exactly 500 updates from common step0; all 512000 stream positions and LR records matched the original E2 trajectory. No continuation to 1217/2434/3651/4868 was started.
Classification is exploratory only; it does not establish a retrieval gain or justify automatic continuation.
