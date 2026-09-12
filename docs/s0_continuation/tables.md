# S0 vs S0_TriMask_HS: 500 -> 1000 optimizer updates

All numbers are the standard evaluators' own output. the frozen promotion gate is defined at 500 optimizer updates only; it is applied to the @500 rows as a reference and is NOT applied to the @1000 rows, which are a 1000-step budget.

## Metrics per checkpoint

| arm | step | COCO I2T R@1 | COCO T2I R@1 | Urban I2T R@1 | Urban T2I R@1 |
| --- | --- | --- | --- | --- | --- |
| S0_smartclip | 500 | 0.6058 | 0.4124 | 0.8700 | 0.8420 |
| S0_smartclip | 1000 | 0.6172 | 0.4199 | 0.8890 | 0.8570 |
| S0_TriMask_HS | 500 | 0.6006 | 0.4121 | 0.8740 | 0.8370 |
| S0_TriMask_HS | 1000 | 0.6148 | 0.4180 | 0.8900 | 0.8550 |

## Differences (pp = percentage points)

### HS_minus_S0_at_500

| metric | left | right | delta | delta (pp) |
| --- | --- | --- | --- | --- |
| coco/image2text_R1 | 0.6006 | 0.6058 | -0.0052 | -0.52 |
| coco/image2text_R10 | 0.8870 | 0.8906 | -0.0036 | -0.36 |
| coco/image2text_R5 | 0.8240 | 0.8220 | +0.0020 | +0.20 |
| coco/text2image_R1 | 0.4121 | 0.4124 | -0.0003 | -0.03 |
| coco/text2image_R10 | 0.7660 | 0.7662 | -0.0002 | -0.02 |
| coco/text2image_R5 | 0.6710 | 0.6709 | +0.0001 | +0.01 |
| urban/i2t_r1 | 0.8740 | 0.8700 | +0.0040 | +0.40 |
| urban/i2t_r10 | 0.9890 | 0.9880 | +0.0010 | +0.10 |
| urban/i2t_r5 | 0.9750 | 0.9710 | +0.0040 | +0.40 |
| urban/t2i_r1 | 0.8370 | 0.8420 | -0.0050 | -0.50 |
| urban/t2i_r10 | 0.9820 | 0.9810 | +0.0010 | +0.10 |
| urban/t2i_r5 | 0.9630 | 0.9670 | -0.0040 | -0.40 |

### HS_minus_S0_at_1000

| metric | left | right | delta | delta (pp) |
| --- | --- | --- | --- | --- |
| coco/image2text_R1 | 0.6148 | 0.6172 | -0.0024 | -0.24 |
| coco/image2text_R10 | 0.8936 | 0.8946 | -0.0010 | -0.10 |
| coco/image2text_R5 | 0.8288 | 0.8304 | -0.0016 | -0.16 |
| coco/text2image_R1 | 0.4180 | 0.4199 | -0.0020 | -0.20 |
| coco/text2image_R10 | 0.7748 | 0.7760 | -0.0012 | -0.12 |
| coco/text2image_R5 | 0.6777 | 0.6789 | -0.0012 | -0.12 |
| urban/i2t_r1 | 0.8900 | 0.8890 | +0.0010 | +0.10 |
| urban/i2t_r10 | 0.9900 | 0.9930 | -0.0030 | -0.30 |
| urban/i2t_r5 | 0.9770 | 0.9770 | +0.0000 | +0.00 |
| urban/t2i_r1 | 0.8550 | 0.8570 | -0.0020 | -0.20 |
| urban/t2i_r10 | 0.9830 | 0.9850 | -0.0020 | -0.20 |
| urban/t2i_r5 | 0.9680 | 0.9720 | -0.0040 | -0.40 |

### S0_smartclip_1000_minus_500

| metric | left | right | delta | delta (pp) |
| --- | --- | --- | --- | --- |
| coco/image2text_R1 | 0.6172 | 0.6058 | +0.0114 | +1.14 |
| coco/image2text_R10 | 0.8946 | 0.8906 | +0.0040 | +0.40 |
| coco/image2text_R5 | 0.8304 | 0.8220 | +0.0084 | +0.84 |
| coco/text2image_R1 | 0.4199 | 0.4124 | +0.0076 | +0.76 |
| coco/text2image_R10 | 0.7760 | 0.7662 | +0.0098 | +0.98 |
| coco/text2image_R5 | 0.6789 | 0.6709 | +0.0080 | +0.80 |
| urban/i2t_r1 | 0.8890 | 0.8700 | +0.0190 | +1.90 |
| urban/i2t_r10 | 0.9930 | 0.9880 | +0.0050 | +0.50 |
| urban/i2t_r5 | 0.9770 | 0.9710 | +0.0060 | +0.60 |
| urban/t2i_r1 | 0.8570 | 0.8420 | +0.0150 | +1.50 |
| urban/t2i_r10 | 0.9850 | 0.9810 | +0.0040 | +0.40 |
| urban/t2i_r5 | 0.9720 | 0.9670 | +0.0050 | +0.50 |

### S0_TriMask_HS_1000_minus_500

| metric | left | right | delta | delta (pp) |
| --- | --- | --- | --- | --- |
| coco/image2text_R1 | 0.6148 | 0.6006 | +0.0142 | +1.42 |
| coco/image2text_R10 | 0.8936 | 0.8870 | +0.0066 | +0.66 |
| coco/image2text_R5 | 0.8288 | 0.8240 | +0.0048 | +0.48 |
| coco/text2image_R1 | 0.4180 | 0.4121 | +0.0059 | +0.59 |
| coco/text2image_R10 | 0.7748 | 0.7660 | +0.0088 | +0.88 |
| coco/text2image_R5 | 0.6777 | 0.6710 | +0.0067 | +0.67 |
| urban/i2t_r1 | 0.8900 | 0.8740 | +0.0160 | +1.60 |
| urban/i2t_r10 | 0.9900 | 0.9890 | +0.0010 | +0.10 |
| urban/i2t_r5 | 0.9770 | 0.9750 | +0.0020 | +0.20 |
| urban/t2i_r1 | 0.8550 | 0.8370 | +0.0180 | +1.80 |
| urban/t2i_r10 | 0.9830 | 0.9820 | +0.0010 | +0.10 |
| urban/t2i_r5 | 0.9680 | 0.9630 | +0.0050 | +0.50 |

## Frozen 500-step gate (reference row, not applied at 1000)

| arm | COCO I2T R@1 | COCO T2I R@1 | verdict |
| --- | --- | --- | --- |
| S0_smartclip | 0.6058 | 0.4124 | the frozen baseline that defines the gate; the "at least one strictly higher" clause is a challenger rule and cannot apply to the reference itself |
| S0_TriMask_HS | 0.6006 | 0.4121 | did not pass the frozen 500-step promotion gate |

## Paired per-query evidence (same queries, identical protocol)

delta = left - right, positive means the left arm is better. Paired bootstrap over the same query indices (10000 replicates, seed 20260911); McNemar is the exact two-sided test on the discordant queries.

| comparison | direction | R@1 left | R@1 right | delta (pp) | 95% CI (pp) | excludes 0 | left-only | right-only | McNemar p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S0_TriMask_HS@1000 vs S0_smartclip@1000 | image2text | 0.6148 | 0.6172 | -0.24 | [-0.80, +0.30] | False | 93 | 105 | 0.43444637987422646 |
| S0_TriMask_HS@1000 vs S0_smartclip@1000 | text2image | 0.4180 | 0.4199 | -0.20 | [-0.38, -0.02] | True | 240 | 289 | 0.03679221092005949 |
| S0_smartclip@500 vs S0_smartclip@1000 | image2text | 0.6058 | 0.6172 | -1.14 | [-1.82, -0.44] | True | 124 | 181 | 0.0013054354702893294 |
| S0_smartclip@500 vs S0_smartclip@1000 | text2image | 0.4124 | 0.4199 | -0.76 | [-1.02, -0.49] | True | 479 | 668 | 2.6486407907991034e-08 |
| S0_TriMask_HS@500 vs S0_TriMask_HS@1000 | image2text | 0.6006 | 0.6148 | -1.42 | [-2.12, -0.72] | True | 121 | 192 | 7.116482210596628e-05 |
| S0_TriMask_HS@500 vs S0_TriMask_HS@1000 | text2image | 0.4121 | 0.4180 | -0.59 | [-0.85, -0.33] | True | 503 | 650 | 1.6686010391989206e-05 |
| S0_TriMask_HS@500 vs S0_smartclip@500 | image2text | 0.6006 | 0.6058 | -0.52 | [-1.06, +0.02] | False | 82 | 108 | 0.06944420068090527 |
| S0_TriMask_HS@500 vs S0_smartclip@500 | text2image | 0.4121 | 0.4124 | -0.03 | [-0.20, +0.15] | False | 239 | 246 | 0.7853125588725774 |


