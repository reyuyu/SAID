# S0-TriMask-HS 损失权重对比（恰好 500 次更新）

新臂 `S0_TriMask_HS_BAL`：三项对齐权重都取 10、两项稀疏权重都取 2；对照组是冻结的 `S0_TriMask_HS`（10/1/1 + 2/0.2）与门槛基准 `S0_smartclip`。全部数字来自标准评估器。

| 臂 | 权重 | COCO I2T R@1 | COCO T2I R@1 | Urban I2T R@1 | Urban T2I R@1 | 500 步门槛 |
| --- | --- | --- | --- | --- | --- | --- |
| S0_smartclip@500 (gate baseline) | 10 / 1 / 1 + 2 / 0.0 (reference objective) | 0.6058 | 0.4124 | 0.8700 | 0.8420 | the frozen baseline that defines the gate; the challenger clause cannot apply to the reference itself |
| S0_TriMask_HS@500 (frozen v0.2) | 10 / 1 / 1 + 2 / 0.2 | 0.6006 | 0.4121 | 0.8740 | 0.8370 | did not pass the frozen 500-step promotion gate |
| S0_TriMask_HS_BAL@500 (balanced) | 10 / 10 / 10 + 2 / 2 | 0.5892 | 0.4083 | 0.8650 | 0.8280 | did not pass the frozen 500-step promotion gate |

## 差值（百分点）

### S0_TriMask_HS_minus_S0@500

| 指标 | 左 | 右 | 差值 | 差值(pp) |
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

### S0_TriMask_HS_BAL_minus_S0@500

| 指标 | 左 | 右 | 差值 | 差值(pp) |
| --- | --- | --- | --- | --- |
| coco/image2text_R1 | 0.5892 | 0.6058 | -0.0166 | -1.66 |
| coco/image2text_R10 | 0.8812 | 0.8906 | -0.0094 | -0.94 |
| coco/image2text_R5 | 0.8184 | 0.8220 | -0.0036 | -0.36 |
| coco/text2image_R1 | 0.4083 | 0.4124 | -0.0041 | -0.41 |
| coco/text2image_R10 | 0.7654 | 0.7662 | -0.0008 | -0.08 |
| coco/text2image_R5 | 0.6692 | 0.6709 | -0.0018 | -0.18 |
| urban/i2t_r1 | 0.8650 | 0.8700 | -0.0050 | -0.50 |
| urban/i2t_r10 | 0.9860 | 0.9880 | -0.0020 | -0.20 |
| urban/i2t_r5 | 0.9720 | 0.9710 | +0.0010 | +0.10 |
| urban/t2i_r1 | 0.8280 | 0.8420 | -0.0140 | -1.40 |
| urban/t2i_r10 | 0.9800 | 0.9810 | -0.0010 | -0.10 |
| urban/t2i_r5 | 0.9620 | 0.9670 | -0.0050 | -0.50 |

### BAL_minus_HS_default

| 指标 | 左 | 右 | 差值 | 差值(pp) |
| --- | --- | --- | --- | --- |
| coco/image2text_R1 | 0.5892 | 0.6006 | -0.0114 | -1.14 |
| coco/image2text_R10 | 0.8812 | 0.8870 | -0.0058 | -0.58 |
| coco/image2text_R5 | 0.8184 | 0.8240 | -0.0056 | -0.56 |
| coco/text2image_R1 | 0.4083 | 0.4121 | -0.0038 | -0.38 |
| coco/text2image_R10 | 0.7654 | 0.7660 | -0.0007 | -0.07 |
| coco/text2image_R5 | 0.6692 | 0.6710 | -0.0018 | -0.18 |
| urban/i2t_r1 | 0.8650 | 0.8740 | -0.0090 | -0.90 |
| urban/i2t_r10 | 0.9860 | 0.9890 | -0.0030 | -0.30 |
| urban/i2t_r5 | 0.9720 | 0.9750 | -0.0030 | -0.30 |
| urban/t2i_r1 | 0.8280 | 0.8370 | -0.0090 | -0.90 |
| urban/t2i_r10 | 0.9800 | 0.9820 | -0.0020 | -0.20 |
| urban/t2i_r5 | 0.9620 | 0.9630 | -0.0010 | -0.10 |

## 配对逐查询证据（COCO，同一批查询）

delta = 左 − 右，负值表示左侧更差；配对 bootstrap 10000 次重采样，seed 20260911。

| 比较 | 方向 | R@1 左 | R@1 右 | Δ(pp) | 95% CI (pp) | 排除 0 | 仅左对/仅右对 | McNemar p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S0_TriMask_HS_BAL@500 vs S0_TriMask_HS@500 | image2text | 0.5892 | 0.6006 | -1.14 | [-1.74, -0.56] | True | 89/146 | 0.00024315374854476837 |
| S0_TriMask_HS_BAL@500 vs S0_TriMask_HS@500 | text2image | 0.4083 | 0.4121 | -0.38 | [-0.57, -0.19] | True | 244/339 | 9.578249156357118e-05 |
| S0_TriMask_HS_BAL@500 vs S0_smartclip@500 | image2text | 0.5892 | 0.6058 | -1.66 | [-2.40, -0.92] | True | 136/219 | 1.2379218276048482e-05 |
| S0_TriMask_HS_BAL@500 vs S0_smartclip@500 | text2image | 0.4083 | 0.4124 | -0.41 | [-0.62, -0.20] | True | 298/400 | 0.00012857740325593738 |

