# Agreement-Mask Step1217 晋级判断

Agreement-Mask 从完整 Step500 checkpoint 恢复，严格新增 717 updates 到 Step1217。没有改变模型、loss、optimizer、RNG、采样器、数据或 scheduler horizon。

- Step500 parent SHA256：`4908184d40c2db5886a9169dd57ac7698de8e617b58851cd76b5ad390ccc2b76`
- Step1217 training checkpoint SHA256：`f8a6695ab935a3488f21a5ac30e8cd8e3f824e89dbcc1dab7f590abc38e27fb5`
- Step1217 bare student SHA256：`b794ad3670cbf6be2ef843e5c17aa97135fc06b53b98e8aee07012bb83be5a64`
- 717/717 updates 完成，四 rank 参数差异为 0，NCCL all-reduce 为 10
- 裸学生严格导出通过，image/text 最大绝对误差均为 0

## 结果

| 指标 | A-Matched@1217 | Agreement@1217 | 差值 |
|---|---:|---:|---:|
| Score5 | 72.829267 | 72.517009 | -0.312258 pp |
| J_long3 | 76.984112 | 76.639682 | -0.344430 pp |
| J_long | 85.905003 | 85.635003 | -0.270000 pp |
| Short4 | 66.597 | 66.333 | -0.264 pp |
| Urban I2T | 93.100 | 92.700 | -0.400 pp |
| Urban T2I | 91.600 | 91.100 | -0.500 pp |
| Urban Mean | 92.350 | 91.900 | -0.450 pp |

| 数据集 | Agreement I2T R@1/5/10 | Agreement T2I R@1/5/10 |
|---|---|---|
| COCO | 61.08 / 83.54 / 89.34 | 42.472 / 68.144 / 77.680 |
| Flickr30k-Test1K | 89.10 / 98.00 / 99.30 | 72.68 / 91.84 / 95.28 |
| DOCCI | 79.02 / 96.12 / 98.22 | 79.72 / 95.50 / 98.02 |
| Long-DCI | 57.643 / 76.769 / 82.649 | 59.655 / 77.664 / 83.123 |
| Urban-1k | 92.70 / 98.90 / 99.60 | 91.10 / 98.70 / 99.30 |

## Mask 状态

Step1217 诊断批次的 F/Dall/D3 keep ratio 为 80.177%/79.714%/76.478%。sigmoid 饱和比例约 2.68%–2.73%，没有全开或全关。`theta` 梯度范数为 0.08551，有界偏置最大绝对值为 0.00420，参数仍在正常学习。

Dall→F 和 D3→Dall 硬包含违反率为 3.912% 和 6.670%，IoU 为 0.902 和 0.810。固定文本、替换图像后的 mask switch rate 为 F/Dall/D3 = 13.23%/13.62%/15.81%，说明视觉条件仍然参与 mask 生成；这些数值没有显示训练崩溃，但 D3 层次约束比 Step500 更松。

## 固定晋级门槛

本轮预先固定的条件是 Score5 至少 `+0.10 pp`、J_long3 至少 `+0.10 pp`、Urban Mean 下降不超过 `0.20 pp`，并且 Mask 数值正常。

- Score5：未通过（-0.312 pp）
- J_long3：未通过（-0.344 pp）
- Urban Mean：未通过（-0.450 pp）
- Mask 数值健康：通过

最终判定：**`STOP_1217`**。Agreement-Mask 不满足晋级条件，不自动继续 Full4。
