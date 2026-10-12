# E — Vanilla Long-Caption Fine-tuning（Step3651）

E 从共同 Step0 独立训练至 3651 updates，使用 Full4 Scheduler Horizon=4868；目标为 `10 × L_native(F)`。Mask、Fusion、Dall/D3、Sparsity、HNS、Inclusion 和辅助监督均关闭。

## 训练与导出审计

- 源码 commit：`60873ce6b29dd8347c9d99c73f34a4112e4729cb`；共同 Step0 SHA：`54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`。
- 训练完成：3651/3651 updates；horizon=4868；四卡参数最大差异 `0.0`；最终 NCCL all-reduce=10；峰值 allocated 约 6.85 GiB/卡。
- Checkpoint SHA：`1b53391b2f9fef0a9ba68cab42d539f56d7647902a5378a534f850bf96d89cd6`；裸学生 SHA：`709b60a5c893229226f8eaa7b7f706f5b0360d632fb64bf991529d67b5d17a62`。
- 严格导出：317 个 CLIP tensor，image/text embedding max abs 均为 0；optimizer step=[3651]。

## E 原生五集结果

| 数据集 | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 |
|---|---:|---:|---:|---:|---:|---:|
| COCO | 60.7600 | 83.0200 | 89.7600 | 41.8520 | 67.6680 | 77.2120 |
| Flickr30k-test1k | 88.6000 | 97.7000 | 99.1000 | 72.5400 | 91.2800 | 95.0600 |
| DOCCI | 78.9000 | 96.0800 | 98.3600 | 79.1800 | 95.5000 | 98.1400 |
| Long-DCI | 57.8400 | 75.9537 | 81.8863 | 56.0905 | 74.5330 | 80.1763 |
| Urban-1k | 92.7000 | 98.9000 | 99.7000 | 91.1000 | 98.4000 | 99.1000 |

## 汇总分数（百分比）

| Arm | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I | Urban Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| A.Full | 73.774418 | 78.188029 | 87.040003 | 67.154000 | 94.100000 | 92.600000 | 93.350000 |
| B.Joint-Mask Only | 72.849872 | 76.789120 | 86.165001 | 66.941000 | 93.300000 | 91.500000 | 92.400000 |
| C.Nested Alignment Only | 73.232981 | 77.618968 | 86.535003 | 66.654000 | 93.500000 | 91.800000 | 92.650000 |
| D.Text-only SmartMask | 73.749518 | 78.126531 | 86.905003 | 67.184000 | 94.000000 | 92.200000 | 93.100000 |
| E.Vanilla Long-Caption | 71.956256 | 75.968426 | 85.470003 | 65.938000 | 92.700005 | 91.100007 | 91.900006 |

E 相对历史 A/B/C/D 的解释：E 的 Score5 为 **71.956256**，J_long3 为 **75.968426**，J_long 为 **85.470003**，Short4 为 **65.938000**；Urban I2T/T2I 为 **92.700005/91.100007**。

## Flickr30k-Full

| Direction | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| I2T | 54.538590 | 77.535160 | 84.655319 |
| T2I | 35.035082 | 57.418117 | 66.587799 |

Flickr30k-Full 使用已验证的 31,783 图像/158,915 caption 协议、FP32 native CLIP、无重排；结果是探索性公开基准比较。

## 结论与限制

纯长文本 CLIP 微调达到了中等的综合检索表现，但在本次单 seed、Epoch3 stop 条件下，Score5/J_long3 均低于 Full Sema-CLIP A 与 Text-only SmartMask D；Urban 双向也低于 A/D。它说明仅用完整长 Caption 的 native CLIP 目标可以保持可用表示，但没有证据支持替代当前 Full Sema-CLIP 主模型或自动延长训练。所有公开测试集结果应视为探索性，不能据此宣称无偏 SOTA。
