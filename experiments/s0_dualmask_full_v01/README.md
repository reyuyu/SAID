# S0-DualMask-Full v0.1

维护者：[reyuyu](https://github.com/reyuyu)。实验分支：`codex/s0-dualmask-full-v01`。

Full v0.1 使用 `10 L_S + 2 S_S + 10 L_U + 2 S_U`；相对 [Clean v0.1](../s0_dualmask_masked_3epoch/README.md)，后缀对齐权重从 1 提高到 10，同时新增系数 2 的有效正配 U 门稀疏约束。两项一起改变，因此只讨论组合效果。S0 前缀目标、候选前缀 mS_j、门结构与全开初始化保持原定义。

## 已完成实验

| 阶段 | 真实训练 SHA | 更新次数 |
|---|---|---|
| 共同初始化 → 500 | `dcd33877f1f77a901d292190834f1ffd83a049a8` | 500 |
| 从 500 严格续训至 3651 | `873b43a5bc000528311e22aac01e92045ac898e0` | 新增 3151，总计 3651 |
| step2000 裸学生补评 | 同一条 3 epoch 训练轨迹的中间检查点 | 没有新增训练更新 |

2026-09-14 同步时，实验分支 HEAD 是 `c6e9280559bbef202f5d17e94adf896a266f6546`，包括之后的干跑跳过与调度保护工具。**分支 HEAD 不等于产生已发表结果的训练 SHA。**

## 结果

每格为百分比 R@1 / R@5 / R@10；原生归一化学生向量做全池内积。

| 协议 | Full step3651 图→文 | Full step3651 文→图 |
|---|---:|---:|
| COCO canonical | 59.420 / 81.520 / 88.740 | 40.296 / 65.576 / 75.296 |
| Urban-1k | 91.800 / 98.600 / 99.200 | 90.700 / 98.300 / 99.100 |
| Flickr30k test1K | 86.900 / 98.100 / 99.300 | 70.200 / 90.620 / 94.720 |
| DOCCI test5K | 77.660 / 95.540 / 98.100 | 78.440 / 95.340 / 98.000 |
| DCI full | 49.263 / 70.455 / 77.732 | 49.263 / 70.096 / 76.784 |
| Long-DCI（重建版） | 57.722 / 76.888 / 82.676 | 60.116 / 78.203 / 83.320 |

[Full step2000 六项结果及与 step3651 对比](STEP2000.md) · [Clean/Full 总表](../RESULTS.md)

## 复现与证据

- [复现命令](REPRODUCE.md)：固定版本、共同初始化、4 卡配置与严格续训。
- [模型实现说明](../../docs/dual_mask_suffix_full_v01/README.md)：四项 loss、正配稀疏定义及梯度测试。
- [500 步原始报告](../../docs/dual_mask_suffix_full_v01/report.md)及[完整 JSON](../../docs/dual_mask_suffix_full_v01/masked_full_formal500_report.json)。
- [3 epoch 原始报告](../../docs/dual_mask_suffix_full_v01/continuation_3epoch_report.md)及[完整 JSON](../../docs/dual_mask_suffix_full_v01/continuation_3epoch_report.json)。
- [step3651 原生评测证据](evidence/step3651/)；[step2000 原生评测证据](evidence/step2000/)。
- [权重 SHA256 与数据清单](../ASSETS.md)。权重与数据不放入 Git。

原 3 epoch 报告写作时中间检查点尚未评测；2026-09-14 已补齐 step2000，原文保留采集时状态，新状态以本目录为准。没有重训 S0、没有 native 正式对照，也没有在本次同步中追加训练。
