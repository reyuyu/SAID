# SAID 原生检索结果

实验定义：[Clean v0.1](s0_dualmask_masked_3epoch/README.md)（后缀 1、U 稀疏 0）、[Full v0.1](s0_dualmask_full_v01/README.md)（后缀 10、U 稀疏 2）。S0 的 `10 L_S + 2 S_S` 在两版中保留。

单位：百分比；各格依次 R@1 / R@5 / R@10。JSON 保留原始浮点精度。

| 协议 | 版本 / 更新次数 | 图片 / 文本 | 图→文 R@1/5/10 | 文→图 R@1/5/10 |
|---|---|---:|---:|---:|
| COCO canonical | Clean-v0.1 / 3651 | 5000 / 25000 | 61.340 / 83.340 / 89.480 | 42.272 / 67.764 / 77.380 |
| COCO canonical | Full-v0.1 / 3651 | 5000 / 25000 | 59.420 / 81.520 / 88.740 | 40.296 / 65.576 / 75.296 |
| COCO canonical | Full-v0.1 / 2000 | 5000 / 25000 | 59.560 / 81.500 / 88.740 | 40.392 / 65.660 / 75.428 |
| Urban-1k | Clean-v0.1 / 3651 | 1000 / 1000 | 91.400 / 98.400 / 99.400 | 89.500 / 97.800 / 98.900 |
| Urban-1k | Full-v0.1 / 3651 | 1000 / 1000 | 91.800 / 98.600 / 99.200 | 90.700 / 98.300 / 99.100 |
| Urban-1k | Full-v0.1 / 2000 | 1000 / 1000 | 91.900 / 98.500 / 99.300 | 90.600 / 98.200 / 99.100 |
| Flickr30k test1K | Clean-v0.1 / 3651 | 1000 / 5000 | 88.700 / 98.800 / 99.300 | 71.960 / 91.160 / 95.260 |
| Flickr30k test1K | Full-v0.1 / 3651 | 1000 / 5000 | 86.900 / 98.100 / 99.300 | 70.200 / 90.620 / 94.720 |
| Flickr30k test1K | Full-v0.1 / 2000 | 1000 / 5000 | 87.000 / 97.900 / 99.200 | 70.040 / 90.540 / 94.640 |
| DOCCI test5K | Clean-v0.1 / 3651 | 5000 / 5000 | 78.140 / 95.120 / 98.100 | 78.920 / 95.180 / 97.720 |
| DOCCI test5K | Full-v0.1 / 3651 | 5000 / 5000 | 77.660 / 95.540 / 98.100 | 78.440 / 95.340 / 98.000 |
| DOCCI test5K | Full-v0.1 / 2000 | 5000 / 5000 | 77.480 / 95.420 / 98.180 | 78.320 / 95.320 / 98.000 |
| DCI full | Clean-v0.1 / 3651 | 7805 / 7805 | 50.032 / 70.647 / 77.604 | 49.058 / 69.865 / 76.336 |
| DCI full | Full-v0.1 / 3651 | 7805 / 7805 | 49.263 / 70.455 / 77.732 | 49.263 / 70.096 / 76.784 |
| DCI full | Full-v0.1 / 2000 | 7805 / 7805 | 49.199 / 70.391 / 77.527 | 49.315 / 69.840 / 76.784 |
| Long-DCI（重建版） | Clean-v0.1 / 3651 | 7602 / 7602 | 58.471 / 76.940 / 82.755 | 58.669 / 76.835 / 82.373 |
| Long-DCI（重建版） | Full-v0.1 / 3651 | 7602 / 7602 | 57.722 / 76.888 / 82.676 | 60.116 / 78.203 / 83.320 |
| Long-DCI（重建版） | Full-v0.1 / 2000 | 7602 / 7602 | 57.722 / 76.703 / 82.741 | 59.879 / 77.848 / 83.044 |

## 原始来源

- Clean step3651：[训练配置](s0_dualmask_masked_3epoch/evidence/step3651/config.json)、[严格导出](s0_dualmask_masked_3epoch/evidence/step3651/export_report_step3651.json)、[COCO](s0_dualmask_masked_3epoch/evidence/step3651/coco_canonical_step3651.json)、[Urban](s0_dualmask_masked_3epoch/evidence/step3651/urban1k3651.json)、[扩展检索](s0_dualmask_masked_3epoch/evidence/step3651/extended_eval_step3651/extended_summary.json)。
- Full step3651：[3 epoch 完整报告](../docs/dual_mask_suffix_full_v01/continuation_3epoch_report.md)、[COCO](s0_dualmask_full_v01/evidence/step3651/S0_DUALMASK_FULL_V01_step003651_canonical.json)、[Urban](s0_dualmask_full_v01/evidence/step3651/S0_DUALMASK_FULL_V01_step003651_urban1k.json)、[扩展检索](s0_dualmask_full_v01/evidence/step3651/extended_summary.json)。
- Full step2000：[完整汇总与身份记录](s0_dualmask_full_v01/evidence/step2000/results.json)、[新评测原始记录](s0_dualmask_full_v01/evidence/step2000/new_evaluations/)、[单独报告](s0_dualmask_full_v01/STEP2000.md)。COCO/Urban 复用已核验记录，扩展四项为 2026-09-14 实测，四项退出码均 0。

## 解释边界

同为 step3651 时，Clean 的 COCO R@1 高于 Full，Full 的 Urban R@1 高于 Clean；扩展检索各有优势，不能用一个数据集代表总体。Full step2000 与 step3651 在这六项协议的双向 R@1 差异绝对值均小于 0.24 个百分点。

两版同时改变后缀对齐权重和 U 稀疏约束，属于组合对照。没有 native 后缀正式训练臂，不能单独宣称 mask 优于普通后缀监督；单 seed 也不能据此声称统计显著。

Long-DCI 使用同一重建版 manifest（7602 对），不等同于未取得的官方 CSV。Flickr30k test1K 按 1000 图、每图 5 句评估。DCI full 为已取得的 7805 对，按原记录的 all split 报告。

所有得分均为原生归一化图文向量的全池内积。不同裸学生导出容器可产生不同文件 SHA；按 [资产清单](ASSETS.md) 与原始来源区分，不把这些文件当成同一个字节流。
