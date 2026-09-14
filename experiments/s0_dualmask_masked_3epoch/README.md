# S0-DualMask-Clean v0.1：500 → 1000 → 3 epoch 复现实验包

本目录只记录 `reyuyu/SAID` 的这一个 masked 实验。目录内含配置、锁定环境、数据/权重校验、完整命令、原生评估代码、已完成的测评结果和迁移说明。训练代码仍引用仓库中的固定提交，不在本目录另写模型或训练目标。

**路线：从共同初始化训练到 500 → 同一状态续到 1000 → COCO canonical 和 Urban-1k 原生评估 → 同一 1000 检查点续到 3651（3 epoch）。** 没有 native 正式训练臂。旧 S0 只读取既有结果。

## 从哪里开始

1. [完整方法和全部训练参数](METHOD_AND_PARAMETERS.md)：精确到候选前缀条件、梯度边界、K 抽样、尾批、三组 optimizer、cosine 公式和 checkpoint 位置。
2. [复现命令](REPRODUCE.md)：准备环境、验证输入、从零复现以及 500/1000 断点继续。
3. [数据/权重迁移与 GitHub 上传](TRANSFER_AND_UPLOAD.md)：必须另外准备的文件、目录布局、校验和上传方法。
4. [机器可读配置](manifests/experiment.json)、[实际运行环境](environment/runtime.json)、[输入文件 SHA256](manifests/assets.json)。

需要把本目录与整个 SAID Git 仓库一起使用。GitHub 包含代码与小型证据，**不包含 1.5 GB 标注、训练/评估图片或模型权重**。尤其 `cvssl_initial.pt` 是本实验的共同初始化，不是任选一个公开 CLIP checkpoint；需要迁移原文件并匹配 SHA256。缺少这些输入时，单独 clone GitHub 不能完整复现实验。

## 固定代码与有效轨迹

| 部分 | 固定版本 |
|---|---|
| 原 0–500 步 | `ff5ad1d4b918d56c6bfa48a2870dc5223e757237` |
| 500→1000、1000→3651 续训 | `11af80b344c623b27b93069f9be526970c9c950c` |
| 训练模式 | masked，4 GPUs × 256，accumulation=1 |
| horizon | 3 × 1217 = 3651 optimizer updates |
| 原始成功 500 目录 | `dual_mask_suffix_masked_formal500_flatgather_v01` |
| 有效后续目录 | `dual_mask_suffix_masked_from500_continuation_v02` |

已完成 **3651 次 optimizer updates（3 epoch）**，最终配置、严格导出与六项原生检索结果已保存到 [step3651](evidence/step3651/)。[status.json](evidence/status.json) 已按最终证据更新。首次发布时的“训练中”快照保留在 Git 历史。

本目录明确指 **Clean v0.1（后缀权重1、U稀疏0）**。与 [Full v0.1（后缀权重10、U稀疏2）](../s0_dualmask_full_v01/README.md) 的完整对比见 [结果总表](../RESULTS.md)。main 中共享训练入口保持 Full 分支实现；重现本实验的严格续训必须使用上表的固定 Clean SHA。

`dual_mask_suffix_masked_continuation_v01` 的误重训 344 步，以及 `dual_mask_suffix_masked_from500_continuation_v01` 的错误数据游标接续 54 步，均已停止、保留并排除。它们的 checkpoint 不用于本实验。修复与失败记录见 [continuation_repair.md](evidence/continuation_repair.md)。

## 已完成的原生结果

单位为百分比；均为 `normalize(student.encode_image)` 与 `normalize(student.encode_text)`，无条件 gate、融合或 rerank。

| 模型 | COCO I2T R@1/5/10 | COCO T2I R@1/5/10 | Urban I2T R@1/5/10 | Urban T2I R@1/5/10 |
|---|---|---|---|---|
| 已有 S0@500 | 60.580 / 82.200 / 89.060 | 41.236 / 67.092 / 76.620 | 87.000 / 97.100 / 98.800 | 84.200 / 96.700 / 98.100 |
| masked@500 | 60.540 / 82.480 / 89.180 | 41.464 / 66.672 / 76.552 | 88.200 / 97.500 / 99.200 | 85.800 / 97.500 / 99.100 |
| masked@1000 | 61.260 / 83.220 / 89.300 | 41.964 / 67.392 / 77.296 | 90.400 / 98.300 / 99.400 | 87.900 / 97.700 / 98.900 |
| Clean v0.1@3651 | 61.340 / 83.340 / 89.480 | 42.272 / 67.764 / 77.380 | 91.400 / 98.400 / 99.400 | 89.500 / 97.800 / 98.900 |

来源为本目录 `evidence/masked_formal500_report.json`、`evidence/step1000/coco_canonical_step1000.json`、`evidence/step1000/urban1k1000.json` 及最终 [step3651 原始记录](evidence/step3651/)。1000 步训练、严格导出、COCO、Urban 四阶段退出码均为 0。没有普通后缀监督的正式对照臂，不能用这些结果单独归因于新 mask。

## 能复现到什么程度

本包固定算法、数据顺序、K 的随机调用方式、更新计数、优化器、评估和输入校验。换服务器时优先迁移完全相同的输入文件、Python/PyTorch/torchvision/Pillow 环境，并保持 4 卡拓扑与 rank 顺序。

**不能承诺跨 GPU、CUDA、CPU 数学库逐位相同。** 原训练使用 BF16 autocast，未开启全局 deterministic algorithms；不同 kernel、归约顺序及检索边界附近的相似度可能产生差异。这里的目标是尽可能复现同一实验协议和结果，并报告真实误差，而不是编造百分之百数值保证。训练图片尚无本包采集的完整逐文件字节清单，需额外通过原文件迁移校验保证一致。

最终训练已完成。文档发布在独立工作区完成；实际脚本、配置和历史结果中的服务器绝对路径是出处信息，可移植命令见 `REPRODUCE.md`。
