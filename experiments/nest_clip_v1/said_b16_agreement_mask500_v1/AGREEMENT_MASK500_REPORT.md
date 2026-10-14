# Agreement-Mask @500

本实验在共同 Step0 独立训练了一个 Agreement-Mask 臂，恰好完成 500 次同步 optimizer update，scheduler horizon 保持 4868。唯一结构变量是：`u_hat_v=tanh(RMSNorm(u_v))`、`u_hat_t=tanh(RMSNorm(u_t))`、`l=u_hat_v*u_hat_t+0.2*tanh(theta)`，原生 mask logits 为 `5*l`，Hard-ST 前向为 `1[l>0]`。RMSNorm 无 affine，epsilon 为 `1e-6`；`theta` 是 512 维可学习通道参数，初始为零。正配对和负配对都经过同一个 `pair_logits` 实现。

## 训练与身份

- 分支：`experiment/said-b16-agreement-mask500-v1`
- 训练与评测源码提交：`2320136bd733094fb0812682a6671bd2fea0021f`
- common Step0 SHA256：`54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`
- 完整 Step500 checkpoint SHA256：`4908184d40c2db5886a9169dd57ac7698de8e617b58851cd76b5ad390ccc2b76`
- 裸学生 SHA256：`090a515270e5f4b1ffc690969a7483f839797d0bfb690ae1296dfaed3938c813`
- global batch：1024（4 × 256），总样本位置 512000
- 正式训练：500/500 updates，四 rank 参数差异为 0，NCCL all-reduce 为 10
- 峰值显存：24.773 GiB/GPU；训练耗时约 1109.1 秒；无 NaN、Inf、OOM 或 DDP 错误
- 裸学生严格加载通过，image/text 导出前后最大绝对误差均为 0

5-step smoke 也通过，`agreement_theta`、视觉 Mask 和文本 Mask 梯度均非零。之前三个 smoke 目录只包含启动或 receipt 失败证据，没有执行 update，未被当作实验结果。

## 五集原生检索结果

所有结果来自同一个 Step500 裸学生；推理使用原生 CLIP embedding，没有 Mask、Fusion、reranking 或 TTA。A-Matched 是同为 500 steps、horizon 4868 的历史基线。

| 数据集 | 方案 | I2T R@1/5/10 | T2I R@1/5/10 |
|---|---|---|---|
| COCO | A-Matched | 60.56 / 82.84 / 89.34 | 41.56 / 67.324 / 76.94 |
| COCO | Agreement | 60.56 / 82.80 / 89.22 | 41.768 / 67.236 / 76.988 |
| Flickr30k-Test1K | A-Matched | 87.30 / 97.50 / 99.10 | 72.26 / 91.40 / 95.24 |
| Flickr30k-Test1K | Agreement | 87.30 / 97.70 / 99.10 | 72.14 / 91.32 / 95.18 |
| DOCCI | A-Matched | 77.74 / 95.46 / 98.06 | 77.70 / 95.10 / 97.80 |
| DOCCI | Agreement | 77.98 / 95.64 / 98.18 | 78.60 / 95.48 / 97.98 |
| Long-DCI | A-Matched | 57.0245 / 75.7432 / 81.8206 | 56.8929 / 76.0326 / 82.0179 |
| Long-DCI | Agreement | 57.4454 / 76.3746 / 82.1889 | 57.0245 / 76.0984 / 82.0705 |
| Urban-1k | A-Matched | 91.30 / 98.60 / 99.70 | 89.40 / 98.70 / 99.40 |
| Urban-1k | Agreement | 91.20 / 98.60 / 99.70 | 89.10 / 98.40 / 99.40 |

| 汇总指标 | A-Matched | Agreement | 差值（Agreement - A-Matched） |
|---|---:|---:|---:|
| Score5 | 71.173740 | 71.311789 | +0.138049 pp |
| J_long3 | 75.009567 | 75.224981 | +0.215414 pp |
| J_long | 84.035003 | 84.220002 | +0.185000 pp |
| Short4 | 65.420 | 65.442 | +0.022 pp |
| Urban I2T | 91.300 | 91.200 | -0.100 pp |
| Urban T2I | 89.400 | 89.100 | -0.300 pp |
| Urban Mean | 90.350 | 90.150 | -0.200 pp |

四卡评测调度真实并行启动 COCO/GPU0、Flickr/GPU1、DOCCI/GPU2、Long-DCI/GPU3；Flickr 释放 GPU1 后立即在 GPU1 启动 Urban。第一次运行因 worktree 缺少大体积 `local_assets` 而失败，没有使用任何估算成绩；显式指定服务器 `local_assets` 的 `retry-v2` 五项任务全部 return code 0。

## Agreement-Mask 数值审计

独立脚本 `recovery/agreement_mask_audit.py` 从 Step500 完整 checkpoint 重建原生 ViT-B/16，并从真实 ShareGPT4V `nested_detail_d3` 索引抽取 1024 个有效样本（K=3：1021 个，K=2：3 个）。负配对采用确定性的 `visual.roll(1, 0)`，仅用于审计，不进入训练。

| 配对 | 视角 | Hard keep | sigmoid(5l) 饱和 | product→最终符号变化 | 视觉/文本 L2 比例中位数 |
|---|---|---:|---:|---:|---:|
| 正 | F | 84.661% | 3.127% | 11.298% | 0.468 |
| 正 | Dall | 84.521% | 3.141% | 11.100% | 0.466 |
| 正 | D3 | 83.974% | 3.148% | 10.845% | 0.476 |
| 负 | F | 84.708% | 3.127% | 11.275% | 0.468 |
| 负 | Dall | 84.608% | 3.139% | 11.138% | 0.465 |
| 负 | D3 | 84.053% | 3.148% | 10.856% | 0.475 |

`theta` 的均值为 0.001606、标准差为 0.001458，最大绝对值为 0.007033；有界偏置的均值为 0.000321、最大绝对值为 0.001407，远未接近 ±0.2 边界。训练最后一步记录的 `agreement_theta` 梯度范数为 0.06563，说明偏置参数真实参与了学习。

三视角符号三者完全一致的比例约为 86.48%（正）和 86.51%（负）；F/Dall、F/D3、Dall/D3 的两两一致率分别约为 93.09%/89.07%/90.78%。这些统计说明 Agreement-Mask 并非恒定全开或全关，且 D3 的保留率略低于 F/Dall。约 10.5–11.0% 的元素中，乘积为负而有界偏置把最终 logit 推到正侧；约 0.34% 被推到相反方向。由于偏置仍很小，这种变化主要来自特征乘积接近零的通道，而不是偏置饱和。

训练最后一步的四组梯度范数为：backbone 130.703、text mask/shared pool 0.5365、visual mask 0.2231、fusion adapter 0.1031；其中 `agreement_theta` 为 0.06563、`visual_adapter` 为 0.07945。HNS keep ratio 为 F/Dall/D3 = 84.454%/84.186%/83.751%，Dall→F 与 D3→Dall 硬包含违反率为 3.413%/4.130%，对应 IoU 为 0.9200/0.9025。

## 结论

在这一个 seed、500-step 预算下，Agreement-Mask 在 Score5、J_long3、J_long 和 Short4 上略高于 A-Matched，但 Urban I2T 与 T2I 同时下降，且下降幅度（尤其 T2I -0.3 pp）超过了“全面超过 BalancedStack”的表述范围。公开五集已被反复观察，不能把这些差值视为无偏泛化证据。

最终分类：**`NO_CLEAR_GAIN`**。当前证据不支持把 Agreement-Mask 替代 BalancedStack，也不授权自动延长训练或调整任何 Agreement-Mask 超参数。

详细原始统计见 `AGREEMENT_MASK500_MASK_AUDIT.json`，训练证据见 `AGREEMENT_MASK500_TRAINING_AUDIT.json`，评测调度证据见 `AGREEMENT_MASK500_EVAL_PROVENANCE.json`。
