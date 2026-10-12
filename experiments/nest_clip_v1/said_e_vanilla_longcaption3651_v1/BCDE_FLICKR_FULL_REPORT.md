
# Clean V3 B/C/D/E Flickr30k-Full 补充评测

本报告补齐 Clean V3 四个模型在 Flickr30k-Full 上的原生检索结果。此次只读取已经完成的 Step3651 裸学生权重，不训练、不修改模型或推理数学，也不覆盖 A/B/C/D/E 的历史五集结果。

## 评测协议

- 数据集：Flickr30k-Full，31,783 张图像、158,915 条 caption。
- 协议：`Flickr30k-Full-official-token-31783x158915`，状态为 `VERIFIED_SOURCE_PROTOCOL`。
- Manifest SHA256：`bfed72a299e4d8cbfef817c13c836cd5aa189e18dd2fd29b632f5808e8dd12bc`。
- Record SHA256：`5e80d06747e21ddf1c2aabd3fdf744d6b506a9473eb769d4d3567adb7e99e34c`。
- 图像 ID/路径 SHA256：`27ad197845327d1fcb58fc6e3f65bd17358e6e155c14694c41605c4765df54a3`。
- 图像内容 SHA256：`e4fef70dd3f73034970b1ec0e20d81a4b421d1871e350e3266e41b744d3ab159`。
- Tokenizer：`model.longclip.tokenize(truncate=True)`，context length 248。
- FP32 model/input/features，autocast off；cuDNN TF32 开启，matmul TF32 关闭；图像/文本 batch 均为 64。
- 检索 tie rule、encoder source hashes 和完整逐次 receipt 见 [`BCDE_FLICKR_FULL_RESULTS.json`](./BCDE_FLICKR_FULL_RESULTS.json) 及各 runtime 原始 JSON。

四个成功结果均满足：`status=COMPLETED`、协议已验证、样本计数一致、模型状态前后 SHA 相同。四卡并行启动时 B/C/D/E 分别分配到物理 GPU 0/1/2/3；worker 在各自 `CUDA_VISIBLE_DEVICES` 命名空间内报告为 `cuda:0`。

## Flickr30k-Full 结果

| 模型 | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 |
|---|---:|---:|---:|---:|---:|---:|
| B · Joint-Mask Only | 56.5711 | 78.8472 | 85.8824 | 36.1659 | 58.5672 | 67.6091 |
| C · Nested Alignment Only | 55.9419 | 78.5042 | 85.1965 | 35.9784 | 58.5502 | 67.7557 |
| D · Text-only SmartMask | **56.7347** | **78.8755** | 85.7754 | **36.6844** | **59.2713** | **68.4076** |
| E · Vanilla Long-Caption | 54.5386 | 77.5352 | 84.6553 | 35.0351 | 57.4181 | 66.5878 |

D 在六个方向/Recall 项中的五项最高，并且在 T2I 三项均最高；B 的 I2T R@10 高于 D，C 仅在 T2I R@10 略高于自身其他方向但整体低于 D。E 在四个模型中六项均最低。

## 模型身份与运行摘要

| 模型 | 裸学生 SHA256 | Step3651 来源 | 运行耗时(s) | 峰值 CUDA allocated | 结果文件 |
|---|---|---|---:|---:|---|
| B | `68716f0d106eaeca97b854990c1c526b46b9f01c08d1a7c84e6e34e6d2080d2c` | `said-b16-component-ablation-clean-v3/B/formal3651-recovery-v1` | 347.70 | 1,586,150,912 B | `B/FLICKR_FULL_RESULTS_R2.json` |
| C | `e3a859f173ec5912f948041a200ee90bb43bdeff0097dddc0118700f2a63f742` | `said-b16-component-ablation-clean-v3/C/formal3651-recovery-v1` | 349.27 | 1,586,150,912 B | `C/FLICKR_FULL_RESULTS_R2.json` |
| D | `cdaf413f810898e2b71da2255f5a189bfb128a7f441524fbc2b942c2b4949e91` | `said-b16-component-ablation-clean-v3/D/eval3651-root` | 348.03 | 1,586,150,912 B | `D/FLICKR_FULL_RESULTS_R2.json` |
| E | `709b60a5c893229226f8eaa7b7f706f5b0360d632fb64bf991529d67b5d17a62` | `said-e-vanilla-longcaption3651-v1/eval3651` | 351.07 | 1,586,150,912 B | `E/FLICKR_FULL_RESULTS_R2.json` |

文本编码约 246.3–248.5 秒，图像编码约 75.8–76.3 秒，评分约 9.5–10.1 秒；没有出现 OOM、非有限值或模型状态变化。

## 运行故障保留

早期 B 空 SHA 启动、C/D/E 输出目录写入失败，以及一次 JOB 参数类型错误的失败证据均保留在 runtime 的 `FLICKR_FULL_RESULTS.failure.json` 或终端记录中。它们没有被当作有效指标；最终只采用四个 `FLICKR_FULL_RESULTS_R2.json`，并在汇总 JSON 中列明。

## 与既有结果的关系

这些是 Flickr30k-Full 的补充原生评测，不能与 Test1K 的 R@1/5/10 直接混比，也不改变既有五集 Score5、J_long3、J_long、Short4 或 Urban 结果。公开测试集此前已被反复观察，Flickr-Full 数值应作为同协议探索性比较，不单独构成无偏泛化结论。
