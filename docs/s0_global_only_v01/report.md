# S0-GlobalOnly v0.1 实验报告：只训练原生图像—文本全局对齐的最简对照

**臂（arm）**：`S0_GLOBAL_ONLY`　**objective**：`global_alignment_only`　**phase**：`s0-global-only-v0.1`
**实现提交**：`01db2225fd38514b4509c24046f8e6996f481926`（分支 `codex/s0-global-only-v01`，由 `8f694c7` 分出）
**训练**：4×A800，每卡 256（全局 1024，梯度累积 1），**恰好 500 次 optimizer update**，无扫描、无第二个配置
**一句话结论**：**把 S0 的 mask 与稀疏项全部拿掉、只留 10× 双向全局对齐之后，性能反而下降** —— COCO I2T R@1 0.5992（比 S0@500 低 **0.66pp = 33 个图像查询**）、T2I R@1 0.40884（低 **0.352pp = 88 个文本查询**）、Urban T2I 低 **1.3pp = 13 个文本查询**、Urban I2T 与 S0 完全相同（0.8700）。也就是说：**在 500 步预算内，S0 的 mask/局部机制不是负担，而是有效成分**。

---

## 1. 这个臂是什么

完全按规格实现，唯一目标就是全局对齐：

```
g = normalize(clip.encode_image(image))            # 原生视觉塔 + 原生 visual.proj
t = normalize(clip.encode_text(prefix_caption))    # 原生文本塔 + 原生 text_projection
Q = 100 * g @ t.T
L = 10 * (cross_entropy(Q, y) + cross_entropy(Q.T, y))
```

- 双向**相加**，绝不乘 0.5；评分尺度**固定 100**，`logit_scale` 从不参与计算；
- **完全没有 mask**：不调用 `mask_net`、不产生任何 mask 张量，也**没有**用"全 1 mask"顶替，直接矩阵乘；
- **没有稀疏项**（S0 是 `lambda_sparse = 2.0 * mean(|m_s|)`），也没有重建/一致性/后缀/U 项；
- 视觉塔、文本塔与两个原生投影正常微调；`mask_net` 与 `logit_scale` 仅为 state-key 兼容保留，
  **冻结、不前向、不进 optimizer**（`partition_parameters` 会断言冻结清单存在、且可训练集合恰好等于 optimizer 组）；
- 不用后缀、不改成完整 caption 或仅第一句：前缀抽法沿用共享数据集代码
  （`caption.replace('\n',' ').split('. ')`，`K ~ randint(1, len(sentences))`，`prefix = '. '.join(sentences[:K])`）。

## 2. 与冻结 S0 的核对结果（先核对、后开跑）

核对来源：S0 系续训 run 自己的 `config.json`（`/root/SAID-s0-continue-v01/runs_salu/said_cls_cvssl/s0_continue_1000/config.json`）
与 S0 臂训练器 `train/train_said_cls_cvssl.py` / `model/said_cls_cvssl.py` 的实现。

| 项 | 冻结 S0 | 本臂 | 是否一致 |
| --- | --- | --- | --- |
| 初始化 | shared_init `c1a4a2be…`，载入后状态摘要 `caf61198…` | 同一个文件、同一摘要 | ✅ 一致 |
| 数据 | ShareGPT4V 清单 + 共享数据集类 + 前缀抽法 | 同一份清单、同一个数据集类 | ✅ 一致 |
| 视图 | `view_a` 参考 openai-clip `_transform(224)` | 同一变换，单视图（不生成 view_b） | ✅ 一致（S0 的 view_b 只服务于 `lambda_U=0` 的项与诊断） |
| 批 / DP | 4 卡 × 256 = 1024，梯度累积 1 | 同 | ✅ 一致 |
| 优化器 | AdamW lr 1e-6、wd 1e-2、betas(0.9,0.999)、eps 1e-8 | 同 | ✅ 一致 |
| warmup / 调度 | warmup 200，cosine horizon = 3 × len(loader) = **3651** | 同（horizon 未压缩） | ✅ 一致 |
| 步数 | 恰好 500 次更新 | 同（第 501 次不执行） | ✅ 一致 |
| 精度 | fp32 主参数 + bf16 autocast | 同 | ✅ 一致 |
| seed | 0 | 0 | ✅ 一致 |
| **评分** | `100 * cos(normalize(v_i * m_j), t_j)`（带 mask） | `100 * <normalize(v_i), normalize(t_j)>`（`m_j ≡ 1`） | ⚠️ **本臂刻意不同**（这就是实验变量） |
| **稀疏项** | `lambda_sparse = 2.0 * mean(|m_s|)` | 无 | ⚠️ **本臂刻意移除** |
| 对齐权重 | `lambda_align = 10.0`，双向相加不乘 0.5 | 同 | ✅ 一致 |
| 其他项 | `lambda_U = 0.0`、`arm_mask = 'none'`（U/互补项本就不参与） | 同（不含） | ✅ 一致 |
| 分布式 | 逐 rank 本地锚点 + **可导** `nn_dist.all_gather` + 标准 DDP 平均，目标 `rank*B+i` | 同 | ✅ 一致 |

**与你的记录无出入**（lr/wd/warmup/3651/4×256/seed 0/fp32+bf16/双向相加/固定 100 全部对上）；
唯一两处差异就是实验变量本身（mask 与稀疏项），已在代码注释、`config.json` 与本节中写明。

## 3. 实现与必要验证

- `train/train_global_only.py`：极简训练器，复用既有 dataset/collate/DistributedSampler/`cosine_lr`；
  自描述检查点、逐 rank 数据流摘要、梯度有限性集体门（非有限即拒绝该步，不做裁剪或跳过）。
- `tools/diag/export_global_only_student.py`：裸学生导出（校验 `completed_steps`/arm/状态摘要）。
- `tools/globalonly_runner.py`：`train → verify → export → coco → urban → report` 六阶段，GPU 预检、原子锁、
  真实退出码、`train` 阶段拒绝重训、**没有** `allow-missing` 类旁路。
- 独立分支/工作树 `codex/s0-global-only-v01` / `/root/SAID-globalonly-v01`，未改动既有 S0 与其他实验产物。

**测试**：`tests/test_global_only.py` + `tests/_global_only_ddp_worker.py`，**8 项全部通过**（真实 CLIP 权重，非 mock）：

1. 双向 loss 的**数值**与**梯度**都与独立写出的参考一致（loss 与 5 个梯度张量逐项比对）；
2. 评分尺度确实是 100（`logit_scale` 被扰动也不影响评分），`logit_scale` 不参与；
3. `mask_net` 与 `logit_scale` 冻结、无梯度、**主干确实更新**（前后参数不等）；
4. 真实训练步的梯度健康报告显示 `mask_net_grad_tensors = 0`、`logit_scale_grad_present = false`；
5. **两 rank DDP 平均梯度严格等于两个 rank 各自本地梯度的算术平均**（并验证两者确实不同，避免空测试），
   同时与单进程全局批参考一致 —— 证明没有 world_size 倍率；
6. 单进程 loss 等于本地锚点定义（CE(Q,y)+CE(Qᵀ,y)，10×）；
7. checkpoint 与**裸学生**都能 `strict=True` 加载，摘要可复算；
8. `partition_parameters` 在缺少兼容键的模型上明确拒绝（冻结清单是被验证的，不是假设的）。

## 4. 正式 run 的事实

| 项 | 值 |
| --- | --- |
| 恰好更新次数 | **500**（`expected_at_500` 与 `synchronized_pair_presentations` 都是 512,000 对，二者吻合） |
| 训练耗时 | **425.9 s**（7.1 分钟），均值 **0.718 s/step** |
| 峰值显存 | **39.3 GiB** |
| 阶段退出码 | train 0 / verify 0 / export 0 / coco 0 / urban 0 / report 0 |
| 各阶段耗时 | train 425.9 s（脚本整体 461.7 s，含预检与初始化）/ export 16.3 s / COCO 491.8 s / Urban 92.7 s |
| 存档 | step 0 / 100 / 250 / **500**（只对 500 步做正式评估） |
| 训练检查点 | `S0_GLOBAL_ONLY_step000500.pt`，sha256 `f389f25c0ef14896d08125a111c3f26ac0d153e57f30f5d674b299761c940eb2` |
| 裸学生 | `student_export/s0_global_only_student.pt`，sha256 `49eb7629732f00a513b2417ad7ef1c4d2bd645cf14118076561d57504c3447fe`，317 张量 |
| 最终状态摘要 | `620ca693e2f1f8f6bad3ec8224851a7f383b87d132794f8fb0f6283f62f35762` |
| 初始化 | 文件 sha256 `c1a4a2be…`，载入后摘要 `caf61198…`（与冻结值逐位一致） |
| 过程中的损失 | 末步 `L = 2.9044 = 10 × (0.1216 + 0.1688)`；训练内 top1：I2T 0.9492 / T2I 0.9414 |
| 运行中证据 | 每一步都记录 `mask_net_forwarded = false`、`mask_tensors_in_loss = 0`、`mask_net_grad_tensors = 0` |

**步时对比**（本机同一批卡）：本臂 **0.718 s/step** 是目前最快的臂（无 mask、无辅助项）；此前在旧机测得的
参考点为 S0 baseline 1.00、TriMask-HS ~0.91、PG-CLIP 1.27、CG-CLIP 2.21 s/step。

## 5. 数据流核对（明确范围：rank0）

本臂日志每条记录都带四种流的**累计摘要**（`caption` / `image_id` / `prefix_k` / `sample`）与四个**逐步摘要**
；其他臂记录的字段不同，因此下面每一项都写明"在什么字段上、与谁、比了多少步"。

| # | 对照（字段 → 对象） | 结果 |
| --- | --- | --- |
| 1 | 逐步 `batch_caption_sha256` + `batch_image_id_sha256` → **PG-CLIP v0.1 @500**（原 run） | **51 / 51 逐步相同** |
| 2 | 逐步 `batch_caption_sha256` + `batch_image_id_sha256` → **PG-CLIP v0.1 @500**（复现 run） | **51 / 51 逐步相同** |
| 3 | 逐步 `batch_caption_sha256` + `batch_image_id_sha256` + `epoch/step_in_epoch` → **CG-CLIP v0.1 @500** | **51 / 51 逐步相同** |
| 4 | **四种累计流全部**（caption / image_id / prefix_k / sample）→ **CG-CLIP v0.1 @500**（键名不同，逐项映射后比较） | **四个流各 51 / 51 相同**（即前缀流也逐位相同） |
| 5 | **独立重建** epoch-0 rank0 采样流（清单 + `DistributedSampler(seed=0)`，不解码图像）→ 本 run 逐步 `image_id` 摘要 | **51 / 51 相同**（用训练器 float32 约定） |
| 6 | 同上 → 本 run 逐步 `sample_id` 摘要（`index + total_len`） | **51 / 51 相同** |
| 7 | 前缀流 → **PG-CLIP** | **无法比对**：PG 的日志没有 prefix_k 字段（只有 image_id 与 caption 两个逐步摘要） |
| 8 | 逐步/累计流 → **S0 系续训 run**、**PG-CLIP 续训 run**（步号 510–1000） | **0 步可对**（步号与本 run 无交集），不能用作核对 |
| 9 | S0-TriMask v0.1、S0-TriMask-HS v0.2（step500 / bal500 / cont1000）的日志 | **完全不可解析**：22 / 51 / 51 / 50 行**全部**为坏行，可解析记录 0 条，无法用于比对 |

**范围声明**：以上全部是 **rank0 证据**（双方日志都只记录 rank0 行，字段自述 `statistics_scope = rank0_local_batch`）。
本臂没有记录逐 rank 摘要；`run_status.json` 记录 `world_size = 4`，DDP 测试另外证明了 4 卡梯度归约语义正确，
但"四个 rank 各自的数据流都与 S0 一致"这句话**本轮没有证据，不作声称**。

**摘要约定必须写明（否则会误读）**：项目既有的 `tensor_digest` 会把张量先转 **float32** 再取字节摘要，
而 image id 是约 2.6e18 的整数、超出 float32 精度 —— 所以逐步摘要是"同一约定下的相等性检验"，不是逐字节相等。
我用 int64 约定重建时得到 **0 / 51**、换成训练器的 float32 约定才是 **51 / 51**（第 5 行就是这个对照实验，
两种约定在同一脚本里同时给出，可复现）。累计流（第 4 行）用的是另一种（原始整数字节）约定，因此第 4 行与
第 5 行的"相同"是两个不同约定下的两个结论，都成立、不可混用。

**结论**：与 CG-CLIP 的四个累计流逐一相同，说明本臂消费的**图像序列、caption 序列与"实际送入文本塔的前缀 K 值序列"
都与同族实验一致**；与 PG-CLIP 只能在 image_id / caption 两个逐步摘要上比对（51/51 相同）；
排名重建进一步独立证明了图像与样本序号流可复现。**这是"数据流相同"的证据，不是"训练结果相同"的证据。**

## 6. 结果与对比（只用冻结协议，无 mask、无 rerank、无评分融合）

**裸学生 → 原生 CLS/EOS 读出**，两项冻结协议：COCO canonical（5000 图 × 25000 文本）、Urban-1k（1000 × 1000）。

| 协议 | 指标 | S0_GLOBAL_ONLY@500 | S0@500（复用，未重训） | 差 | 换算成查询条数 |
| --- | --- | --- | --- | --- | --- |
| COCO val2017 | **I2T R@1** | **0.5992** | 0.60580 | **−0.660 pp** | **−33 / 5000 图像查询** |
| COCO val2017 | **T2I R@1** | **0.40884** | 0.41236 | **−0.352 pp** | **−88 / 25000 文本查询** |
| Urban-1k | **I2T R@1** | **0.8700** | 0.87000 | ±0.000 pp | **0 / 1000** |
| Urban-1k | **T2I R@1** | **0.8290** | 0.84200 | **−1.300 pp** | **−13 / 1000 文本查询** |
| COCO | I2T R@5 / R@10 | 0.81700 / 0.88560 | — | — | — |
| COCO | T2I R@5 / R@10 | 0.66736 / 0.76268 | — | — | — |
| Urban-1k | I2T R@5 / R@10 | 0.9710 / 0.9880 | — | — | — |
| Urban-1k | T2I R@5 / R@10 | 0.9590 / 0.9820 | — | — | — |

换算规则（分母不同，同一 pp 数字对应条数不同）：COCO I2T 1 个查询 = 0.02pp、COCO T2I 1 个查询 = 0.004pp、
Urban-1k 1 个查询 = 0.1pp。

**放进现有臂队列（500 步档，COCO R@1）**：

| 臂 | COCO I2T R@1 | COCO T2I R@1 | Urban I2T | Urban T2I |
| --- | --- | --- | --- | --- |
| S0 baseline（下限） | **0.60580** | **0.41236** | 0.8700 | **0.8420** |
| PG-CLIP v0.1（复现 run） | 0.60560 | 0.41236 | 0.8730 | 0.8300 |
| PG-CLIP v0.1（原 run） | 0.60480 | 0.41256 | 0.8700 | 0.8330 |
| S0-TriMask v0.1 (soft) | 0.60280 | 0.41444 | 0.8710 | 0.8370 |
| S0-TriMask-HS v0.2 | 0.60060 | 0.41208 | **0.8740** | 0.8370 |
| **S0-GlobalOnly v0.1（本臂）** | **0.59920** | **0.40884** | 0.8700 | **0.8290** |
| CG-CLIP v0.1 | 0.59520 | 0.40852 | 0.8700 | 0.8300 |
| S0-TriMask-HS 均衡权重 | 0.58920 | 0.40828 | 0.8650 | 0.8280 |
| S0 XPool v0.1 | 0.57420 | 0.38508 | 0.7670 | 0.7090 |

**读法**：本臂在 COCO 两个方向上都低于 S0，但高于 CG-CLIP；Urban I2T 与 S0 完全打平（0.8700），
Urban T2I 0.8290 是"保留原生对齐"这一族里最低的。结合同配置 run 间波动（0.02–0.30pp）来看，
−0.66pp / −0.352pp 都**大于**波动上限，所以"取消 mask 与稀疏项会掉点"这个方向是可信的；
但**没有跑控制臂**，不能进一步断言是哪一部分机制在起作用（掩码本身？稀疏正则？还是两者共同）。

## 7. 产物、SHA 与可追溯性

| 产物 | 路径 / 值 |
| --- | --- |
| 训练代码 SHA | `01db2225fd38514b4509c24046f8e6996f481926` |
| 训练检查点 | `runs_salu/s0_global_only_v01/S0_GLOBAL_ONLY_step000500.pt`，sha256 `f389f25c…` |
| 裸学生 | `student_export/s0_global_only_student.pt`，sha256 `49eb7629…`，317 张量（含冻结的 `mask_net` 兼容键） |
| 训练日志 | `runs_salu/s0_global_only_v01/salu_log.jsonl`（51 条记录） |
| 两份评测原文 | `evaluation/S0_GLOBAL_ONLY_step000500_canonical.json`、`..._urban1k.json` |
| 学生元数据 | `student_export/s0_global_only_student_metadata.json` |
| 阶段退出码 | train 0 / verify 0 / export 0 / coco 0 / urban 0 / report 0（见 `run_status.json`） |
| 数据流核对脚本（已入库、已实跑） | `tools/diag/compare_arm_streams.py`、`tools/diag/rebuild_stream_digests.py` |
| 结果汇总 | `docs/s0_global_only_v01/results.json` |

**一次诚实更正（元数据语义）**：导出工具原先把"未训练 vs 已训练"的探针差值记成了 `lossless_probe`，
名字与含义都错了（那个数本来就应该是大的）。已更正为两个字段：
`probe_after_two_independent_loads_max_abs_diff = 0.0`（真正的无损性检验：同一学生加载进两个新模型，
`encode_image`/`encode_text` 逐位相同）与 `probe_trained_vs_untrained_max_abs_diff`
（期望非零，证明训练后的状态确实被加载）。**任何权重与指标都没有改动。**

## 8. NOT RUN（本轮没有做，也不声称）

- 任何控制臂：只加回 mask、只加回稀疏项、换稀疏权重、更长步数下的同一对照。
- 超参搜索、多 seed、超过 500 步的续训（cosine horizon 3651 只走到 500）。
- 两路/多路融合、rerank、评分融合（两项评测都是单一原生读出）。
- 除 rank0 之外的数据流逐 rank 比对（本臂未记录逐 rank 摘要）。
- 与 PG-CLIP 的**前缀流**比对（PG 日志没有 prefix_k 字段，不是不一致，而是不可比）。
- 与 S0 原始 500 步日志的逐位比对（那份记录不在这台机器上；可用 S0 记录步号与本 run 无交集，0 步可对）。
- 与 S0-TriMask / S0-TriMask-HS 日志的数据流比对（这四份日志行**完全不可解析**，与本臂无关，属既有文件问题）。
