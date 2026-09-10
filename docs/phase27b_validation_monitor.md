# Phase 2.7B：标准检索验证监视器（Validation Monitor）

本阶段把"训练中途的标准检索验证"从**声明**变成**可用**：训练脚本现在能按固定间隔、用固定协议、
在不使用 Said Router 的前提下评测标准 CLIP 检索，并把每次结果写成可追溯的 JSONL。

## 目标与边界

- 只做验证与记录：不参与 loss，不改变优化器、数据采样或模型结构。
- 验证一律走标准推理路径 `encode_image()` / `encode_text()`；Said Router 只用于训练目标，
  不进入任何检索指标。
- 协议固定、结果确定：同一 checkpoint 重复评测得到相同数值。

## 组件

| 文件 | 作用 |
| --- | --- |
| `eval/retrieval/coco_retrieval.py` | `retrieval_metrics()`：i2t / t2i 的 R@1/5/10；`evaluate_coco()`：COCO val2017 标准 5-caption 协议（5,000 图 → 25,000 文本） |
| `eval/retrieval/sharegpt4v_retrieval.py` | `evaluate_sharegpt4v()`：固定 ShareGPT4V 验证集（审计 split 的前 1,000 条，每图 1 条 caption → 1,000 路检索） |
| `train/train_salu.py` | `--val_every` / `--val_sharegpt4v` / `--eval_coco_initial` / `--eval_coco_each_epoch` / `--eval_coco`（别名 `--val_coco`）/ `--val_batch_size` / `--legacy_eval_coco`；`plan_validation()`、`run_validation_job()`、`run_planned_validation()`、`run_initial_validation()`、`append_validation_record()` |
| `eval/validation_protocol.py` | `build_manifest()` / `load_or_create_manifest()`（冻结的 ShareGPT4V-1K 三变体清单）、`evaluate_variant()` / `evaluate_all_variants()`、`rng_guard()` / `rng_snapshot()` |

检索指标本身与特征来源无关：`retrieval_metrics()` 先把特征做 L2 归一化，再按方向统计命中率，
因此对特征整体缩放不敏感。

## 训练中的接入方式

- **初始（step 0）**：开启 `--eval_coco_initial` 时，`main()` 在进入训练循环**之前**调度一次
  canonical COCO 评测（`reason='initial'`、`step=0`、`epoch=0`）。
- **间隔**：每 `--val_every N` 步（`N > 0`）跑一次 ShareGPT4V-1K 三变体。
- **epoch end / final**：ShareGPT4V-1K 各跑一次；COCO 只在 `--eval_coco_each_epoch` /
  `--eval_coco` 打开时跑，且 epoch end 与 final 重合时只跑一次。
- 评测只在 rank 0 执行，前后各有一次 `dist.barrier()`：其它 rank 在 barrier 处等待，
  因此不会有人在 rank 0 评测时提前进入下一步的集合通信（不会出现 NCCL 超时或错位）。
- 评测期间模型切到 `eval()`，结束后在 `finally` 中切回 `train()`；评测在
  `torch.inference_mode()` 下运行，并包在 `rng_guard()` 中。
- 每次评测追加一行 UTF-8 JSON 到 `<output_dir>/validation_history.jsonl`：

```json
{"step": 0, "epoch": 0, "dataset": "coco_val2017", "caption_variant": "coco_5captions",
 "reason": "initial", "protocol": "coco-val2017-5captions-v1", "similarity_chunk": 512,
 "wall_sec": 103.4, "metrics": {"image2text_R1": 0.5842, "...": 0.0}}
```

`reason` 为 `initial` / `interval` / `epoch_end` / `final`。

## 运行方式

```bash
cd /root/SAID
export CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export SHARE4V_DATA_ROOT=/root/datasets/ShareGPT4V
export SHARE4V_JSON=share-captioner_coco_lcs_sam_1246k_1107.json
export COCO_DATA_ROOT=/root/datasets/coco
export SHARE4V_FULL_AUDIT=/root/SAID/outputs/data_audit/sharegpt4v_full_audit.json

torchrun --nproc_per_node=4 --master_port=25961 train/train_salu.py \
  --base_model B16 --batch_size 256 --epochs 1 --said_loss_mode identifiable \
  --backbone_lr 1e-6 --head_lr 1e-4 --tau_said 0.07 --amp_dtype bf16 \
  --val_every 100 --val_sharegpt4v --eval_coco_initial --eval_coco_each_epoch --val_coco \
  --val_batch_size 64 --output_dir runs_salu/<run_name>
```

`SHARE4V_FULL_AUDIT`（或 `--strict_manifest`）会启用 full-data Gate：JSON SHA256、数据根目录、
split manifest SHA256 与 train/val overlap 必须在加载图片之前全部匹配，否则 fail-fast。
`SHARE4V_JSON` 按数据集既有约定解释为**相对 `SHARE4V_DATA_ROOT` 的路径**，绝对路径同样接受。

## 全量数据 smoke（本次验证）

- 数据：完整 ShareGPT4V 1,246,901 条（COCO 118,287 + LLaVA 558,128 + SAM 570,486），
  `sam/images` 570,486 张全部就位。
- 4 × A800-80GB，`--batch_size 256`、global batch 1,024、`--max_steps 20`、
  `--val_every 20 --val_sharegpt4v --val_coco`。
- full-data Gate：通过（V1 曾在 `SHARE4V_JSON` 为相对路径时误判为文件缺失，V2 已修正为
  先按 `SHARE4V_DATA_ROOT` 拼接再校验 SHA256）。

结果（V2 实跑）：

| 项目 | 数值 |
| --- | --- |
| steps | 20 |
| wall sec/step | 0.9715（compute 0.7975） |
| wall samples/sec | 1054.1（compute 1284.0） |
| val 记录数 | 2（`step=20 / tag=interval` 与 `tag=final`，两者数值完全一致 → 确定性成立） |
| ShareGPT4V（1,000 路） | i2t R@1 0.786 / R@5 0.935 / R@10 0.966；t2i R@1 0.795 / R@5 0.957 / R@10 0.979 |
| COCO（5,000 图 × 5 caption） | i2t R@1 0.5210 / R@5 0.7688 / R@10 0.8444；t2i R@1 0.3293 / R@5 0.5803 / R@10 0.6855 |
| 报错 | 无 NaN / OOM / 超时 |

全量数据的单步耗时明显高于此前的 no-SAM 子集（wall 0.97 vs 0.57 s/step）：SAM 图片分辨率高，
CPU 侧解码与 resize 成为瓶颈，GPU 会出现等待。这只影响吞吐，不影响指标或正确性。

> 上面的绝对值只用于确认流水线可用；20 步的 checkpoint 不是有意义的模型质量结论。

## 已知限制

- 间隔验证会串行化训练：rank 0 评测时其余 rank 停在 barrier，训练吞吐在该步会出现一次尖峰。
  COCO 协议（5,000 图）比 ShareGPT4V 协议（1,000 图）慢，建议间隔设置得比 COCO 评测耗时长。
- 评测特征在 CPU 上以 fp32 汇总；`similarity_chunk` 只决定相似度块的形状，**除极少数 near-tie
  行外不改变指标数值**（实测 COCO i2t R@5 有 1/5000 行的差异，详见下节与"固定验证 Protocol"）。
- 验证集固定为审计 manifest 定义的 split，不随训练随机种子变化，因此不同 run 之间的指标可直接比较。

## Chunked 相似度（本阶段实现）

`retrieval_metrics()` 不再构造完整的 `5000 × 25000` 相似度矩阵：

- I2T：每次取 `image_features[start:end]` 与**全部**文本做矩阵乘，按行取 top-10；
- T2I：每次取 `text_features[start:end]` 与**全部**图像做矩阵乘，按行取 top-10；
- 每行先用 1-D `argsort()` 选候选（与 legacy `train_utils.eval_coco` 的 `argsort()[-k:]` 规则一致），
  三档 Recall 都从同一份 top-10 中切出；
- `similarity_chunk` 现在真正生效：在小尺寸测试特征上，chunk = 1 / 2 / 7 / 13 / 53 / 4096 / None
  的结果**完全相同**（单元测试断言字典相等，不是近似）；真实 COCO 特征上见下节的 1/5000 near-tie。

内存：默认 `chunk = 512` → I2T 块 512×25,000×4B ≈ 48.8 MB（完整矩阵为 476.8 MB），
T2I 块 512×5,000×4B ≈ 9.8 MB。实测峰值 RSS：legacy 3153 MB → new 3279 MB。

## legacy COCO 协议等价性（正式验证）

同一 checkpoint（`runs_salu/phase22/salu_said_only_last.pt`，step 659，residual final）、
同一 `preprocess`、同一 tokenizer、同一 COCO val2017 顺序与 caption 顺序，两次独立进程分别运行：

| 指标 | legacy `train_utils.eval_coco` | new（chunk 512, image_batch 1） | 是否一致 |
| --- | --- | --- | --- |
| I2T R@1 | 0.5842 | 0.5842 | ✅ |
| I2T R@5 | **0.8128** | **0.8130** | ❌ 差 1 张 |
| I2T R@10 | 0.8850 | 0.8850 | ✅ |
| T2I R@1 | 0.39984 | 0.39984 | ✅ |
| T2I R@5 | 0.65884 | 0.65884 | ✅ |
| T2I R@10 | 0.75708 | 0.75708 | ✅ |

耗时：legacy 143.9 s（其中逐图编码约 132 s）；new（image_batch 1）148.4 s；
new（image_batch 64）103.0 s。六个指标**未完全一致**，按指令 STOP 定因，结论如下。

**已排除的原因**（均有实测证据）：

1. 特征不一致 —— 打桩捕获 legacy 实际使用的特征后逐位比较：图像 max abs Δ = **0.0**、
   文本 max abs Δ = **0.0**、差异行数 0；即两侧特征完全相同。
2. 排序 / 顺序 / 归一化 / protocol 定义 —— caption 顺序、5-caption 映射、L2 归一化、
   `argsort()[-k:]` 选择规则均一致；在同一批特征上，legacy 循环与 new 循环的命中数相同。
3. 分块与整块 GEMM 的算术差异 —— 512 行块与整块逐位比较差异元素 **0**；
   512 行块的每行 top-5 集合与整块矩阵 top-5 集合 **0 处不匹配**。

**根因**：COCO val2017 的**第 1414 行**相似度对 GEMM 分块形状敏感——整块 5000 行与单行块
在该行给出不同 top-5（整块 `{243, 6245, 6247, 12255, 23328}`，单行块
`{243, 6245, 6247, 7073, 23328}`）。legacy 在整块矩阵上按行 `argsort`，该行的命中判定为
"miss"，而分块路径判为 "hit"（`7073 ∈ truth = {7070..7074}`），于是 R@5 为 4065 而非 4064。
差异规模为 **1 行 / 5000 = 0.02%**，只出现在 R@5。

**权衡（需要 Review 决策）**：六项逐位相同要求计算完整 `5000 × 25000` 相似度矩阵（476.8 MB），
而本阶段明确要求不构造完整矩阵。当前实现选择"有界显存 + 选择规则与 legacy 一致"，
代价是这一行的浮点敏感性。若两者都要满足，可加一个仅用于认证的显式 `--exact_full_matrix`
模式；本阶段未实现，等 Review 决定。

## COCO val2017 / ShareGPT4V 训练集 overlap 审计

对完整 manifest（1,246,901 条；training 1,245,901 + validation 1,000）按 canonical 标识比较：

| 项目 | 数值 |
| --- | --- |
| COCO val2017 标注 id / 目录 stem | 5000 / 5000（差集 0） |
| COCO 来源训练 stem 数 | 118,287 |
| 训练集唯一 stem 数 | 1,245,901（无重复） |
| **overlap：COCO 源 ∩ val2017** | **0** |
| **overlap：任意来源 ∩ val2017** | **0** |
| examples | 无 |

结论：COCO val2017 的 5,000 张图没有任何一张出现在 ShareGPT4V 训练 split 中，
检索指标不存在 val 泄漏。报告落在 `outputs/data_audit/coco_val_overlap_report.json`（未提交）。

## 固定验证 Protocol（本阶段定稿）

**Canonical similarity chunk = 512**，对所有模型、所有数据集统一；不为了对齐 legacy 的
1/5000 R@5 浮点边界差异而切回完整相似度矩阵。`coco_retrieval.py` 的 docstring 已相应改写：
不再声称分块与整块在有限精度下 bitwise identical，只声明 Recall@K 定义相同、chunk 固定、
极少数 near-tie 可能因 FP32 GEMM 形状产生排序差异，因此**公平性来自固定的 evaluator + chunk**。

**ShareGPT4V-1K 冻结 cohort**（`outputs/validation/sharegpt4v1k_manifest.json`，首次生成后只读）：

- cohort：JSON 前 1,000 条（审计 held-out split），每图 1 caption → 1,000 路检索
- 每个样本保存 `json_index` / `image_path` / 三种 caption，manifest 记录 `seed = 26`、
  `dataset_json_sha256`、`similarity_chunk`
- `first_sentence`：第一句
- `fixed_sparse`：与训练相同的 prefix 规则（`'. '` 切分上的均匀随机前缀），但使用**私有 RNG**
  （`seed + json_index`），一次生成后永久复用，绝不重采样
- `full_dense`：完整 caption
- 重建同一 JSON/seed 得到完全相同的 manifest；已存在的 manifest 若不匹配则报错而非覆盖
- 变体重合（本 cohort）：`fixed_sparse == full_dense` 132/1000、`fixed_sparse == first_sentence`
  141/1000；平均句数 4.48（sparse）vs 8.12（full）——trainng 规则的固有性质，如实记录

**验证节奏**：

- `--val_every N`：只驱动 ShareGPT4V-1K（三种变体），另在 epoch end 与 final 各跑一次
- COCO 只按显式节奏：`--eval_coco_initial`（step 0）/ `--eval_coco_each_epoch`（epoch end）/
  `--eval_coco`（final）；`--val_coco` 保留为 `--eval_coco` 的别名
- **`--eval_coco_initial` 在第一次 optimizer update 之前**由 `main()` 直接调度：所有 rank 先
  `dist.barrier()`，rank 0 调 `run_initial_validation()` 跑 canonical COCO（`similarity_chunk=512`），
  记录为 `step=0 / epoch=0 / reason='initial'`，评测结束由 `run_validation_job` 恢复 `train()`，
  再一次 `dist.barrier()` 之后才进入训练循环。step 0 只调度 COCO，ShareGPT4V-1K 仍按
  间隔 / epoch end / final 走；resume 到 step > 0 的 run 跳过该初始评测（它已经越过 step 0）
- 同一 `(step, dataset, caption_variant)` 只写一次：epoch end 与 final 落在同一步时 COCO
  与 ShareGPT4V-1K 都只跑一次（实测 Run B：6 条记录而非 9 条）

**Balancing Gain 定义（全项目统一）**：

```
Balancing Gain = Full Pair Gap - Said Pair Gap
positive = Said better    negative = Said worse
relative_balancing_gain = Balancing Gain / Full Pair Gap
```

这是 `gap_comparison()` / `batch_representation_gaps()` 一直以来的定义（代码未改动任何数值），
本轮只统一了措辞：定义写在 `eval/validation_protocol.py` 的 `BALANCING_GAIN_DEFINITION` /
`BALANCING_GAIN_SIGN`，同名字段随每条诊断写入（`balancing_gain_definition` /
`balancing_gain_sign`），文档、测试与用户可见 label 共用同一句话。

**RNG 安全**：每次评测都包在 `rng_guard()` 中，退出时恢复 Python `random`、NumPy、
torch CPU 以及**全部 CUDA** RNG 状态。`validation_history.jsonl` 统一记录
`step / epoch / dataset / caption_variant / metrics / wall_sec / protocol / similarity_chunk`
（UTF-8，`ensure_ascii=False`）。

## 实测：ShareGPT4V-1K 三变体（residual final，step 659）

| 变体 | I2T R@1/5/10 | T2I R@1/5/10 | Full Gap | Full RMG | Said Gap | Said RMG | Balancing Gain | Conditioning Margin | wall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| first_sentence | 0.631 / 0.864 / 0.934 | 0.610 / 0.836 / 0.912 | 0.6755 | 0.5418 | 0.6675 | 0.5720 | +0.0080 | +0.0988 | 17.7 s |
| fixed_sparse | 0.878 / 0.973 / 0.986 | 0.852 / 0.960 / 0.982 | 0.6460 | 0.4997 | 0.6756 | 0.5345 | −0.0297 | +0.1640 | 17.5 s |
| full_dense | 0.947 / 0.995 / 0.997 | 0.931 / 0.996 / 0.997 | 0.6378 | 0.5010 | 0.6827 | 0.5409 | −0.0449 | +0.1835 | 18.2 s |

Balancing Gain 一列即 `Full Gap − Said Gap`：只有 `first_sentence` 为正（Said 更好），
`fixed_sparse` 与 `full_dense` 为负（Said 更差）。

检索只使用 `encode_image` / `encode_text`；Said 特征只出现在 Gap / RMG / conditioning 诊断中，
从不进入检索排序（evaluator 内部断言 `encode_router_input == encode_image`，实测 max abs Δ = 0.0）。
评测前后 RNG 快照完全一致（`rng_unchanged = True`）。

## RNG matched smoke（4 × A800，full 1,245,901 训练集）

- Run A：20 步，无验证
- Run B：10 步 → 在 step 10 插入 ShareGPT4V-1K 三变体验证 → 继续到 20 步

| 检查 | 结果 |
| --- | --- |
| 每步 batch 图像摘要（`batch_image_sha256`，全部 20 步） | 差异 0 |
| 每步 caption 摘要（`batch_caption_sha256`，全部 20 步） | 差异 0 |
| steps 11–20 摘要差异 | 0 |
| rank 0–3 `sampler_order_sha256` | 全部一致 |
| rank 0–3 `caption_stream_sha256` | 全部一致 |
| rank 0–3 `initial_state_sha256` | 全部一致 |
| Run A / Run B 验证记录数 | 0 / 6（step 10 与 step 20 各 3 变体） |

上面的 matched smoke 未开启 `--eval_coco_initial`；step 0 的初始评测走同一条
`run_validation_job()` 路径（同一个 `rng_guard()` + `eval()/train()` 恢复），其 RNG 中性由
`test_initial_validation_hook_emits_step0_coco_before_any_update` 断言（调用前后 `rng_snapshot()` 相等），
因此开启该 flag 不会改变训练流。
