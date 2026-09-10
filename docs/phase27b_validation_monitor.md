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
| `train/train_salu.py` | `--val_every` / `--val_sharegpt4v` / `--val_coco` / `--val_batch_size`；`run_standard_validation()`；`append_val_record()` |

检索指标本身与特征来源无关：`retrieval_metrics()` 先把特征做 L2 归一化，再按方向统计命中率，
因此对特征整体缩放不敏感。

## 训练中的接入方式

- 每 `--val_every N` 步（`N > 0`）以及训练结束时各评测一次；`--val_every 0` 表示只做结束时的评测。
- 评测只在 rank 0 执行，前后各有一次 `dist.barrier()`：其它 rank 在 barrier 处等待，
  因此不会有人在 rank 0 评测时提前进入下一步的集合通信（不会出现 NCCL 超时或错位）。
- 评测期间模型切到 `eval()`，结束后切回 `train()`；评测在 `torch.inference_mode()` 下运行。
- 每次评测追加一行 JSON 到 `<output_dir>/val_metrics.jsonl`：

```json
{"kind": "standard_retrieval", "step": 20, "tag": "interval", "world_size": 4,
 "sharegpt4v": {"image2text_R1": 0.0, "image2text_R5": 0.0, "image2text_R10": 0.0,
                "text2image_R1": 0.0, "text2image_R5": 0.0, "text2image_R10": 0.0},
 "coco": {"image2text_R1": 0.0, "...": 0.0}}
```

`tag` 为 `interval`（按间隔触发）或 `final`（训练结束）。

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
  --val_every 100 --val_sharegpt4v --val_coco --val_batch_size 64 \
  --output_dir runs_salu/<run_name>
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
- 评测特征在 CPU 上以 fp32 汇总；`similarity_chunk` 只决定相似度块的形状，不改变指标数值。
- 验证集固定为审计 manifest 定义的 split，不随训练随机种子变化，因此不同 run 之间的指标可直接比较。

## Chunked 相似度（本阶段实现）

`retrieval_metrics()` 不再构造完整的 `5000 × 25000` 相似度矩阵：

- I2T：每次取 `image_features[start:end]` 与**全部**文本做矩阵乘，按行取 top-10；
- T2I：每次取 `text_features[start:end]` 与**全部**图像做矩阵乘，按行取 top-10；
- 每行先用 1-D `argsort()` 选候选（与 legacy `train_utils.eval_coco` 的 `argsort()[-k:]` 规则一致），
  三档 Recall 都从同一份 top-10 中切出；
- `similarity_chunk` 现在真正生效：chunk = 1 / 2 / 7 / 13 / 53 / 4096 / None 的结果**完全相同**
  （单元测试断言字典相等，不是近似）。

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
