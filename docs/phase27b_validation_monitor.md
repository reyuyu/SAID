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
- 评测特征在 CPU 上以 fp32 汇总；`similarity_chunk` 参数保留为协议标记，不改变指标数值。
- 验证集固定为审计 manifest 定义的 split，不随训练随机种子变化，因此不同 run 之间的指标可直接比较。
