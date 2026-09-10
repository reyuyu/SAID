# Phase 2.7A.2：数据补全和 Full Data Gate

本阶段只做数据准备、审计和 DataLoader smoke。没有模型 forward、backward、optimizer 或正式训练。

## 来源与可复现标识

- Hugging Face 数据集：`hanlincs/InternVL-SA1B-Caption-WebDataset`
- 固定 revision：`4cdaea026f51899bb88d24d423121d2106a943ba`
- 仅 `data/sa_000000.tar` 至 `data/sa_000050.tar`。
- Windows 通过 `hf-mirror.com` 取得固定 revision 的文件；每个 tar 对照固定文件清单中的 LFS SHA256 和 size。
- 上传后独立计算服务器 SHA256，只有本地、固定清单和服务器三方相等时，才允许删除 Windows 流水目录里的该 tar。
- 服务器保留全部 tar。已有 `000000` 复用，未重新下载。

这是与所需 SA-1B ID 兼容的候选 RGB 来源。分辨率多样只排除“全部固定 512×512”，不能排除等比例 resize 或重新编码。当前没有 Meta 原始图像，不能宣称 bytes 一致。

## 每个 shard 的处理

`tools/data/transfer_hf_sa1b.py` 使用有界并发下载和上传，支持 curl 续传与 SFTP `.partial` 续传。SSH 凭据只从环境读取，host key 必须已在 known_hosts 中。Windows 原生 SFTP 可通过 `--native-sftp` 使用；非交互认证由运行环境配置 SSH_ASKPASS，不在仓库保存密码。

`tools/data/prepare_hf_sa1b.py` 先检查文件大小和 SHA256，再直接扫描 tar。每个 shard 保存完整 ID 集及摘要。全局文件锁保证并发上传完成后的 ID 合并和抽取顺序一致。重复、额外或跨 shard 重复 ID 立即失败并记录错误。

Selective extraction 只写 JSON 引用的 `sam/images/sa_<id>.jpg`，拒绝越界名称和非普通文件。临时文件完成 `PIL.verify`、重新打开、RGB 转换和 load 后才原子替换。已有有效图像不覆盖；中断后的重跑校验现有文件并继续缺失项。

`sam_shards.json` 中的 `image_verified` 指抽取时全部 required 图片通过校验；最终仍需独立全量图片审计。ID union 数值从实际扫描计算，不用乘法推断完整性。

## 全量审计、split 与 Gate

`tools/data/audit_sharegpt4v.py` 使用 spawn worker 和最多 2,048 个任务的有界批次，避免继承父进程锁或一次排队百万个任务。每张唯一图片必须存在且可完整解码，失败按 records 权重分类计数，完整错误写入 JSONL。未发生静默过滤。

Split 始终为 JSON 前 1,000 条 validation、其余 training。manifest 保存原始 JSON index、相对路径、来源、split 和解析符号链接后的 canonical identifier。报告包含 JSON SHA256、split manifest SHA256、overlap 和审计时间。若 overlap 非零，保留所有 records 等待 Review。

训练 Dataset 可显式提供 `strict_manifest`，或通过 `SHARE4V_FULL_AUDIT` 设置；完整数据的正式实验必须设置该变量。Gate 检查完整审计、JSON SHA256、数据根目录、manifest SHA256 和 overlap；任何不一致在加载图片/模型之前失败。没有 `skip_missing` 选项。审计只能证明扫描当时的状态；之后移除或破坏图片仍会由现有 loader 报错。

示例（路径由环境指定）：

```bash
python -m tools.data.audit_sharegpt4v --root "$SHARE4V_DATA_ROOT" \
  --json "$SHARE4V_DATA_ROOT/$SHARE4V_JSON" --workers 24

python -m tools.data.loader_smoke --root "$SHARE4V_DATA_ROOT" \
  --output outputs/data_audit/loader_single.json

torchrun --standalone --nproc_per_node=4 -m tools.data.loader_smoke \
  --root "$SHARE4V_DATA_ROOT" --output outputs/data_audit/loader_4gpu.json
```

Smoke 使用实际 training Dataset 的 `__getitem__`、同一 CLIP 224 预处理和原 caption sampling。单进程固定随机读取 1,000 条；4 GPU 使用 DistributedSampler、每卡 20 batches × 16 samples，并验证全局 sampler 不重复。无模型构建、forward、backward 或 optimizer。

## 中文只读页面

“数据完整性”页面读取小型 audit/shard/disk artifact，不打开 split 全量条目、不扫描图片。完整审计、51 shard 验证、ID union 和无 overlap 同时满足才显示绿色 Gate。未完成、缺文件或错误 artifact 都显示未通过。字体继承既有中文 fallback。

输出、图片、tar、日志和凭据不提交 Git。
