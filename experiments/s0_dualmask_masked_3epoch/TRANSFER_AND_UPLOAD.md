# 输入迁移、数据准备和 GitHub 上传

## 必需资产

完整原路径、文件大小和SHA256见 [assets.json](manifests/assets.json)。不上传模型/图片/大型标注到Git普通提交，也不在文档中写SSH密码、token或签名下载地址。

| 资产 | 用途 | 本包是否包含 |
|---|---|---|
| `cvssl_initial.pt`（612,068,926 bytes） | 完整共同初始化，含原S0 mask | 否，迁移原文件 |
| `ViT-B-16.pt`（350,837,078 bytes） | 模型构造用OpenAI CLIP base缓存 | 否，原缓存或官方固定URL |
| `share-captioner_coco_lcs_sam_1246k_1107.json` | 训练caption与图片相对路径 | 否，迁移或官方数据源 |
| ShareGPT4V图片树 | 全部原训练图片 | 否 |
| COCO val2017 + captions_val2017.json | 5000×25000冻结评估 | 否 |
| Urban1k image/ + caption/ | 1000×1000冻结评估 | 否 |
| 完整masked step500或1000 | 直接接续原训练时需要 | 否，迁移原文件 |
| `bare_student_step1000.pt` | 只做原生检索；不能续训 | 否，迁移或严格导出 |

公开CLIP base不等于`cvssl_initial.pt`；新建seed=0模型也不能替代已冻结、含S0 mask的共同初态。没有共同初态原文件时，能复现算法但不能声称起点完全相同。

## 官方来源

- [ShareGPT4V官方数据说明](https://github.com/ShareGPT4Omni/ShareGPT4V/blob/master/docs/Data.md) 和 [标注数据仓库](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V)。本实验仅用1.246M captioner文件及它引用的 COCO/LLaVA-pretrain/SAM 图片，保持JSON字节/顺序。
- [COCO官方主页/下载](https://cocodataset.org/#download)。训练部分为train2017，评估部分为val2017及captions_val2017.json，不能混成2014 1k/fivefold协议。
- [Urban1k官方数据仓库](https://huggingface.co/datasets/BeichenZhang/Urban1k)，来源是[Long-CLIP官方项目](https://github.com/beichenzbc/Long-CLIP)。原评估记录的 revision=null，因此本包使用实际图片/文本SHA256锁定内容，不编造数据revision。
- base CLIP 的固定URL和校验已写入生产 `model/longclip.py`，下载路径为 `https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt`；必须匹配该SHA256。

官方入口于2026-09-14核对。公开数据下载若与本包SHA不一致，先找差异，不要自动重排、重写JSON或跳过样本。数据源对应的使用条款仍适用。

## 目录布局

```text
ShareGPT4V/
  share-captioner_coco_lcs_sam_1246k_1107.json
  coco/train2017/*.jpg
  llava/llava_pretrain/images/...
  sam/images/...
coco/
  val2017/*.jpg
  annotations/captions_val2017.json
Urban1k/Urban1k/
  image/<stem>.jpg
  caption/<stem>.txt
```

训练图片相对路径以原JSON `image` 字段为准；不要仅根据上面的示意图重命名文件。COCO/Urban逐文件字节清单在 `manifests/evaluation_files_sha256.json`（7000文件）。训练标注已做SHA与全部行数/来源统计，但训练全图片本轮没有完整哈希扫描，避免影响正在进行的数据读取；新机器需通过迁移校验补足。

## 从原服务器迁移

使用自己已有SSH配置或密钥，不将凭据提交到GitHub。以下只是模板，SSH_HOST/PORT及目标目录由你填写：

```bash
rsync -a --info=progress2 -e 'ssh -p PORT' SSH_HOST:/root/datasets/ShareGPT4V/ /your/datasets/ShareGPT4V/
rsync -a --info=progress2 -e 'ssh -p PORT' SSH_HOST:/root/datasets/coco/ /your/datasets/coco/
rsync -a --info=progress2 -e 'ssh -p PORT' SSH_HOST:/root/datasets/Urban1k/ /your/datasets/Urban1k/
scp -P PORT SSH_HOST:/root/SAID-gap-completion/runs_salu/said_cls_cvssl/shared_init/cvssl_initial.pt /your/assets/cvssl_initial.pt
scp -P PORT SSH_HOST:/root/.cache/clip/ViT-B-16.pt /your/cache/ViT-B-16.pt
```

直接续原1000还要复制：

```bash
scp -P PORT SSH_HOST:/root/SAID-s0-dualmask-clean-v01/runs_salu/dual_mask_suffix_masked_from500_continuation_v02/s0_dual_mask_suffix_masked_step001000.pt /your/assets/
```

不要复制正在写入的`.tmp`文件。1000 checkpoint已完成且固定；其他最新checkpoint先确认对应步保存完成。

数据树迁移后可在原训练不受影响的时段运行只读checksum比较：

```bash
rsync -acn --itemize-changes -e 'ssh -p PORT' SSH_HOST:/root/datasets/ShareGPT4V/ /your/datasets/ShareGPT4V/
```

`-n`是dry-run，`-c`按内容比较；不要附加`--delete`。训练图片数量大，完整checksum比较会消耗两端IO，结果/退出码需要另存。对于软链接，确保目标也可访问；普通`-a`保留链接本身。

## 保存和同步新机器结果

为每次复现用独立结果目录，保存命令、阶段真实exitcode、config、环境、连续更新日志、每rank数据摘要、checkpoint SHA、测评JSON和实际硬件。将新结果命名为新replica，不能覆盖本包`evidence/`里的原始结果。可用`sha256sum`给完整checkpoint和bare导出生成新的迁移清单。

本GitHub目录将说明、必要的可移植入口、固定环境、hash manifests和小型证据放在一起；未上传模型和数据。不要对模型运行`git add .`，不要提交个人`env.sh`、SSH配置、私钥或密码。

运行中仓库HEAD必须冻结，所以同步文档推荐在另一工作区/clone完成：

```bash
git clone --branch codex/s0-dualmask-clean-v01 https://github.com/reyuyu/SAID.git SAID-docs-update
cd SAID-docs-update
# 只修改 experiments/s0_dualmask_masked_3epoch/ 下的文档/清单/新replica结果
git add -- experiments/s0_dualmask_masked_3epoch
git diff --cached --stat
git diff --cached --check
git commit -m 'docs: record masked reproduction evidence'
git push origin HEAD:codex/s0-dualmask-clean-v01
```

这次同步使用独立detached工作区普通push，原在训工作区仍为`11af80b`，不重置或中断当前训练。3epoch最终结果未完成时必须标RUNNING/NOT AVAILABLE；不能预写成功退出码或最终检索数值。
