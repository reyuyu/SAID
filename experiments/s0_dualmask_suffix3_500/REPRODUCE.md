# 复现3/0 @500

目标是从共同初态训练恰好500步并运行六项原生检索。训练代码固定为 `dcd33877f1f77a901d292190834f1ffd83a049a8`；发布文档的提交不是训练提交。

## 1. 获取文档与固定代码

```bash
git clone https://github.com/reyuyu/SAID.git SAID
export SAID_DOCS_REPO="$PWD/SAID"
export SAID_BUNDLE="$SAID_DOCS_REPO/experiments/s0_dualmask_suffix3_500"
git -C "$SAID_DOCS_REPO" worktree add --detach ../SAID-suffix3-train dcd33877f1f77a901d292190834f1ffd83a049a8
export SAID_TRAIN_REPO="$PWD/SAID-suffix3-train"
```

不要在训练中的工作区切分支。四卡配置需要能容纳约69GiB/卡的显存，本轮实际使用4×A100 80GB PCIe。

## 2. 环境

```bash
conda create -n said-suffix3 --file "$SAID_BUNDLE/environment/conda-explicit-linux-64.txt" -y
conda activate said-suffix3
python -m pip install -r "$SAID_BUNDLE/environment/requirements.txt"
python -m pip check
export SAID_PYTHON="$(command -v python)"
```

本次80项包版本与原环境锁匹配，Python3.10.21、torch2.5.1+cu124、torchvision0.20.1+cu124、NumPy2.2.6、Pillow12.3.0。实际安装将OpenAI CLIP固定提交以源码ZIP构建；本包可安装requirements将这一项规范化为同一Git提交URL，其他版本不变。`environment/cpu-smoke-and-versions.json` 保留原验证结果，`runtime.json` 保留发布前运行环境检查；后者的采集时点明确单列。

当前与原实验的GPU/驱动不同。固定版本、张量初态及数据身份能提高可比性，不能保证跨硬件整个轨迹逐位一致。包源将来若移除版本，应保留失败记录并说明实际环境差异。

## 3. 完整数据与构造缓存

准备并核对：

- `share-captioner_coco_lcs_sam_1246k_1107.json`：SHA256 `5c5f0f4ee58d7b7467f9e49eb5b17f930890a8a0c18a4e2a5be6b15714ef8b3c`，1,246,901条。
- COCO train2017、LLaVA pretrain、SAM图片树；跳过前1000条后为1,245,901条训练记录。`--total-len 1000` 是切片起点。
- SAM采用原实验的数据来源及固定revision：见[数据补全记录](../../docs/phase27a2_full_data_gate.md)。51个分片对应570,486张图片，训练部分569,486张。该来源未与Meta原始图片逐字节比较。
- COCO val2017及annotations、Urban-1k，结构见[输入迁移说明](../s0_dualmask_masked_3epoch/TRANSFER_AND_UPLOAD.md)。
- 四项扩展数据按 [data_verification.json](evidence/evaluation/data_verification.json) 的固定来源、文件大小、SHA及manifest身份准备。DCI/Long-DCI共用图片。

Hugging Face下载使用 `https://hf-mirror.com`，保留固定revision和LFS文件SHA。图片、完整标注和模型没有包含在Git中。

先在当前数据根目录生成并通过完整审计：

```bash
export SHARE4V_DATA_ROOT=/your/datasets/ShareGPT4V
export SHARE4V_JSON=share-captioner_coco_lcs_sam_1246k_1107.json
export SHARE4V_FULL_AUDIT=/your/audit/sharegpt4v_full_audit.json
cd "$SAID_TRAIN_REPO"
"$SAID_PYTHON" -m tools.data.audit_sharegpt4v \
  --root "$SHARE4V_DATA_ROOT" --json "$SHARE4V_DATA_ROOT/$SHARE4V_JSON" \
  --output-dir /your/audit --workers 24
```

审计文件绑定实际根目录、标注和split manifest，不可把另一台机器的PASS文件直接挪用。

CLIP base置于运行用户的 `~/.cache/clip/ViT-B-16.pt`，固定URL及SHA见 [assets.json](manifests/assets.json)。它用于构造模型。

## 4. 重建并验证共同初态

本轮原始 `cvssl_initial.pt` 容器未取得。按seed0和固定LongCLIP构造代码，得到与历史317个模型张量相同的摘要，包含14个原S0门张量：

`caf61198def1b78654d6b70ef16db9a3475ea81b3a16ab8a799989f574de2faf`

```bash
export SAID_INIT=/your/assets/common_init_reconstructed.pt
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 "$SAID_PYTHON" "$SAID_BUNDLE/tools/rebuild_initial_state.py" \
  --repo "$SAID_TRAIN_REPO" --output "$SAID_INIT" --report /your/assets/initialization.json
```

脚本验证base SHA、固定源码、张量摘要及保存重载；摘要不匹配时失败。新的保存文件有自己的文件SHA，不能套用历史容器 `c1a4a2be...` 或本机重建文件 `6e9cb606...` 的文件身份。后续以该次实际报告为准。

## 5. 恰好500步

```bash
export SAID_RUN=/your/new-runs/suffix3_u0_500
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash "$SAID_BUNDLE/tools/train500.sh"
```

脚本固定4×256、accumulation1、workers8、seed0、CLIP/mask/U LR=1e-6/1e-3/1e-4、warmup200、BF16、3 epochs且max-steps500。因此余弦horizon仍为3651。S0的10/2目标保留，只设 `--lambda-suffix 3 --lambda-u-sparse 0`。

源码与初态身份在启动前检查。输出目录必须全新；训练失败即返回真实退出码，保留失败目录，不自动续接失败轨迹。成功后应有500条连续更新日志、step000500完整checkpoint、config和退出码0。本轮500步耗时、逐rank流摘要及真实命令见 `evidence/training/`。

## 6. 严格导出与六项评测

```bash
"$SAID_PYTHON" "$SAID_BUNDLE/tools/export_student.py" \
  --repo "$SAID_TRAIN_REPO" \
  --checkpoint "$SAID_RUN/s0_dual_mask_suffix_masked_step000500.pt" \
  --output "$SAID_RUN/bare_student_step500.pt" --report "$SAID_RUN/export.json"

"$SAID_PYTHON" "$SAID_BUNDLE/tools/evaluate.py" \
  --repo "$SAID_TRAIN_REPO" --checkpoint "$SAID_RUN/bare_student_step500.pt" \
  --export-report "$SAID_RUN/export.json" \
  --coco-root /your/datasets/coco --urban-root /your/datasets/Urban1k/Urban1k \
  --benchmark-root /your/datasets/retrieval_benchmarks \
  --output-dir /your/new-evaluations/suffix3_u0_500 --gpus 0,1,2,3
```

导出器检查实际500更新、系数3/0、horizon、模型及门严格加载、三组优化器计数和状态有限性，并验证裸学生保存前后张量相等。迁移本次已公布的完整checkpoint时，还可用 `--expected-sha256` 绑定 [assets.json](manifests/assets.json) 的身份。

评测入口调用仓库原冻结函数，不重写度量：

- COCO：图像batch64、对应文本通常320，CPU归一化，similarity chunk512和逐行argsort。
- Urban：图像batch64，1000条文本一次编码，设备内归一化与完整矩阵topk。
- 扩展四项：图文各batch64，设备上FP32归一化后转CPU，完整相似度矩阵topk；双向全候选池。

不使用gate评分、特征融合或rerank。每项分目录保留实际命令、原始结果和退出码，避免 `extended_summary.json` 相互覆盖。参数身份和manifest校验失败不会继续评估。

### 扩展清单的历史口径

四份manifest分别是 `docci_test.jsonl`、`flickr30k_test1k.jsonl`、`dci_full.jsonl`、`long_dci_reconstructed.jsonl`；SHA见数据验证记录。DOCCI与Flickr可用仓库的对应parser从官方英文JSONL和显式test1K CSV生成。

**DCI必须使用历史 `8f7f1a5` 版本的 `parse_dci`（short_caption，否则回退extra_caption）。** 当前拼接两字段的parser不能复现本实验清单。示例：

```bash
git -C "$SAID_DOCS_REPO" show 8f7f1a5:tools/prepare_retrieval_benchmarks.py > /your/tools/said_prepare_dci_historical.py
"$SAID_PYTHON" /your/tools/said_prepare_dci_historical.py --dataset dci \
  --input /your/dci/annotations/densely_captioned_images/annotations \
  --manifest /your/datasets/retrieval_benchmarks/manifests/dci_full.jsonl
"$SAID_PYTHON" "$SAID_DOCS_REPO/tools/prepare_retrieval_benchmarks.py" --reconstruct-long-dci \
  --input /your/dci/annotations/densely_captioned_images/annotations \
  --manifest /your/datasets/retrieval_benchmarks/manifests/long_dci_reconstructed.jsonl
```

Long-DCI是7602条extra_caption重建版；其图片与DCI共用。图片从校验过的归档按manifest文件名映射，保留字节并完整解码，不能只把压缩包存在当作数据已准备好。任何重建结果都必须达到本包列出的清单SHA和图片/文本数量。
