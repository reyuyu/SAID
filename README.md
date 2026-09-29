# SAID · S0 Dual-Mask

由 [reyuyu](https://github.com/reyuyu) 维护的视觉–语言检索实验仓库。SAID 基于 SmartCLIP / LongCLIP，研究用描述前缀选择视觉特征，并利用剩余后缀补充监督。这里保存我的实验实现、固定训练配置、检查点校验信息和原生检索评测结果。

## 实验配置

| 版本 | 后缀对齐权重 λ_suffix | 新 U 门稀疏权重 λ_U | 已完成训练 | 入口 |
|---|---:|---:|---|---|
| **S0-DualMask-Clean v0.1** | 1 | 0 | 500 → 1000 → 3651（3 epoch） | [方法、参数与复现](experiments/s0_dualmask_masked_3epoch/README.md) |
| **S0-DualMask-Full v0.1** | 10 | 2 | 500 → 3651（3 epoch），另有 step2000 评测 | [方法、参数与复现](experiments/s0_dualmask_full_v01/README.md) |
| **S0-DualMask 3/0** | 3 | 0 | 500 → 3651（3 epoch） | [结果、参数与复现](experiments/s0_dualmask_suffix3_3epoch/README.md) |

这些配置保留相同的 S0 前缀目标、候选列前缀条件 `mS_j`、全局正配标签及后缀有效样本的 `W/V` 归约。U 门为 `Linear(1024,512) → GELU → Linear(512,512)`，正式初始化全开。

```text
L_S0    = 10 L_S + 2 S_S
L_clean = L_S0 + 1 L_U
L_full  = L_S0 + 10 L_U + 2 S_U
L_3_0   = L_S0 + 3 L_U
```

`S_U` 只统计有效正配的 U 门稀疏项。**Clean 的 U 稀疏系数为 0，不代表 S0 自带的前缀稀疏项被移除。** Full 同时改变后缀权重和 U 稀疏约束，结果只能解释为组合差异，不能单独归因于其中一项。

## Clean/Full原历史检索结果

单位为百分比，每格为 **I2T R@1 / T2I R@1**（图→文 / 文→图）。3651 次更新等于本数据流的完整 3 epoch。

| 协议（图 / 文） | Clean v0.1 · 3651 | Full v0.1 · 3651 | Full v0.1 · 2000 |
|---|---:|---:|---:|
| COCO canonical（5000 / 25000） | 61.340 / 42.272 | 59.420 / 40.296 | 59.560 / 40.392 |
| Urban-1k（1000 / 1000） | 91.400 / 89.500 | 91.800 / 90.700 | 91.900 / 90.600 |
| Flickr30k test1K（1000 / 5000） | 88.700 / 71.960 | 86.900 / 70.200 | 87.000 / 70.040 |
| DOCCI test5K（5000 / 5000） | 78.140 / 78.920 | 77.660 / 78.440 | 77.480 / 78.320 |
| DCI full（7805 / 7805） | 50.032 / 49.058 | 49.263 / 49.263 | 49.199 / 49.315 |
| Long-DCI（重建版）（7602 / 7602） | 58.471 / 58.669 | 57.722 / 60.116 | 57.722 / 59.879 |

完整双向 R@1/5/10、数据清单哈希和来源见 [评测总表](experiments/RESULTS.md)；机器可读版本见 [results.json](experiments/results.json)。Full step2000 的 COCO/Urban 为已核验历史记录，其余四项在 2026-09-14 补评，退出码均为 0。

所有结果使用裸学生的 `normalize(encode_image(I))` 与 `normalize(encode_text(C))` 做全候选池内积检索。没有条件 gate 评分、特征融合或 rerank。Long-DCI 标为**重建版**，与官方 CSV 不混用；本仓库不据单 seed 结果宣称统计显著性或全面领先。

## 500步调参补充：3/0

新增 [S0-DualMask 3/0 @500](experiments/s0_dualmask_suffix3_500/README.md)：后缀对齐权重3、新U门稀疏权重0，固定horizon3651但实际只训练500步。独立页面保存连续训练日志、六项原始检索结果、同为500步的S0/Clean/Full/1/2对照及复现命令。

3/0在COCO/Urban开发集的四方向R@1均值为69.345%，但COCO图→文59.940%未达到原选型门槛60.380%，因此未替代原先选中的1/2。追加扩展评测显示相对Clean在Urban、DCI、Long-DCI的双向R@1提高，COCO和Flickr下降；不宣称全面领先或统计显著。这里的500步结果与上表3651/2000步属于不同训练预算。

## 3/0完整3 epoch结果

3/0已从500步完整状态续训到3651步，并完成六项评测。下表为**同一台A100机器的复现对照**；上面的原历史A800数值保持不变。每格为图→文 / 文→图R@1（%）。

| 协议 | 3/0 · 3651 | Clean · 3651（本机） | Full · 3651（本机） |
|---|---:|---:|---:|
| COCO canonical | 60.680 / 41.700 | 61.280 / 42.300 | 59.380 / 40.256 |
| Urban-1k | 91.400 / 90.300 | 91.700 / 89.700 | 91.800 / 90.600 |
| Flickr30k test1K | 87.600 / 71.640 | 88.500 / 71.900 | 86.800 / 70.120 |
| DOCCI test5K | 78.300 / 79.280 | 78.260 / 78.860 | 77.600 / 78.500 |
| DCI full | 49.942 / 49.558 | 50.083 / 48.930 | 49.263 / 49.327 |
| Long-DCI（重建版） | 58.748 / 59.813 | 58.366 / 58.603 | 57.695 / 60.195 |

相对本机Clean，12个R@1方向6升6降，等权平均仅+0.040个百分点；相对本机Full，9升3降，平均+0.619个百分点。3/0改善了部分长描述检索表现，仍有COCO/Flickr取舍，不据此宣称全面领先或统计显著。

[完整R@1/5/10、训练日志及复现](experiments/s0_dualmask_suffix3_3epoch/README.md) · [逐项差值CSV](experiments/s0_dualmask_suffix3_3epoch/comparison.csv)

## 获取代码与复现

```bash
git clone https://github.com/reyuyu/SAID.git
cd SAID
conda create -n said-smartclip python=3.10 -y
conda activate said-smartclip
python -m pip install -r requirements.txt
```

`main` 提供项目导航、共享实现及这些配置的完整 Git 历史。共享 `model/dual_mask_suffix.py` 和训练入口保留 Full 分支扩展，默认系数仍为 1/0；Full 必须显式使用 10/2。**复现已有轨迹时，使用各实验说明中的固定训练 SHA 和原初始化，不把今天的 main SHA 当成历史训练 SHA。** Clean/Full 的严格续训入口分别在其固定提交中。

```bash
# 两份互不覆盖的固定训练代码；0→500 的起点版本见各实验复现说明。
git worktree add --detach ../SAID-clean 11af80b344c623b27b93069f9be526970c9c950c
git worktree add --detach ../SAID-full 873b43a5bc000528311e22aac01e92045ac898e0
```

正式训练配置为 ViT-B/16、4 卡 × 256、accumulation=1、seed=0、FP32 主参数 + BF16 autocast，完整 cosine horizon 为 3651。CLIP / S0 门 / U 门三组 LR 为 `1e-6 / 1e-3 / 1e-4`。原随机前 K 句与“后缀排除末句”规则固定。

环境、数据、初始化和续训步骤见 [Clean 复现命令](experiments/s0_dualmask_masked_3epoch/REPRODUCE.md)、[Full 复现命令](experiments/s0_dualmask_full_v01/REPRODUCE.md)。仅 clone 仓库不含训练图片、完整标注或权重；缺失输入不能用任意 CLIP 初始化替代。跨硬件复现应报告真实浮点误差。

## 原生学生推理

将已校验的 **bare student** 放到本地 `checkpoints/`。完整训练检查点需先按实验说明严格导出。

```python
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from model import longclip

model, preprocess = longclip.load_from_clip(
    "ViT-B/16", device="cpu", args=argparse.Namespace()
)
state = torch.load("checkpoints/bare_student.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()
images = preprocess(Image.open("assets/cat.webp").convert("RGB")).unsqueeze(0).to(device)
tokens = longclip.tokenize(["A cat holding a sign.", "A dog on a bicycle."],
                           context_length=248, truncate=True).to(device)
with torch.no_grad():
    scores = (
        F.normalize(model.encode_image(images).float(), dim=-1)
        @ F.normalize(model.encode_text(tokens).float(), dim=-1).T
    )
print(scores)
```

该示例先构造与训练一致的模型，再严格加载全部学生张量。首次构造可能下载 OpenAI CLIP base；此 base 只用于构造，所有模型状态随后由裸学生覆盖。

## 目录与产物

- [experiments/](experiments/README.md)：各实验导航、参数、结果和复现资料。
- [model/dual_mask_suffix.py](model/dual_mask_suffix.py)、[train/train_dual_mask_suffix.py](train/train_dual_mask_suffix.py)：共享模型与训练入口。
- [docs/](docs/)：历次验收、500 步与 3 epoch 原始报告。
- [检索协议](docs/extended_retrieval/README.md)：COCO、Urban、Flickr30k、DOCCI、DCI 相关准备与评测说明。
- [权重与数据说明](experiments/ASSETS.md)：文件身份、SHA256、独立存储与迁移边界。
- [本次 main 同步记录](docs/main_sync_20260914.md)：合并来源、冲突处理和验证结果。

代码和小型证据使用 Git 管理；模型、数据与大缓存独立存储。仓库没有公开权重下载附件时，不把私人云盘备份当成公开下载链接。

## 来源与许可

本项目在 [SmartCLIP](https://github.com/Mid-Push/SmartCLIP)、[LongCLIP](https://github.com/beichenzbc/Long-CLIP) 和 [OpenAI CLIP](https://github.com/openai/CLIP) 的基础上开展。原 SmartCLIP 项目说明及其引用保存在 [上游 README](docs/upstream/SmartCLIP-README.md)。本仓库保留 [Apache-2.0 LICENSE](LICENSE) 与上游署名；各数据集遵循各自的使用条款。
