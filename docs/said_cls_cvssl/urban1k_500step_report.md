# Urban-1k 评测：S0@500 / C0@500 / C1@500（含 COCO 对照）

本轮只做一件事：把已完成的 500 步臂在 **Urban-1k** 上评测，并与已有 COCO 结果合并成一张
500-step SOTA 表。**没有新增训练**，COCO 未重跑（沿用之前 canonical 评测的 JSON）。

后续约定：**不再比较 ShareGPT4V，只看 COCO 与 Urban-1k。**

## 1. 数据集来源与指纹

| 项 | 值 |
| --- | --- |
| 来源 | HuggingFace `BeichenZhang/Urban1k`（Long-CLIP 官方发布，Urban-200 的扩容版） |
| revision（固定） | `953fcf1d1a3bde031d9424a91a0154e594b5a586` |
| 归档 | `Urban1k.zip`，80,586,923 B，**sha256 `08e42b3fada77abf7f890a087ed6e9f9fbba1dc143f9f09801ed83187ae617b8`**（与 HF LFS 记录的 oid 逐位一致） |
| 本地路径 | `/root/datasets/Urban1k/Urban1k/{image,caption}` |
| 规模 | **1000 图 / 1000 caption**，stem 集合完全一致（上游脚本按文件名配对） |
| caption 长度 | 词数 min 74 / p50 106 / p90 120 / max 179；单行（与上游 `readlines()[0]` 一致） |
| 下载方式 | `hf-mirror.com`（`huggingface.co` 在本机不可达，GitHub 可达）；下载与解压脚本已记录 |

caption 中位 106 词、最长 179 词：这是**长 caption 基准**，正好压 LongCLIP 248-token 的能力，
也是选择它替代 ShareGPT4V-1K 作为第二评测集的理由。

## 2. 评测程序如何对齐 SmartCLIP 的 Urban1k 协议

上游参考：`eval/retrieval/Urban1k.py`（Mid-Push/SmartCLIP，自仓库 initial commit 起未改动）。它做的是：

```python
model, preprocess = longclip.load(checkpoint)
text_feature = model.encode_text(longclip.tokenize(captions, truncate=True))
text_feature /= text_feature.norm(dim=-1, keepdim=True)
image_embeds = stack(model.encode_image(preprocess(image)))
image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
T2I R@1 = mean(argmax(text_i @ image_embeds.T) == i)
I2T R@1 = mean(argmax(image_i @ text_feature.T) == i)
```

新增的 `tools/urban1k_retrieval.py` **逐条复刻**该口径：

* 主表示 = `normalize(model.encode_image(preprocess(image)))` / `normalize(model.encode_text(tokenize(...)))`；
* 全 1000 候选池、纯内积、对角为正样本、无重排序、无 masked Said/U 特征、无 decoder 预测向量；
* 同一个 248-token 截断 tokenizer 与 224 预处理（本仓库的 reference transform 已与 LongCLIP 的
  `_transform(224)` 做过 `torch.equal` 验证）；
* 上游**只打印 top-1**，这里补 R@5/R@10；报告中明确 `upstream_metrics_reproduced = [T2I R@1, I2T R@1]`，
  `added_here = [R@5, R@10]`，所以 R@1 可与上游数字直接比较。

**严格加载（不静默跳过）**：`tools/eval_urban1k_cls.py` 用 `strict=True` 加载；`--expect-steps` 会核对
checkpoint 内的 `completed_steps`，不符即失败；`clip.` / `module.` 前缀只在"剥离后键集合是模型键的
子集"时才剥离，否则报错。三个 500-step checkpoint 均为 `missing=[] unexpected=[]`。

COCO 一列**没有重跑**，直接读取既有 canonical JSON：
`outputs/cvssl_screening/S0_canonical.json`、`C0_canonical.json`、
`/root/SAID-c1-tcr/outputs/cvssl_screening/c1_tcr/C1_step500_canonical.json`。

## 3. 结果表（500-step，I2T/T2I R@1/5/10）

| 数据集 | 指标 | Initial | S0@500 | C0@500 | C1@500 | C1 − S0 |
| --- | --- | --- | --- | --- | --- | --- |
| **Urban-1k** | I2T R@1 | 0.6820 | **0.8700** | **0.8730** | 0.8700 | +0.0000 |
| **Urban-1k** | I2T R@5 | 0.8950 | 0.9710 | 0.9730 | 0.9730 | +0.0020 |
| **Urban-1k** | I2T R@10 | 0.9440 | 0.9880 | 0.9890 | **0.9910** | +0.0030 |
| **Urban-1k** | T2I R@1 | 0.5280 | **0.8420** | 0.8340 | 0.8370 | −0.0050 |
| **Urban-1k** | T2I R@5 | 0.7740 | **0.9670** | 0.9620 | 0.9630 | −0.0040 |
| **Urban-1k** | T2I R@10 | 0.8480 | 0.9810 | **0.9840** | 0.9810 | +0.0000 |
| COCO val2017 | I2T R@1 | 0.5170 | **0.6058** | 0.6042 | 0.6038 | −0.0020 |
| COCO val2017 | I2T R@5 | 0.7662 | 0.8220 | 0.8224 | **0.8248** | +0.0028 |
| COCO val2017 | I2T R@10 | 0.8428 | **0.8906** | 0.8896 | 0.8880 | −0.0026 |
| COCO val2017 | T2I R@1 | 0.3269 | 0.4124 | 0.4072 | 0.4122 | −0.0002 |
| COCO val2017 | T2I R@5 | 0.5776 | 0.6709 | 0.6655 | **0.6719** | +0.0010 |
| COCO val2017 | T2I R@10 | 0.6823 | 0.7662 | 0.7636 | **0.7664** | +0.0002 |

### J = (I2T R@1 + T2I R@1) / 2

| | Urban-1k J | COCO J |
| --- | ---: | ---: |
| Initial | 0.6050 | 0.4220 |
| S0@500 | **0.8560** | **0.5091** |
| C0@500 | 0.8535 | 0.5057 |
| C1@500 | 0.8535 | 0.5080 |

## 4. 三个读数（按 R@1，百分点）

**S0 − Initial（训练的总收益）**

| 数据集 | I2T R@1 | T2I R@1 |
| --- | ---: | ---: |
| Urban-1k | **+18.80 pp** | **+31.40 pp** |
| COCO | +8.88 pp | +8.54 pp |

**C0 − S0**

| 数据集 | I2T R@1 | T2I R@1 |
| --- | ---: | ---: |
| Urban-1k | +0.30 pp | **−0.80 pp** |
| COCO | −0.16 pp | −0.51 pp |

**C1 − S0**

| 数据集 | I2T R@1 | T2I R@1 |
| --- | ---: | ---: |
| Urban-1k | ±0.00 pp | **−0.50 pp** |
| COCO | −0.20 pp | −0.02 pp |

## 5. 结论（只陈述测到的东西）

1. **Urban-1k 上没有任何臂超过 S0@500。** I2T R@1：S0 0.8700，C0 0.8730（+0.3 pp，唯一为正），
   C1 0.8700（持平）；T2I R@1：S0 0.8420 最高，C0 −0.8 pp、C1 −0.5 pp。J 也是 S0 最高（0.8560）。
2. **这与 COCO 的结论一致**：C0 与 C1 都在 S0 附近 ±0.5 pp 内，无一致方向的优势。
   两个数据集都没有给出"C0/C1 优于 S0"的证据。
3. **两集都远超 Initial**：Urban-1k I2T +18.8 pp / T2I +31.4 pp，COCO +8.9 / +8.5 pp。
   即 500 步训练本身有效，差异只出现在臂之间。
4. **Urban-1k 的初始基线（0.6820）明显高于 COCO（0.5170）**，且所有臂在 Urban-1k 上的绝对值
   都高得多（I2T R@1 ~0.87 vs ~0.60）。这与"Urban-1k 的 caption 是描述性长文本、与
   LongCLIP 的 248-token 能力天然契合"一致；COCO 是 5 条短 caption 的经典协议，难度不同。
   **两集数字不可横向比较**，只能各自纵向比臂。
5. Urban-1k 的 caption 中位 106 词，因此它比 ShareGPT4V-1K 的 `first_sentence` 更接近"长文本"
   场景；不过它仍是被反复用于选型的开发集，不构成独立泛化证明。

**未证明**：Unsaid 语义保留、互补 mask 优于全局（G0）/随机（R0）SSL、统计显著性 —— 均 **NOT ESTABLISHED**
（G0/R0 未运行，无 withheld/概念基准）。

## 6. 产物

| 项 | 路径 |
| --- | --- |
| Urban-1k 原始结果 | `outputs/cvssl_screening/baseline_urban1k/{Initial,S0_step500,C0_step500,C1_step500}_urban1k.json` |
| 数据集 | `/root/datasets/Urban1k/`（含 `REVISION.txt` 固定 revision） |
| 评测模块 | `tools/urban1k_retrieval.py` |
| 评测 CLI | `tools/eval_urban1k_cls.py` |
| 表格生成 | `tools/diag/build_urban1k_table.py` |

## 7. 未执行

| 项 | 状态 |
| --- | --- |
| 从 S0/C0/C1 继续训练或 3 epoch | **NOT RUN** |
| G0 / R0 | **NOT RUN** |
| 超参搜索 | **NOT RUN** |
| step100/250 的 Urban-1k | **NOT RUN**（本轮只评 500 步终点） |
| Urban-1k 上的 R@5/R@10 与上游对齐性 | 上游只报 R@1，本轮的 R@1 与上游定义一致；R@5/R@10 为本项目补充，无上游数字可比 |
