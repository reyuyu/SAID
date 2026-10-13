# DOCCI Modality Gap 双簇异常诊断报告

## 方法与审计边界

本次诊断基于已冻结的 1,000 对 DOCCI 样本、同一 Long-CLIP 官方权重和 HNS-v1 B16 Step3651 完整 checkpoint。没有训练、没有创建 optimizer、没有写入 checkpoint，也没有修改生产模型。三种 caption 设置为 Full、50% BPE prefix 和 First Sentence。逐样本补充推理直接调用现有 `BalancedStack` 的 `pair_logits(..., paired=True)`，然后使用原实现的 sigmoid、`hard_st` 和 `normalize(z * mask)`。

数据审计通过：样本 ID 唯一（1000/1000），v1 与 v2 样本顺序、图片路径和原始 caption 完全一致，caption 无重复；三种设置的特征均为 1000×512、有限且 L2 范数为 1（误差约 1e-7）。逐样本 mask 二进制向量也没有重复模板：每个设置 1000 个 unique mask row，最大重复比例 0.1%。

## 1. 双簇是否存在于 512 维空间

对原始 512 维归一化图像特征做 KMeans k=2（50 次初始化）并进行不同随机种子稳定性检查：

| Features | Major / minor | 512-D silhouette | 512-D centroid distance | KMeans seed ARI min–mean |
|---|---:|---:|---:|---:|
| Ours masked — Full | 800 / 200 | 0.1803 | 0.6325 | 1.000 / 1.000 |
| Ours masked — 50% | 879 / 121 | 0.1947 | 0.6233 | 1.000 / 1.000 |
| Ours masked — First sentence | 560 / 440 | 0.0425 | 0.2996 | 0.675 / 0.891 |
| Ours native | 631 / 369 | 0.0448 | 0.3144 | 0.984 / 0.992 |
| Long-CLIP native | 603 / 397 | 0.0405 | 0.2411 | 0.933 / 0.955 |

因此，Full 和 50% 的双簇不是纯粹由二维 PCA 制造的：在原始 512 维空间仍有中等 silhouette、较大的质心距离，而且 KMeans 结果稳定。First Sentence 的二分结构很弱且不稳定，说明图上看到的“第二簇减少”主要是两簇间距离收缩与边界重排，而不是一个固定样本群简单消失。

## 2. Caption 设置之间的样本迁移

下表使用每个设置自己的多数簇/少数簇标签，避免 KMeans 原始 0/1 标签置换影响。矩阵行是 source（major, minor），列是 target（major, minor）。

| Source → target | 迁移矩阵 | ARI | 少数簇 Jaccard |
|---|---|---:|---:|
| Full → 50% | `[[781, 19], [98, 102]]` | 0.479 | 0.466 |
| Full → First sentence | `[[474, 326], [86, 114]]` | 0.025 | 0.217 |
| 50% → First sentence | `[[509, 370], [51, 70]]` | 0.016 | 0.143 |

Full 的 200 个少数簇样本中只有 114 个仍在 First Sentence 的少数簇；同时有 326 个原 Full 多数簇样本转入 First Sentence 少数簇。这个迁移模式不支持“同一异常样本簇只是被压缩”的解释，而更像是 caption-conditioned mask 改变后，样本级条件边界整体重排。

## 3. Mask 机制统计

| Setting / cluster | Count | Keep ratio | Selected channels | Logit std | Saturated fraction | Paired cosine |
|---|---:|---:|---:|---:|---:|---:|
| Full major | 800 | 0.6121 | 313.39 | 0.2852 | 0.0000% | 0.3329 |
| Full minor | 200 | 0.6096 | 312.09 | 0.3429 | 0.0010% | 0.3309 |
| 50% major | 879 | 0.5999 | 307.12 | 0.3164 | 0.0056% | 0.3130 |
| 50% minor | 121 | 0.5843 | 299.16 | 0.3802 | 0.0097% | 0.3084 |
| First sentence major | 560 | 0.6208 | 317.88 | 0.5169 | 0.1936% | 0.2867 |
| First sentence minor | 440 | 0.5923 | 303.28 | 0.4153 | 0.0692% | 0.3040 |

Full 的两个簇 keep ratio 几乎相同（0.6121 vs 0.6096），所以 Full 双簇不是由明显不同的稀疏度造成。50% 有较小的 keep ratio 差异，但仍不足以解释完整特征分离。mask channel centroid 的平均 L1 差异为 0.0719 / 0.0783 / 0.0802，二值 channel disagreement 为 12.1% / 14.5% / 14.3%；并没有发现固定通道模板、全开、全关或近乎重复的 mask。

mask 本身、概率、logit 和 gate 的二分聚类与图像特征聚类的 ARI 如下：

| Setting | Binary mask ARI | Probability ARI | Logit ARI | Gate ARI |
|---|---:|---:|---:|---:|
| Full | 0.017 | 0.240 | 0.328 | 0.075 |
| 50% | 0.001 | 0.238 | 0.249 | 0.044 |
| First sentence | 0.282 | 0.104 | 0.053 | 0.074 |

Full/50% 的 binary mask ARI 极低，说明“mask 选了哪几个通道”本身不是两个图像簇的充分解释。Full/50% 少数簇的 logit 标准差和 text/visual condition 分量范数较高，且 sigmoid 基本没有饱和；这支持一个较谨慎的机制假设：大量接近 Hard-ST 阈值的通道经过 caption-conditioned 的连续 logit 改变后，产生不同的 mask×image 特征位移。该因果链与当前观测一致，但仅凭本次无梯度推理不能证明因果。

## 4. Native 对照

Ours native 和 Long-CLIP 的 512 维 silhouette 分别约 0.0448 和 0.0405，远低于 Ours masked Full/50% 的 0.1803/0.1947。Independent KMeans 与 Ours masked 的 ARI 为：Full -0.003、50% 0.000、First Sentence 0.255。换言之，Full/50% 的明显二分主要在 mask 后出现，而不是原生 image feature 已经存在的同一分群。

Mask 对 native image 的改变也集中在 Full/50% 的少数簇：

| Setting / cluster | Masked–native L2 | Masked–native cosine |
|---|---:|---:|
| Full major / minor | 0.539 / 0.796 | 0.853 / 0.682 |
| 50% major / minor | 0.518 / 0.788 | 0.864 / 0.688 |
| First sentence major / minor | 0.477 / 0.512 | 0.883 / 0.867 |

这解释了为什么 First Sentence 的图像簇明显合并：mask 后特征仍然更接近 native，并且两个 cluster 的 mask-induced displacement 差异大幅缩小。

## 5. PCA 检查

| PCA fit | PC1 | PC2 |
|---|---:|---:|
| Original shared PCA（Long-CLIP + Ours masked，image/text） | 23.14% | 9.64% |
| Ours masked-only PCA（3 settings，image/text） | 16.12% | 3.41% |
| Ours masked + native shared PCA | 16.97% | 3.38% |

使用 512-D KMeans 标签回看二维投影时，Ours masked Full/50% 的二维 silhouette 为 0.677/0.611，而对应 512-D silhouette 只有 0.180/0.195；First Sentence 为 0.326 对 0.043。PCA 明显放大了可视分离，因此图上的距离不能当作原空间距离。但原空间的 Full/50% 结构仍存在，不能简单归因于 PCA 假象。

## 6. 代表样本与数据审计

`representative_samples.json` 保存了每个设置两个 cluster 中距 512-D 质心最近的 10 个样本、ID、路径、原始 caption、第一句、keep ratio 和 paired cosine。代表样本涵盖建筑、消防栓、云、树、动物、地面纹理等多种内容，没有出现一个明显单一语义类别解释所有少数簇成员。Full/50% 的 cluster 与 caption 长度只有弱关系：Full 两簇完整 caption 平均长度约 135.1 vs 138.3；50% 少数簇较长（77.1 vs 67.6 BPE），但 First Sentence 两簇第一句长度几乎相同（23.2 vs 22.6），而分离已经很弱。

未发现样本重复、顺序错位、Batch 拼接错位、非有限值、全零特征或 checkpoint 哈希错误。诊断代码记录了模型和数据哈希，且没有改动 ABCDE 或旧分析结果。

## 最终判断

1. **双簇在 Full/50% 的原始 512 维空间有真实但中等强度的结构；PCA 进一步夸大了视觉分离。** First Sentence 的结构显著减弱且 KMeans 稳定性下降。
2. **Mask 参与了双簇形成，但证据不支持“异常稀疏或固定 mask 模板”是原因。** Full 两簇 keep ratio 几乎相同，binary-mask 与 feature-cluster ARI 很低，也没有饱和或重复模板。
3. **更合理的解释是 caption-conditioned BalancedStack/Hard-ST 产生了样本依赖的 mask-induced feature displacement。** Full/50% 少数簇距 native 更远、logit 与条件分量波动更大；First Sentence 时这种差异收缩，簇合并。
4. **没有发现已验证的数据或实现 Bug。** 仍不能排除训练 checkpoint 本身学到的某种数据分布效应；这需要跨 checkpoint、更多样本或连续 soft-mask/反事实实验才能进一步区分。
5. **该现象不否定 Modality Gap Reduction 的量化结果，但限制了 PCA 图的解释。** centroid gap 和 PCA 可视化必须结合原始 512-D 距离、native 对照和样本迁移；不能把“簇更近”单独当作语义对齐改善，也不能把二维图的簇大小当作统计显著性。
