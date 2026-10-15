# Full Caption 主文图说明

- 数据集：DOCCI test，固定的 1000 个样本，仅 Full Caption。
- 输入：`../input/features.npz`、`../input/modality_gap_metrics.csv`；来自 GitHub 提交 `124e89574d6ec586f5eaf4f5ff63f9679b3c89ff`，没有训练或重新提特征。
- 展示映射：`longclip_full_*` → Long-CLIP；`ours_native_full_*` → Sema-CLIP (Global)；`ours_masked_full_*` → Sema-CLIP (Masked)。后两个仅修改图中名称，特征内容保持不变。
- 样本顺序：复用单一导出的 shared samples manifest；1000 个唯一 image IDs 与 caption_lengths.json 的顺序一致，Native/Masked 的 Full text 数组逐元素一致。NPZ 没有逐模型独立的 ID 数组，三组的对应关系依据共享导出 provenance；未重排或筛选。输入 hash 和 ID 序列 hash 见参数文件。
- 第一行：每个模型单独用 Full image+text 共 2000 个向量拟合 PCA，PC1–PC2，full SVD，seed=20261013。三列坐标范围一致、纵横等比例。所有点可见，s=12，alpha=0.38；线段仅连接投影 centroid。
- 第二行：每模型轴 `u=(μ_image−μ_text)/||μ_image−μ_text||`，投影为 `x·u`；没有平移分布。共同横坐标范围与 50 个等宽 bins，density=True；KDE 使用 Scott 带宽、1000 点网格。虚线为两个 projected centroid，双向箭头长度为 gap。纵轴密度分别自动缩放，比较间距应看共同的横轴，不应比较分布峰值高度。
- 标注：Gap 只显示在第二行；Gap 在原始归一化 512D 空间计算，并与源 CSV 校验，误差 <1e-7。由于 Full gap-axis 正是对应 centroid 差方向，该行投影 centroid gap 与 512D gap 数学上相等。PCA 连线长度不能作为完整 512D gap。Paired cosine、Paired L2、centroid norms 和 mask keep ratio 保存在配套 metrics CSV；原有 bootstrap CI 在 `../tables/metrics_with_bootstrap_ci.csv` 保留。
- 风格：白底浅灰网格，Image #2F5D8A，Text #D9822B，centroids #173A5E/#9A4E00；PNG 400 dpi，PDF 嵌入字体和矢量文字/KDE，散点栅格化。

| 表示 | Gap (512D) |
|---|---:|
| Long-CLIP | 1.048990 |
| Sema-CLIP (Global) | 0.778372 |
| Sema-CLIP (Masked) | 0.686359 |

该图适合作为双栏主文图。第二行是更直接的 gap 证据，第一行提供分布形态；PCA 为逐模型独立拟合，不能用跨列点的相对位置解释模型间的语义对应关系。Masked 减小 global centroid gap，但 Native 的 paired cosine 更高，应避免把 global gap 改善写成逐样本匹配也全面改善。
