# Phase 2.6 实测结果

固定 512 张验证图像，fp32 推理、TF32 关闭，正式指标均为 512D。
复用 reviewed Phase 2.5 residual arm 的五个 checkpoint，未重复完整训练。
主 caption 是固定的训练式随机句子前缀；detail ladder 为独立离线诊断。

Manifest SHA256：`01c1e189d7ed09c75c5eb0c9b96685c3985163d46bca3f39524b27d3ab4c40fc`。
五个阶段共用同一 manifest；z_base 每个阶段逐元素相同，Base 指标与当前 t 配对。
512 个 probe image ID 与实际训练索引区间的 image ID 重叠为 0。

## 固定 Probe

| Checkpoint | Base Pair Gap | Full Pair Gap | Said Pair Gap | Gain | Base L2M | Full L2M | Said L2M | Base RMG | Full RMG | Said RMG | Conditioning Margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| initial | 0.680187 | 0.680187 | 0.704193 | -0.024006 | 0.803368 | 0.803368 | 0.911545 | 0.563132 | 0.563132 | 0.656763 | 0.000172 |
| step100 | 0.672099 | 0.675416 | 0.734586 | -0.059170 | 0.716812 | 0.720326 | 0.869359 | 0.517937 | 0.514339 | 0.604870 | 0.008460 |
| step200 | 0.676821 | 0.658731 | 0.686012 | -0.027281 | 0.744502 | 0.737114 | 0.830675 | 0.518691 | 0.500439 | 0.546856 | 0.139702 |
| step400 | 0.677708 | 0.655026 | 0.672166 | -0.017140 | 0.741369 | 0.747357 | 0.811989 | 0.516905 | 0.497861 | 0.530365 | 0.169382 |
| final | 0.677719 | 0.654524 | 0.671848 | -0.017324 | 0.740892 | 0.749241 | 0.813317 | 0.516845 | 0.497508 | 0.530212 | 0.174632 |

## Final Caption Detail

| Detail | Base Gap | Full Gap | Said Gap | Gain | 平均使用 tokens | 截断数 / 512 |
|---|---:|---:|---:|---:|---:|---:|
| 第一句 | 0.690375 | 0.675125 | 0.662290 | 0.012835 | 19.96 | 0 |
| 25% | 0.681300 | 0.660039 | 0.672976 | -0.012937 | 52.67 | 0 |
| 50% | 0.673789 | 0.649928 | 0.673055 | -0.023127 | 92.71 | 0 |
| 75% | 0.673142 | 0.646274 | 0.673702 | -0.027429 | 137.39 | 0 |
| 完整文本 | 0.674604 | 0.647444 | 0.678769 | -0.031326 | 174.22 | 15 |

共 2,555 个唯一 detail caption；五条曲线各为同一组 512 张图像。
Full Gap 从第一句 0.675125 总体下降到完整文本 0.647444，但 75% → 100% 回升 0.001170，存在非单调。
Said 优势仅在最稀疏的第一句条件出现（Gain +0.012835），25% 及更详细条件均为负。
增益随 detail 降低，但不是从正优势逐渐收敛至零：它转为对 Said 不利。
完整文本中 15/512 被截断；不把文本长度等同真实语义覆盖率，也不作单个图像或总体显著性推断。

## Caption 条件化与总体结构

Final：Own Gap 0.671848，Shuffle Gap 0.846480，Conditioning Margin +0.174632。
条件化能力明显，但固定主 caption 的 Said Pair Gap 高于 Full 0.017324；
Said RMG 0.530212 也高于 Full 0.497508，Said L2M 0.813317 高于 Full 0.749241。
因此不能宣称主 probe 上的过滤改善了整体模态结构。
五个 checkpoint 的主 caption Gain 全部为负，不能只展示第一句的正结果。
Scientific Observation：**F（Mixed）**。固定主 caption 更接近 B，detail 的整体下降方向支持 D；
最稀疏条件有局部支持，但不足以选择一般性的 A。没有新损失或新算法来迎合假设。

## 在线 Batch smoke

4 × A800，256 样本/卡，10 步，bf16 autocast + fp32 master weights；seed=25。
rank 0 最后一条记录 step=9 / completed_steps=10 / epoch=0：
Full Pair Gap 0.661189；Said Pair Gap 0.734203；Gain -0.073014；Relative Gain -0.110429。
Route Top-1 0.015625；Route Margin -0.000456。全部新增日志字段为有限数。
这是独立启动 smoke，不与 Phase 2.5 final 或固定 probe 混合解释。

## 全局检索

复用同一 residual final checkpoint 的 Phase 2.5 标准 COCO 5K 评估；本阶段没有重新运行 COCO。
checkpoint SHA256：`d98135b6805ca6e9b80b90001cd40024b2749e8a56c24db6af46352747884532`。
I2T R@1/5/10：58.54% / 81.36% / 88.50%。
T2I R@1/5/10：40.172% / 65.904% / 75.532%。
其他四个 checkpoint 未导入检索结果，前端显示未运行提示。

## 验证

五个 checkpoint 的 standard global 接口与 probe 的 global 输出最大绝对差均为 0。
显式 512×512 模态内距离矩阵独立核验 Pair Gap、L2M、RMG，误差 < 1e-12。
五组 PCA 的四种表征均按保存的公共 basis / mean 重投影核验，共用坐标范围与固定前 30 条连接线。
缓存 Base 数组与五个 NPZ 逐元素相同；去重 aliases 的 embeddings 逐元素相同。
现存 26 个 SALU checkpoint（25 个历史文件 + 本轮 smoke）全部 strict load 成功，包含 metadata-free 旧格式和 attention_delta。
全套 pytest：103 passed、1 skipped、3 个既有 warnings；errors=0。
默认首页不需要 heatmap artifact；初始阶段、缺失 COCO、缺失诊断文件与四个历史页面均有 AppTest 覆盖。
浏览器核验 1920×1080、1366×768、616×900：页面横向溢出为 0，JavaScript page errors 为 0；
三图在宽屏并排、窄屏纵向排列，中文正文、图例与字体图标正常显示。

原始结果位于 `outputs/representation_balance/`，没有提交输出、权重、日志或图像到 Git。
实现与公式说明见 [phase26_representation_balance.md](phase26_representation_balance.md)。
等待 Review；不进入 Prototype / Unsaid。
