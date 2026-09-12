# FineLIP-Prefix FP0：两 epoch 实验报告

依据FineLIP方法与公开技术说明独立实现的前缀监督基线。

代码独立实现，不复制或动态导入外部 FineLIP checkout。上游参考固定为 `2118312c9d640c71904379e90129649a46e6f2dd`；许可证未识别、直接复用授权未确认；UPSTREAM_RUNTIME_PARITY: NOT RUN。

实际训练：4×32 microbatch，128 候选，累积4；一组平均四个独立 microbatch 的 hinge SUM，不是512候选。数据样本数 1245901，每epoch 9734 microbatch、2434 updates；6epoch horizon=14604，本次在4868 updates结束。

保存 init/1000/epoch1/epoch2；按用户后续要求跳过1000步评估，仅评估init/epoch1/epoch2。严格导出317个学生 tensors，主评估只用原生 CLS/EOS。两次epoch尾部均实际累积2个microbatch、208对样本，已flush。

相对上游的明确区别：随机前k句、共同初始化与seed、bf16主干/FP32核心、重复image和有效前缀过滤、EOS边界（包含SOT和内容ID=0）、optimizer-step scheduler、每epoch shuffle、实际累积尾部、非有限值硬失败。

## 原生检索（百分数）

### COCO val2017：5000图/25000文本

| 模型 | I2T R@1/5/10 | T2I R@1/5/10 |
|---|---|---|
| Initial | 51.70 / 76.62 / 84.28 | 32.69 / 57.76 / 68.23 |
| S0@500 | 60.58 / 82.20 / 89.06 | 41.24 / 67.09 / 76.62 |
| T1_fix_v2@500 | 57.28 / 80.80 / 87.88 | 37.82 / 63.10 / 73.68 |
| AM@500 | 55.02 / 78.58 / 86.70 | 38.87 / 64.19 / 74.16 |
| FP0_update0 | 51.70 / 76.62 / 84.28 | 32.69 / 57.76 / 68.23 |
| FP0_update2434 | 56.78 / 79.90 / 87.50 | 39.86 / 65.73 / 75.57 |
| FP0_update4868 | 56.52 / 79.88 / 87.30 | 39.90 / 66.04 / 76.41 |

### Urban-1k：1000图/1000文本

| 模型 | I2T R@1/5/10 | T2I R@1/5/10 |
|---|---|---|
| Initial | 68.20 / 89.50 / 94.40 | 52.80 / 77.40 / 84.80 |
| S0@500 | 87.00 / 97.10 / 98.80 | 84.20 / 96.70 / 98.10 |
| T1_fix_v2@500 | 76.40 / 92.90 / 96.30 | 71.30 / 89.40 / 93.80 |
| AM@500 | 72.80 / 92.00 / 95.30 | 75.10 / 90.80 / 94.30 |
| FP0_update0 | 68.20 / 89.50 / 94.40 | 52.80 / 77.40 / 84.80 |
| FP0_update2434 | 82.50 / 95.70 / 98.10 | 80.40 / 94.50 / 96.90 |
| FP0_update4868 | 83.50 / 96.40 / 98.10 | 81.70 / 94.50 / 97.10 |

ShareGPT4V 固定1K 的 first_sentence/fixed_sparse/full_dense 各自结果完整保存在 results.json 的 sharegpt4v1k 字段，不混入 COCO 或 Urban-1k。

## 预算与性能

FP0@1000呈现512000对，checkpoint保留但按用户要求不评估。第一/第二epoch分别呈现1245904/2491808对，超过旧S0/T1/AM@500的512000对预算；常规候选池为128对1024，不能称等预算比较。FP0每epoch末的最后一个microbatch有80个全局样本；按实际大小处理。

复用的T1/AM同步训练计时分别折算为1.105/1.096 GPU小时，计时范围见JSON；S0原复用指标中无GPU耗时，标记NOT RECORDED。

两epoch呈现 2491808 对（含DistributedSampler每epoch为对齐rank补齐的3条重复），训练墙钟 2409.5 秒，累计四卡计算 2.406 GPU小时；峰值显存 7.84 GiB，日志更新耗时中位数 0.490 秒。

评估分阶段单卡墙钟记录在results.json的evaluation_timings。训练gpu_compute_hours为rank0同步计时乘四卡的估算；检查点时间戳累计耗时另在每个provenance记录，包含数据等待和保存开销。

## 固定64前缀诊断

| update | image scale | text scale | image slot cos | text slot cos | native I2T/T2I R1 | finegrain I2T/T2I R1 |
|---|---|---|---|---|---|---|
| 0 | 1.00000 | 1.00000 | 0.999624 | 0.998355 | 0.953 / 0.953 | 0.406 / 0.656 |
| 2434 | 1.05564 | 1.04672 | 0.999871 | 0.999269 | 0.984 / 0.984 | 1.000 / 0.984 |
| 4868 | 1.05653 | 1.04297 | 0.999952 | 0.999455 | 1.000 / 1.000 | 1.000 / 0.984 |

此固定训练cohort只用于解释局部与全局行为，不是独立验证集主指标，不能替代canonical。聚合权重熵和固定manifest另存。

## 来源与验收

6项必要测试通过，涵盖公式、负cosine、梯度符号/平移、EOS/重复样本、两rank真实累积与尾部AdamW更新、严格状态恢复。共享初始化GPU验收确认native接口一致、40-token集合和全主干backward。测试详情见 validation.json。

论文外部参考：[FineLIP, arXiv v1 表1](https://arxiv.org/html/2504.01916v1)：B/16 Urban-1k I2T=90.7/98.3/99.5，T2I=89.3/97.5/98.7。论文使用6epoch长caption训练；本次是2epoch随机前缀、项目共同初始化。仅列外部参考，不当成同框架对照；未用FineLIP*混合细粒度推理结果代替原生CLS。

实现SHA：`2dec367f7e2194365a504b690c7db6ebab502602`。各checkpoint与裸学生SHA见results.json provenance。

NOT RUN：1000步评估（按用户后续要求省略）；官方运行时数值对照；完整正式worker恢复续训（已有恢复单测）；六epoch及任何新变体。未证明Said/Unsaid语义分离。完成本次两epoch与评估后STOP。

## 结果判断

1. 原生检索总体上升，但不是单调：COCO 从初始化 51.70/32.69 提升到 epoch1 56.78/39.86，epoch2 为 56.52/39.90（I2T/T2I R@1）；Urban-1k 从 68.20/52.80 提升到 82.50/80.40，再到 83.50/81.70。epoch2 对 COCO 轻微回落，对 Urban-1k 继续改善，属于数据集混合趋势，不能据此外推饱和点。

2. 固定 cohort 上两侧 slot 平均余弦均升高（图像 0.999624→0.999952，文本 0.998355→0.999455），表明 slot 差异能量下降；本轮诊断未计算有效秩，因此不能声称“差异能量和有效秩同时改善”。聚合权重熵、slot 统计和原始 JSON 一并保留。

3. FP0 没有 router 或 learned top-k 选择，router caption 依赖性与随机选择比较均为不适用；fine-grained 局部分数在固定 cohort 上提高，但不能替代 canonical。

4. 改善传递到原生 CLS/EOS：COCO 和 Urban-1k 均高于共同初始化，尤其 Urban-1k；仍低于复用的 S0@500，且训练预算和候选池不同，不能做等预算结论。

5. 这组证据支持“前缀监督确实学到检索改善”，但 COCO 在第二 epoch 已出现平台迹象、Urban-1k仍在上升。是否继续需要新的有限预算决策；本轮按授权在第二 epoch 停止，不自动延长。
