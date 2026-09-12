# FineLIP-Prefix FP0

依据 FineLIP 方法与公开技术说明独立实现的前缀监督基线。学术方法与版本参考：https://github.com/tiiuae/FineLIP/tree/2118312c9d640c71904379e90129649a46e6f2dd 。上游来源已在 upstream_audit.md 核查；本实现不复制、动态导入或依赖外部 FineLIP 源码。

LICENSE_NOT_IDENTIFIED / REUSE_PERMISSION_UNCONFIRMED；UPSTREAM_RUNTIME_PARITY: NOT RUN。数学验收采用独立循环公式参考，不称为官方源码数值对照或官方训练轨迹复现。既有 CLIP、数据门禁、native retrieval 工具沿用 SAID 的来源记录。

结构：共同初始化 ViT-B/16、248-token 文本，双侧 LN/Linear(512,102)/GELU/Linear(102,39)，各自可学习 scale=1。视觉 CLS 与文本 EOS 独立保留；文本聚合包含 SOT 和 EOS 前内容。输出 token 归一化后，cosine 经 LeakyReLU(0.1)，两个方向 max-mean 相加。合法有序负例的双向 hinge SUM 为反传目标。没有 router、mask、U、reference、decoder 或额外 global loss。

明确区别：随机前 k 句、共同初始化与 seed、FP32 核心/bf16 主干、重复 image/有效前缀过滤、按 EOS 而非 nonzero 计数、每 epoch shuffle、按 optimizer update 的 scheduler、实际大小的累积尾部、非有限值硬失败。

正式配置：4x32 microbatch，128 候选，每四个 microbatch 的 SUM 目标平均后更新；不是 512 候选。AdamW 两组 lr=1e-6/2e-4，wd=.01，betas=.9/.999，eps=1e-8。warmup=200 updates；动态六 epoch LR horizon；运行两 epoch。epoch0 每个 micro 的 alpha=i/L。图像行分片反传 W*alpha*H_local/group_size。无 clipping。

数据只输出一张标准 image_a 和随机前缀；未选后缀不返回模型。每 rank 每批保存 sample_id/image_id/k/token_length 与 caption hash。DataLoader 使用明确 rank/epoch generator，恢复重放至实际游标并核对已有 caption hash；不声称与旧模型流逐位一致。checkpoint 只在更新边界保存，全状态严格恢复；epoch 尾部不会流入下一 epoch。

验证：聚合公式/可学习 scale、负 cosine、ranking 符号/公共平移/空集合、EOS/ID=0/重复前缀、累积尾部/恢复游标、两 rank 的真实 no_sync 累积梯度和 AdamW 更新。分布式更新参考使用 FP64，避免 softmax 平移不变的末层 bias 的近零舍入梯度被 AdamW eps 放大；没有扩大容差。生产核心仍 FP32。实际共享初始化 GPU 验收另核对 native CLS/EOS、40-token 形状与全主干 backward。

运行与评估：tools/run_finelip_prefix.sh 新目录；先检查资源，再运行唯一两 epoch 配置，最后只评估 init/1000/epoch1/epoch2 的裸学生 CLS/EOS。COCO、Urban-1k 和现成 ShareGPT4V 固定1K结果分别命名。完整结果与预算在运行完成后写入报告。不自动延长六 epoch；UNSAID_SEMANTICS: NOT ESTABLISHED。
