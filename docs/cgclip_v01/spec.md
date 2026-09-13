# CG-CLIP v0.1 — 规格说明（Caption-Gated Final-CLS Attention）

**臂标识（arm）**：`CG_CLIP_V01`
**目标名（objective）**：`clip_native_caption_gated_cls`
**阶段**：`cgclip-v0.1`
**门控类型**：`caption_gated_final_cls_attention`
**实验口号**：原生对齐 ＋ 文本门控的最后层 CLS 注意力对齐
**代码基线**：`codex/cgclip-v01`，由 PG-CLIP 分支尖端 `6887793` 建立工作树 `/root/SAID-cgclip-v01`

---

## 1. 动机与允许/禁止清单

本臂要问的问题是：**在不改变原生编码器结构、不新增第二套 Value 来源、不动投影层的前提下，让文本只在最后一个视觉块里"选择 CLS 该读哪些 patch"，能否在 500 步内形成可测量的检索改动？**

允许：
- 保留全部原生结构：12 层视觉 Transformer、`ln_pre`、`ln_post`、`visual.proj`、文本 Transformer 与 `text_projection` 全部原样；
- 新增一个极小的门控网络（64 维键空间），只作用于**最后一个视觉块**的 CLS 注意力权重；
- 两条路径（原生、条件）共享同一份参数：同一个 CLS Query、同一套 Key/Value、同一个 `out_proj`、同一个残差、同一个 `ln_2`/MLP、同一个 `ln_post`、同一个 `visual.proj`。

明确禁止（本臂的"不做"清单，代码中亦不提供相应开关）：
- 文本 Query 替换原生的 CLS Query；
- 把文本向量或文本 bias 加到视觉输出上；
- 第二套 Value 来源（例如再用文本生成一份 V）；
- 复制最后一块或复制投影层；
- PG-CLIP / SmartCLIP 式的投影后通道掩码（本臂与 `pgclip` 是**不同**的实验，代码不共享门控实现）；
- 文本掩码、`U` 重建、教师模型、第三条路径。

## 2. 记号与计算图

设进入第 12 个（最后一个）视觉块的 token 序列

```
X11 = [CLS11, patch11_1 … patch11_196]        # [B, 197, 768]，由一次视觉前向（块 0..10）给出
```

**原生路径（native）**

```
c   = X11[:,0]
U   = last.ln_1(X11)
q_cls = Q(U[:,0]);  K = K(U);  V = V(U)
a[h,p] = softmax_p( q_cls[h]·K[h,p] / sqrt(64) )
oG  = out_proj( concat_h Σ_p a[h,p] V[h,p] )
cG  = c + oG + mlp( ln_2( c + oG ) )
vG  = Norm( ln_post(cG) @ W )
```

**条件路径（conditional，仅训练期存在）**

```
t      = Norm( EOT(ln_final(text)) @ text_projection )      # 512 维，fp32
qT_j   = A( stop_grad(t_j) )                                # 512 → 64，无 bias，零初始化
kG_i,p = B( stop_grad(U_i,p) )                              # 768 → 64，无 bias，xavier
logits = qT_j · kG_i,p / sqrt(64) + b                       # b 是可训练标量，初始 log(8)
p      = sigmoid(logits)
hard   = (p >= 0.5)
m      = hard + (p - p.detach())                            # 196 个 patch 门，直通估计
m_full = [1, m]                                             # CLS 槽固定 1
a_cond = (a * m_full) / Σ_p (a * m_full)                     # 无 detach、无 clamp
oA     = out_proj( concat_h Σ_p a_cond[h,p] V[h,p] )         # 同一个 out_proj
cA     = c + oA + mlp( ln_2( c + oA ) )                      # 同一组残差 / ln_2 / MLP
vA     = Norm( ln_post(cA) @ W )                             # 同一个 ln_post / W
```

要点：
- 一次 `m_full` 由 196 个 patch 门与 1 个固定 CLS 门组成；**CLS 门永远为 1**，不参与稀疏项，也不画进 14×14 栅格；
- 门的**唯一**作用对象是 CLS 的注意力权重；文本因此只改变"CLS 读哪些 Value"；
- `a_cond` 的求和**不做 detach、不做 clamp**：分母在计算图内，梯度同时流经分子与分母；
- 门的两路输入都 `stop_grad`：门不会把梯度推回文本主干或 `U`；文本主干仍通过最终相似度训练，视觉主路径的 `U/Q/K/V/c` 仍然活跃。

## 3. 目标函数

```
L_total = 5 · LG + 5 · LA + 1 · LS

LG = CE(QG, y) + CE(QGᵀ, y)          QG[i,j] = 100 · <vG_i, t_j>
LA = CE(QA, y) + CE(QAᵀ, y)          QA[i,j] = 100 · <vA_i,j, t_j>
LS = mean( 196 个 patch 门，仅真实正样本对 )      # 不含 CLS 槽
```

- 相似度一律乘 **固定尺度 100**（不使用 `logit_scale`；该参数保留仅为状态兼容并在优化器分组中冻结）；
- 正样本与负样本使用**完全相同**的规则（同一条 100 尺度、同一个 pair 门）；
- `LS` 的正样本对来自**真实配对**（不是 `abs`、不是 `mean(p)`、不是 hard 值的平均值），且只统计 patch 门。

**LSE margin 的定义（必须严格区分）**

```
LSE_margin = positive − logsumexp(仅负样本)
CE         = softplus(−LSE_margin)
```

`logsumexp(全体) − positive` 是交叉熵，**不得**被称为 LSE margin。每个方向都记录 `CE` 与
`softplus(−LSE_margin)` 的最大绝对差作为自检字段（`ce_from_lse_margin_max_abs_diff`）。

## 4. 分布式的正样本列

- DDP 下每个 rank 的本地批大小为 `B_local`；
- 本地第 `i` 张图的**全局正样本列** = `rank · B_local + i`（全局候选集是全部 rank 的文本）；
- I2T 用本地图片行做锚点、全局文本做候选；T2I 用本 rank 的文本做锚点、全局图片做候选；
- 稀疏项取**正样本那一列**的门，且直接从已经算过的 tile 里切片，**不第二次跑门控网络**；
- 梯度路线是成熟路线：逐 rank 锚点均值 ＋ 自动微分感知的 gather ＋ 标准 DDP 参数梯度平均，**任何地方都不乘 world_size 系数**。

## 5. 精度与分块

- 主权重 fp32；
- bf16 autocast 只覆盖**前 11 个视觉块**与**文本 Transformer**；
- 最后一个视觉块的 CLS 核心（`ln_1`、Q/K/V、原生 softmax、门控、重归一化、`out_proj`、两个残差、`ln_2`/MLP、`ln_post`、`visual.proj`）、文本投影、归一化、相似度与全部损失**显式 fp32**；
- 训练脚本显式关闭 `torch.backends.cuda.matmul.allow_tf32`，并把 TF32 的真实取值写进 config 与日志；
- 条件路径按 `(image_chunk × text_chunk)` 分块，使用 **non-reentrant** 激活检查点，闭包不捕获任何循环变量；
- 任何时刻都不物化 `[B,B,12,197]`、`[B,B,768]`、`[B,B,3072]` 这类中间量。

**可调项**：只允许调整分块大小与检查点开关。批大小、池化方式、权重、门控类型均**固定**。

## 6. 门控的初始化与不变量

| 量 | 取值 | 后果 |
| --- | --- | --- |
| `A.weight` | 全 0（无 bias） | `qT = 0` ⇒ `logits = b` |
| `B.weight` | xavier_uniform（无 bias） | 第一步 `B` 梯度为 0，`A` 一旦移动即恢复 |
| `b` | 可训练标量，初始 `log 8` | `p = 8/9 ≥ 0.5` ⇒ 196 个门**全开** |
| 随机种子 | `gate_seed = 0`，且在 `isolated_rng` 内构造 | 不动全局 Python/NumPy/CPU/CUDA 随机流 |
| CLS 槽 | 固定 1 | 不训练、不进稀疏项 |

初始化时刻的精确结论：
- `a_cond ≡ a`，`QA ≡ QG`（数值上到浮点误差），`LS = 1`；
- `A` 的梯度非零（`logits` 对 `A` 线性，即使 `qT = 0`），`b` 的梯度非零，`B` 的梯度**恰好为 0**；
- 视觉/文本共享参数的梯度满足：`∇(5·LG + 5·LA) = 10·∇LG`（到浮点误差）。

注意：门"全开"只是**初始化**的事实。训练会移动 `A`/`b`，门会关掉一部分 patch；"条件路径在推理期不用"**并不**说明共享主干在训练中没有被改变，本实验也不声称原生能力被保持。

## 7. 检查点

检查点必须自描述，且**键不得冲突**：

| 键 | 内容 |
| --- | --- |
| `clip` | 共享 CLIP 状态字典 |
| `gate_state` | 门的**张量**（`query.weight`、`key.weight`、`bias`） |
| `gate_config` | 门的**文字描述**（与张量分离，永不覆盖） |
| `gate_state_digest` / `clip_state_digest` | 真实状态字典的 sha256 摘要 |
| `optimizer_clip` / `optimizer_gate` | 两个优化器的完整状态 |
| `completed_steps` / `epoch` / `step_in_epoch` | 步数游标（已完成步数，不是步数+1） |
| `lr_horizon_steps`、`loss_weights`、`fixed_scale`、`norm_eps` | 目标函数身份 |
| `attention_route`、`caption_scope`、`visual_spec`、`chunking`、`precision` | 结构与精度来源 |
| `digests`（含逐 rank 流摘要）、`rng`、`grad_health` | 可追溯性与可复现性 |

写入是**原子**的（先写 `.tmp` 再 `os.replace`）。恢复时用 `check_resume_compatible` 拒绝跨目标/跨门控类型/跨系数的静默续训。

## 8. 单次实验配置（唯一配置）

| 项 | 值 |
| --- | --- |
| 初始状态 | `/root/SAID-gap-completion/runs_salu/said_cls_cvssl/shared_init/cvssl_initial.pt`（sha256 `c1a4a2be…`），**唯一**起点 |
| GPU | 4 × A800-80GB，`torch.distributed.run --nproc_per_node=4` |
| 每卡批 | 256（全局 1024） |
| 优化器 | AdamW ×2：CLIP `lr 1e-6` / `wd 1e-2` / warmup 200；门控 `lr 1e-3` / `wd 0` / warmup 0；`betas (0.9,0.999)`，`eps 1e-8` |
| 调度 | cosine，horizon = `3 × len(loader)`（真实 loader，不写死 3651） |
| 步数 | **恰好 500 次优化器更新**（`--max_steps 500` 只做截断） |
| 存档 | `0, 20, 100, 250, 500` |
| AMP | bf16（前 11 块 + 文本 Transformer） |
| 分块 | `image_chunk = 32`，`text_chunk = 64`，non-reentrant 激活检查点开（分块是规格里唯一允许调整的旋钮：正式开跑前在 4×A800 上实测 `8×32` 为 6.3 s/step、`32×64` 为 2.0 s/step，显存同为 40.7 GiB，损失一致到 1e-3 量级的浮点求和顺序差异，故采用后者） |
| 视图 | 只用 `image_a`（参考 openai-clip `_transform(224)`），不加第二个视图、不加新增强任务 |
| 字幕流 | 参考前缀抽法 `caption.split(". ")[:k]`，`k ~ randint(1, n)` |

**不做**：不扫权重、不扩展 epoch、不自动补其他训练对照、不追加 seed。

## 9. 验收测试（`tests/test_cgclip.py`）

| 组 | 检验 |
| --- | --- |
| A | 原生 CLS 行与**未经改动的原生块**（以及 `encode_image` 端到端）在数值与梯度上都一致；12 头 × 64 维、文本 8 头不混淆 |
| B | 初始化门全开：`a_cond ≡ a`、`QA ≡ QG`、`LS = 1`；`∇(5LG+5LA) = 10∇LG`；`A`/`b` 梯度非零、`B` 梯度恰好为 0 |
| C | 分块（含激活检查点）与逐对循环在分值与梯度上一致；正样本门取自全局正样本列，且不重复评估门控 |
| D | 重归一化**等于**独立的"掩码 logits + softmax"定义；四个具名错误变体（不重归一化的 Value 掩码、用软概率代替直通硬门、pair 索引转置、文本 Query 替换 CLS Query）都被测出显著差异 |
| E | 直通梯度同时流经分子与分母：门 bias 的解析梯度与有限差分一致（评价点选在阈值之外） |
| F | 极值：196 门全关 ⇒ 只剩 CLS 自读、分母有限且为正、分值有限；注意力出现非有限/非正值时**报错拒绝**，不 clamp |
| G | LSE margin 定义、`CE = softplus(−LSE margin)`、权重 5/5/1、稀疏项不含 CLS 槽、门初始化与 RNG 隔离 |
| H | 优化器分组覆盖每一个可训练参数且恰好一次，最后一块与 `visual.proj` 在且仅在一组，`mask_net`/`logit_scale` 冻结 |
| I | 用**生产写检查点函数**做真实保存/恢复：非初始门 ＋ 优化器状态 ＋ 步数游标；摘要可复算；篡改/缺失可检测；跨目标/跨门控恢复被拒绝 |
| J | CLI 面：训练脚本参数与默认值；runner 无扫描/补跑/`allow-missing` 开关 |
| K | 真实两 rank DDP 单步梯度与单进程全批参考一致（混合门，覆盖 `A`/`B`/`b` 三组梯度） |

## 10. Runner 阶段

`tools/cgclip_runner.py` 固定六个阶段，逐阶段真实退出码：

`train → verify → export → coco → urban → report`

- 启动前确认 4 张 GPU 空闲（`nvidia-smi` 失败或可见卡数不足即**拒绝启动**）；
- 原子锁、PID/时间戳/argv/git HEAD 记录；
- `setsid` 分离运行，stdout/stderr 落到 `logs/` 下，SSH 断开不影响训练；
- `verify` 阶段逐项校验 step-500 检查点（`completed_steps == 500`、`gate_state` 非空且形状匹配、`gate_config` 是描述、两个优化器状态真实、摘要可复算、日志里有 step 500 记录）；**没有** `allow_missing` 类旁路；
- 任一阶段失败即停止流水线并保留现场，不静默继续；
- 只有通过校验的检查点才能续训。

## 11. 日志字段（前端只读这些字段）

两条路径各自记录：I2T/T2I 分值、`LG`/`LA` 之和、`5·LG`、`5·LA`、`LS`、`L_total`，以及
正样本均值、最强负样本均值、最大 margin（均值/最小）、LSE margin（均值/最小）、`CE`、
`top1`、正样本获胜比例。

patch 门统计：保留个数（均值/最小/最大）、保留比例、全关/全开比例、软概率分布（均值/标准差/
最小/最大/近阈值比例/5 个分位）、同一文本跨图像与同一图像跨文本的门变化（固定小样本）。

CLS 读数诊断：原生与条件的 CLS 自质量、patch 质量、逐头自质量、`‖attention_output‖/‖CLS11‖`、
条件与原生 CLS 方向的余弦、条件与原生 512 维余弦。

呈现：196 个 patch 门的 14×14 栅格（**CLS 自读单独呈现**，仅显示层面按头平均）；
`rank0-local` 与 `global` 两种口径都标注清楚；逐 rank 流摘要；全局样本呈现数 = 500 × 1024 =
512,000（128,000 是单 rank 口径）；同时给出"同步步时间"与"512000/总墙钟"；内存单位明确写
GB / GiB。

## 12. 判定规则（500 步冻结后）

- COCO：I2T R@1 ≥ **0.6058** 且 T2I R@1 ≥ **0.41236**，且至少一项**严格大于** ⇒ `PROMISING_AT_500`，否则 `FAIL`；
- Urban-1k 单独报告（不与 COCO 混合成单一结论）；
- 一律使用**未取整**的分值与命中数比较；
- 0.10 pp 的 COCO I2T 差额 = **5 条查询**（每条 1/5000）；0.02 pp 的 COCO T2I 差额 = **5 条查询**（每条 1/25000）；0.02 pp 的姊妹 run 差额 = **1 条 I2T 查询**。表述必须按查询条数写清楚，不得写成"只有一张图"。
