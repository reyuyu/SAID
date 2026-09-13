# S0-Suffix v0.1 规格（Prefix-Conditioned Suffix Readout / 前缀条件下的后缀视觉读出）

**objective**：`said_prefix_suffix_v01`　**两个固定臂**：`S0_SUFFIX_NATIVE`、`S0_SUFFIX_MASK`
**工作树**：`/root/SAID-s0-suffix-v01`　**分支**：`codex/s0-suffix-v01`　**base SHA**：`5676666`（S0/CV-SSL 血统，已存在的工作树，无同名实验、无未提交修改）
**共同初始化**：`/root/SAID-gap-completion/runs_salu/said_cls_cvssl/shared_init/cvssl_initial.pt`
sha256 `c1a4a2be1b212f38677f729a4f03160d788a6d6db79ca5021a43022af0f5cba8`，317 张量，含 `mask_net.*`（S0 视觉掩码参数）、`logit_scale`、248-token 位置编码。两臂都从这里开始，**不得**从 S0@500 或任何微调检查点续训。

不改写旧 S0 数学：`L_S0 = 10·(L_SIDM + L_DISM) + 2·L_sparse_S`，`mS_j = S0Mask(H(P_j).detach())`，硬直通（sigmoid + Hard-ST）、候选 detach / anchor live、双向相加不乘 0.5、固定 100 倍尺度、逐 rank anchor 均值 + 可导 gather + 标准 DDP 平均（无 world_size 因子）。公共初始化中原 `mask_net` 参数是 S0 门的起点，**不做全开初始化、不冻结**。

## 2. 数据：前缀不变，新增合法训练后缀

仍用原 ShareGPT4V 清单、标准 `image_a`、seed=0。前缀**逐字符串**沿用既有采样：`caption = caption.replace('\n',' ')`；`sentences = caption.split('. ')`；`K = rng.randint(1, len(sentences))`；`P = '. '.join(sentences[:K])`。**不允许**为保证后缀非空而改 K 的分布、换分句器、删空片段后重抽、或重抽到后缀非空。

新增后缀 `R`（**只用于新的后缀监督**，不改变 P，也不从任何路径永久删除最后一句）：
```
last = 原 sentences 中最后一个非空片段的下标
R = '. '.join(s for s in sentences[K:last] if s 非空)        # 确定性跳过空片段
suffix_valid = (R 非空 且 tokenize 后有有效内容 token)
```
例子（binding）：`[A,B,C,D]` K=1→R=B+C；K=2→R=C；K=3→R 空；K=4→R 空；`[A,B,C,D,""]` K=2→R=C（不得把 D 收进来）。空后缀**不补位、不换图、不重抽**，该样本照常参加 S0 路，只是不参加后缀路。P 与 R **分别 tokenize、分别过共享文本编码器**（禁止编码整条 caption 再切 hidden）。248 截断策略不变；EOT 用项目真实约定（`argmax(token ids)`），不用 `token_id != 0` 计数。两臂共享图片顺序、K、P、R、valid 标记与增强策略；**逐 rank** 记录 sample/prefix/suffix/valid 流摘要（不能只记 rank0 就宣称全局一致）。

## 3. 原 S0 目标完整保留

`L_S0` 按既有实现（`model/said_cls_cvssl.py` 的 S0 路径）计算，包含两个方向与原有稀疏项，使用完整 global batch；**无效后缀样本不得从 S0 候选池移除**。`lambda_suffix=0` 时，S0 loss 与旧 mask／共享参数梯度必须与直接调用原 helper 一致（测试 E），且新模块初始化不得污染原数据随机流。

## 4. 新掩码 F（完整图像 + 前缀-masked 图像）

- 原生图像只编码一次：`g = Norm(clip.encode_image(I))`；`Norm(x)=F.normalize(x.float(), -1, eps=1e-6)`；`g` 是当前学生输出（视觉主干继续训练，不是冻结 reference）。
- `rS = g_det * stop_grad(mS_j)`（**不再归一化**；不得用 `Norm(g*mS)` 替代，也不得把两个已归一化向量之差称为纯 Unsaid）。
- F 输入：`xU = stop_grad(concat[g, rS])`，形状 `[1024]`。F 固定为 `Linear(1024,512,bias)` → GELU → `Linear(512,512,bias)`；第一层 Xavier uniform、bias=0；最后一层 weight=0、bias=log(8)；seed=0，构造在**完整隔离且可恢复**的 RNG 上下文（CPU+所有已初始化 CUDA 生成器）。
- 门：`pU = sigmoid(F(xU))`；`hardU = (pU>=0.5)`；`mU = hardU + (pU - pU.detach())`。初始 `pU=8/9`、`mU` 严格全 1。不 top-k、不固定保留数、不设 soft 下界。
- 读出：`u = Norm(g * mU)`（**乘完整 g**，不是 `g*(1-mS)`；不强制 `mU=1-mS`；不施加正交/互斥/低相似约束，允许两份 mask 重叠）。
- F 只能看：完整图像表示 `g`、由**候选前缀**产生的 masked 表示 `rS`。**禁止**读 `R_j`、`H(R_j)`、`tR_j`、完整 caption hidden、其他未列入 P 的文本、正负配对标签。

## 5. 两臂评分与 loss

候选协议：**第 j 列对应 (P_j,R_j)**；任意图像 i 与 R_j 比较都使用 **P_j** 生成的 `mS_j`，再由 `[g_i, g_i*mS_j]` 生成 `mU_i,j`。禁止"用图片自己的正确前缀 P_i 生成 u_i 后整行与所有 R_j 比较"这种等价替换。

- 后缀文本独立编码：`tR_j = Norm(clip.encode_text(R_j))`（单独 forward，不混入完整 caption 上下文）。
- **A（native）**：`QN[i,j] = 100·dot(g_i,tR_j)`；`L_suffix_native = CE(QN,y)+CE(QN.T,y)`；`L = L_S0 + 1.0·L_suffix_native`。无新掩码网络参与评分或训练（`suffix_mask_state` 必须为 `None`）。
- **B（mask）**：`QU[i,j] = 100·dot(u_i,j,tR_j)`；`L_suffix_mask = CE(QU,y)+CE(QU.T,y)`；`L = L_S0 + 1.0·L_suffix_mask`。
- `lambda_suffix=1.0` 固定（首版预选值，非校准最优）；两方向相加不乘 0.5；**不加**新稀疏项（旧 S0 稀疏项保留，但不移植到 mU）。
- 梯度职责：`L_S0` 训练视觉/文本/原 S0 mask（按原实现）。后缀 loss：经 `g*mU` 训练视觉主干、经 `tR` 训练文本主干、训练 F；**不经** `xU` 中的 `g/rS` 回传视觉或原 S0 mask，**不直接**训练原 S0 mask 参数。注意共享主干会被后缀目标更新（"没有直接梯度到旧 mask" ≠ S0 被冻结）。
- 全开 mU 时（同 checkpoint、同有效候选池）`QU` 必须与 `QN` 一致到合理浮点误差。

## 6. 有效后缀与 DDP 缩放（本轮重点）

每 rank 原始 microbatch `B` 不变，但每 rank 有效后缀数 `n_r` 可能不同。先 `all_gather(suffix_valid)` 得全局有效索引 `J`、`V=|J|`；后缀候选图片/文本都只取 `J`；得到 `V×V` 评分矩阵；**两臂使用完全相同的 J**。原 S0 仍算完整 global batch。为免变长 gather：每 rank 后缀 token 保持 `B` 行，无效位置放空串占位并标 `valid=false`，**占位不得进入 loss 或候选池，也不得当作真实后缀统计或正例**。

DDP：目标是全局有效 anchor 均值
`L_U_global = (Σ_valid I2T_CE + Σ_valid T2I_CE) / V`；每 rank 本地只算自己的有效 anchor，反传时乘
**`world_size / V`**（`L_U_backward_local = (W/V)·(local_I2T_sum + local_T2I_sum)`），随后标准 DDP 平均。
等价地：本地均值 × `world_size·n_r/V`。**不得**直接平均各 rank 的有效均值（n_r 不同会错权重）；所有 rank 有效数相同时退化为普通 local mean，**不得再多乘一次 world_size**。

边界：`V<2` → 全局后缀 loss 记为 0（不重抽 K、不补伪后缀、不用单候选 CE 假装训练）；某 rank `n_r=0` 但全局 `V≥2` → 仍参加所有必要 collective，贡献**可导的零**，不得提前 return 造成死锁。大 gather 前共同检查 microbatch 长度；**不得在 rank0-only 日志路径里执行 collective**。

## 7. 分块与数值边界

`image_chunk=16`、`text_chunk=32`（初始值，只允许调分块）。块内：`g_det=g.detach()`、`mS_det=mS.detach()`、`rS=g_det[:,None,:]*mS_det[None,:,:]`、`xU=concat([broadcast(g_det), rS], -1)`、`mU=hard_st(sigmoid(F(xU)))`、`u=Norm(g_live[:,None,:]*mU)`、`scores=100·(u*tR[None,:,:]).sum(-1)`。**最终 g 保持 live，F 输入停梯度**。（可把第一层权重拆成"完整 g 部分 + masked 部分"预计算可复用项，但必须与直接拼接参考在前向与 F 梯度上等价。）不常驻 `[B_global,B_global,1024]`；不在每 rank 重算全局全部图像行；必要时用非重入 activation checkpoint，**collective 放在重计算块之外**。

全关/近零 `u`：eps 安全归一化，保留样本与候选，不自动全开、不回退原生评分、不补开维度；记录全关率与近零范数率。非有限 loss/梯度：`optimizer.step` 之前全 rank 共同停止并报告（不 nan_to_num、不自动改 LR/阈值/lambda、不跳过后继续）。

## 8. 预算与优化器

ViT-B/16、248 token、单视图、4 卡 × 256 pairs、原 S0 候选池 1024、gradient accumulation=1、**每臂恰好 500 次 optimizer update**（第 501 次不执行）。参数组与既有冻结 S0 设置一致：CLIP 主干与两侧 projection `lr=1e-6, wd=1e-2, warmup=200`；原 S0 mask 网络 `lr=1e-3, wd=0, warmup=0`；新 F **仅在 mask 臂** `lr=1e-4, wd=0, warmup=0`；均 AdamW `betas=(0.9,0.999), eps=1e-8`。按 Parameter ID 检查无遗漏、无重复；**旧 S0 mask 必须继续训练**。`horizon = 3*len(loader)` 动态计算（预计 3651），500 只是停止位置。FP32 master；主干沿用既有 bf16 autocast；**新 F、g 归一化、后缀评分与 CE 核心显式 FP32**，两臂后缀精度一致。保存 `0/20/100/250/500`；前 20 步属于同一次 run（不另起 smoke 再重训）。**先跑 native，再跑 mask，顺序执行不抢卡**；不因第一臂未过门而取消第二臂，也不因某臂过门而延长训练。

## 9. 必要测试（`tests/test_said_prefix_suffix.py`）

A 前后缀切分（K=1/中间/N-1/N；1 句/2 句 caption；末尾空片段；最后非空句段被排除；P 与旧规则逐字符串一致；空 R 不补位不重抽）——例子 `[A,B,C,D]` 与 `[A,B,C,D,""]` 必须逐条通过。
B 文本输入隔离（P 的 token/hidden 只来自 P；R 单独编码；固定图像与 P 时改 R 不得改变 mS 或 mU；只改变后缀目标与评分）。
C 新门输入（`rS=g*mS` 不再归一化；F 输入全 detach；最终 g live；`mU` 作用于完整 g 而非硬补集）。
D 初始化（`pU=8/9`、`mU` 严格全 1；同 checkpoint 下与原生后缀评分一致；第一步零输出层梯度行为）。
E S0 回归（`lambda_suffix=0` 时 S0 loss、旧 mask 与共享参数梯度与原 helper 一致；新模块初始化后全局 RNG 恢复）。
F 逐 pair 与分块（两种候选方向、全局对角索引正确；比较 `g`、`tR`、F 参数梯度；原 mS 不从后缀 loss 直接得梯度）。
G 有效子集与 DDP（真实两 rank：有效数相等 / 不等 / 一个 rank 为 0 / 全局 V=0 / V=1；验证 `world_size/V` 缩放与 `lambda_suffix` 未丢失；单进程同全局样本与同有效子集参考，比较**梯度与一次更新**而非只看 forward）。
H 全关/近零（数值有限、无隐藏 fallback；学到的全关门不得被当作无效文本剔除）。
I 完整 checkpoint（调用**生产保存函数**；非初始 F 与原 S0 mask 严格 round-trip；重载后 prefix score、suffix score、两份 mask 与原生输出对应）。

正式两臂各自从共同初始化重新加载；临时测试更新不得作为正式起点。接受合理浮点误差，**不得放宽容差掩盖倍率错误**。

## 10. 检查点与 runner

检查点键（张量与描述**必须分开**，避免再发生权重被 metadata 覆盖）：`clip_state`、`suffix_mask_state`（native 臂显式 `None`）、`suffix_mask_config`、`optimizer_clip`、`optimizer_mask`、`optimizer_suffix`（native 臂可为 `None`）、`scheduler_config`、`scheduler_state`、`completed_steps`、`data_cursor`、`rng_states`、`loss_config`、`sampling_config`、`provenance`（含两臂共同的 base SHA、init sha256、流摘要、逐 rank 摘要）。step 0/20/500 校验完整性；step 20 必须能验证**已更新**的 F 实际加载。缺 F 时不得以随机/初始化 F 代替并宣称条件模型可恢复。裸学生只含原 clip 兼容 state（原 `mask_net` 保留、新 F 不进入）。

runner：明确解释器启动 `torch.distributed.run`；资源（4 卡空闲）重新检查；原子锁；后台与 SSH 解耦；stdout/stderr、PID、run_id 持久化。每臂顺序 `train → 校验 step500 → export → COCO → Urban → report`，四阶段真实退出码分别保存，失败后只补对应阶段（不为修 export 重训 500 步），attempt 历史保留。

## 11. 主评估与冻结门

只用原生 CLS/EOS：`Norm(student.encode_image(I))`、`Norm(student.encode_text(C))`；**不用** P/R 条件、新 F 或新 mU，不做融合/rerank/辅助读出替换。COCO canonical（5000×25000）与 Urban-1k（1000×1000）各报 R@1/5/10。严格导出加载（missing/unexpected 为空；导出前后原生接口输出一致）。冻结门（未取整）：**COCO 两方向均不低于 S0@500（I2T 0.6058 / T2I 0.41236），且至少一个严格提高**；Urban 单列不改变门。**重点比较 `S0_SUFFIX_MASK − S0_SUFFIX_NATIVE`**：即使超过 S0，若未超过同样后缀监督的 native 对照，也**不能**把提升归因于新掩码；若两臂都超 S0 且近似打平，结论应优先指向"后缀监督收益"，而非选择机制成立。单 seed 小差值不宣称稳定显著。

## 12. 小型只读机制诊断（不替代主评）

**队列（已核实的诚实结论）**：训练管线 `train/said_cvssl_data.py` **没有任何 holdout 机制**，既有 `sharegpt4v1k_manifest.json`（1000 条）与 `sharegpt4v1k_usr_manifest.json`（868 条）都是从**同一份训练清单**按 `json_index 0..999` 取出的切片。因此本诊断**不是训练集外的验证集**，只能称为"训练分布内的固定诊断划分"，报告中必须如此标注，且不得据此宣称泛化。诊断按固定顺序取最多 64 个在**本轮规则**下 R 非空的样本（不足 64 就报实际数量，不从训练集补齐），固定一次 P/R 划分（预先固定中间切分 K，例如 `K=max(1, L//2)`，并注明这只是诊断划分、未改变训练随机 K），保存 manifest，两臂共用。

四种模式（对 mask 臂 checkpoint）：**NORMAL**（真实图像 + 正确候选前缀 mask → mU → 后缀评分）；**U_ALL_ONES**（强制 mU≡1，应等于同 checkpoint 的原生后缀评分）；**PREFIX_SHUFFLED**（仅置换候选前缀 mask 来源，保持图片/后缀/标签不变，重新生成 mU）；**IMAGE_SHUFFLED**（固定 P/R 与标签，置换输入图像，`g/rS/mU` 全按置换后图像重算，只作视觉依赖干预检查）。正常对齐时每个候选 j 用其对应 `P_j`（不得改成整行共享正确 `P_i`）。记录 I2T/T2I 的 R@1、正负 margin、配对排名变化、mU 保留率与全开/全关率、mS/mU 重叠与读出余弦。只作诊断：mask 不同 ≠ 语义解耦；打乱变差不自动证明纯视觉补充信息；全开表现相近可能意味着新选择增量有限。只读、独立 status，不改旧 run_status。

## 13. 日志、前端与报告

复用既有只读前端（不新建平台、不动 8765 与隧道），登记两个新 run。训练日志主记：S0 两方向与原稀疏、后缀两方向与 `lambda_suffix`、total、各 rank 有效后缀数与全局 V、空后缀比例、P/R 长度与截断量、旧 mS 与新 mU 保留率（标明 positive-pair/local 或 global 范围）、原生/条件后缀正负 margin、各组 LR、同步步时、GPU 内存。**LSE margin 定义必须为 `positive − logsumexp(合法负配)`**，`logsumexp(全部)−positive` 是 CE，不得错命名。前端两臂对照并显示：S0 冻结结果、native 后缀对照、新 mask 方法、R@1/5/10 及"方法−对照"差值；明确显示"新增后缀监督 / lambda_suffix=1 / 新 U 稀疏系数=0 / 原 S0 目标未改 / 有效候选池 V / 原生主评与条件诊断分开"。页面只读日志与小型结果，不 import torch、不加载 checkpoint、不触发 GPU。实际服务 JS 做 `node --check`；无浏览器工具则视觉验收记 NOT RUN。

报告必须含：P/R 与原最后一句排除的精确定义；空后缀处理；两臂公式与梯度边界；有效子集 DDP 缩放；数据流身份与监督预算差异；完整 checkpoint 与裸学生 SHA；训练 SHA 与报告 SHA 分别记录；两数据集结果、固定门与方法增量；小池诊断与 NOT RUN。**不宣称**：后缀等于全部 Unsaid；两个 mask 互相正交或完整覆盖图像；只用训练后缀就自动获得纯语义解耦；辅助分支不会改变 S0 能力；超过旧 S0 就证明新 mask 有用。

## 14. 冻结 API 名单（实现与测试共同依赖）

`model/said_prefix_suffix.py`：`OBJECTIVE`、`ARM_NATIVE`、`ARM_MASK`、`LAMBDA_SUFFIX`、`IMAGE_CHUNK_DEFAULT=16`、`TEXT_CHUNK_DEFAULT=32`、`SUFFIX_MASK_SEED=0`、`SUFFIX_GATE_BIAS_INIT=log(8)`、`split_prefix_suffix(caption, k)`（返回 dict：`prefix/suffix/valid/k/n_sentences/last_nonempty_index`）、`sample_k(n_sentences, rng)`、`class SuffixMask(nn.Module)`（`forward(x)` 返回 `pU`）、`class SuffixMaskGate`（`gate_from_pU(pU)` 返回 `mU`）、`suffix_readout_scores(...)`、`global_valid_indices(...)`、`suffix_scaling(world_size, V)`、`checkpoint_metadata(...)`、`state_digest(...)`。
`train/train_said_prefix_suffix.py`：`ARMS` 字典、`main()`、模块级生产保存函数（供测试调用）。
诊断：`tools/diag/suffix_mechanism_diagnostic.py`。
