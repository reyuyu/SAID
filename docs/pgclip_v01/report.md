# PG-CLIP v0.1 报告（Pre-Projection Gated CLIP：原生对齐 ＋ 投影前 768 维文本条件视觉选择）

- objective：`clip_native_preproj_mask`；arm：`PG_CLIP_V01`；phase：`pgclip-v0.1`
- branch / worktree：`codex/pgclip-v01` / `/root/SAID-pgclip-v01`（本 worktree 建立时的实际提交 `894599d`）
- **正式训练使用的实现 SHA：`7bafe2f3ecc3bd46d1bc9a1a7c3a61071eabac5c`**（记录在 run 的 `config.json` / `run_status.json.implementation_sha`）
- 本轮只跑**一个**配置，没有任何参数搜索；训练恰好 500 次 optimizer update，没有第 501 次
- 结论一句话：**500 步后与 S0@500 基线基本打平，冻结门判定 FAIL**（COCO I2T R@1 0.6048 < 0.6058 下限 0.10pp；T2I 0.41256 高出下限 0.02pp）；gate 确实学到了非平凡的条件 mask（平均保留 611/768 个坐标、没有 caption 仍全开），两路表示也已明显分化（余弦 0.941、LP 一度高于 LG），但原生 CLS/EOS 表示在 500 步内没有被这套辅助目标改变到可测出提升的程度。

---

## 1. 精确取点与两路公式（与实现逐条对应）

```
h_i  = ln_post( CLS_output_of_visual_block_12 )        # [B, 768]，一次视觉主干前向
v_raw = h_i @ clip.visual.proj                          # [B, 512]，同一个 W，不复制
H_j   = ln_final( full_token_sequence )                 # [B, 248, 512]，一次文本主干前向
eot_j = argmax(token ids)                               # 仓库既有 EOT 约定（不是非零 token 计数）
t_raw = H_j[eot_j] @ clip.text_projection               # FP32

W = clip.visual.proj                                    # 唯一 Parameter，两路共享
QG[i,j] = 100 * dot( Norm(h_i @ W),             t̂_j )
QP[i,j] = 100 * dot( Norm((h_i * mask_j) @ W),  t̂_j )    # t̂ = Normalize(t_raw), eps = 1e-6
```

- `encode_image_with_preprojection` / `encode_text_final_hidden` 为新增的**可选**接口；测试用计数器证明一次调用中 trunk 与 `ln_post` 各执行 1 次，且 `v_raw == h @ proj == encode_image(x)`（同精度逐位相等）、`encode_text(..., return_full=True)` 的隐藏序列与 `encode_text_final_hidden` 逐位相等。
- `encode_image` / `encode_text` 的默认行为与 **317 个 state_dict 键**完全不变（与仓库内 `tests/state_dict_keys_baseline.txt` 逐行比对）。
- 条件投影范数用**显式分块** `Norm((h*m)@W)`，不做 `sum(h²·m²)` 的对角简化；单元测试给出非对角 `W@Wᵀ` 的手工反例（`W@Wᵀ` 非对角占比 > 0.5，对角捷径与真实投影范数的相对误差 > 1%），并验证实现与二次型 `(h*m)@(W@Wᵀ)@(h*m)ᵀ` 逐元素一致。

## 2. gate：结构、初始化与梯度边界

- 结构：`MaskNetwork(width=512, layers=1, heads=8)`（仓库既有实现：1 层 ResidualAttentionBlock + 原 AttentionPool）+ `Linear(512, 768, bias=True)`。
- 初始化：stem 正常初始化；输出层 `weight = 0`、`bias = log 8` ⇒ 初始 `p = 8/9`、`mask ≡ 1`。实测 `GATE_INIT {"mask_min": 1.0, "mask_max": 1.0, "p_mean": 0.888889, "projection_weight_norm": 0.0}`。
- 构造在隔离且可恢复的 RNG 中完成（Python/CPU/当前 CUDA 设备状态全部还原），不污染主模型与数据流。
- 前向 `hard = (p >= 0.5)`，反向走 sigmoid 直通；`mask = hard + (p - p.detach())`；不 detach 整份 mask、不做 soft 下界、不 top-k、不固定保留数。
- 梯度职责（测试逐项固定）：`LG` 对 gate 的梯度**全为 None**；`LP` 对共享参数与 gate 都非零；`LS = mean|mask|` 对视觉/文本主干**全为 None**、只作用于 gate。训练日志显示 step 1 的 `gate_stem_grad_norm = 0.0`（输出层权重为 0 所致，属预期），step 10 起变为 0.23，step 500 为 0.38。
- `W` 只通过 `clip.visual.proj` 引用；按 parameter id 验证优化器两组不重叠、覆盖全部可训练参数、旧 `mask_net`(14 个参数) 与 `logit_scale` 被冻结排除、`W` 只出现在一组里。

## 3. 损失与 DDP 约定（实际配置）

```
LG = CE(QG, y) + CE(QGᵀ, y)          LP = CE(QP, y) + CE(QPᵀ, y)          LS = mean(|mask|)
L_total = 5*LG + 5*LP + 1*LS          # 双向相加，不乘 0.5，总 loss 外不再乘任何系数
```

- **step 1 的三重恒等式实测成立**：`LG = LP = 1.304433`（全开 mask 时两路等价）、`LS = 1.0`、`L_total = 14.044331 = 10×1.304433 + 1`。报告明确区分“总 loss 数值”与“共享参数的梯度对应 10×LG”。
- DDP：本地 anchor mean + autograd-aware gather + 标准 DDP 参数梯度平均，**任何地方都没有乘/除 world_size**。
- **真实两 rank 验收（测试 F）**：同一全局 batch、同一超参、非平凡 gate（输出层已扰动）。梯度比值：norm 1.0000007–1.0000045、max 0.99999–1.00002、余弦 1.00002–1.00022。一步 AdamW 之后的参数差只出现在梯度极小的元素上：

  | \|g_ref\| 区间 | 元素数 | 有差异的元素 |
  |---|---|---|
  | < 1e-6 | 25,730,056 | 15,343 |
  | 1e-6 – 1e-4 | 493,857 | 333 |
  | 1e-4 – 1e-3 | 2,075,762 | 3 |
  | 1e-3 – 1e-2 | 14,232,617 | **0** |
  | > 1e-2 | 110,722,877 | **0** |

  读法：两条路线在数学上一致；残余差异来自 fp32 累加噪声层面上梯度符号的翻转，而 AdamW 的首步 `lr·g/(|g|+eps)` 会把这种翻转放大成整步 ±lr。**这是被实测确认的结论，不是注释里的声明**（任务第 7 节要求）。
- 全关/近零向量边界：`mask≡0` 与 `h·1e−12` 情形下分数保持有限、候选池不缩小、坐标不被自动重新打开；`projected_output_energy_ratio` 允许 > 1（单测用近乎抵消的投影构造出 > 1 的情形，验证不被截断）。

## 4. 训练预算、吞吐与显存（真实数值）

| 项 | 值 |
|---|---|
| 4 卡 × 每卡 batch | 4 × 256 = **1024** 全局匹配 batch，gradient accumulation = 1 |
| optimizer updates | **恰好 500**（checkpoint：0/20/100/250/500） |
| 训练墙钟 | **632.55 s**（10.5 分钟），平均 **1.13 s/step**，≈ 996 样本/秒 |
| 峰值显存 | **42.52 GB / 卡**（A800-80GB） |
| 数据 | Full ShareGPT4V 清单 1,245,901 条；每 epoch 1217 个 batch；LR horizon = 3×1217 = **3651**（按真实 loader 计算，非硬编码）；本轮只截断到 500 |
| 学习率 | CLIP 1e-6（warmup 200，step 1 实测 5e-9）、gate 1e-3（warmup 0，step 500 实测 9.55e-4），betas (0.9,0.999)，eps 1e-8，wd 1e-2 / 0 |
| 截断 | 1022/128,000 条 caption 超过 248 token（0.80%），最大有效长度 248，按参考方式截断并记录 |
| 精度 | bf16 autocast 主干 + **FP32 投影前核心**（原生/条件投影、最终文本投影、gate、normalize、score、loss），参数全部 FP32 master |

**数据流身份（实测，不是“seed 相同”的推断）**：PG-CLIP 的 `caption_stream_sha256 = 799f8efa…c0fd4` 与 `sample_stream_sha256 = 39c9885c…422856` 与 S0@500 的 `run_summary.json` **逐位相同**；同时记录了 `image_id` 与 `prefix_k` 流摘要。也就是说这套 500 步实验与 S0 基线**吃的是完全相同的样本与 caption 流**。

## 5. gate 是否真正形成条件选择（本轮实测）

| 量 | step 75 | step 500 |
|---|---|---|
| mask 平均保留坐标数（/768） | 673.3（87.7%） | **611.2（79.6%）** |
| 全开 caption 比例 | 0.0 | **0.0** |
| 全关 caption 比例 | 0.0 | 0.0 |
| 逐 caption 保留数范围 | 576–731 | 487–699 |
| gate 概率 均值 / 最小 / 标准差 | 0.745 / 0.0055 / 0.196 | 0.599 / 0.0046 / 0.126 |
| 概率分位数 p5/p50/p95 | — | 0.405 / 0.590 / 0.827 |
| 跨 caption 的逐坐标概率变化（均值） | 0.087 | 0.091 |
| `preproj_retained_energy`（‖h·m‖²/‖h‖²） | 0.890 | **0.820** |
| `projected_output_energy_ratio` 均值 / 最大 / >1 比例 | 0.834 / 0.958 / 0.0 | 0.729 / 0.943 / **0.0** |
| 原生与条件输出余弦（均值） | 0.967 | **0.941** |

- **是**：500 步后没有 caption 仍处于“全开”，平均关闭 157 个坐标，逐 caption 差异明显（487–699），跨 caption 的逐坐标概率也有系统变化（0.091），说明 gate 依赖 caption 内容而非退化成一个常数门。
- **但是**：本轮数据里 `projected_output_energy_ratio` 从未超过 1（>1 的情形只在单测中构造出来），且两路 top1/margin 的差距不大（step 500：原生 I2T top1 0.9531、条件 0.9492；max margin 8.48 vs 7.87；LSE margin 0.122 vs 0.146）。也就是说条件路径确实“不同”，但在这 500 步内**没有变成更强的表示**。

## 6. 500 步后的评估（只用原生 CLS/EOS）

主评估只用 `Norm(student.encode_image(I))` 与 `Norm(student.encode_text(C))`：**不用 gate、不做两路融合、不做 reranking、不用任何辅助表示替换 CLS/EOS**。裸学生 `student_000500.pt`（`PreProjectionGate` 不进入裸学生 state；导出器验证 `missing/unexpected` 为空、键集合与原生 CLIP 一致、导出前后 `encode_image`/`encode_text` 输出逐位一致）SHA256 `ffaaf79f217e536da2b2cba3e197fa6f6128c1299ed2d3c29d073e18b17ca804`。

| 数据集 | 模型 | I2T R@1 / R@5 / R@10 | T2I R@1 / R@5 / R@10 |
|---|---|---|---|
| COCO canonical（5000×25000） | **PG-CLIP v0.1 @500** | **0.6048** / 0.8210 / 0.8908 | **0.41256** / 0.66940 / 0.76544 |
| COCO canonical | S0@500（冻结参照） | 0.6058 / 0.8220 / 0.8906 | 0.41236 / 0.67092 / 0.76620 |
| 差值（pp） | | **−0.10** / −0.10 / +0.02 | **+0.02** / −0.15 / −0.08 |
| Urban-1k（1000×1000） | **PG-CLIP v0.1 @500** | **0.87000** / 0.97000 / 0.99000 | **0.83300** / 0.96200 / 0.98200 |
| Urban-1k | S0@500（冻结参照） | 0.87000 / 0.97100 / 0.98800 | 0.84200 / 0.96700 / 0.98100 |
| 差值（pp） | | **0.00** / −0.10 / +0.20 | **−0.90** / −0.50 / +0.10 |

**冻结晋级门**（只在 500 步、只看 COCO 原始精度）：I2T R@1 ≥ 0.6058 且 T2I R@1 ≥ 0.41236，且至少一项严格更高 → `PROMISING_AT_500`，否则 `FAIL`。
本轮 **COCO I2T R@1 = 0.6048（差 0.10pp）→ FAIL**；T2I 高出下限 0.02pp，不足以翻转判定。换算成查询条数：COCO I2T 的 5000 个候选**图像查询**中 0.10pp = **5 个**（每个查询占 1/5000）；COCO T2I 的 25000 个**文本查询**中 0.02pp = **5 个**（每个查询占 1/25000）。两份协议的分母不同，同一个 pp 数字对应的查询条数也不同，下文一律按这个换算写清楚。Urban-1k 单列，不参与判定（T2I R@1 −0.90pp 是两份协议里最大的负差）。

**归因限制**：本轮**没有**运行 native-only 或 native+post-projection 控制臂，因此即便出现提升也只能归因于“PG-CLIP 这一整套组合”，不能单独声称“投影前位置优于投影后”；反之，本轮的 FAIL 也不能单独归因到取点位置上。单 seed、单配置、500 步（余弦 horizon 为 3651 步，训练时远未走完调度）——这些都是结论的边界。

## 6.1 同配置第二次运行（sibling run）与 run 间波动

原 run 的 checkpoint 因为键冲突没有保存 gate 张量（见第 8 节），所以为了能评辅助分支，用**修复后的同一份代码、同一份配置、同一份共享初始化**又跑了 500 步（`runs_salu/pgclip_v01_rep500/step500`，实现 SHA `59e12d6`，训练/导出/两套评估四个阶段全部 exit 0）。它**不是逐位复现**：`final_state_digest` = `700db0fb…`（原 run `b532599d…`），gate digest `4572f35e…`（原 run `5b8c7383…`）；数据与 caption 流仍然与原 run 和 S0 逐位相同（`799f8efa…` / `39c9885c…`）。也就是说它是**同一配置的第二个样本**，而不是原 run 的恢复。

| 指标 | 原 run @500 | sibling @500 | S0@500（冻结） | 原 run vs S0 | sibling vs S0 |
|---|---|---|---|---|---|
| COCO I2T R@1 | 0.6048 | 0.6056 | 0.6058 | −0.10 pp | −0.02 pp |
| COCO T2I R@1 | 0.41256 | 0.41236 | 0.41236 | +0.02 pp | ±0.00 pp |
| Urban I2T R@1 | 0.8700 | 0.8730 | 0.8700 | ±0.00 pp | +0.30 pp |
| Urban T2I R@1 | 0.8330 | 0.8300 | 0.8420 | −0.90 pp | −1.20 pp |

**这个对照很有价值**：同一配置两次运行的 R@1 波动为 0.02–0.30 pp（COCO 0.08 pp，Urban I2T 0.30 pp，Urban T2I 0.30 pp）。据此重新读原来的结论：
- **COCO 的差异（−0.10 / +0.02 pp）落在 run 间波动之内**，不能读成“PG-CLIP 伤害了原生表示”；
- **Urban T2I 两次都是负的（−0.90 / −1.20 pp），大于 run 间波动（0.30 pp）**，这是本轮唯一看起来稳定的变化，方向是**变差**（长 caption 的文本→图像检索）；
- COCO / Urban 的 I2T 基本不变；sibling 的 COCO I2T 0.6056 距离冻结门下限 0.6058 只差 0.02 pp（5000 个图像查询里的 1 个），两个同配置 run 都 FAIL，说明 500 步的判定就贴在下限附近，对噪声很敏感。

## 6.2 新增：辅助分支图文检索（diagnostic，不参与冻结门）

按用户要求新增一种评估方式：**直接用辅助（投影前条件）分支做图文检索**。工具 `tools/diag/pgclip_aux_retrieval.py`（+ `tests/test_pgclip_aux_retrieval.py`，9 个 CPU 测试）在同一遍编码上同时算两条路径：

```
native     QG[i,j] = 100 · ⟨Norm(h_i @ W),             t̂_j⟩
auxiliary  QP[i,j] = 100 · ⟨Norm((h_i * mask_j) @ W),  t̂_j⟩
```

- 候选规则与训练一致：I2T 每条候选文本用**它自己的** mask；T2I 固定文本用**同一个** mask（两个方向读同一张矩阵）。
- 指标口径：COCO 采用历史的**多正例**约定（每张图的 5 条 caption 任一中即命中），T2I 正例为 `j//5`；Urban-1k 一对一；排名用仓库既有规则 `rank = 1 + #{s_j > max_pos} + #{j < min_pos : s_j == max_pos}`（并列时下标小的排前面）。池大小与冻结协议一致（COCO 5000×25000、Urban 1000×1000），输出里显式写入 `not_a_gate_candidate: true`、`new_optimizer_updates: 0` 与池大小，绝不冒充冻结评估。
- 只读：不带梯度、不更新参数（前后 parameter digest 一致），只写 run 目录内一个小 JSON（9 KB）。运行耗时：COCO 76 s、Urban 11 s（单卡）。

| 数据集 | 读出 | I2T R@1 / R@5 / R@10 | T2I R@1 / R@5 / R@10 | I2T / T2I MRR |
|---|---|---|---|---|
| COCO（5000×25000） | 原生 | **0.6066** / 0.8212 / 0.8920 | **0.41168** / 0.66960 / 0.76532 | 0.70262 / 0.53117 |
| COCO | **辅助分支** | **0.5538（−5.28 pp）** / 0.7942（−2.70） / 0.8682（−2.38） | **0.39356（−1.81 pp）** / 0.65108（−1.85） / 0.74712（−1.82） | 0.66226（−4.04）/ 0.51324（−1.79） |
| Urban-1k（1000×1000） | 原生 | **0.8710** / 0.9720 / 0.9910 | **0.8300** / 0.9650 / 0.9800 | 0.91720 / 0.88999 |
| Urban-1k | **辅助分支** | **0.8020（−6.90 pp）** / 0.9540（−1.80） / 0.9820（−0.90） | **0.8190（−1.10 pp）** / 0.9410（−2.40） / 0.9700（−1.00） | 0.87058（−4.66）/ 0.87349（−1.65） |

逐查询配对证据（辅助 vs 原生，同一批特征）：

| 数据集 / 方向 | R@1 命中增加 | R@1 命中丢失 | 排名改善 / 不变 / 变差 | 最大变差 | 平均排名变化 |
|---|---|---|---|---|---|
| COCO I2T | 223 | **487** | 811 / 2855 / **1334** | 426 | **+1.11**（变差） |
| COCO T2I | 756 | **1209** | 4902 / 12184 / **7914** | 1603 | **+2.12**（变差） |
| Urban I2T | 20 | **89** | 43 / 825 / **132** | 30 | **+0.30** |
| Urban T2I | 40 | **51** | 61 / 819 / **120** | 78 | **+0.46** |

被测文本的 mask 状态（同一快照）：COCO 平均保留 **631.0/768**（范围 467–739，全开 0.000、全关 0.000，p 均值 0.630、最小 0.002）；Urban 平均保留 **575.2/768**（444–708，p 均值 0.584、最小 0.084）——辅助分支确实是**带条件掩码**的读出，没有退化成全开。

**结论（明确，且与“辅助分支能带来提升”的直觉相反）**：**把辅助分支单独当作检索读出用，明显比原生路径差**——I2T R@1 低 5.3 pp（COCO）/ 6.9 pp（Urban），T2I 低 1.8 pp / 1.1 pp，MRR 与平均排名一致变差，且丢命中远多于得命中（COCO I2T 487 vs 223，COCO T2I 1209 vs 756）。这个 5–7 pp 的差距比同配置 run 间波动（0.02–0.30 pp）大一个数量级以上，因此不是噪声。

必须一起读的三点：
1. **这不改变冻结结论**：冻结门只用原生 CLS/EOS；辅助分支是训练时的辅助目标。本评估只回答“若把它当读出会怎样”，答案是更差。
2. 一个自然的解释：mask 是在**训练目标**（1024 候选池内对比 + `mean|mask|` 稀疏）下学出来的，被优化成“让 QP 这个训练损失下降”，而不是“让 QP 成为更好的检索表示”；gate 学习率 1e-3 远高于主干 1e-6，500 步内 mask 已经关掉约 18%（COCO）/ 25%（Urban）的坐标，把投影后向量的一般性信息削掉了一部分（`preproj_retained_energy` 0.82、原生/条件输出余弦 0.94）。
3. 本评估**没有**测试任何“两路融合 / 用辅助分数 rerank 原生候选 / 用原生与辅助的分数组合”，所以它不能说明融合的上限，也不能说明继续联合训练更久会怎样。

原生列由本工具自己的编码路径算出（batch 128），与冻结 evaluator 的同 run 结果相差 ≤0.13 pp（COCO I2T 0.6066 vs 0.6056，T2I 0.41168 vs 0.41236）；辅助比较是**同一批特征内的配对比较**，这个 0.1 pp 量级的协议差异不影响 5–7 pp 的结论。

## 7. 真实退出码与 runner 记录

| 阶段 | 退出码 |
|---|---|
| train（500 步） | **0** |
| export（裸学生，含无损性证明） | **0** |
| COCO canonical | **0** |
| Urban-1k | **0** |

- runner：原子锁 + 启动前重新查询四卡（`nvidia-smi` 失败视为“不空闲”）+ 数据路径预检 + `setsid` 脱离 SSH + 记录 PID/命令/run_id/实现 SHA；每阶段写真实退出码，失败即停止且绝不评估不完整权重。
- **attempt 记录（4 次，全部如实保留在 `run_status.json.attempt_history`）**：
  1. 训练成功（exit 0），但 checkpoint 校验读错了 gate 身份键的位置并比较了缺失的顶层 `lr_horizon_steps` → 校验失败、未评估；
  2. 修正后再次校验，发现 checkpoint **没有 gate 张量** → 拒绝继续；
  3. 修正 runner 的校验并加 `--allow-missing-gate-state`，仅补跑 `export,coco,urban`（**没有重新训练**）；
  4. 完成，四个阶段全部 exit 0。

## 8. 本轮发现并修复的工程问题（不重新训练）

1. **checkpoint 键冲突（最重要）**：trainer 把 gate 的**张量**与 gate 的**描述**写进了同一个 `gate` 键，`checkpoint_metadata` 覆盖了 `gate.state_dict()`，导致第一次正式运行的 **gate 权重没有被保存**。已修复：张量写到 `gate_state`，描述留在 `gate`，并在 trainer 里加断言；`tests/test_pgclip.py` 增加回归断言（两个键必须同时存在、`gate` 里不能再出现 `projection.weight`）。**后果必须说明**：本 run 无法再复现训练后的 gate，因此本 run 的逐坐标 768 维网格**未运行**（页面显示“未运行 + 原因”，不伪造网格）；`tools/diag/pgclip_mask_snapshot.py` 已实现该只读快照，供后续 run 使用。
2. **checkpoint schema v1.1**：把 gate 身份（`gate_mode/gate_width/gate_out/gate_layers/gate_heads/gate_seed`）与 `lr_horizon_steps` 提升为顶层字段，校验器不必再从 `config` 里取。
3. **runner 校验加强**：除元数据外，还核对 `gate.projection.weight` 形状 (768,512)、`projection.bias` (768)、stem 张量存在、`clip.visual.proj` 形状 (768,512)、gate 与 clip 不共享同名张量；对 v1.0 checkpoint 显式要求 `--allow-missing-gate-state`，并把 `gate_state_missing` / `gate_state_note` 写入 `run_status.json`（不隐藏损失）。
4. 本次修复发生在训练之后，因此**训练 SHA（`7bafe2f`）与后续交付/前端提交 SHA 分开记录**（见第 10 节）。

## 9. 前端（已有只读平台，不重建）

- 新增只读端点 `GET /api/run/<id>/pgclip`：只读 run 目录内白名单文件（`config.json`、`salu_log.jsonl`、`run_summary.json`、`run_status.json`、可选的 `pgclip_mask_snapshot.json`）与 `evaluation/` 下的两份结果；不 import torch、不加载 checkpoint、不触发 GPU 前向、不写任何文件（测试用 monkeypatch 核对打开的文件集合与目录不变性）。
- 新区域 **「I · PG-CLIP v0.1」**显示：两路损失与权重 5/5/1、两个方向的 top1/max margin/LSE margin、44 条曲线字段的 首/末/最小/最大、gate 与 mask 统计、**768 维坐标网格（只代表维度索引，不代表图像空间）**、两个能量指标（明确标注 `projected_output_energy_ratio` 可 > 1 且从不截断）、训练进度与冻结门结果（COCO/Urban 对照 S0@500 与差值）。**旧 L1/L2/L3 字段不再出现**（测试断言 `loss_1`/`loss_2` 不在返回的 series 里）。
- 该 run 已登记（`pgclip_v01`），原 5 个 run 与 SSH 隧道保留；端口仍为 8765。
- 验证：`node --check` 服务端 `app.js` 通过；`pytest tests/test_training_dashboard.py` **43 passed**（含 4 个 PG-CLIP 端点测试与 node 无头渲染测试）；真实 run 的端点返回 54.7 KB，`available=true`、61 条记录、权重 5/5/1、COCO/Urban 数值与判定 FAIL 一致；非 PG run 在该区域显示 未运行。
- **视觉验收 NOT RUN**：本会话没有浏览器工具，我只做了 HTTP/API 级验证，**不把 HTTP 200 说成视觉通过**。

## 10. 提交与交付物（训练 SHA 与后续 SHA 分开）

| 提交 | 内容 |
|---|---|
| **`7bafe2f`** | PG-CLIP v0.1 实现 + 验收测试 + spec（**正式训练使用的 SHA**，已 push 到 `codex/pgclip-v01`） |
| `a4646fb` | checkpoint schema v1.1、`gate_state` 键修复、runner 校验加强、`pgclip_mask_snapshot.py`、第一版报告与 `results.json` |
| `59e12d6` | 候选 mask 测试的顺序无关性修正 |
| 后续（本次追加，同一分支） | `tools/diag/pgclip_aux_retrieval.py`（辅助分支检索诊断）+ 9 个 CPU 测试、`mask_statistics` 的分位数分块修复、本报告的 6.1/6.2 两节与 `results.json` 的对应字段 |
| 前端（另一 worktree `codex/s0-trimask-hs-v02`） | `tools/dashboard_data.py`（`pgclip()` + 白名单 + `auxiliary_retrieval`）、`tools/serve_training_dashboard.py`、`web/training_dashboard/{app.js,index.html,style.css}`（区域 I 增加 B2 辅助分支检索块）、`tests/test_training_dashboard.py`（+5 测试） |

交付文件：`model/model_longclip.py`（两个可选接口）、`model/pgclip.py`、`train/train_pgclip.py`、`tools/pgclip_runner.py`、`tools/diag/export_pgclip_student.py`、`tools/diag/pgclip_mask_snapshot.py`、`tests/test_pgclip.py`、`tests/_pgclip_ddp_worker.py`、`docs/pgclip_v01/{spec.md,report.md,results.json}`。**不提交 checkpoint、数据、caption 缓存或大型日志**（run 目录在 `runs_salu/`，已被 `.gitignore` 覆盖）。

## 11. 明确 NOT RUN / 不可宣称

- **NOT RUN**：浏览器视觉验收；本 run 的逐坐标 768 维 mask 快照（原因见第 8 节；sibling run 的 checkpoint 现在带 gate 张量，快照工具可直接用，但尚未跑）；native-only 与 native+post-projection 控制臂；两路融合 / rerank / 分数组合类实验；任何参数搜索、额外 seed、更长训练、跑满 3 epoch；文本 mask、U/教师/decoder/重建、交叉注意力、软门、动态映射。
- **关于辅助分支检索**：本轮只测了“单独用辅助分支当读出”这一种用法，结果是比原生差 5.3–6.9 pp（I2T R@1）。**不能**由此推断“辅助分支没有用”——它在训练中参与共享主干；也不能推断“融合也没用”，因为融合没有被测。
- **不可宣称**：辅助（条件）分支不损害原生能力——它参与训练共享主干与原生投影，本轮 COCO I2T 就低了 0.10pp；768 维天然比 512 维语义更纯；稀疏表示必然实现语义解耦；继承 SmartCLIP 的全部理论保证；单 seed 过门就等于稳定 SOTA（本轮连门都没过）。
- 结论适用范围：**一个配置、500 次更新、单 seed、余弦 horizon 未走完**；FAIL 与“打平”都应读作“在这个预算内没有观察到相对 S0@500 的提升”，而不是“投影前条件选择这条路不可能有效”。
