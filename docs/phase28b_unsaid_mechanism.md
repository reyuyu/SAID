# Phase 2.8B：Unsaid Mechanism & Reachability

> **本轮（reachability 加固）修正了本文档中的 gap 列**：早先 `optimization_gap` / `positive_mixture_gap`
> 把一个 64 样本的 oracle 均值与 256 样本的 current/span 均值相减，属于统计错误。请以下文
> 「Reachability 加固（本轮修正）」一节的同 subset 数字为准；`C_current` / `C_patch_max` / `C_oracle`
> 三个量本身未受影响（`C_span` 的 tolerance 也已改用标准判据，差别 ≤ 4.7e-4）。

本阶段不设计新方法，只验证 Phase 2.8A 的 Minimal Unsaid 在真实训练中"能不能被优化、能不能从 Said 分离、
target 是否在 patch 表征的可达范围内"，并回答瓶颈在哪一层。

## 冻结的定义（未改动）

```text
s_i = q^T k_i          A_s = softmax(s / tau_said)      A_u = softmax(-s / tau_unsaid)
z_s = normalize(sum_i A_s_i h_i)                        z_u = normalize(sum_i A_u_i h_i)
g = sg(normalize(z_G))  s = sg(normalize(z_S))
r_u = g - (g^T s) s     target_u = sg(normalize(r_u))   valid = ||r_u|| > 1e-4
L_u = mean_valid[1 - cos(z_u, target_u)]
```

参数固定：`lambda_unsaid = 0.25`、`tau_unsaid = 0.07`、`unsaid_residual_eps = 1e-4`（无 sweep）。
未修改 `A_unsaid`、complement target、Said / Global loss、caption sampling、validation protocol、
canonical chunk、dashboard、standard inference。

## 新增工具

| 文件 | 作用 |
| --- | --- |
| `eval/unsaid_mechanism_probe.py` | 离线 checkpoint 评测：基础 mechanism 指标 + 三层 reachability（best patch / linear span / approximate softmax-mixture oracle）。只读，不训练、不写 checkpoint、不 backward 到模型参数 |
| `train/train_salu.py` 的 `--lr_total_steps` | backward-compatible：未设置时行为与旧代码完全一致；设置后 `--max_steps` 只截断训练，cosine schedule horizon 仍按完整 run 计算 |
| `tests/test_unsaid_mechanism_probe.py` | 11 个 synthetic 测试（span 可达/不可达、best patch、oracle、确定性、不改模型、invalid residual） |
| `tests/test_lr_total_steps.py` | 5 个测试（默认行为不变、解耦、非法值、CLI 默认） |

Probe 默认设置：ShareGPT4V-1K frozen manifest 前 **256 张图**（三种 caption 全评，`first_sentence`
为 primary）、**fp32** geometry（避免 bf16 噪声）、oracle 64 个 valid 样本 × 100 步 Adam（lr 0.1，
从当前 Unsaid logits 初始化）。输出目录 `outputs/unsaid_mechanism/`（gitignored）。

## Matched 500-step 实验

| 项目 | 值 |
| --- | --- |
| 数据 | 完整 ShareGPT4V 1,245,901（Full Data Gate PASS） |
| steps_per_epoch | 1216（真实 loader 实测：1,245,901 / 1024） |
| lr_total_steps | 3648 = 3 × 1216（用实测值，未手工猜） |
| 实际训练步数 | 500（Arm S 与 Arm SU 相同） |
| batch/GPU、global batch | 256、1024 |
| 硬件 / 精度 | 4 × A800-80GB、bf16、ViT-B/16、residual、identifiable Said |
| seed | 0 |
| 超参 | backbone_lr 1e-6、head_lr 1e-4、tau_said 0.07、tau_unsaid 0.07、eps 1e-4、weight_decay 1e-2、warmup 200 |
| Arm S / Arm SU | `lambda_unsaid = 0.0` / `0.25`（唯一差异） |
| checkpoint 节点 | initial / 50 / 100 / 200 / 300 / 500（两 arm 完全相同，无 cherry-pick） |
| 训练期间 | 关闭 ShareGPT4V / COCO validation、关闭 dashboard export |

`--max_steps 500` 若不配 `--lr_total_steps` 会把 schedule horizon 也压到 500，第 500 步学习率已衰减到 ~2.6e-5；
本阶段两个 arm 都用 `--lr_total_steps 3648`，因此前 500 步等价于未来 3-epoch 正式 run 的前 500 步
（第 499 步 lr 仍为峰值的 0.98）。

## 数据流公平性

| 检查 | 结果 |
| --- | --- |
| `initial_state_sha256`（rank 0–3，两 arm） | 全部 `4621b8d027f482ea…` |
| `sampler_order_sha256` rank0/1/2/3 | S = SU：`953402b3…` / `0ca2c1dd…` / `3258c1e9…` / `6f31a118…` |
| `caption_stream_sha256` rank0/1/2/3 | S = SU：`7563aafb…` / `86c30b90…` / `70460b50…` / `c27336a4…` |
| `batch_image_sha256`（每 10 步，0–490 共 50 个点） | 两 arm **逐位相同**，mismatch 0 |
| `batch_caption_sha256`（同上） | 两 arm **逐位相同**，mismatch 0 |

结论：两 arm 同初始状态、同 sampler、同 caption 流、逐步同 batch → 匹配实验成立。

## 训练轨迹

Arm S（`lambda_unsaid = 0`）：

| step | loss_global | loss_said | route top1 | evidence top1 | route margin | evidence margin | Full Gap | Said Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.6591 | 3.3339 | 0.0039 | 0.7461 | −0.0186 | 8.1577 | 0.6674 | 0.7164 |
| 100 | 0.2902 | 2.4106 | 0.1719 | 0.9102 | 1.6021 | 12.7249 | 0.6574 | 0.7456 |
| 300 | 0.1768 | 0.1109 | 0.9688 | 0.9727 | 19.1729 | 26.1501 | 0.6449 | 0.6765 |
| 499 | 0.1758 | 0.1005 | 0.9766 | 0.9609 | 20.2895 | 26.8628 | 0.6430 | 0.6764 |

Arm SU（`lambda_unsaid = 0.25`）相同节点：

| step | loss_global | loss_said | loss_unsaid | loss_total | route top1 | evidence top1 | route margin | evidence margin | Full Gap | Said Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.6591 | 3.3339 | 0.9838 | 4.2390 | 0.0039 | 0.7461 | −0.0186 | 8.1577 | 0.6674 | 0.7164 |
| 100 | 0.2909 | 2.4190 | 1.0307 | 2.9676 | 0.1523 | 0.9062 | 1.5734 | 12.7844 | 0.6574 | 0.7443 |
| 300 | 0.1798 | 0.1060 | 1.0832 | 0.5566 | 0.9688 | 0.9805 | 19.0453 | 26.0288 | 0.6488 | 0.6741 |
| 499 | 0.1790 | 0.0976 | 1.0099 | 0.5290 | 0.9805 | 0.9648 | 20.0268 | 26.6900 | 0.6532 | 0.6743 |

Arm SU 的 Unsaid 轨迹（训练 batch，rank 0，每 50 步）：

| step | L_u | cos(z_u,target) | cos(z_s,z_u) | overlap | JSD | residual norm | valid ratio | Unsaid entropy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.9838 | +0.0162 | 0.9891 | 0.6880 | 0.0748 | 0.7789 | 1.0000 | 5.1972 |
| 100 | 1.0307 | −0.0307 | 0.9609 | 0.4914 | 0.1947 | 0.7410 | 1.0000 | 5.0669 |
| 200 | 1.1012 | −0.1012 | 0.8298 | 0.2283 | 0.4231 | 0.6464 | 1.0000 | 4.6919 |
| 300 | 1.0832 | −0.0832 | 0.7838 | 0.1925 | 0.4590 | 0.5784 | 1.0000 | 4.6411 |
| 400 | 1.0566 | −0.0566 | 0.7834 | 0.1946 | 0.4582 | 0.5466 | 1.0000 | 4.6906 |
| 499 | 1.0099 | −0.0099 | 0.7716 | 0.1875 | 0.4669 | 0.5201 | 1.0000 | 4.6495 |

读法（只陈述观察）：`L_u` 先上升（0.98 → 1.10，step 200）后回落（→ 1.01，step 499），但**始终没有低于
step 0**；`cos(z_s,z_u)` 单调下降（0.989 → 0.772）、overlap 下降（0.688 → 0.188）、JSD 上升
（0.075 → 0.467）——`z_u` 确实与 `z_s` 分离，主要来自 `A_u` 变尖锐（entropy 5.197 → 4.642）。

吞吐：Arm S compute 0.4259 / wall 0.5992 s/step（wall_total 327.3 s），Arm SU 0.4284 / 0.5962 s/step
（322.9 s）；peak GPU 39.98 GB / 40.02 GB。两 arm 均 `EXIT=0`，无 NaN / Inf / OOM / NCCL 超时。

## Reachability 三层分解

对每个 valid 样本（256/256 valid，无 invalid）：

- `C_current = cos(z_u, target_u)`
- `C_patch_max = max_i cos(normalize(h_i), target_u)`
- `C_span = ||V_r^T V_r target_u||`（fp32 SVD，`H` 行空间，rank 由稳定阈值确定）——只是**任意带符号线性组合**
  的宽松上界，不代表 positive softmax pooling 能达到
- `C_softmax_oracle`：**approximate softmax-mixture oracle**（只优化自由 logits，Adam 100 步，从当前
  Unsaid logits 初始化）
- `optimization_gap = C_oracle − C_current`，`positive_mixture_gap = C_span − C_oracle`

### first_sentence（primary，mean / median / P10 / P90）

| step | Arm | L_u(eval) | C_current ↑ | C_patch_max | C_oracle | C_span | span rank(med) | opt gap | pos gap | cos(S,U) ↓ | overlap ↓ | JSD ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 共享 initial | 0.9841 | +0.0159 (0.013/−0.003/0.039) | 0.197 | 0.202 | 0.816 | 196 | 0.186 | 0.614 | 0.991 | 0.689 | 0.075 |
| 50 | S | 1.0027 | −0.0027 | 0.189 | 0.199 | 0.814 | 196 | 0.202 | 0.615 | 0.990 | 0.680 | 0.080 |
| 50 | SU | 1.0011 | −0.0011 | 0.192 | 0.199 | 0.814 | 196 | 0.200 | 0.615 | 0.990 | 0.677 | 0.081 |
| 100 | S | 1.0192 | −0.0192 | 0.207 | 0.208 | 0.811 | 196 | 0.228 | 0.603 | 0.972 | 0.548 | 0.154 |
| 100 | SU | 1.0148 | −0.0148 | 0.208 | 0.212 | 0.811 | 196 | 0.227 | 0.599 | 0.973 | 0.552 | 0.152 |
| 200 | S | 1.0679 | −0.0679 | 0.337 | 0.348 | 0.837 | 196 | 0.416 | 0.489 | 0.831 | 0.286 | 0.363 |
| 200 | SU | 1.0662 | −0.0662 | 0.326 | 0.337 | 0.830 | 196 | 0.403 | 0.493 | 0.828 | 0.284 | 0.366 |
| 300 | S | 1.0885 | −0.0885 | 0.353 | 0.350 | 0.843 | 196 | 0.438 | 0.494 | 0.802 | 0.266 | 0.383 |
| 300 | SU | 1.0738 | −0.0738 | 0.329 | 0.329 | 0.833 | 196 | 0.403 | 0.503 | 0.800 | 0.266 | 0.383 |
| 500 | S | 1.1004 | −0.1004 (0.111/−0.184/−0.010) | 0.386 | 0.380 | 0.855 | 196 | 0.481 | 0.475 | 0.780 | 0.250 | 0.398 |
| 500 | SU | 1.0658 | −0.0658 (0.075/−0.150/0.030) | 0.347 | 0.347 | 0.838 | 196 | 0.413 | 0.491 | 0.783 | 0.272 | 0.377 |

`cos(z_s, target_u)`（target 构造 sanity check）在 256/256 valid 样本上 median = `−0.0000`（initial 与
step500、两 arm 相同），说明 complement target 的正交性在整个训练过程中保持。

分位数（step 500，first_sentence）：

| 量 | Arm S mean/median/P10/P90 | Arm SU mean/median/P10/P90 |
| --- | --- | --- |
| `C_current` | −0.1004 / −0.1111 / −0.1836 / −0.0102 | −0.0658 / −0.0749 / −0.1502 / +0.0297 |
| `C_patch_max` | 0.386 / 0.393 / 0.234 / 0.532 | 0.347 / 0.356 / 0.195 / 0.486 |
| `C_span` | 0.855 / 0.860 / 0.814 / 0.893 | 0.838 / 0.843 / 0.782 / 0.884 |
| `cos(z_s,z_u)` | 0.780 / 0.801 / 0.634 / 0.887 | 0.783 / 0.804 / 0.638 / 0.887 |
| residual norm | 0.676 / 0.679 / 0.588 / 0.761 | 0.549 / 0.544 / 0.465 / 0.631 |
| span rank | 196 / 196 / 196 / 196 | 196 / 196 / 196 / 196 |

### fixed_sparse / full_dense（step 0 / 100 / 300 / 500）

| 变体 | step | Arm | C_current | C_patch | C_oracle | C_span | cos(S,U) | overlap | JSD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_sparse | 0 | 共享 | +0.0148 | 0.196 | 0.201 | 0.816 | 0.991 | 0.688 | 0.076 |
| fixed_sparse | 100 | S | −0.0371 | 0.186 | 0.187 | 0.805 | 0.953 | 0.477 | 0.207 |
| fixed_sparse | 100 | SU | −0.0325 | 0.187 | 0.190 | 0.805 | 0.955 | 0.484 | 0.202 |
| fixed_sparse | 300 | S | −0.1217 | 0.271 | 0.279 | 0.820 | 0.752 | 0.221 | 0.432 |
| fixed_sparse | 300 | SU | −0.0903 | 0.251 | 0.263 | 0.807 | 0.757 | 0.224 | 0.429 |
| fixed_sparse | 500 | S | −0.1392 | 0.273 | 0.281 | 0.822 | 0.717 | 0.209 | 0.445 |
| fixed_sparse | 500 | SU | −0.0646 | 0.242 | 0.257 | 0.798 | 0.743 | 0.230 | 0.424 |
| full_dense | 0 | 共享 | +0.0160 | 0.197 | 0.202 | 0.816 | 0.990 | 0.689 | 0.076 |
| full_dense | 100 | S | −0.0427 | 0.179 | 0.180 | 0.803 | 0.948 | 0.458 | 0.222 |
| full_dense | 100 | SU | −0.0380 | 0.180 | 0.184 | 0.803 | 0.950 | 0.465 | 0.216 |
| full_dense | 300 | S | −0.1332 | 0.236 | 0.233 | 0.810 | 0.736 | 0.206 | 0.448 |
| full_dense | 300 | SU | −0.0956 | 0.218 | 0.217 | 0.797 | 0.744 | 0.209 | 0.445 |
| full_dense | 500 | S | −0.1508 | 0.223 | 0.215 | 0.807 | 0.693 | 0.193 | 0.462 |
| full_dense | 500 | SU | −0.0596 | 0.198 | 0.199 | 0.781 | 0.728 | 0.214 | 0.442 |

三变体结论一致：`C_span` 稳定在 0.78–0.86（rank 恒为 196 = 全部 patch 行独立），而 `C_current` 在两个 arm
里都随训练变负。

### Oracle 预算敏感度（是否只是没优化够）

同一 256 图 cohort 的前 64 个 valid 样本、first_sentence、lr 0.1：

| 模型 | oracle 100 步 | 500 步 | 2000 步 | C_patch_max | C_span |
| --- | --- | --- | --- | --- | --- |
| Arm S step500 | 0.3802 | 0.3819 | 0.3820 | 0.3800 | 0.8574 |
| Arm SU step500 | 0.3470 | 0.3488 | 0.3489 | 0.3439 | 0.8416 |

oracle 在 100 → 2000 步之间只变化 0.002，并且**停在 best single patch 附近**（0.382 ≈ 0.380）：
因此"`C_oracle` 低"不是优化预算问题，而是 positive softmax mixture 在**这批 patch 特征**上的表达限制。

## Canonical Validation（initial / Arm S step500 / Arm SU step500）

ShareGPT4V-1K（1,000 图，三变体，`similarity_chunk = 512`，manifest dataset SHA `5c5f0f4ee58d7b74…`）：

| 变体 | 模型 | I2T R@1/5/10 | T2I R@1/5/10 | Full Gap | Said Gap | Full RMG | Said RMG | Balancing Gain | Cond. Margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| first_sentence | initial | 0.5440/0.7740/0.8590 | 0.5140/0.7480/0.8140 | 0.7044 | 0.7008 | 0.5966 | 0.6829 | +0.0036 | 0.0001 |
| first_sentence | Arm S | 0.6840/0.9140/0.9590 | 0.6500/0.8820/0.9400 | 0.6788 | 0.6700 | 0.5509 | 0.5779 | +0.0088 | 0.1184 |
| first_sentence | Arm SU | 0.6840/0.9080/0.9560 | 0.6500/0.8820/0.9370 | 0.6868 | 0.6712 | 0.5594 | 0.5771 | +0.0156 | 0.1179 |
| fixed_sparse | initial | 0.7460/0.9200/0.9530 | 0.7400/0.9180/0.9490 | 0.6776 | 0.7138 | 0.5614 | 0.6532 | −0.0362 | 0.0000 |
| fixed_sparse | Arm S | 0.9140/0.9870/0.9960 | 0.8840/0.9740/0.9850 | 0.6521 | 0.6793 | 0.5023 | 0.5295 | −0.0272 | 0.1956 |
| fixed_sparse | Arm SU | 0.9160/0.9850/0.9960 | 0.8880/0.9730/0.9850 | 0.6612 | 0.6783 | 0.5126 | 0.5308 | −0.0171 | 0.1931 |
| full_dense | initial | 0.7580/0.9200/0.9520 | 0.7760/0.9480/0.9730 | 0.6783 | 0.7410 | 0.5599 | 0.6596 | −0.0627 | 0.0000 |
| full_dense | Arm S | 0.9740/0.9980/0.9990 | 0.9600/0.9980/0.9990 | 0.6439 | 0.6835 | 0.5018 | 0.5316 | −0.0396 | 0.2174 |
| full_dense | Arm SU | 0.9710/0.9980/0.9990 | 0.9630/0.9980/0.9990 | 0.6535 | 0.6810 | 0.5132 | 0.5336 | −0.0275 | 0.2131 |

COCO val2017（5-caption，`similarity_chunk = 512`）：

| 模型 | I2T R@1/5/10 | T2I R@1/5/10 |
| --- | --- | --- |
| initial | 0.5170/0.7662/0.8428 | 0.3269/0.5776/0.6823 |
| Arm S | 0.5870/0.8132/0.8832 | 0.4030/0.6579/0.7586 |
| Arm SU | 0.5860/0.8100/0.8828 | 0.4003/0.6577/0.7575 |

三组评测 `rng_unchanged = True`。Arm S 与 Arm SU 在 500 步后的标准能力差异 ≤ 0.006 R@K（COCO I2T R@1
0.5870 vs 0.5860、T2I R@1 0.4030 vs 0.4003），`conditioning_margin` 0.1184 vs 0.1179（first_sentence）——
即 500 步内 **Unsaid 没有造成可测量的标准能力损伤**，也没有带来提升。这里只是检查损伤，不是 SOTA 结论。

## Said 冲突检查（共享 q_proj / k_proj）

| step | Arm | route top1 | evidence top1 | route margin | evidence margin |
| --- | --- | --- | --- | --- | --- |
| 0 | S = SU | 0.0039 | 0.7461 | −0.0186 | 8.1577 |
| 100 | S | 0.1719 | 0.9102 | 1.6021 | 12.7249 |
| 100 | SU | 0.1523 | 0.9062 | 1.5734 | 12.7844 |
| 300 | S | 0.9688 | 0.9727 | 19.1729 | 26.1501 |
| 300 | SU | 0.9688 | 0.9805 | 19.0453 | 26.0288 |
| 499 | S | 0.9766 | 0.9609 | 20.2895 | 26.8628 |
| 499 | SU | 0.9805 | 0.9648 | 20.0268 | 26.6900 |

固定 validation probe 上的 `conditioning_margin`（ShareGPT4V-1K 三变体）见上一节的 validation 表。

## 解释（只依据测量，按预先定义的规则）

- **规则 A（target geometry）不成立**：`C_span ≈ 0.78–0.86`、span rank 恒为 196，target 有很大一部分落在
  patch 行的带符号张成空间内。
- **规则 B 成立**：`C_span` 高但 `C_oracle` 低（0.20–0.38，且 oracle 预算敏感度为 0.002、停在 best patch 附近）
  → target 可以由带符号线性组合表示，但**positive softmax pooling 表达不出来**。
- **规则 C 部分成立**：`C_oracle`（0.35–0.38）明显高于 `C_current`（≤ 0.02，且随训练转负）
  → patch geometry 比 router 当前达到的水平更好；但同时 `optimization_gap` 巨大（0.41–0.48），
  说明 Router/优化也没有把已有的可达部分拿到。
- **规则 D 不成立**：probe 上 `C_current` 从 step 50 起即为负，两 arm 均未向 oracle 收敛；训练 batch
  上的 `L_u` 只在 step 200 → 499 之间从 1.1012 回落到 1.0099（仍未低于 step 0 的 0.9838）。

六个问题的测量答案（详见上面的表）：

1. `loss_unsaid` 是否稳定下降：**没有**。训练 batch 上先升后降，probe 上 `L_u(eval)` 单调变差。
2. `cos(z_u, target)` 是否持续上升：**没有**。训练 batch 上 −0.101 → −0.010（部分回升），probe 上 +0.016 → −0.066。
3. `cos(z_s, z_u)` 是否持续下降：**是**。0.989 → 0.772（训练 batch）/ 0.991 → 0.783（probe）。
4. Arm SU 是否明显破坏 Global / Said / route / evidence：**没有观察到明显破坏**（见" Said 冲突检查"与
   validation 表）。
5. `C_span` 有多高：**0.78–0.86**（rank 196/196）。
6. `C_softmax_oracle` 有多高：**0.20–0.38**（≈ best single patch）。
7. 瓶颈归属：**主要是 positive-mixture 表达力**（`C_span − C_oracle ≈ 0.47–0.59`），其次才是 router 优化
   （`C_oracle − C_current ≈ 0.41–0.48`）；target geometry 不是主要瓶颈。

## Reachability 加固（本轮修正，不改方法、不重训）

本轮只修 `eval/unsaid_mechanism_probe.py` 的诊断，未改模型 / loss / Said / Unsaid / validation / dataset，
也未重新训练；用现有 Arm S / Arm SU 的 initial / 100 / 300 / 500 checkpoint 重新离线评测
（`outputs/unsaid_mechanism_v2/`，256 图 cohort，oracle 与 cone 都用同一 64 个 valid 样本）。

### 1. oracle subset 统计修正

所有 gap 现在只在同一个 oracle subset 上计算：

```text
router_gap (== optimization_gap) = mean(C_oracle - C_current[selected])
oracle_optimization_gap          = mean(C_cone   - C_oracle)
positive_mixture_gap             = mean(C_span[selected] - C_oracle)
positive_constraint_gap          = mean(C_span[selected] - C_cone)
```

并额外输出 `oracle_subset_current` / `oracle_subset_patch_max` / `oracle_subset_span`
（mean / median / P10 / P90）；全 256 样本的 `c_current` / `c_patch_max` / `c_span` 仍单独报告，两者不再混算。
`oracle_subset_valid_indices` 记录具体样本下标。单元测试
`test_oracle_gaps_use_only_the_oracle_subset` 用 2/4 子集验证每条 gap 只由同一 subset 复现，
并保留一条"旧定义会给出不同数"的回归断言。

### 2. 标准 numerical rank tolerance

tolerance 由 `sigma_max * max(N,D) * eps * 1e-6` 改为标准基准 `sigma_max * max(N,D) * eps`
（`span_tolerance()`；`tol_factor` 可调）。离线敏感度（0.1× / 1× / 10×，first_sentence）：

| Arm / step | rank 均值 0.1× / 1× / 10× | C_span 均值 0.1× / 1× / 10× | rank 全样本一致比例 | C_span 相对跨度 | stable |
| --- | --- | --- | --- | --- | --- |
| S initial | 196.0 / 196.0 / 195.49 | 0.81646 / 0.81646 / 0.81608 | 0.797 | 4.65e-4 | False |
| S step100 | 196.0 / 196.0 / 195.57 | 0.81129 / 0.81129 / 0.81097 | 0.820 | 3.95e-4 | False |
| S step300 | 196.0 / 196.0 / 195.59 | 0.84337 / 0.84337 / 0.84299 | 0.840 | 4.49e-4 | False |
| S step500 | 196.0 / 196.0 / 195.60 | 0.85468 / 0.85468 / 0.85439 | 0.848 | 3.39e-4 | False |
| SU step500 | 196.0 / 196.0 / 195.67 | 0.83839 / 0.83839 / 0.83818 | 0.852 | 2.52e-4 | False |

判据是本轮固定下来的（`rank_identical_fraction == 1.0` 且 `C_span` 相对跨度 < 1e-3）：由于 10× tolerance
下平均每个样本会多剔除约 0.4 个奇异值，**严格按判据标记为 not stable**；但 `C_span` 的相对变化
≤ 4.7e-4（< 0.05%），0.1× 与 1× 完全一致，因此**结论层面稳定**。没有为了好看挑选 tolerance。

### 3. 非负锥（exact cone）可达性

`nonnegative_cone_reachability(H, target)`：`a* = argmin_{a>=0} ||H^T a - target||²`，
`p = H^T a*`，`C_cone = cos(normalize(p), target)`；`||p|| <= 1e-8` 时 `C_cone = 0` 且状态记为 `zero`。
因为 `z_u`（softmax 混合后归一化）落在同一个非负锥内，`C_cone` 是 positive-mixture 方向可达性的
**松散但精确求解的参考上界**（投影性质 `cos(p*, t) = ||p*|| >= cos(p, t)` 对锥内任意 `p` 成立）。

环境里没有 `scipy`（未安装，且本轮不修改冻结的环境文件），因此用 Lawson–Hanson(1974) active-set 算法
在 numpy/float64 上实现等价 NNLS（`nnls()`），并用 KKT 条件与已知构型做单元测试
（`test_nnls_satisfies_kkt_conditions_and_is_deterministic`）。

### 4. 同 subset 的最终分解（first_sentence，n=64/256）

| step | Arm | C_current(sub) | C_patch_max | C_softmax_oracle | C_cone | C_span(sub) | router_gap | oracle_opt_gap | pos_constraint_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 共享 | +0.0184 | 0.193 | 0.202 | 0.203 | 0.820 | 0.1840 | 0.0006 | 0.6168 |
| 100 | S | −0.0215 | 0.201 | 0.208 | 0.209 | 0.812 | 0.2300 | 0.0007 | 0.6033 |
| 100 | SU | −0.0169 | 0.205 | 0.212 | 0.213 | 0.813 | 0.2289 | 0.0007 | 0.6005 |
| 300 | S | −0.0905 | 0.348 | 0.350 | 0.351 | 0.846 | 0.4401 | 0.0016 | 0.4950 |
| 300 | SU | −0.0773 | 0.329 | 0.329 | 0.331 | 0.837 | 0.4066 | 0.0015 | 0.5061 |
| 500 | S | −0.1030 | 0.380 | 0.380 | 0.382 | 0.857 | 0.4831 | 0.0019 | 0.4754 |
| 500 | SU | −0.0667 | 0.344 | 0.347 | 0.349 | 0.842 | 0.4137 | 0.0019 | 0.4927 |

`C_cone` 与 `C_softmax_oracle` 几乎相同（差 0.001–0.002），且都停在 best single patch 附近 →
**违约的是非负约束本身，而不是 softmax/simplex 参数化**。

固定 subset 判据：`C_current(sub) = 0.0184/0.0166/−0.0024/0.0415`（mean/median/P10/P90，step 0）。

### 5. matched causal delta（step 500，SU − S）

| caption | ΔC_current（cohort / subset） | ΔC_patch | ΔC_oracle | ΔC_cone | ΔC_span | Full Gap SU−S | Said Gap SU−S | Cond. SU−S | I2T R@1 SU−S | T2I R@1 SU−S |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| first_sentence | **+0.0346 / +0.0363** | −0.0361 | −0.0331 | −0.0331 | −0.0158 | +0.0080 | +0.0013 | −0.0006 | +0.0000 | +0.0000 |
| fixed_sparse | **+0.0745 / +0.0716** | −0.0294 | −0.0239 | −0.0246 | −0.0231 | +0.0092 | −0.0009 | −0.0025 | +0.0020 | +0.0040 |
| full_dense | **+0.0911 / +0.0890** | −0.0231 | −0.0153 | −0.0163 | −0.0261 | — | — | — | — | — |
| COCO（SU−S） | — | — | — | — | — | — | — | — | −0.0010 | −0.0027 |

即：三个 caption 下 SU 的 `C_current` 都高于（更接近 0）matched S，而同一批 checkpoint 的
patch/oracle/cone/span 几何量反而略低——所以这部分提升不能由几何漂移解释，是 `L_u` 的对齐压力。

### 6. 本轮结论（只回答四个问题）

- **A. `L_u` 相对 matched Said-only 是否产生 target-alignment pressure？** 是，但幅度小且绝对水平仍为负：
  `ΔC_current = +0.035 / +0.072 / +0.089`（first_sentence / fixed_sparse / full_dense，step 500），
  同时几何量（patch/oracle/cone/span）反向变化 −0.015…−0.036。Said-only 对照下 `C_current` 随训练变负，
  说明这个"压力"只是减缓恶化，并未把 `z_u` 拉向 target。
- **B. 标准 tolerance 下 `C_span` 是否仍然高？** 是：0.80–0.86（rank 中位数 196/196），
  0.1×/1×/10× tolerance 下 `C_span` 相对变化 ≤ 4.7e-4。
- **C. exact `C_cone` 更接近谁？** 接近 **`C_softmax_oracle`**（差 0.001–0.002），远离 `C_span`
  （差 0.48–0.62）。softmax oracle 已经在非负锥最优解附近，simplex/归一化不是限制。
- **D. 瓶颈归属（最终）**：**positive / non-negative mixture capacity** 是主瓶颈
  （`positive_constraint_gap` 0.48–0.62）；**oracle/router optimization** 是次瓶颈
  （`router_gap` 0.18–0.48，且 `C_current` 随训练变负）；**target geometry 不是瓶颈**
  （`C_span` 高、跨 tolerance 稳定、`cos(target, z_s) ≈ 0`）。


## 未做的事

没有实现 prototype bank、text semantic retrieval、hidden caption supervision、concept bank、新 learnable
module、新 Unsaid projection；没有修改 Said / Global / caption sampling / validation protocol / dashboard /
standard inference；没有 sweep `lambda_unsaid` 或 `tau_unsaid`；没有做 3-epoch 正式训练；没有进入 2.8C。
下一步策略选择留待 Review。
