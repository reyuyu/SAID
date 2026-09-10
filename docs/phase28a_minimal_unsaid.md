# Phase 2.8A：最小 Unsaid V1 核心

本阶段在**完全保留 Global + Said** 的前提下，加入第一个最小的 Unsaid 学习分支。这是"核心数学路径 +
backward compatibility"阶段：不做正式训练、不调 Said、不加 prototype bank、不加任何可学习模块。

## 目标与边界

- **不改**：`L_global` 定义、`L_said` 定义、route / evidence loss、pairwise routing、`q_proj` / `k_proj`
  参数、`tau_said`、`encode_image()` / `encode_text()`、标准检索推理、validation protocol、dashboard、
  ShareGPT4V caption sampling（仍是 random prefix）。
- **不加**：新的视觉 backbone / projection、prototype bank、spatial grounding / entropy /
  orthogonality / reconstruction 目标。Unsaid 分支**没有任何参数**。
- Unsaid 只是把已经存在的量重新组合：Said 的 raw score `s`、同一套 patch features `H`、`z_G`、`z_S`。

## 数学定义

对当前正样本 `(I_i, C_i)`，patch features `H = {h_1..h_N}` 就是 Said 使用的同一套 router input
（`said_feature_source=residual` 时 Unsaid 同样用 residual）。

```text
s_i        = q^T k_i                      q = normalize(W_q t), k_i = normalize(W_k h_i)   （复用 Said）
A^S_i      = softmax(s_i / tau_said)      （Said，未改动）
A^U_i      = softmax(-s_i / tau_unsaid)   （Unsaid anti-routing，tau_unsaid 默认 0.07）
z_U        = normalize(sum_i A^U_i h_i)   （与 Said 相同的 patch features）
```

**禁止 `A_unsaid = 1 - A_said`**：`A_said` 是归一化分布，`1 - A_said` 的和是 `N - 1`（不是 1），
近似常数，不能作为 complement selector。单元测试同时验证"等于 `softmax(-s/tau)`"与"不等于 `1 - A_said`"。

互补视觉目标（两条支路都 stop-gradient）：

```text
g      = sg(normalize(z_G))
s      = sg(normalize(z_S))
r_U    = g - (g^T s) s
n_U    = ||r_U||_2
valid  = n_U > unsaid_residual_eps        （默认 1e-4）
target = sg(r_U / n_U)                    （invalid 样本不除零，直接置 0）
```

损失（只用 valid 样本）：

```text
L_U     = mean_valid[ 1 - cos(z_U, target_U) ]
L_total = lambda_global * L_global + lambda_said * L_said + lambda_unsaid * L_U
```

默认 `lambda_unsaid = 0.0`；正式候选值 `lambda_unsaid = 0.25`。

**只用 own caption**：`z_{i,j} = Said(I_i, C_j)` 的 pairwise 逻辑完全保留，但 Minimal Unsaid V1 只对真实
正样本 `(I_i, C_i)` 计算 `A^U_i` / `z_U_i` / `target_U_i` / `L_U`，**不构造** `Unsaid(I_i, C_j)`，
也不加 B×B 的 Unsaid pairwise loss（测试 `test_unsaid_adds_no_pairwise_routing` 断言 `route_pairwise`
在一次 forward 中只被调用一次）。

## Router API

`SaidRouter.forward_with_details(text_feature, patch_features)` 返回
`{'scores': s, 'attention': A_s', 'said': z_s'}`：`s` 是未除温度的 cosine score，attention / Said 特征
与 `forward()` 用**完全相同的算子与顺序**（`s = q^T k`，`A = softmax(s / tau_said)`），因此两者逐位相同
（测试 `test_enabled_and_disabled_said_attention_are_bit_identical`）。

默认旧调用方式不变：

```python
if lambda_unsaid == 0:      # 完全走原 Said-only 路径
    A_own, z_s_own = self.said_router(t, patch_features)
else:                       # 才额外返回 Unsaid 需要的 raw score（不重复投影 q/k）
    details = self.said_router.forward_with_details(t, patch_features)
```

`lambda_unsaid == 0` 不是"先算再乘 0"：Unsaid 分支的函数根本不会被调用（测试用 monkeypatch 让
`complement_attention` / `forward_with_details` 抛错来证明这一点）。

## Backward compatibility（第一 Gate）

| 检查项 | 结果 | 证据 |
| --- | --- | --- |
| `loss_global` | 逐位相同 | `test_lambda_zero_reproduces_the_said_only_objective_exactly`（测试内独立重写 Said-only 前向，`torch.equal`） |
| `loss_said` | 逐位相同 | 同上 |
| `loss_route` / `loss_evidence` | 逐位相同 | 同上 |
| `loss_total` | 等于 `lambda_global*L_G + lambda_said*L_S` | 同上 |
| route / evidence accuracy、margin | 逐位相同 | 同上 |
| Said attention 统计（entropy / effective patches / max / min / feature norm） | 逐位相同 | 同上 |
| `encode_image` / `encode_text` / `encode_router_input` | 未改动 | `test_standard_inference_paths_are_untouched` |
| 参数数量 | 不变（`said_router` 仍只有 `q_proj` + `k_proj`） | `test_no_new_parameters_or_state_dict_keys` |
| state_dict learnable keys | 不变，无任何 `unsaid` 键 | 同上 |
| 旧 Said-only checkpoint | `strict=True` 加载成功，missing / unexpected 均为空 | `test_legacy_said_only_checkpoint_loads_with_empty_key_diff` |
| Unsaid 路径是否真的被跳过 | 是（函数调用被 monkeypatch 拦截即报错） | `test_lambda_zero_skips_the_unsaid_path_entirely` |

## CLI

`train/train_salu.py` 新增（旧参数默认值不变，旧命令不加任何新 flag 仍是 Said-only）：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--lambda_unsaid` | `0.0` | Unsaid 互补损失权重；0 = 完全跳过该分支 |
| `--tau_unsaid` | `0.07` | Unsaid anti-routing softmax 温度 |
| `--unsaid_residual_eps` | `1e-4` | residual 范数低于该阈值视为 invalid |

三者随 `vars(args)` 一起写入 checkpoint 的 `args`。

## 训练日志监控

`lambda_unsaid > 0` 时记录：`loss_unsaid`、`unsaid_valid_ratio`、`unsaid_residual_norm_mean`、
`unsaid_target_cosine`、`said_unsaid_cosine`、`unsaid_attention_entropy`、
`unsaid_effective_patch_count`（= `exp(entropy)`）、`unsaid_attention_max`、
`said_unsaid_attention_overlap`、`said_unsaid_attention_jsd`、`unsaid_enabled = true`。

`overlap = sum_i min(A^S_i, A^U_i)`（batch mean，范围 `[0, 1]`）；JSD 是 `A^S || A^U` 的
Jensen-Shannon 散度（nats）。

`lambda_unsaid == 0` 时记录 `unsaid_enabled = false`、`loss_unsaid = 0`，其余 Unsaid 诊断一律写 `null`
（JSON `None`），避免"跳过"被误读成真实测量值。

**这些量只做监控**：`overlap`、entropy、`said_unsaid_cosine`、JSD 都不进入任何 loss。

## 数值稳定性

- Unsaid 分支内部（softmax、互补目标、cosine 损失、overlap/JSD/entropy）用 **fp32** 计算；
  Global / Said 的数值路径未被改动。
- `n_U` 用 `clamp_min(eps)` 后再除，invalid 样本的 target 直接置 0：不会出现 0/0、NaN 或 Inf。
- 全 batch invalid 时（例如 `z_G == z_S`），`L_U = 0 / 1 = 0`，仍是合法的 autograd 图，backward 正常。
- 掩码均值都用 `weights.sum().clamp_min(1)`，`masked_mean` 同理会返回有限 0。
- **梯度方向**：target 支路 stop-gradient；`z_U` 支路可反传 → 梯度进入 Said Router 的 `q_proj` /
  `k_proj` 以及产生 patch features 的视觉骨干；`z_G` 只喂 target，因此不接收 `L_U` 的梯度。

## 单元测试（`tests/test_unsaid_core.py`，18 个）

| 组 | 测试 | 覆盖点 |
| --- | --- | --- |
| A | `test_anti_routing_puts_mass_on_low_scores` | `s=[高,中,低]` → `A_said` 高 score 权重最大、`A_unsaid` 低 score 权重最大；和为 1；有限；温度更小更尖锐 |
| B | `test_unsaid_is_negative_softmax_not_one_minus_said` | 与 `softmax(-s/tau)` 逐位相等；`1 - A_said` 的和是 N−1，数值上明显不同 |
| C | `test_complement_target_is_unit_norm_and_orthogonal_to_said` | `cos(target, z_S) ≈ 0`（<1e-5）、`||target|| ≈ 1`、residual 范数与公式一致 |
| D | `test_target_branch_is_stop_gradient_and_prediction_is_differentiable` / `test_unsaid_gradients_reach_q_proj_k_proj_and_the_visual_backbone` | target / residual 无 grad；`q_proj`、`k_proj`、视觉 patch 投影梯度有限且非零；`z_G` 投影无梯度 |
| E | `test_degenerate_residual_is_invalid_and_loss_is_zero_without_nan` / `test_mixed_validity_averages_only_over_valid_samples` | `z_G == z_S` → 全部 invalid、`L_U == 0`、无 NaN/Inf、backward 合法；混合 batch 只对 valid 求均值 |
| F | `test_attention_overlap_bounds_and_separation` | `0 <= overlap <= 1`；同分布 ≈ 1；分离分布更小；entropy / effective patch / JSD 边界 |
| G | `test_lambda_zero_skips_the_unsaid_path_entirely` / `test_lambda_zero_reproduces_the_said_only_objective_exactly` / `test_enabled_and_disabled_said_attention_are_bit_identical` / `test_unsaid_adds_no_pairwise_routing` / `test_loss_total_includes_lambda_unsaid_only_when_enabled` | backward compatibility Gate（见上表） |
| H | `test_no_new_parameters_or_state_dict_keys` / `test_legacy_said_only_checkpoint_loads_with_empty_key_diff` | 无新参数 / 无新 key；`runs_salu/phase22/salu_said_only_last.pt` strict 加载成功 |
| 额外 | `test_unsaid_attention_is_not_identical_to_said_attention` / `test_unsaid_is_finite_under_bf16_autocast` | `A_u ≠ A_s`（overlap < 1、JSD > 0）；bf16 autocast 下所有 Unsaid 诊断有限 |

## 5-step Smoke（4 × A800）

配置：完整 ShareGPT4V（1,245,901 条）+ Full Data Gate、ViT-B/16、`--batch_size 256`（global 1,024）、
bf16、seed 0、`--log_every 1`、**关闭全部 validation / COCO**、`--max_steps 5`。
Run A：`--lambda_unsaid 0.0`；Run B：`--lambda_unsaid 0.25 --tau_unsaid 0.07 --unsaid_residual_eps 1e-4`。

Run A：`--lambda_unsaid 0.0`；Run B：`--lambda_unsaid 0.25 --tau_unsaid 0.07 --unsaid_residual_eps 1e-4`。
两个 run 同 seed、同数据、同 sampler、同 caption stream。5 step 只能确认"能跑通"，不能判断优劣。

### 数据流一致性

| 检查 | 结果 |
| --- | --- |
| 5 步 `batch_image_sha256`（A vs B） | 逐位相同（`a85956f9cfc37803, 60137619844eb058, 8123ef2b79bee5c4, 881e0d19d210f316, c9e162b98f106302`） |
| 5 步 `batch_caption_sha256`（A vs B） | 逐位相同（`65543d645877a5b8, 30d1ad2aecc2433b, cafce3412452cae9, a4d772fc78c1d8d1, f38c6a85837504f6`） |
| rank0 `initial_state_sha256` | A = B = `4621b8d027f482ea…` |
| rank0 `sampler_order_sha256` | A = B = `953402b3c3d7322e…` |
| rank0 `caption_stream_sha256` | A = B = `de506eb21f6c0dbd…` |

**真实规模上的 backward compatibility**：step 0 两个 run 的 `loss_global = 0.659137`、
`loss_said = 3.333893` 完全相同 —— 开启 Unsaid 不扰动 Global / Said 的数值。step ≥ 1 因为 Run B 多了一项
损失、参数更新不同而自然分叉（A step1 `loss_global 0.554174` vs B `0.553487`），这属于预期行为。

### Run A（`lambda_unsaid = 0.0`）

| 项目 | 值 |
| --- | --- |
| steps | 5 |
| `compute_sec_per_step` | 1.4030 |
| `wall_sec_per_step` | 1.8769 |
| `wall_sec_total` | 18.7 s |
| 每步 `loss_total` | 3.993031 / 3.757577 / 3.757620 / 3.666068 / 3.693206 |
| 每步 `loss_unsaid` | 0.000000（`unsaid_enabled = false`，其余 Unsaid 诊断全为 `null`） |
| peak GPU memory | 38.85 → 39.98 GB |
| 错误 | 无 NaN / Inf / OOM / 超时，`EXIT=0` |

### Run B（`lambda_unsaid = 0.25`）

| 项目 | step 0 | step 4（last） |
| --- | --- | --- |
| `loss_global` | 0.659137 | 0.479914 |
| `loss_said` | 3.333893 | 3.217614 |
| `loss_unsaid` | 0.983787 | 0.989479 |
| `loss_total` | 4.238977 | 3.944898 |
| `unsaid_valid_ratio` | 1.000000 | 1.000000 |
| `unsaid_residual_norm_mean` | 0.778913 | 0.774698 |
| `unsaid_target_cosine` | 0.016213 | 0.010521 |
| `said_unsaid_cosine` | 0.989118 | 0.989873 |
| `said_attention_entropy` | 5.198310 | 5.197982 |
| `unsaid_attention_entropy` | 5.197235 | 5.197579 |
| `said_effective_patch_count` | 181.020447 | 180.964722 |
| `unsaid_effective_patch_count` | 180.831299 | 180.889374 |
| `said_unsaid_attention_overlap` | 0.687987 | 0.688507 |
| `said_unsaid_attention_jsd` | 0.074751 | 0.074633 |
| `route_top1_acc` | 0.003906 | 0.007812 |
| `evidence_top1_acc` | 0.746094 | 0.773438 |
| `route_margin` | −0.018583 | −0.005382 |
| `evidence_margin` | 8.157661 | 9.127837 |
| `said_attention_max` | 0.013372 | 0.013479 |
| `pair_gap_full` / `pair_gap_said` / `balancing_gain` | 0.667377 / 0.716355 / −0.048977 | 0.669061 / 0.734335 / −0.065275 |
| `peak_gpu_mem_gb` | 38.896671 | 40.023647 |

| 检查 | 结果 |
| --- | --- |
| `loss_unsaid` 有限 | 是（5 步 0.983787–0.989479） |
| 所有 Unsaid 诊断有限 | 是（全 log 无 NaN / Inf；脚本逐字段扫描 `math.isfinite`） |
| `A_unsaid` 与 `A_said` 不同 | 是（JSD ≈ 0.0747 > 0，overlap ≈ 0.688 < 1） |
| `target_U` 与 `z_S` 余弦 | 由 C 组测试断言 `< 1e-5`；smoke 中 `unsaid_residual_norm_mean ≈ 0.775 > eps`，valid ratio = 1 |
| `z_U` 与 `target_U` 余弦 | ≈ 0.0105–0.0162（**只报告观察值**：初始阶段两者只是有限、可反传，不构成质量结论） |
| DDP / OOM / 超时 | 无，`EXIT=0` |
| throughput | A `compute 1.4030 / wall 1.8769` s/step，B `compute 1.4190 / wall 1.8318` s/step |
| peak GPU | A 39.98 GB / B 40.02 GB（单卡 80 GB） |

参考量：ViT-B/16 有 196 个 patch，均匀注意力的熵上界为 `ln 196 = 5.278`；smoke 中 Said / Unsaid 熵都在
5.198 附近，与 overlap ≈ 0.688 是同一现象的两种读数（此处只记录数值）。

**关于步时**：5 步平均包含首步 DDP / allocator warmup，不能与更长 run 的步时直接比较
（Phase 2.7B 的 20 步 smoke 为 compute 0.7975 / wall 0.9715 s/step，Phase 2.7C 的 21 步续训为
0.687466 / 0.870126 s/step）。A 与 B 的差异在测量噪声量级内，本阶段不对性能下结论。


## 未做的事

- 没有做长训练，没有根据 5 step 调 `lambda_unsaid`，没有判断模型优劣。
- 没有改 Said（结构 / 目标 / 超参）、没有改 validation protocol 与 dashboard、没有改 caption sampling。
- 没有引入第二套 caption identification 目标；Unsaid 只有 own-caption 支路。
