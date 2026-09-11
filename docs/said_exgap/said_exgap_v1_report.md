# SAID-ExGAP v1 实施报告（Explanatory-Gap Guided Masked Representation Learning）

本报告只写实测值。未运行/未通过的项目一律显式标注 **NOT RUN / FAIL**。

- 仓库：`/root/SAID-gap-completion`（HEAD `9b317c1`，分支 `codex/phase3.0a-gap-completion-core`）
- 目标模式：`--objective_mode said_exgap`（新增，独立于 `legacy` / `gap_completion`）
- 总损失：`L = lambda_S * L_S + lambda_ExGAP * L_ExGAP`，**没有** `L_G`（global-text CLIP loss），**没有** Phase 2.x Unsaid 分支，**没有** `C_U` 训练
- 未合并 main、未 force push、未 squash、未删除任何旧实验/报告/检查点

---

## 1. 修改文件清单

新增：

| 文件 | 行数 | 内容 |
| --- | --- | --- |
| `model/exgap.py` | 399 | ExGAP 全部数学（Global / Said / Mask / Masked-Unsaid / Gap / Loss）+ `ExGapModule`（image-only attention pooling）+ checkpoint 配置校验 |
| `tests/test_said_exgap.py` | 634 | 28 个新测试（Forward / Mask / Gap / Loss 单调性 / 梯度隔离 / 退化分支 / 目标模式校验） |
| `tools/exp_said_exgap_smoke.sh` | 47 | 6 臂 smoke（mean/attention × 20/100/500 步），Full Data Gate + 4×A800 |
| `docs/said_exgap/said_exgap_v1_report.md` | 本文件 | 交付报告 |

修改：

| 文件 | 变更 | 内容 |
| --- | --- | --- |
| `model/salu_model.py` | +377 / −5 | `OBJECTIVE_MODES` 增加 `said_exgap`；新增 `_forward_said_exgap`、`encode_said_exgap`、`_exgap_collapse_metrics`、`build_exgap_module`、`gradient_route_report`；`forward_train` 新增 ExGAP 超参与日志字段 |
| `train/train_salu.py` | +205 / −12 | `resolve_objective_mode`、`build_exgap_log_fields`、`objective_checkpoint_metadata`、`save_checkpoint` 的 `exgap_config`、`validate_resume_objective`、`validate_objective_args` 分派；`said_exgap` 训练循环与 CLI |
| `tests/test_gap_completion.py` | +5 / −1 | `test_c_gap_mode_rejects_an_unknown_objective_mode` 改为成员检查（`said_exgap` 是合法第三模式）并加一个非法模式用例 |
| `tools/phase30a_fixed_cohort_eval.py` | +14 / −6 | 评测器容忍 `exgap.*` 作为合法多余/缺失键（canonical CLS 检索不经过该头），并把 `absent_exgap_keys` 记入 JSON |

未改动（刻意）：`model/salu_modules.py`（既有 `SaidRouter` 一行未改）、`model/unsaid_core.py`、所有 Phase 3.0A / Phase 2.x 报告、检查点与已验证结果。`runs_salu/`、`runs_smartclip/` 仍为未跟踪（不入库）。

---

## 2. 数学 → 代码映射

`H = normalize(patch_features)`（`model/exgap.py:64`），Global/Said/Masked-Unsaid 全部读同一个 `H`，**没有** `CLS → global` / `patch → said` 双空间。

| 数学 | 代码 | 位置 |
| --- | --- | --- |
| `h_p = H_p / ||H_p||` | `normalize_patches` | `exgap.py:64` |
| `p_G = mean_p h_p`，`A^G_p = 1/N` | `mean_global_weights` | `exgap.py:72` |
| `A^G_p = softmax_p( (q_G W_Q)(h_p W_K)^T / √d )`，`q_G` 只来自图像 | `attention_global_weights` / `ExGapModule.forward` | `exgap.py:81`, `exgap.py:341` |
| `z_G = Norm(Σ_p A^G_p h_p)` | `compute_global_representation` | `exgap.py:98` |
| `ell^S_p = (q̂ · k̂_p) / τ_said`，`q̂ = Norm(t W_Q)`，`k̂_p = Norm(h_p W_K)` | `said_relevance_logits` | `exgap.py:121` |
| `r^S_p = sigmoid(ell^S_p)`；`A^S_p = r^S_p/(Σ_q r^S_q+ε)`；`z_S = Norm(Σ A^S_p h_p)` | `compute_said_representation` | `exgap.py:147` |
| `M_p = 1(r^S_p ≤ τ_M)`（detach，退化时保留最低相关性的 1 个 patch） | `compute_said_mask` | `exgap.py:174` |
| mean 模式 `z_U = Norm(Σ M_p h_p/(Σ M_p+ε))`；attention 模式 `A^U_p = M_p A^G_p/(Σ_q M_q A^G_q+ε)` | `compute_masked_unsaid_representation` | `exgap.py:219` |
| `gap_raw = [S_sc − S_gc]_+`（**没有**取绝对值）；`G_exp = sg(clamp(gap_raw/(1−S_gc+ε),0,1))` | `compute_explanatory_gap` | `exgap.py:241` |
| `L_ExGAP = mean_valid( G_exp · τ · softplus((S_uc − sg(S_gc))/τ) )` | `compute_exgap_loss` | `exgap.py:265` |
| `S_gc = cos(z_G,t)`、`S_sc = cos(z_S,t)`、`S_uc = cos(z_U, sg(t))` | `_forward_said_exgap` | `salu_model.py:1014` |
| `L = λ_S L_S + λ_ExGAP L_ExGAP` | `_forward_said_exgap` | `salu_model.py:1030` |
| checkpoint `exgap_config` 写入 / 恢复校验 | `checkpoint_config` / `validate_exgap_config` | `exgap.py:362`, `exgap.py:376` |
| CLI 前置校验分派 | `validate_objective_args` | `train_salu.py:904` |

Said 路由复用的就是既有 `SaidRouter.q_proj/k_proj`（`model/salu_modules.py:53-54`），**没有**新增第二个 router，也没有 prototype bank / decoder / relation head / teacher。

---

## 3. 梯度路径（实测）

`python probe_exgap_gradient_route.py --pool {mean,attention}`（CPU stub，真实模型函数；`d L_ExGAP|G=1` 是把 `G_exp` 强制置 1 的路径探针，用来在 gap 尚未激活时也能量出梯度去向）：

| 参数组 | d L_S | d L_ExGAP（本 batch 真实 G_exp） | d L_ExGAP (G=1) |
| --- | --- | --- | --- |
| `clip.patch_proj` | 4.49057 | 0.00395365 | 0.524021 |
| `clip.text_proj` | 4.37544 | 0 | 0 |
| `clip.global_proj` | 0 | 0 | 0 |
| `clip.logit_scale` | 1.50056 | 0 | 0 |
| `said_router` | 4.74713 | 0 | 0 |
| `exgap`（pooling，mean 模式） | 0 | 0 | 0 |

| 参数组 | d L_S | d L_ExGAP（真实） | d L_ExGAP (G=1) |
| --- | --- | --- | --- |
| `clip.patch_proj` | 6.96675 | 0.175961 | 0.521131 |
| `clip.text_proj` | 19.0031 | 0 | 0 |
| `clip.global_proj` | 0 | 0 | 0 |
| `clip.logit_scale` | 1.45155 | 0 | 0 |
| `said_router` | 23.9287 | 0 | 0 |
| `exgap`（pooling，attention 模式） | 0 | 0.000955156 | 0.00288473 |

结论（与设计契约一致）：

- `L_ExGAP` 的梯度只到 **掩码后仍然保留的 patch → 共享视觉主干（`clip.patch_proj`）**；attention 模式下额外到 `exgap` pooling 自己的参数（`A^G` 被用进 `z_U`），这是该模式唯一合法的额外去向。
- `L_ExGAP` 到 **text encoder 恒为 0**（`t_gap = t.detach()`）、到 **Said router 恒为 0**、到 **CLS/global 分支恒为 0**、到 `logit_scale` 恒为 0。因此 ExGAP 无法通过移动路由器/文本编码器来"缩小 gap"。
- `L_S` 才训练 `said_router` 与文本侧；两个目标的梯度路径完全不重叠（除共享的 `clip.patch_proj`）。

---

## 4. 测试

```
python -m pytest tests/ -q
427 passed, 2 skipped, 4 warnings in 29.81s
```

新增测试文件名：`tests/test_said_exgap.py`（28 个）：

```
test_forward_shapes_ranges_and_finiteness                     (Test 1 forward 正确性)
test_forward_works_for_both_pooling_modes
test_attention_pooling_starts_uniform_and_ignores_the_caption
test_said_relevance_is_alive_at_the_router_scale               (回归：掩码必须能触发)
test_full_model_mask_is_alive_end_to_end
test_mean_global_equals_plain_mean_pooling
test_mask_selects_patches_that_are_not_said                    (Test 2 mask 正确性)
test_masked_pooling_ignores_the_said_dominant_patches
test_mean_mode_masked_pooling_is_the_masked_mean
test_all_patches_masked_falls_back_and_never_produces_nan
test_mask_threshold_validation_and_extremes
test_explanatory_gap_values_and_normalisation                  (Test 3 explanatory gap)
test_explanatory_gap_is_detached_and_clamped
test_exgap_loss_is_monotone_in_the_violation                   (Test 4 loss 单调性)
test_exgap_loss_scales_with_the_gap_weight
test_exgap_loss_respects_the_valid_mask
test_exgap_only_backward_touches_only_the_masked_path          (Test 5 梯度隔离)
test_exgap_gradient_route_with_an_active_gap
test_attention_pooling_hits_a_dead_saddle_if_both_sides_are_zero
test_exgap_backward_in_attention_mode_does_not_touch_the_route
test_inference_api_does_not_leak_exgap_gradient_into_the_text_encode
test_gap_weight_cannot_be_used_to_shrink_the_gap
test_full_objective_moves_the_router_only_through_the_said_loss
test_mask_is_detached_from_the_relevance_graph
test_said_exgap_rejects_mixed_objectives                        (目标模式/CLI 校验)
test_encode_said_exgap_api
test_checkpoint_config_round_trip_and_validation
test_no_forbidden_ingredients_in_the_new_module
```

修改的既有测试：`tests/test_gap_completion.py::test_c_gap_mode_rejects_an_unknown_objective_mode`（`OBJECTIVE_MODES` 由 2 个变 3 个，改为成员断言 + 追加一个非法模式用例，未放宽原断言）。

---

## 5. Smoke（20 / 100 / 500 步，Full ShareGPT4V，4×A800，batch 256/GPU，seed 0）

命令：`bash tools/exp_said_exgap_smoke.sh`（6 臂：`mean`/`attention` × 20/100/500 步，串行）。
公共参数：`--objective_mode said_exgap --lambda_global 0 --lambda_unsaid 0 --lambda_said 1.0 --lambda-exgap 1.0 --exgap-mask-threshold 0.6 --exgap-temperature 0.05 --exgap-collapse-every 20 --said_loss_mode identifiable --said_feature_source residual --base_model B16 --lr_total_steps 3648 --warmup_length 200 --backbone_lr 1e-6 --head_lr 1e-4 --amp_dtype bf16`。
`FULL_DATA_GATE_PASS`、`steps_per_epoch 1216`、`COLLAPSE_COHORT (64, 3, 224, 224) from dataset indices 0-63` 三行均出现；6 臂全部 `exit=0`，日志 `/tmp/said_exgap_smoke.log`（444 行，130 条 LOG 记录）。

三臂（20/100/500）是同配置的独立运行，seed 与数据顺序相同；实测在 step 21 时三者数值完全一致（`global_pairwise_cos = 0.8046875`），之后因 bf16/非确定性 kernel 出现 ~1e-4 量级的运行间差异，因此跨臂比较只在约 4 位有效数字内成立。

| 指标 | mean 20 | mean 100 | mean 500 | attention 20 | attention 100 | attention 500 |
| --- | --- | --- | --- | --- | --- | --- |
| `route_top1_acc` | 0.0078 | 0.2266 | 0.9766 | 0.0078 | 0.2109 | 0.9844 |
| `evidence_top1_acc` | 0.8867 | 0.8945 | 0.9688 | 0.8906 | 0.8867 | 0.9648 |
| `S_sc − S_gc`（>0 即"解释性增益"成立） | **+0.00112** | **+0.00766** | **+0.03233** | **+0.00115** | **+0.00768** | **+0.04585** |
| `S_uc − S_gc`（掩码后应更低） | −0.00017 | −0.00930 | −0.12924 | −0.00014 | −0.00941 | −0.11728 |
| `explanatory_gap_positive_fraction` | 0.7266 | 0.9609 | 1.0000 | 0.7383 | 0.9570 | 1.0000 |
| `mask_keep_ratio` | 0.9922 | 0.8516 | 0.3594 | 0.9922 | 0.8516 | 0.3691 |
| `masked_patch_count_mean`（N=196） | 194.9 | 166.7 | 70.5 | 194.9 | 166.6 | 72.3 |
| `fallback_fraction` | 0 | 0 | 0.0039 | 0 | 0 | 0.0039 |
| `said_relevance_std` | 0.0963 | 0.1548 | 0.2035 | 0.0964 | 0.1547 | 0.2043 |
| `loss_said` | 3.0519 | 2.2536 | 0.0904 | 3.0507 | 2.2578 | 0.0914 |
| `loss_exgap` | 6.57e-05 | 2.89e-04 | 2.02e-04 | 6.69e-05 | 2.89e-04 | 3.34e-04 |
| `global_pairwise_cos`（最近一次 collapse 监控，step ≤ 481） | 0.8047 (step 21) | 0.7617 (step 101) | 0.7344 (step 481) | 0.8047 (step 21) | 0.7617 (step 101) | 0.7383 (step 481) |
| `global_pairwise_cos_max` | 0.9102 | 0.8906 | 0.9023 | 0.9102 | 0.8906 | 0.9023 |
| `global_attention_l1_deviation`（attention 模式） | 0（精确均匀） | — | — | 6.35e-07 | 4.25e-05 | **0.0537** |
| `global_attention_entropy`（ln 196 = 5.27811） | 5.27811 | 5.27811 | 5.27811 | 5.27811 | 5.27811 | 5.27194 |

读法（只写实测能得到的东西）：

1. **`S_sc > S_gc` 会成立，而且随步数稳定增长**：`+0.0004`（step 1）→ `+0.0011`（20）→ `+0.0077`（100）→ `+0.0323`（mean 500）/ `+0.0459`（attention 500），`explanatory_gap_positive_fraction` 同步升到 1.0。也就是说"当前 caption 下 Said 表示比 Global 更能解释这张图"这一前提在训练中真实出现，而不是靠构造。
2. **硬掩码真的开始工作**：`mask_keep_ratio` 从 0.996（step 1）降到 0.852（100）、0.359（mean 500）/ 0.369（attention 500）；`said_above_threshold_fraction` 从 0.0058 升到 0.64。`fallback_fraction = 0.0039`（约 0.4% 的样本出现"全部 patch 都被告知"的退化情形，已按契约回退为保留最低相关性 patch，无 NaN、无样本被丢弃）。
3. **ExGAP 的作用方向正确**：`S_uc` 被压到远低于 `S_gc`（−0.129 / −0.117），即由非 Said patch 构成的掩码表示不再复述同一句 caption；`unsaid_pairwise_cos`（0.691）低于 `said_pairwise_cos`（0.762），说明两路表示确实分化。
4. **没有塌缩**：`global_pairwise_cos` 0.816 → 0.734/0.738，始终远低于 Full-Base gap_completion 臂的 0.977；`global_pairwise_cos_max` ≤ 0.9023，未触发 `WARNING REPRESENTATION_COLLAPSE`（阈值 0.9 作用在均值上）。
5. **`loss_exgap` 数值很小（1e-4 量级）且在 500 步时略有回落，这不是健康指标，不能当作收敛证据。** 它的定义是 `G_exp · τ · softplus((S_uc − sg(S_gc))/τ)`：当 `S_uc − S_gc` 已经远低于 0（`−0.13`，`τ=0.05`）时 softplus 饱和到 0.07 附近，而 `G_exp = gap_raw/(1−S_gc)` 只有 0.04 量级，两者相乘自然得到 1e-4 量级。判断健康与否必须看第 1–4 条的几何量。
6. **两种 pooling 模式确实不同**：`attention` 臂的 `global_attention_l1_deviation` 从 0（init 精确均匀，与 mean 臂逐位同起点）单调升到 0.0537，熵由 5.27811 降到 5.27194，即 `A^G` 在 ExGAP 信号下真的离开了均匀解（这正是 `test_attention_pooling_hits_a_dead_saddle_if_both_sides_are_zero` 保护的机制）；500 步时 attention 臂的 `S_sc − S_gc`（0.0459）大于 mean 臂（0.0323）。

**未做（NOT RUN）**：COCO/ShareGPT4V 检索评测（canonical `encode_image`/`encode_text` 表）在本次交付中**没有**运行 —— smoke 阶段按要求只做 20/100/500 步核心指标，不做 3 epoch 完整训练，也不产出检索数字。因此本报告不对 ExGAP 的最终检索收益做任何断言。

---

## 6. git 状态（提交后实测）

分支：`codex/said-exgap-v1`（从 `codex/phase3.0a-gap-completion-core` 的 HEAD `9b317c1` 新建，**未合并 main**）。

```
$ git status --short
?? runs_smartclip/

$ git diff --stat
(空 —— 工作树干净)

$ git log -3 --oneline
8b7a546 feat: SAID-ExGAP v1 explanatory-gap guided masked representation learning
9b317c1 exp: reproduce SmartCLIP full-training baseline
1eba014 feat: decompose unsaid semantics into common mode and contrast
```

本次提交（`8b7a546`）的 `git show --stat`：

```
 docs/said_exgap/said_exgap_v1_report.md | 219 +++++++++++
 model/exgap.py                          | 399 ++++++++++++++++++++
 model/salu_model.py                     | 382 ++++++++++++++++++-
 tests/test_gap_completion.py            |   6 +-
 tests/test_said_exgap.py                | 634 ++++++++++++++++++++++++++++++++
 tools/exp_said_exgap_smoke.sh           |  47 +++
 tools/phase30a_fixed_cohort_eval.py     |  20 +-
 train/train_salu.py                     | 217 ++++++++++-
 8 files changed, 1900 insertions(+), 24 deletions(-)
```

对应的 `--numstat`（新增/删除行）：`model/exgap.py 399/0`、`tests/test_said_exgap.py 634/0`、`tools/exp_said_exgap_smoke.sh 47/0`、`docs/said_exgap/said_exgap_v1_report.md 219/0`（自引用文件，最后一次修订本文件时该数字会随之变化）、`model/salu_model.py 377/5`、`train/train_salu.py 205/12`、`tools/phase30a_fixed_cohort_eval.py 14/6`、`tests/test_gap_completion.py 5/1`。

`runs_smartclip/`、`runs_salu/` 为运行产物，保持未跟踪，不入库；检查点与原始日志未被提交。

---

## 7. 与 brief 的偏离（全部先实测、后修改，并已加回归测试）

1. **Said relevance 的尺度 `/√d` → `/τ_said`。**
   brief 写的是 `r^S_p = sigmoid((tW_Q)(h_pW_K)^T/√d)`。用默认初始化的投影实测该 logit 只有 `~1e-4` 量级：`said_relevance_std = 4e-5`、`r^S_p = 0.5 ± 4e-5` 恒小于 `τ_M = 0.6`，于是 `mask_keep_ratio ≡ 1.0`、`loss_exgap ≡ 0.0`（100 步全程），整个 ExGAP 目标静默失效。
   现改为复用既有 Said router 自己的 logit 尺度：`ell^S_p = (q̂·k̂_p)/τ_said`（`q̂ = Norm(tW_Q)`、`k̂ = Norm(h_pW_K)`，`τ_said = 0.07`），即"把 router 已经在优化的那个 logit 过一遍 sigmoid"，与 `SaidRouter.forward`/`route_pairwise` 完全同一个分数，只是 softmax → sigmoid。实测 `said_relevance_std` 从 `4e-5` 升到 `0.154`（100 步），掩码真正开始丢 patch。
2. **`ExGapModule` 的初始化不能两侧同时为 0。**
   原实现把 `w_query`/`w_key` 全部零初始化。零 logit 确实让 `A^G` 在 init 时**精确均匀**（这是必须的：attention 模式必须与 mean 模式从同一点出发），但双零是死鞍点——`d logit/d W_query ∝ k = 0` 且 `d logit/d W_key ∝ q = 0`，任何 ExGAP 信号都无法让 pooling 离开均匀解。现改为：query 投影零初始化（保证 logit 恒为 0、`A^G` 精确均匀），query token 与 `w_key` 小随机初始化（给出逃逸方向，实测 `d L/d W_query ≠ 0`）。新增测试 `test_attention_pooling_hits_a_dead_saddle_if_both_sides_are_zero` 同时固定"init 精确均匀"和"梯度非零"两条性质。
3. **`encode_said_exgap(return_details=True)` 的 `s_uc` 泄漏文本梯度。**
   训练路径用 `t_gap = t.detach()`，但推理 API 的 `s_uc` 用了活的 `t`。路径探针实测 `d s_uc / d clip.text_proj = 0.67`（训练路径为 0）。已改为 detach，并新增测试 `test_inference_api_does_not_leak_exgap_gradient_into_the_text_encoder`。
4. **pooling agent 的注册时机。** 原实现在第一次 forward 里惰性创建，此时 DDP 已经包装完毕，模块会留在 CPU（实测报错 `Expected all tensors to be on the same device ... cpu and cuda:1`）并且不进 DDP reducer（梯度静默不同步）。现改为训练脚本在 `DDP(...)` 之前调用 `salu.build_exgap_module()`，mean / attention 两臂的初始状态摘要在同一时刻生成，因此逐位可比。
5. **评测器容错。** `tools/phase30a_fixed_cohort_eval.py` 原本把任何 unexpected key 视为错误，ExGAP 检查点会因 `exgap.*` 直接报错。现把 `exgap.*` 与 `said_router.*` 同等对待（canonical CLS 检索不经过这些头，不可能影响测量），并把 `absent_exgap_keys` 写进结果 JSON。该改动不可能改变任何历史测量。
