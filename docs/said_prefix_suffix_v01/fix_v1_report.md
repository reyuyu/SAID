# S0-Suffix Correctness Fix v1 — 修复报告（**部分完成，验收未通过**）

> 本轮范围：定点修复 → 测试 → 生产入口短流程验收 → 提交与普通 push。
> **结论：修复已落地并显著推进，但验收未通过；两臂 500 步、COCO、Urban 一律未运行。**

- 问题版本：`74bf9c0`（WIP 分支 `codex/s0-suffix-v01-wip-unfinished`）
- 本轮提交：见本次 commit（父提交 `74bf9c0`，**未 force push**）
- 正式分支 `codex/s0-suffix-v01` 本轮**未移动**，仍在 `5676666`
- `formal_optimizer_updates = 0`

## 1. 实际验收结果（证据，非结论）

| 层 | 内容 | 结果 |
| --- | --- | --- |
| 1+2 | `pytest tests/test_said_prefix_suffix.py` | **29 passed / 19 failed**（修复前为 10 passed / 29 failed） |
| 3a | 真实 CLIP 小批（batch=16，2 步）`S0_SUFFIX_NATIVE` | **EXIT=1（失败）** |
| 3b | 真实 CLIP 小批（batch=16，2 步）`S0_SUFFIX_MASK` | **EXIT=1（失败）** |
| 4a | 生产入口 4 卡×256，2 步 `S0_SUFFIX_NATIVE` | **EXIT=1（失败）** |
| 4b | 生产入口 4 卡×256，2 步 `S0_SUFFIX_MASK` | **EXIT=1（失败）** |

四个临时运行目录均在 `runs_salu/DEBUG_NOT_FORMAL_*`，**不得作为任何正式训练的起点**。

## 2. 仍然失败的测试（19 条，原文名）

```
test_d_first_backward_gives_the_zero_initialised_output_layer_no_gradient
test_e_lambda_suffix_zero_reproduces_the_original_s0_objective_and_gradients
test_f_readout_signature_and_return_type_are_the_frozen_ones
test_f_index_mapping_counterexample_W2_B4_J_1_3_4_7
test_g_keep_ratio_is_the_hard_gate_fraction_and_never_the_thresholded_mean
test_g_norm_ratio_is_per_pair_and_divided_by_the_plain_g_norm
test_g_statistics_report_an_empty_pool_as_count_zero_and_value_null
test_g_lse_margin_is_positive_minus_logsumexp_of_valid_negatives_only
test_g_returned_dict_carries_no_live_tensor_that_cannot_reach_the_loss
test_i_unequal_valid_counts_two_rank_update_matches_the_single_process_oracle
test_i_equal_valid_counts_degenerate_to_a_plain_local_mean
test_i_rank_with_zero_valid_suffixes_contributes_a_differentiable_zero
test_i_global_valid_pool_below_two_gives_an_exactly_zero_loss_with_no_nan
test_i_world_size_factor_is_present_exactly_once
test_j_production_writer_roundtrips_a_non_initial_f_and_mask
test_j_state_digest_separates_tensors_from_descriptions_and_detects_tampering
（另 3 条见 debug/accept_pytest_fixv1.log 的 FAILED 列表）
```

已知的两类具体原因（原文报错）：

- `test_j_production_writer_roundtrips_...`：`AttributeError: 'MaskNetwork' object has no attribute 'out'`
  → 测试用了错误的 S0 mask 子模块名（真实结构是 `mask_net.resblocks.*` / `mask_net.attn_pool.*`）。
- `test_j_state_digest_...`：`DID NOT RAISE` → 生产 digest 函数对测试构造的篡改输入未按预期报错（测试或实现的一方需要对齐）。
- 五条 `test_i_*`（两 rank 生产入口）全部因为 worker 进程 `returncode=1` 失败 → 需要看 worker 的 stderr。

## 3. 本轮实际改动（文件与摘要哈希）

| 文件 | 修复前 | 修复后（本轮） | 摘要前缀 |
| --- | --- | --- | --- |
| `model/said_prefix_suffix.py` | 91 708 B | 127 042 B | `10494600e6655eb9` |
| `train/train_said_prefix_suffix.py` | 60 812 B | 94 139 B | `bec83f9bfdc8bccca` |
| `tests/test_said_prefix_suffix.py` | 104 117 B | 150 192 B | `6ad3db0e945c4a44` |
| `tests/_suffix_ddp_worker.py` | 33 271 B | 33 271 B（未变） | `3ab13d13f94c8a1e` |

（上面的摘要是**验收时**计算的真实值；commit 后以提交内文件为准。）

## 4. 明确未完成 / 未验证

- 生产入口（2 步）**未跑通**，因此第 12 节第 4 层"至少一个 step 的有效后缀 `V>=2` 且确认不是只跑了 S0"**未验证**。
- 第 2 节标准归一化反向、第 3 节图像×候选前缀广播、第 4 节三种索引空间、第 5 节 CE/缩放、第 7 节精度、
  第 8 节重日志统计这些修复**已写入代码**，但**没有通过本轮验收证据**，不得视为已修好。
- 第 11 节 resume 边界、第 10 节 holdout 重述，均**未在本轮验证**。
- 两个 500 步正式 run、COCO、Urban、机制诊断、前端登记：**全部 NOT RUN**。
- 测试结果未按"生产实现错误 / 测试 reference 错误 / 环境问题 / 未定位"逐项归类（需要三类代理的收尾报告）。

## 5. 勘误：上轮"已入库的 debug 日志"其实不可读

`74bf9c0` 声称包含两个 debug 日志，实际被 `.gitignore:25 (*.log)` 静默忽略（`git ls-files` 为空，
`git check-ignore -v` 指向该规则）。本轮用 `git add -f` 强制加入，并已做敏感信息扫描
（`password|token|secret|key|credential|PRIVATE KEY` 命中数均为 0）。此前"本地存在"不等于"仓库可读"，
这一条更正为本轮事实。

## 6. 下一步（待授权）

1. 看两个 rank worker 的 stderr，修 `test_i_*` 五条（这是 DDP 缩放的正确性核心）。
2. 对齐 `test_j_*` 两条与生产接口（mask 子模块名、digest 拒绝行为）。
3. 修完重跑分层验收，直到四层全绿；再写完整 `fix_v1_report.md`（按 1–10 项逐条"问题→修改→证据"）。
4. 仍然**不启动**两臂 500 步，等审查结论与单独授权。
