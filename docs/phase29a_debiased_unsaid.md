# Phase 2.9A：Global + Said + Debiased Unsaid Alignment Core

新主方法的 core：三个 caption view（Full / Said-prefix / withheld suffix）驱动的 tri-alignment。
本阶段只做 core + 单元测试 + 极短 smoke，不做长训练、不做质量结论。旧 residual Unsaid 作为 ablation 保留。

## 算法

```text
C_full = (S1..SN)                      # ShareGPT4V 完整 caption
K ~ Uniform{1..N}                      # 旧 SmartCLIP prefix 采样，未改动
C_said = S1 + ... + SK
J ~ 独立 stateless 采样 in [K+1, N]     # 仅当 K < N；K == N 时 has_unsaid = False
U = S_J                                # withheld suffix sentence（Unsaid semantic target）
```

- **Global（Full view）**：`L_global = symmetric CLIP InfoNCE(normalize(E_img(I)), normalize(E_text(C_full)))`
  （`--global_caption_view full`；默认 `prefix` = 旧行为）。
- **Said（prefix view，完全复用）**：`A^S = softmax(q_s^T k_p / tau_said)`、`z_s = normalize(sum A^S h)`，
  identifiable route/evidence 目标与诊断原样保留。
- **Debiased Unsaid**：
  ```text
  s^S = q_s^T k_p                                    # own-prefix Said raw score
  gate = gate_floor + (1-gate_floor) * sigmoid(-s_hat / tau_gate),  s_hat = (s^S-mean)/ (std+eps)
  gate = stop_gradient(gate),  gate_floor <= gate <= 1        # 软门，无 hard mask
  r^U_{i,j,p} = normalize(W_q t^U_j)^T normalize(W_k h_{i,p})  # 同一套 W_q / W_k，无新投影
  logit^U = r^U / tau_unsaid + beta * log(gate + eps)
  A^U = softmax_p(logit^U),  z^U_{i,j} = normalize(sum_p A^U h_{i,p})
  score^U_{i,j} = scale * cos(z^U_{i,j}, t^U_j)
  L_unsaid = 0.5*(CE(score^U, labels) + CE(score^U.T, labels))     # 仅 has_unsaid 样本
  ```
  `B_u < 2` 时 `L_unsaid = 0`（可微、有限）。`L_total = λG·L_G + λS·L_S + λU·L_U`，无其它损失项。

## RNG / 兼容

- prefix 采样仍是 `random.randint(1, num_sentences)`（同一调用、同一顺序）→ 旧 prefix caption stream 不变。
- suffix 用 `sample_unsaid_index(index, N, K, seed) = sha256(seed:index:K)` 派生，**不消耗 worker RNG**。
- 单元测试：同 seed 下开/关 suffix view，`K` 与 `C_said` 逐字符串相同；smoke 中 Run A/B 的
  `batch_caption_sha256` 前缀流与 Phase 2.8A 完全一致（`65543d645877a5b8, 30d1ad2aecc2433b, …`）。

## 新 API

- `share4v_train_dataset(caption_views=True)` → dict（image / caption_full / caption_said / caption_unsaid /
  has_unsaid / num_sentences / prefix_k / unsaid_sentence_index）；默认 `False` 仍是旧的 `(image, prefix)`。
- `SALUModel.score_said_conditioned(patches, texts)`：Said pair score 矩阵。
- `SALUModel.score_unsaid_candidates(patches, prefix_texts, candidate_texts, return_details=...)`：
  `[B, C]` pair scores（可选 attention / gate / hidden logits）。

## CLI

`--unsaid_mode {residual,debiased_suffix}`（默认 residual）、`--global_caption_view {prefix,full}`（默认 prefix）、
`--unsaid_gate_floor 0.1`、`--unsaid_gate_temperature 1.0`、`--unsaid_suppression_beta 1.0`；全部进入 checkpoint args。

## 诊断（debiased_suffix）

`loss_{global,said,unsaid,total}`、`unsaid_enabled`、`unsaid_valid_ratio/batch_size`、
`unsaid_retrieval_top1_{i2t,t2i}` / `margin_{i2t,t2i}`、Said/Unsaid entropy 与 effective patch count、
`said_unsaid_attention_overlap`、`unsaid_gate_{mean,min,max}`、
`said_coverage_under_raw` / `said_coverage_under_gated`（coverage = `1 - gate`，仅诊断），
以及数据统计 `data_{num_sentences_mean,median,prefix_k_mean,suffix_length_mean,unsaid_position_mean,has_unsaid_ratio}`。
residual 模式的旧诊断键在新模式下写 `null`（不伪造）。

## 极短 smoke（4 × A800，完整 ShareGPT4V + Full Data Gate，B16，bs 256/G = 1024，bf16，seed 0）

Run A（legacy：`prefix` + `lambda_unsaid 0`，5 步）：`loss_global` 0.6591→0.4802、`loss_said` 3.3339→3.2130、
`loss_total` 3.9930→3.6932，与 Phase 2.8A legacy Run A 的 step 0/1 数值逐位一致；
capture stream 一致；compute 1.0627 / wall 1.4476 s/step；peak GPU 39.97 GB；`EXIT=0`。

Run B（`full` + `debiased_suffix`，λU=1，20 步）：

| step | loss_global | loss_said | loss_unsaid | loss_total | valid ratio | valid B | top1 i2t/t2i | margin | gate mean/min/max | cov raw/gated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.5173 | 3.3339 | 3.9889 | 7.8401 | 0.871 | 223 | 0.157 / 0.305 | 3.888 | 0.5498 / 0.1131 / 0.9800 | 0.4881 / 0.4213 |
| 19 | 0.2750 | 3.1208 | 3.5414 | 6.9372 | 0.883 | 226 | 0.199 / 0.367 | 4.111 | 0.5494 / 0.1138 / 0.9819 | 0.4907 / 0.4244 |

Said/Unsaid entropy ≈ 5.19 / 5.20，effective patches ≈ 180 / 181，overlap ≈ 0.735；全部 loss 有限、无 NaN/Inf；
compute 1.0650 / wall 1.5687 s/step；peak GPU 74.26 GB（`[B,B,P]` hidden logits/attention 使显存明显上升，仍 < 80 GB）；
`EXIT=0`。gate 后 coverage 一致低于 raw（0.42 vs 0.49）。

数据统计（Run B 最后一步 batch）：句数 mean 7.98 / median 8，prefix K mean 4.45，suffix 长度 mean 3.53，
withheld 句位置 mean 6.38，has_unsaid ratio 0.883。

**本阶段不对质量下结论**；`lambda_unsaid = 1.0` 仅用于确认 CE scale / 梯度 / 接线。

## 测试

`tests/test_debiased_unsaid.py`（20 个）：caption views 的 RNG 兼容、Unsaid 确为真实 suffix 句、
stateless 采样边界、gate 范围/均匀分数有限、gate detach、等 hidden 相似度下低 coverage patch 权重更高、
soft gate 不硬删强匹配 patch、`beta=0` 退化为纯 hidden 语义注意力、coverage 诊断方向、
pairwise retrieval top1/margin、`B_u<2` 安全零损失、pair score 检索正确性、`score_said_conditioned`、
梯度到 q/k/视觉/文本路径、诊断有限性、单 valid 样本安全零、prefix+λU=0 与 legacy 目标逐位一致、
full view 只改 global 对齐、无新参数/state_dict、`encode_image/text` 未被改动。

## 未做的事

prototype bank / concept bank / phrase extraction / text retrieval memory / spatial grounding /
entropy·orthogonality·overlap·reconstruction 损失 / 新 learnable module / 独立 q_u,k_u /
500 步或 3 epoch 训练 / lambda·tau·beta sweep / 修改 canonical validation 与 dashboard /
删除旧 residual Unsaid。
