# SAID-ExGAP Phase ExGAP-1B 报告：Causal Attribution + Retrieval Gate

本报告只写实测值；未运行的项目一律显式标注 **NOT RUN**。所有数字来自本次运行在
`/root/SAID-gap-completion` 上产出的日志与 JSON，路径与 sha256 见第 9 节 provenance。

- 分支 `codex/said-exgap-v1`，HEAD `6350e33`（训练与检索都在该提交上运行；本阶段代码改动在本报告之后提交）
- M0 = `runs_salu/said_exgap_phase1b/M0_said_only_masking`
- M1 = `runs_salu/said_exgap_phase1b/M1_full_exgap`
- 检索结果 = `outputs/said_exgap/phase1b_canonical_retrieval.json`
- 表格生成器 = `tools/exp_said_exgap_phase1b_tables.py`

**本阶段没有修改任何 ExGAP 数学、没有改 mask threshold、没有加 USS / global CLIP loss /
prototype / reconstruction，也没有跑 3 epoch。**

---

## 1. Experimental question

1. **Does ExGAP add causal separation beyond Said masking?**
   即 `D_U = S_gc − S_uc` 在 M1（`L = L_S + L_ExGAP`）是否严格大于 M0（`L = L_S`，但**完整计算**同一套
   forward 几何，只是 `λ_ExGAP = 0` 不参与 backward）。
2. **Does patch-derived global representation support retrieval?**
   即 `z_G = Pool(H) = Norm(mean_p h_p)` 是否具备可用的检索能力，以及它在 M0/M1 训练后是否退化。

---

## 2. Matched configuration

两个臂除 `--lambda-exgap`（0.0 / 1.0）外**逐字相同**：

```text
objective_mode said_exgap        lambda_global 0        lambda_unsaid 0
lambda_said 1.0                  exgap-global-pool mean  exgap-mask-threshold 0.6
exgap-temperature 0.05           exgap-collapse-every 20 grad-attribution-steps 20,100,500
save-completed-steps 0,100,500   said_loss_mode identifiable
said_feature_source residual     base_model B16          amp_dtype bf16
batch_size 256/GPU (global 1024) max_steps 500           lr_total_steps 3648
warmup_length 200                backbone_lr 1e-6        head_lr 1e-4
seed 0                           num_workers 8           strict_manifest (Full Data Gate)
world_size 4 (4x A800)
```

Full Data Gate：`FULL_DATA_GATE_PASS`，`steps_per_epoch 1216`，`dataset_size 1245901`，两臂 `exit=0`。

### 2.1 matched-stream audit（硬门槛，任一不一致即 STOP）

| 审计项 | M0 | M1 | 结果 |
| --- | --- | --- | --- |
| `initial_state_sha256` | `d61c429b496cf6cb654687d902108c675d513d8ac1d0e49c4a5dbe7b30c49a90` | 同左 | **PASS** |
| `sampler_order_sha256` | `953402b3c3d7322e3a74b72880d47a94f2aa9147db4004f41b2cad95094c9d55` | 同左 | **PASS** |
| `caption_stream_sha256` | `7563aafb3d5b048d4ab23ee3769885b81630086522cb8eeb2de5bb82a05d6ce3` | 同左 | **PASS** |
| `prefix_caption_stream_sha256` | 同上 | 同上 | **PASS** |
| `objective_mode` / `steps` / `dataset_size` | `said_exgap` / 500 / 1245901 | 同左 | **PASS** |
| 每臂 4 个 rank 的初始摘要唯一值个数 | 1 | 1 | **PASS** |

```text
M0 initial_state == M1 initial_state: PASS
```

随机前缀 caption 也一致（`caption_stream_sha256` 相等），因为 `caption_views=False` 下
`share4v_train_dataset` 的前缀抽样只由 `seed` 决定，两臂同 seed。

---

## 3. Training geometry（completed steps 20 / 100 / 500）

### 3.1 核心因果表（section 18）

| metric | M0 Said-only | M1 ExGAP | M1 − M0 |
| --- | ---: | ---: | ---: |
| route_top1 | 0.9844 | 0.9883 | +0.0039 |
| evidence_top1 | 0.9648 | 0.9648 | 0.0000 |
| loss_said | 0.0895 | 0.0926 | +0.0030 |
| S_sc | 0.2127 | 0.2146 | +0.0019 |
| S_gc | 0.1805 | 0.1844 | +0.0039 |
| S_uc | 0.0521 | 0.0470 | −0.0050 |
| S_sc − S_gc | 0.0322 | 0.0302 | −0.0020 |
| S_uc − S_gc | −0.1284 | −0.1373 | −0.0089 |
| **D_U = S_gc − S_uc** | **0.1284** | **0.1373** | **+0.0089** |
| gap_raw_mean | 0.0322 | 0.0302 | −0.0020 |
| gap_weight_mean | 0.0391 | 0.0368 | −0.0022 |
| gap_positive_fraction | 1.0000 | 1.0000 | 0.0000 |
| mask_keep_ratio | 0.3594 | 0.3223 | −0.0371 |
| mask_drop_ratio | 0.6406 | 0.6797 | +0.0391 |
| said_above_threshold_fraction | 0.6403 | 0.6778 | +0.0375 |
| loss_exgap | 0.000205 | 0.000172 | −3.2e−05 |

早期两个检查点（用于确认两臂同起点）：

| metric | M0 @20 | M1 @20 | M0 @100 | M1 @100 |
| --- | ---: | ---: | ---: | ---: |
| route_top1 | 0.0117 | 0.0078 | 0.2266 | 0.2148 |
| S_sc − S_gc | 0.0011 | 0.0011 | 0.0076 | 0.0076 |
| D_U = S_gc − S_uc | 0.000138 | 0.000142 | 0.0094 | 0.0093 |
| gap_positive_fraction | 0.7266 | 0.7227 | 0.9648 | 0.9609 |
| mask_keep_ratio | 0.9922 | 0.9922 | 0.8477 | 0.8477 |
| loss_exgap | 6.64e−05 | 6.62e−05 | 0.000285 | 0.000287 |

### 3.2 表示几何（固定 64-image cohort；collapse 监控在 completed 21 / 101 / 481）

| metric | M0 @21 | M1 @21 | M0 @481 | M1 @481 |
| --- | ---: | ---: | ---: | ---: |
| global_pairwise_cos | 0.8047 | 0.8047 | 0.7344 | 0.7266 |
| global_pairwise_cos_max | 0.9102 | 0.9102 | 0.8984 | 0.8945 |
| said_pairwise_cos | 0.8047 | 0.8047 | 0.7617 | 0.7539 |
| unsaid_pairwise_cos | 0.8047 | 0.8047 | 0.6875 | 0.6758 |
| unsaid_pairwise_cos_max | 0.9102 | 0.9102 | 0.8750 | 0.8711 |
| global_embedding_std | 0.0188 | 0.0188 | 0.0221 | 0.0224 |
| said_embedding_std | 0.0188 | 0.0188 | 0.0210 | 0.0213 |
| unsaid_embedding_std | 0.0188 | 0.0187 | 0.0240 | 0.0244 |
| global/said/unsaid_feature_norm | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

两臂在 completed 21 时逐位相同（同起点），到 481 时 `global_pairwise_cos` 0.7344 / 0.7266，
`global_pairwise_cos_max` 0.8984 / 0.8945 —— **都在 0.9 以下，未触发 `WARNING REPRESENTATION_COLLAPSE`**，
与已经塌缩的 Full-Base gap_completion 臂（0.977）完全不同。

---

## 4. Gradient attribution

`grad_attr_*` 在 completed 20 / 100 / 500 上用全新诊断图测量（`torch.autograd.grad`，不碰训练梯度；
新增测试 `test_attribution_does_not_disturb_the_gradients_of_the_training_step` 保证这一点）。

### completed step 500

| group | M0 G_S | M0 G_ExGAP | M0 R_grad | M1 G_S | M1 G_ExGAP | M1 R_grad |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| visual_backbone | 10.8116 | 0.0123 | 1.13e−03 | 11.7173 | 0.0105 | 9.0e−04 |
| patch_pathway | 10.8116 | 0.0123 | 1.13e−03 | 11.7173 | 0.0105 | 9.0e−04 |
| text_encoder | 4.4302 | **0.0000** | 0 | 4.5349 | **0.0000** | 0 |
| said_router | 0.6578 | **0.0000** | 0 | 0.6807 | **0.0000** | 0 |
| exgap_pooling | 0.0000 | **0.0000** | 0 | 0.0000 | **0.0000** | 0 |

### completed step 20 / 100（同样的 0 结构）

| group | M0 G_S @20 | M0 G_ExGAP @20 | M1 G_S @20 | M1 G_ExGAP @20 | M0 G_S @100 | M0 G_ExGAP @100 | M1 G_S @100 | M1 G_ExGAP @100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| visual_backbone / patch_pathway | 36.5121 | 0.0016 | 19.6728 | 0.0015 | 127.0067 | 0.0083 | 321.8125 | 0.0084 |
| text_encoder | 21.7262 | **0.0000** | 21.7380 | **0.0000** | 8.2954 | **0.0000** | 8.3832 | **0.0000** |
| said_router | 0.8286 | **0.0000** | 0.8282 | **0.0000** | 1.1922 | **0.0000** | 1.1913 | **0.0000** |
| exgap_pooling | 0.0000 | **0.0000** | 0.0000 | **0.0000** | 0.0000 | **0.0000** | 0.0000 | **0.0000** |

结论：

- **`said_router` 与 `text_encoder` 从 `L_ExGAP` 得到的梯度在三个检查点、两个臂上全部正好是 0.0000**
  （不是"很小"，是 0）；`clip.logit_scale`（`other` 组）同样为 0。没有触发 STOP 条件。
- `L_ExGAP` 唯一非零去处是共享视觉主干（本架构里 "patch pathway" 就是 `clip.visual`，两者数值相同）。
- `R_grad = G_ExGAP / (G_S + ε)` 在视觉主干上从 4.4e−5（step 20）升到 1.1e−3（step 500，M0）/ 9.0e−4（M1），
  即 ExGAP 的相对梯度强度随 gap 变宽而增长约 25 倍 —— 与 §3 里 `S_uc − S_gc` 从 −0.00014 变到 −0.13 一致。
- 注意 M0 的 `G_ExGAP` 也非零：这是**诊断量**（forward 几何完全一致，`λ=0` 只是不进入 backward），
  它证明"如果没有关掉，ExGAP 会往哪里推"。

---

## 5. Retrieval source audit

1. **`PRIMARY: patch_global` 的定义与来源**
   `z_G = Norm(mean_p h_p)`，由新增 API `SALUModel.encode_exgap_global(images, global_pool)` 提供；
   该 API 与训练 forward 用的是**同一对函数**（`encode_router_input` + `exgap.normalize_patches` +
   `exgap.compute_global_representation`），没有第二套实现。
   evaluator 侧只要模型带该 API 就走 API（`patch_global_source = model_api`），否则退回同名共享公式
   （`training_formula`），并在同一次前向里**交叉校验**两者：
   实测每个 variant 的 `z_patch_global_vs_training_formula_max_abs_diff = 5.96e−08`（initial 1.19e−07），
   代码在 `global_pool == 'mean'` 且走 API 时对该差值设了 **1e−5 的硬失败阈值**。
2. **`DIAGNOSTIC: legacy_cls`** 仍然是 `clip.encode_image()`；它与 `patch_global` 在 JSON 中永远是两个
   独立键（`retrieval_by_representation`），从未合并成一列。`retrieval` 顶层键镜像请求的 primary。

   实测（JSON 原文）：9 个 checkpoint×variant 行的 `patch_global_source` **全部为 `model_api`**，
   即 evaluator 走的就是 `SALUModel.encode_exgap_global`；同一行的 drift 为
   5.96e−08（M0_500 / M1_500）/ 1.19e−07（initial）。两次独立检索运行的 24 行数值**逐位相同**
   （`compare_retrieval_runs.py`：`numerical rows differing = NONE`）。
3. **PatchGlobal ≠ legacy CLS（实测）**：Initial 上 COCO I2T R@1 = 0.0678（patch_global）vs
   0.5170（legacy_cls）；`global_pairwise_cos = 0.8047`（patch-mean 有很强的共同方向）。
4. **Initial 的两个 reference 都有**：同一个完成步 0 检查点（`salu_exgap_step000000.pt`，两臂共享），
   分别读 patch_global 与 legacy_cls。
5. **加载硬失败**：evaluator 现在对 0 张量匹配直接 `raise`（禁止再出现
   `strict=False silently loads zero tensors`）；本次三个检查点都是 `loaded_tensors = 326`，
   `missing_keys = []`，`unexpected_keys = []`。

---

## 6. Canonical retrieval

评测器：`tools/phase30a_fixed_cohort_eval.py --canonical --canonical_only
--image-representation patch_global --global-pool mean --coco`，
COCO val2017（5-caption）+ ShareGPT4V-1K 三个 frozen variants，`similarity_chunk = 512`。

### 6.1 完整表（R@1 / R@5 / R@10）

| model | image repr | dataset / variant | I2T R@1 | T2I R@1 | I2T R@5 | T2I R@5 | I2T R@10 | T2I R@10 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| initial | legacy_cls | coco_val2017 | 0.5170 | 0.3269 | 0.7662 | 0.5776 | 0.8428 | 0.6823 |
| initial | patch_global | coco_val2017 | 0.0678 | 0.1609 | 0.1734 | 0.3488 | 0.2504 | 0.4566 |
| initial | legacy_cls | 1K first_sentence | 0.5440 | 0.5140 | 0.7740 | 0.7480 | 0.8590 | 0.8140 |
| initial | patch_global | 1K first_sentence | 0.1280 | 0.3220 | 0.2730 | 0.5730 | 0.3460 | 0.6810 |
| initial | legacy_cls | 1K fixed_sparse | 0.7460 | 0.7400 | 0.9200 | 0.9180 | 0.9530 | 0.9490 |
| initial | patch_global | 1K fixed_sparse | 0.1350 | 0.4960 | 0.2900 | 0.7580 | 0.3600 | 0.8350 |
| initial | legacy_cls | 1K full_dense | 0.7580 | 0.7760 | 0.9200 | 0.9480 | 0.9520 | 0.9730 |
| initial | patch_global | 1K full_dense | 0.1260 | 0.5420 | 0.2640 | 0.7860 | 0.3560 | 0.8790 |
| M0_500 | legacy_cls | coco_val2017 | 0.5402 | 0.3539 | 0.7864 | 0.6110 | 0.8698 | 0.7132 |
| M0_500 | patch_global | coco_val2017 | 0.0102 | 0.2474 | 0.0272 | 0.4842 | 0.0496 | 0.5960 |
| M0_500 | legacy_cls | 1K first_sentence | 0.5940 | 0.5830 | 0.8450 | 0.8150 | 0.9130 | 0.8880 |
| M0_500 | patch_global | 1K first_sentence | 0.0340 | 0.5000 | 0.1330 | 0.7680 | 0.2760 | 0.8570 |
| M0_500 | legacy_cls | 1K fixed_sparse | 0.8360 | 0.8370 | 0.9650 | 0.9560 | 0.9850 | 0.9730 |
| M0_500 | patch_global | 1K fixed_sparse | 0.0660 | 0.7130 | 0.1570 | 0.9210 | 0.2480 | 0.9560 |
| M0_500 | legacy_cls | 1K full_dense | 0.9250 | 0.9230 | 0.9930 | 0.9930 | 0.9960 | 0.9970 |
| M0_500 | patch_global | 1K full_dense | 0.6890 | 0.7760 | 0.8990 | 0.9510 | 0.9480 | 0.9830 |
| M1_500 | legacy_cls | coco_val2017 | 0.5328 | 0.3542 | 0.7848 | 0.6095 | 0.8652 | 0.7129 |
| M1_500 | patch_global | coco_val2017 | 0.0116 | 0.2523 | 0.0304 | 0.4880 | 0.0556 | 0.6006 |
| M1_500 | legacy_cls | 1K first_sentence | 0.5990 | 0.5870 | 0.8390 | 0.8100 | 0.9120 | 0.8830 |
| M1_500 | patch_global | 1K first_sentence | 0.0360 | 0.5000 | 0.1440 | 0.7660 | 0.2890 | 0.8610 |
| M1_500 | legacy_cls | 1K fixed_sparse | 0.8350 | 0.8320 | 0.9620 | 0.9570 | 0.9840 | 0.9740 |
| M1_500 | patch_global | 1K fixed_sparse | 0.0720 | 0.7210 | 0.1660 | 0.9280 | 0.2540 | 0.9570 |
| M1_500 | legacy_cls | 1K full_dense | 0.9200 | 0.9230 | 0.9940 | 0.9920 | 0.9960 | 0.9970 |
| M1_500 | patch_global | 1K full_dense | 0.7020 | 0.7850 | 0.9040 | 0.9570 | 0.9540 | 0.9840 |

### 6.2 representation-source 表（section 19）

| checkpoint | legacy_cls COCO I2T R@1 | patch_global COCO I2T R@1 | delta | legacy_cls COCO T2I R@1 | patch_global COCO T2I R@1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial | 0.5170 | 0.0678 | −0.4492 | 0.3269 | 0.1609 |
| M0-500 | 0.5402 | 0.0102 | −0.5300 | 0.3539 | 0.2474 |
| M1-500 | 0.5328 | 0.0116 | −0.5212 | 0.3542 | 0.2523 |

**CLS 在变好，patch-global 在变坏。** Initial 的 CLS COCO I2T R@1 = 0.5170 与历史 canonical 表里的
`Initial 0.5170 / 0.3269` 完全一致，说明评测器与历史结果同源、可直接比较。

---

## 7. Causal interpretation

把两个效应分开看：

### 7.1 masking / L_S 效应（M0 vs Initial）

M0 里 `λ_ExGAP = 0`，`L_ExGAP` 的梯度为 0（§4 已证），只有 `L_S` + 硬掩码诊断 + 共享主干微调。
结果：COCO patch_global I2T R@1 **0.0678 → 0.0102**，而 CLS **0.5170 → 0.5402**。
所以 patch-global 的退化**不是 ExGAP 造成的**，它来自 `L_S` 对共享视觉主干的微调 + patch-mean 本身
就不是 CLIP 的检索嵌入。这个退化发生在完全没有 ExGAP 梯度的臂上，是 clean 的对照证据。

### 7.2 ExGAP 效应（M1 vs M0）

- 因果分离：`D_U` 0.1284 → 0.1373（+0.0089，+6.9%），`S_uc` 0.0521 → 0.0470。
- 掩码几乎不受影响：`mask_keep_ratio` 0.3594 vs 0.3223（Δ = −0.0371），
  `route_top1` 0.9844 vs 0.9883（Δ = +0.0039 = 4/1024）。差异存在但很小，且 §4 已经证明
  ExGAP 到 router 的梯度**恒为 0**，所以这个差异只能来自共享主干的间接效应（ExGAP 改变了 patch 特征，
  patch 特征再改变 router 的输入），不是"ExGAP 直接操纵了 mask"。
- 检索：COCO patch_global I2T R@1 0.0102 → 0.0116（+0.0014），T2I R@1 0.2474 → 0.2523（+0.0049）；
  1K full_dense patch_global I2T R@1 0.6890 → 0.7020；1K fixed_sparse 0.0660 → 0.0720；
  1K first_sentence 0.0340 → 0.0360。**方向上 M1 ≥ M0（四处全为正），但绝对量极小。**

### 7.3 回答 §15 的问题

> Does Said-only learning become more useful when retrieval reads the same patch space?

**实测答案：不是，反而差得多。** 同一个 M0 模型：
legacy_cls COCO I2T R@1 = 0.5402，patch_global = 0.0102（差 53 倍）；
T2I R@1 0.3539 vs 0.2474。M0 与历史 A Said-only **loss 形式相同但检索读取的表示不同**，
两者不能当作同一个模型：A Said-only（3 epoch）0.5224/0.3423（读 CLS），
M0（500 step）0.5402/0.3539（读 CLS）/ 0.0102/0.2474（读 patch-global）。

### 7.4 问题的性质（按 §17 的三分类）

**representation problem**，不是 optimization / objective problem：

1. 初始化时 patch_global 就已经只有 CLS 的 1/7.6（COCO I2T R@1 0.0678 vs 0.5170），
   而 `global_pairwise_cos = 0.8047` 说明 `mean_p h_p` 由一个跨图共同方向主导，判别成分很小。
2. 训练让它更差（0.0678 → 0.0102），且这一步发生在**没有任何 ExGAP 梯度**的 M0 上。
3. 例外是 1K `full_dense`（长而全的 caption）：patch_global 0.1260 → 0.6890/0.7020。
   短 caption（first_sentence / fixed_sparse / COCO）全部退化。这与"patch-mean = 整图的全部内容"的
   语义一致：它匹配"把整图都说全"的长文本，而不匹配"只说一部分"的短查询 —— 恰好是检索最需要的方向。
4. ExGAP 只把 `D_U` 推大 0.0089，对 patch-global 检索只带来 +0.0014（COCO I2T R@1），不足以改变结论。

按 §17 的要求：**没有**调 threshold / lambda / loss / distillation / attention pool / USS /
gap normalization / margin / projection head；先 STOP 并报告真实结果。

---

## 8. Go / No-Go

| Gate | 判据 | 实测 | 判定 |
| --- | --- | --- | --- |
| **A** causal effect | `D_U^{M1} > D_U^{M0}` | 0.1373 > 0.1284（+0.0089，+6.9%）；两臂 `gap_positive_fraction = 1.0`，`S_uc` 0.0470 < `S_gc` 0.1844 | **PASS** |
| **B** no collapse | M1 `global_pairwise_cos` 不持续冲向 0.9+ | 0.8047（21）→ 0.7266（481），max 0.8945 < 0.9，无 `REPRESENTATION_COLLAPSE` 警告 | **PASS** |
| **C** patch-global retrieval viability | M0/M1 相对 Initial-PatchGlobal **不发生灾难性下降** | Initial 0.0678 → M0 0.0102 / M1 0.0116（COCO I2T R@1，−85%/−83%）；1K first_sentence 0.1280 → 0.0340/0.0360；1K fixed_sparse 0.1350 → 0.0660/0.0720 | **FAIL** |
| **D** ExGAP usefulness | `M1 retrieval ≥ M0` **且** `M1 separation > M0` | 两点都成立（4/4 检索指标 M1 ≥ M0；`D_U` +0.0089） | **PASS（但不足以救 C）** |

### 总判定：**FAIL**

失败点是 **Gate C**：`H → z_G` 产生的 patch-derived global representation **不具备可用检索能力**，
而且 500 步训练让它进一步恶化；"理想"条件（`M0 > Initial-PatchGlobal`、`M1 ≥ M0`）中前者完全反向。

### 建议

**不进入 3 epoch。** 下一步应先解决表示问题（而不是目标函数或超参数），可选方向（本阶段**未实施、
未验证**）：让检索读取 CLS 与 patch space 的连接（例如把 CLS 也放进共享空间），或让 `z_G` 从
一个被检索目标训练的池化得到；这些都属于新的方法改动，必须重新走一轮 matched control 再评估。

```text
SAID-ExGAP 500-step gate: FAIL（Gate A PASS / Gate B PASS / Gate C FAIL / Gate D PASS）
```

---

## 9. Provenance

| checkpoint | step | objective_mode | global_pool | sha256 | loaded tensors | missing | unexpected |
| --- | ---: | --- | --- | --- | ---: | --- | --- |
| `M0.../salu_exgap_step000000.pt` | 0 | said_exgap | mean | `dfb5de8bcd6223c0d68847ced35b929f63ca46bf8fcad35cde9cf7217e9ec01c` | 326 | [] | [] |
| `M0.../salu_exgap_step000500.pt` | 500 | said_exgap | mean | `8eebdfdf991026f44d528552a92617f10b134058f650404a3cc3bf1de72eb897` | 326 | [] | [] |
| `M1.../salu_exgap_step000500.pt` | 500 | said_exgap | mean | `de0894f991f8fe64d950adecd9979ca1c50444dc7281e5d66035de2f79b35b74` | 326 | [] | [] |

- `git_head`（评测时）：`6350e3344a68d3e3420d182510ccbf0badffd1cc`
- ShareGPT4V-1K caption manifest sha256：`c521192355dd7ee5479a6816bd5a3f918318b3af6f1b2704a4f3ed5aeeed8ae9`
- `checkpoint_exgap_config`：M0 `lambda_exgap = 0.0`，M1 `lambda_exgap = 1.0`，其余（pool/threshold/temperature/normalize）完全相同
- 复现命令：`bash tools/exp_said_exgap_phase1b.sh`（训练）→ `bash tools/exp_said_exgap_phase1b_retrieval.sh`（检索）
  → `python tools/exp_said_exgap_phase1b_tables.py`（表格）

### 本阶段新增/修改的代码与测试

| 文件 | 变更 |
| --- | --- |
| `model/salu_model.py` | 新增 `encode_exgap_global`；日志新增 `D_U_mean` / `S_uc_minus_S_gc_mean` / `gap_weight_mean` / `*_embedding_std` |
| `train/train_salu.py` | 新增 `--save-completed-steps`、`--grad-attribution-steps`、`exgap_grad_attribution` / `exgap_grad_groups`；日志触发条件包含归因步 |
| `eval/retrieval/coco_retrieval.py` | 新增 `patch_global_features`（唯一共享实现）与 `evaluate_coco_representations`（一次前向同时产出两种表示） |
| `eval/validation_protocol.py` | `evaluate_variant` / `evaluate_all_variants` 支持 `image_representation` + `global_pool`，返回两列独立结果与 drift 校验 |
| `tools/phase30a_fixed_cohort_eval.py` | `--image-representation` / `--global-pool`；provenance（sha256 / git HEAD / 加载张量数 / missing / unexpected / manifest hash）；0 张量匹配硬失败；`--canonical_only` 不再加载 USR cohort；解析 `--image_root` 默认值 |
| `tools/exp_said_exgap_phase1b.sh`、`tools/exp_said_exgap_phase1b_retrieval.sh`、`tools/exp_said_exgap_phase1b_tables.py` | 新增：匹配对照训练、检索门、报告表格生成 |
| `tests/test_exgap_phase1b.py` | 新增 11 个测试（API 一致性、grad attribution 契约、R_grad、patch_global_features 优先级与回退、归因不污染训练梯度） |
| 测试结果 | `pytest tests/ -q` → **438 passed, 2 skipped** |
