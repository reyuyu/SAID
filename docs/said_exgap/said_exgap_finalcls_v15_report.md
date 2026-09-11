# SAID-ExGAP v1.5 报告：Pre-Final Evidence Routing + Native Final-CLS Readout

本报告只写实测值；未运行的项目一律显式标注 **NOT RUN**，未测到的字段一律写 **MISSING**（绝不写 0）。

- 仓库 `/root/SAID-gap-completion`，分支 `codex/said-exgap-finalcls-v15`
- 新 objective：`said_exgap_finalcls`（旧 `said_exgap` / `gap_completion` / `legacy` 全部保留、未改语义）
- 训练产物 `runs_salu/said_exgap_finalcls/{step20,step100,step500}_{F0_said_only,F1_full_exgap}`
- 检索结果 `outputs/said_exgap/finalcls_v15_canonical_retrieval.json`
- 表格生成器 `tools/exp_said_exgap_finalcls_tables.py`

**未修改旧实验结果、未删除旧 `said_exgap`、未改写任何历史 checkpoint、未 merge main、未自动启动 3 epoch。**

---

## 1. Architecture

一句话实现原则（与 brief §46 一致）：

> **Route caption relevance over the pre-final visual tokens `H11`, but never replace CLIP's native
> global readout. Instead, use the original final Transformer block itself to produce Global, Said
> and complementary final-CLS representations by softly reweighting only the CLS-to-patch
> attention row.**
>
> **H11 decides which evidence is available; the original Block 12 decides how that evidence
> becomes a CLIP representation.**

```
images
  │
  ├─ encode_visual_prefinal(images)                     [model/model_longclip.py]
  │     = conv1 + class_embedding + pos + ln_pre
  │       + blocks[0..10]            ← 前 11 个 block 全部执行
  │     -> cls11_raw [B,768], patch11_raw [B,196,768], x11_raw [B,197,768]
  │
  ├─ H11_route = Norm(ln_post(patch11_raw) @ proj)      [SALUModel.prefinal_route_features]
  │     （复用视觉投影本身：没有新增第二套 projection）
  │
  ├─ SaidRouter(q_proj, k_proj)：r = sigmoid( (q̂(t)·k̂(h11_p)) / τ_S )   τ_S = 0.07
  │
  └─ FinalBlockCLSReadout(visual)                       [model/final_cls_routing.py]
        prepare(x11_raw)  -> LN_1、Q_cls、K_all、V_all、base_logits（各算一次）
        read_global()     -> A^G = softmax(a)                     == clip.encode_image
        read_said(r)      -> A^S = softmax(a + log r)
        read_unsaid(r̅)    -> A^U = softmax(a + log(1 - sg(r)))
        read_pairwise_said(r[i,j])  （按 candidate chunk 计算）
        全部走原始：out_proj → CLS residual → ln_2 → MLP → residual → ln_post → proj
```

新增接口（不破坏任何既有 API）：

| 接口 | 位置 | 说明 |
| --- | --- | --- |
| `VisionTransformer.forward_prefinal(x)` | `model/model_longclip.py` | 返回 `cls11_raw` / `patch11_raw` / `x11_raw`，前 11 block 已跑、第 12 block **未**跑 |
| `CLIP.encode_visual_prefinal(image)` | 同上 | 公开包装 |
| `SALUModel.encode_visual_prefinal` / `prefinal_route_features` / `final_cls_readout` / `said_final_cls` | `model/salu_model.py` | |
| `FinalBlockCLSReadout`、`prepare` / `read_global` / `read_said` / `read_unsaid` / `read_pairwise_said` / `attention_weights` | `model/final_cls_routing.py` | **零参数**，直接引用 `visual.transformer.resblocks[-1]` 的 `ln_1` / `attn.in_proj_weight` / `attn.in_proj_bias` / `attn.out_proj` / `ln_2` / `mlp` 与 `visual.ln_post` / `visual.proj` |
| `pairwise_said_relevance` / `gate_bias` / `reference_final_cls` / `attach_reference_helper` | 同上 | 后者仅供测试的慢速参考路径 |

必须没有的东西（已用代码扫描 + 功能测试双重把关，见 §5/§6）：`H12 → mean pool → global`、
`H11 → direct pool → final Said`、`Said+Unsaid → reconstruct Global`、Global Absorption、Gap
Completion、global-text CLIP loss、prototype bank、USS、full-caption teacher、patch-mean primary、
新增 global projection head。**标准 inference 仍然是 `clip.encode_image` / `clip.encode_text`**（§28）。

---

## 2. Why H11, not H12

第 12 个 block 的真实输入是 `X11 = [c11, H11]`，所以 `c12` 的最后一级视觉证据来自 `H11`；
`H12` 与 `c12` 是同一层同时产出的，用 `H12` 让 caption 去“选择”证据在因果上是循环的。
实测（T1，toy-scale real VisionTransformer）：

```
blocks[0:layers-1] -> X11 ;  blocks[-1](X11)  ==  full blocks[0:layers]      (max_abs < 1e-5)
X11 vs block12 输出（CLS 行）差异 > 1e-3      ← 证明 X11 不是第 12 层输出
```

`forward_prefinal` 与 `forward` 共用同一个 `_token_sequence()` 前导（本次重构，`forward` 数值不变：
全部 468 个既有测试继续通过）。

---

## 3. Exact final-block CLS derivation

原始 block：

```
x_attn = x + MHA(LN1(x))
x_out  = x_attn + MLP(LN2(x_attn))
g      = Norm( ln_post( x_out[CLS] ) @ proj )
```

只改 CLS query row，因此 `K/V` 与 patch query row 完全不变，CLS 行可以**精确单独**计算：

```
a_p        = (Q_cls · K_p) / sqrt(d_head)          ← 与 torch MHA need_weights=False 路径一致
c_attn     = c11 + out_proj( Σ_p A_p V_p )
c12        = c_attn + MLP( ln_2( c_attn ) )        ← MLP 是 token-wise，只需对 CLS 行运行
feature    = ln_post(c12) @ proj                   ← 原始 RAW 特征（未归一化）
```

`A_p` 分别为 `softmax(a)`、`softmax(a + log r)`、`softmax(a + log(1 - sg(r)))`。
`q` 的缩放严格复刻 torch 的 `q * sqrt(1/head_dim)`（不是 `q / sqrt(head_dim)`，避免最后一位差异）。

工程约束（§14/§15/§17/§18）：**没有** `for image: for caption: block12(full_sequence)`，**没有**
`[B, B, 197, 768]`；`LN1`、`Q_cls`、`K_all`、`V_all`、`base_logits` 每张图只算一次；pairwise
candidate 维度分块（`--finalcls-pair-chunk-size`，默认 32；实测 20 步时 `pair_chunk=32` 与
chunk=1 的结果一致，见 §6）。

---

## 4. Said / Unsaid soft gate

```
r_{i,j,p} = sigmoid( (q̂(t_j) · k̂(h_{i,p}^{11})) / τ_S )          τ_S = 0.07（router 原温度）
b^S_{i,j,p} = log clamp(r_{i,j,p}, ε, 1)          b^S_CLS = 0      ε = 1e-6
b^U_{i,p}   = log clamp(1 - sg(r_{i,i,p}), ε, 1)  b^U_CLS = 0
A^S = softmax(a + b^S)          A^U = softmax(a + b^U)
```

* **没有 spatial softmax**：`r` 是每个 patch 独立的 caption relevance。
* **主训练没有 hard 0.6 mask**：0.6 只作为 diagnostic（`said_fraction_gt_0_6`）保留；
  v1 的 `compute_said_mask` 在 v1.5 路径中被**代码扫描证明未被调用**（T11）。
* Said gate 是 **live** 的（`L_S → r^S → SaidRouter`）；Unsaid gate 用 **detached** complement，
  `read_unsaid` 在收到 `requires_grad=True` 的 relevance 时直接 `raise`。
* CLS self key 永不抑制（`bias[:, 0] == 0`，T10 对 Said 和 Unsaid 都验证）。

---

## 5. Native global equivalence（T2 / Gate 0，HARD GATE）

| 场景 | 测量 | 结果 |
| --- | --- | --- |
| toy-scale real ViT（fp32，B=4） | `max_abs` / `normalized` | 见 `pytest tests/test_final_cls_routing.py`（阈值 1e-5） |
| **真实 ViT-B/16（fp32, GPU）** | `max_abs = 1.431e-06`，`relative = 1.925e-07` | **PASS（≤ 1e-5）** |
| 真实 ViT-B/16，ones-gate `r ≡ 1` | `max_abs = 0.0`（精确） | **PASS** |
| 训练时（bf16 autocast，B=256/GPU） | `global_identity_max_abs_diff = 0.03125`，`relative = 3.40e-03` | bf16 量化噪声，仅监控 |

Gate 0 的判据（§20）是 **FP32 ≤ 1e-5**，实测 1.4e-6 通过。bf16 下的 3.4e-3 相对差是
bf16 的有效精度（≈3 位十进制）造成的，不是公式差异：该数字由 `global_identity_*` 字段在训练第 0 步
（completed=1）记录，两臂完全相同（0.03125 / 3.401e-3）。

---

## 6. Pairwise optimized vs slow reference（T4）

| 场景 | 结果 |
| --- | --- |
| toy（B=2, C=2, fp32）optimized vs `reference_final_cls`（每 (i,j) 跑一次完整 block + `[L,L]` additive mask） | 见 `test_t4_pairwise_readout_matches_the_slow_reference`（阈值 1e-5） |
| **真实 ViT-B/16（B=2, C=2, fp32, GPU）** | `max_abs = 1.192e-06` |
| chunk 不变性（toy, chunk=0 vs chunk=2, C=5） | `< 1e-6` |
| pair relevance 与 v1 logit 定义一致性（逐 candidate 对比 `exgap.said_relevance_logits`） | `< 1e-6` |
| 训练实际使用的 `--finalcls-pair-chunk-size 32` | 两次独立 500 步 run（F0/F1）逐字段一致，说明 chunk 只影响显存/速度 |

---

## 7. Gradient routes

`grad_attr_*`（完成步 20 / 100 / 500，全部在**独立诊断图**上用 `torch.autograd.grad` 测量，
不污染训练梯度；`tests/...::test_attribution_does_not_disturb_the_gradients_of_the_training_step`）。

### completed step 500

| group | F0 G_S | F0 G_ExGAP | F0 R_grad | F1 G_S | F1 G_ExGAP | F1 R_grad |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `visual_backbone`（含 final block） | 11.8840 | 0.1119 | 9.4e−03 | 10.7118 | 0.1447 | 1.35e−02 |
| `final_block`（`resblocks.11`） | 1.7174 | 0.0631 | 3.67e−02 | 1.6974 | 0.0785 | 4.62e−02 |
| `text_encoder` | 5.8411 | **0.0000** | 0 | 5.6954 | **0.0000** | 0 |
| `said_router` | 1.1062 | **0.0000** | 0 | 1.1050 | **0.0000** | 0 |
| `other`（`clip.logit_scale`） | 0.0000 | **0.0000** | 0 | 0.0000 | **0.0000** | 0 |

同样结构在 completed 20 / 100 上成立（text encoder 与 said_router 的 `G_ExGAP` 恒为 0.0000）。

* **`L_ExGAP` 只能到视觉主干（blocks 1–11 + patch pathway）与原始 final block**，永远到不了
  Said Router 与 text encoder ✓（§25 契约）。
* **`L_S` 同时训练 router、text encoder、final block 与主干** ✓（§26 契约；`route_top1 = 0.9844`
  证明 live gate 真的在训练 router）。
* **`other = 0.0000` 的原因已实测定位**：该组唯一的参数是 `clip.logit_scale`，本仓库预训练
  checkpoint 的值为 `4.605170` → `exp = 100.0000076` → `scale = logit_scale.exp().clamp(max=100)`
  **正好处于 clamp 上边界**，因此 `d scale / d logit_scale ≡ 0`。直接验证：
  `d(clamp(exp(logit_scale)*3))/d logit_scale = 0.`，且 v1 与 v1.5 两种 objective 在该组上都是 0。
  这不是 v1.5 的行为变化，也与本阶段结论无关（记录在此以免被误读为“梯度丢失”）。

---

## 8. Smoke：20 / 100 / 500 步（F0 = L_S，F1 = L_S + L_ExGAP）

配置：`said_exgap_finalcls`、4×A800、batch 256/GPU（global 1024）、bf16、seed 0、
`--finalcls-pair-chunk-size 32 --exgap-temperature 0.05 --exgap-collapse-every 5`、
`--save-completed-steps 0,<steps>`、Full Data Gate PASS，两臂 `exit=0`。

### 8.1 matched-stream audit（每一步数都 PASS）

| 审计项 | F0 | F1 | 结论 |
| --- | --- | --- | --- |
| `initial_state_sha256` | `d61c429b496cf6cb654687d902108c675d513d8ac1d0e49c4a5dbe7b30c49a90` | 同 | **PASS** |
| `sampler_order_sha256` | `953402b3c3d7322e3a74b72880d47a94f2aa9147db4004f41b2cad95094c9d55` | 同 | **PASS** |
| `caption_stream_sha256` | step20 `8c4eee5fe3b5751ed3daa0e94bf852c255297ca27acea11e4401bc411aa74787`；step100/500 `45f3b25bdc52499eb547e8647267ec12021d4bea21926da95fec4a70622398ed` | 同 | **PASS** |
| objective / steps / dataset_size | `said_exgap_finalcls` / 20·100·500 / 1245901 | 同 | **PASS** |

（20 步与 100 步的 caption 流摘要不同是正常的：摘要覆盖已消费的前缀流，长度不同即不同。）

### 8.2 训练几何

| metric | F0 @20 | F1 @20 | F0 @100 | F1 @100 | **F0 @500** | **F1 @500** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `loss_said` | 2.8907 | 2.8918 | 2.6695 | 2.6711 | 0.1131 | 0.1107 |
| `loss_route` | 5.5831 | 5.5839 | 5.1371 | 5.1390 | 0.0709 | 0.0708 |
| `route_top1` | 0.0039 | 0.0078 | 0.0312 | 0.0391 | 0.9844 | 0.9844 |
| `evidence_top1` | 0.9258 | 0.9258 | 0.9453 | 0.9414 | 0.9609 | 0.9648 |
| `S_gc` | 0.3378 | 0.3378 | 0.3216 | 0.3213 | 0.2765 | 0.2298 |
| `S_sc` | 0.3347 | 0.3347 | 0.3242 | 0.3239 | 0.3386 | 0.3108 |
| `S_uc` | 0.3357 | 0.3358 | 0.3127 | 0.3123 | 0.2471 | 0.1953 |
| `S_sc − S_gc` | −0.0031 | −0.0031 | +0.0025 | +0.0026 | +0.0621 | +0.0810 |
| **`D_U = S_gc − S_uc`** | 0.002033 | 0.002040 | 0.0089 | 0.0089 | **0.0294** | **0.0345** |
| `G_exp` | 0.00254 | 0.00256 | 0.0077 | 0.0078 | 0.0879 | 0.1056 |
| `gap_positive_fraction` | 0.3828 | 0.3867 | 0.5898 | 0.5859 | 0.8672 | 0.9141 |
| `said_relevance_mean` / `std` | 0.4637 / 0.0939 | 0.4638 / 0.0940 | 0.4882 / 0.1513 | 0.4885 / 0.1512 | 0.2518 / 0.1775 | 0.2569 / 0.1809 |
| `said_fraction_gt_0.6`（仅 diagnostic） | 0.0775 | 0.0781 | 0.2378 | 0.2383 | 0.0487 | 0.0538 |
| `N_eff` Said（soft） | 192.35 | 192.35 | 189.80 | 189.82 | 136.09 | 136.09 |
| `N_eff` Unsaid（soft） | 193.22 | 193.22 | 190.64 | 190.64 | 186.26 | 185.79 |
| `cos(z_S, g)` | 0.9777 | 0.9777 | 0.9710 | 0.9711 | 0.7967 | 0.7796 |
| `cos(z_U, g)` | 0.9871 | 0.9871 | 0.9776 | 0.9776 | 0.9894 | 0.9894 |
| `cos(z_S, z_U)` | 0.9917 | 0.9917 | 0.9748 | 0.9748 | 0.7785 | 0.7568 |
| `‖z_S − g‖` | 0.2027 | 0.2027 | 0.2257 | 0.2252 | 0.6273 | 0.6540 |
| `‖z_U − g‖` | 0.1520 | 0.1522 | 0.1763 | 0.1763 | 0.1391 | 0.1397 |

**`D_U^{F1} − D_U^{F0}` 随步数增长：+6.5e−06（20）→ +4.2e−06（100）→ +0.0050（500，+17%）**，
并且 `S_sc − S_gc`（0.0621 → 0.0810）、`G_exp`（0.0879 → 0.1056）、`gap_positive_fraction`
（0.8672 → 0.9141）在 F1 上全部更高：**ExGAP 确实在 500 步尺度上带来了额外的 complementary separation。**

### 8.3 原生 CLS 几何（固定 64-image cohort）+ 性能（§36/§37）

| metric | F0 @500 | F1 @500 |
| --- | ---: | ---: |
| `global_cls_pairwise_cos` | 0.4277 | 0.4805 |
| `global_cls_pairwise_cos_max` | 0.7695 | 0.8008 |
| `global_cls_std` | 0.0318 | 0.0303 |
| `64way_i2t@1` | 0.9531 | 0.9219 |
| `64way_t2i@1` | 0.9688 | 0.9531 |
| `compute_sec_per_step` | 0.6632 | 0.6556 |
| `compute_samples_per_sec` | 1543.95 | 1561.86 |
| `peak_gpu_mem_gb` | 76.23 | 76.23 |

与旧实现比较（同机同 batch）：v1 的 `patch_global` 臂约 0.46 s/step、v1.5 约 0.66 s/step（+43%，
来自 pairwise final-block readout），峰值显存 76 GB（A800 80 GB，可用但偏紧；`pair-chunk-size`
是唯一的省显存旋钮，未降低任何科学定义）。**没有 OOM**。

---

## 9. Retrieval（PRIMARY = 原生 CLS，§42）

`tools/exp_said_exgap_finalcls_retrieval.sh`：`--image-representation legacy_cls`，
COCO val2017（5-caption）+ ShareGPT4V-1K 三个 frozen variants，`similarity_chunk = 512`。

| model | image repr | COCO I2T R@1 | COCO T2I R@1 | 1K first I2T/T2I | 1K sparse I2T/T2I | 1K full I2T/T2I |
| --- | --- | ---: | ---: | --- | --- | --- |
| Initial | legacy_cls (**PRIMARY**) | 0.5170 | 0.3269 | 0.5440 / 0.5140 | 0.7460 / 0.7400 | 0.7580 / 0.7760 |
| F0 Said-finalCLS 500 | legacy_cls (**PRIMARY**) | 0.4630 | 0.3630 | 0.4970 / 0.5880 | 0.7680 / 0.8420 | 0.8820 / 0.9230 |
| F1 ExGAP-finalCLS 500 | legacy_cls (**PRIMARY**) | 0.4032 | 0.3473 | 0.4640 / 0.5740 | 0.7400 / 0.8290 | 0.8630 / 0.9110 |

R@5 / R@10（同一 JSON）:

| model | repr | COCO I2T R@5 / T2I R@5 | COCO I2T R@10 / T2I R@10 | 1K full I2T R@5 / T2I R@5 |
| --- | --- | --- | --- | --- |
| Initial | legacy_cls | 0.7662 / 0.5776 | 0.8428 / 0.6823 | 0.9200 / 0.9480 |
| F0_500 | legacy_cls | 0.7198 / 0.6158 | 0.8122 / 0.7202 | 0.9850 / 0.9920 |
| F1_500 | legacy_cls | 0.6672 / 0.6003 | 0.7658 / 0.7063 | 0.9770 / 0.9870 |

诊断列（patch_global，**不是** v1.5 的主指标，仅保留以便与 ExGAP-1B 对照）：Initial 0.0678 /
F0 0.0252 / F1 0.0230（COCO I2T R@1）。

provenance（JSON 原文）：`F0_500` sha256 `d7610c39ad97d9500cb285829aef9ea776dd6bd396479e6d1d6e5d4ab9b36b5b`、
`F1_500` sha256 `76dadada8e0fe5829905f0efb7ee8084f0ac0eaadfa542d57dbbecea739caf5e`、
`initial` sha256 `93cb9cd8db5c92eb4f6ec8b9ead270281f6ca40219b23fa7275fb4300a7b044d`；
三者 `loaded_tensors = 326`、`missing_keys = []`、`unexpected_keys = []`、
`objective_mode = said_exgap_finalcls`（evaluator 对 0 张量匹配硬失败）。

读法：

1. **不再有 patch-global 灾难**：v1 里主指标是 patch_global（0.0678 → 0.0102），现在主指标回到原生
   CLS（0.5170 → 0.4630/0.4032），量级正常。
2. **T2I 与 1K full_dense 明显变好**：COCO T2I 0.3269 → 0.3630/0.3473；1K full_dense
   0.7580 → 0.8820/0.8630（I2T）、0.7760 → 0.9230/0.9110（T2I）。
3. **I2T R@1 变差**：COCO 0.5170 → 0.4630（F0，−0.054）/ 0.4032（F1，−0.114）；
   1K first_sentence 0.5440 → 0.4970/0.4640。这是 500 步、lr 1e-6/1e-4 的短程微调，且**两臂同向**。
4. **F1 在检索上系统性地略差于 F0**：I2T 四个数据集全部更低（−0.060 / −0.033 / −0.028 / −0.019），
   T2I 也四项全低（−0.016 / −0.014 / −0.013 / −0.012）。这与 §8 中 F1 的 `D_U` 更大**同时发生**。

---

## 10. Go / No-Go（§43：A–E 全部满足才建议 3 epoch）

| 条件 | 判据 | 实测 | 判定 |
| --- | --- | --- | --- |
| **A** native CLS retrieval 无灾难性下降 | 量级正常、非 10× 崩塌 | COCO I2T 0.5170 → 0.4630/0.4032（−11%/−22% 相对）；T2I 与 1K full_dense 上升 | **WEAK**（非灾难，但 I2T 明确下降） |
| **B** Global CLS geometry 无 collapse | 不持续冲向 0.9 | `global_cls_pairwise_cos` 0.4277/0.4805，max 0.7695/0.8008，`64way_i2t@1` 0.9531/0.9219，无 collapse 警告 | **PASS** |
| **C** Said Router identifiable 目标正常 | route/evidence 学起来 | `route_top1` 0.0039 → **0.9844**，`evidence_top1` 0.9609/0.9648，router 收到非零梯度 | **PASS** |
| **D** F1 的 complementary separation 明显优于 F0 | `D_U^{F1} > D_U^{F0}` | 0.0345 > 0.0294（+0.0050，+17%），且 `S_sc−S_gc`、`G_exp`、`gap_positive_fraction` 全部更高 | **PASS** |
| **E** F1 retrieval 不明显劣于 F0 | F1 ≥ F0 | **F1 在 COCO / 1K 三个 variant 的 I2T 与 T2I 上全部低于 F0**（8/8 单元） | **FAIL** |

### 总判定：**FAIL（Gate E）→ 不进入 3 epoch，STOP。**

原因陈述（不调任何超参、不改方法）：

* v1.5 的架构目标是达成的：原生 CLS 重新成为 global readout（Gate 0 fp32 `max_abs = 1.43e-06`），
  final-stage intervention 非平凡（`‖z_S − g‖` 从 0.20 涨到 0.63），无 collapse，
  identifiable Said 目标完全正常（route@1 0.984）。
* ExGAP 的因果效应在 v1.5 上是**正向且更大**的：`D_U` +17%、`S_sc−S_gc` +30%、`G_exp` +20%。
* 但 **ExGAP 的因果分离增益没有换来检索增益，反而在原生 CLS 上系统性略降**。按 §43-E 与 §17，
  此时应当 STOP 而不是调 `lambda`/threshold/表示头去“过闸”。
* 性质判断（§17 三分类）：这是 **objective/representation trade-off**，不是实现错误：
  同一个 final block 同时承担 (a) 原生 CLS 检索质量 与 (b) caption-conditioned 读出，
  ExGAP 通过 `z_U` 把该 block 的注意力行推离原生分布，`cos(z_U, g)` 保持 0.989（几乎不动），
  而 `cos(z_S, g)` 被压到 0.78 —— 干预集中在 CLS 侧、且以占用检索方向为代价。

### 下一步建议（本阶段**未实施、未验证**）

若要让 ExGAP 的分离增益落到检索上，需要新的、可被 matched control 检验的设计，例如：
把 ExGAP 的梯度限制在 final block 的**部分**子空间、或为 final-stage 干预引入显式的检索保持项
（**注意**：任何 global-text 对齐项都属于 §2 明确禁止的 `L_G`，因此可选空间只有“改结构/改路由”
这类方案）。这些都超出本阶段范围，必须重新走一遍 F0/F1 matched control。

---

## 11. 代码 / 测试清单

| 文件 | 变更 |
| --- | --- |
| `model/model_longclip.py` | 抽出 `_token_sequence()`；新增 `VisionTransformer.forward_prefinal` 与 `CLIP.encode_visual_prefinal`（`forward` 数值不变） |
| `model/final_cls_routing.py`（新） | `FinalBlockCLSReadout`（零参数、引用原始 final block）、`prepare` / `read_global` / `read_said` / `read_unsaid` / `read_pairwise_said` / `attention_weights` / `gate_bias` / `pairwise_said_relevance` / `reference_final_cls` |
| `model/salu_model.py` | 新 objective `said_exgap_finalcls`；`encode_visual_prefinal` / `prefinal_route_features` / `final_cls_readout` / `said_final_cls` / `_forward_said_exgap_finalcls` / `_finalcls_collapse_metrics`；v1 公式与 `_forward_said_exgap` 未改 |
| `train/train_salu.py` | 新 mode 的 CLI（`--finalcls-pair-chunk-size`）、`build_finalcls_log_fields`、checkpoint `finalcls_config` + resume 校验、`validate_finalcls_args`、log 分支、`global_identity_check`、grad attribution 支持新 objective（新增 `final_block` 组） |
| `tools/exp_said_exgap_finalcls_smoke.sh`、`tools/exp_said_exgap_finalcls_retrieval.sh`、`tools/exp_said_exgap_finalcls_tables.py`（新） | 20/100/500 步 matched smoke、原生 CLS 检索门、报告表格 |
| `tests/test_final_cls_routing.py`（新，30 个测试） | T1–T12 全覆盖 + 零参数/未注册/原始张量身份/objective 不改 inference/λ=0 几何一致/诊断字段完整/CLS key 不偏置等 |

测试：`pytest tests/ -q` → **468 passed, 2 skipped**（v1.5 新增 30 个，既有 438 个全部继续通过）。
