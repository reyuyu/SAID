# SAID-ExGAP v1.5 — F0 (Said-only) full 3-epoch diagnostic trajectory

诊断运行，不是方法改动：`objective_mode=said_exgap_finalcls`、`lambda_said=1.0`、`lambda_exgap=0.0`。
**没有修改任何模型 / loss / router / mask / ExGAP / 数据 / 评估代码，没有调超参，F1 未启动。**

- 训练：`runs_salu/said_exgap_finalcls_3ep/F0_said_only_3ep`（4×A800，ViT-B/16，LongCLIP 248，
  full ShareGPT4V，batch 256/GPU → global 1024，bf16，seed 0，backbone lr 1e-6，head lr 1e-4，
  warmup 200，3 epochs = 3648 步）
- 与之前 F0/F1 matched 实验**同一 exact initialization**：`initial_state_sha256 =
  d61c429b496cf6cb654687d902108c675d513d8ac1d0e49c4a5dbe7b30c49a90`，step-0 日志逐字段与 matched 臂一致
  （例：`S_gc_mean = 0.3326192796230316`、`D_U_mean = 0.0015679292846471071`）
- Checkpoint：**step0 / step500 / step1216 / step2432 / step3648** 全部落盘（0.616 GB / 1.818 GB ×4）
- 评测：`tools/exp_said_exgap_finalcls_f0_retrieval.sh`，同一个 canonical evaluator，
  **PRIMARY = `legacy_cls` = `clip.encode_image()`**，COCO val2017 + ShareGPT4V-1K 三 variants，
  `similarity_chunk = 512`；5 个 checkpoint 全部 `loaded 326 tensors, missing [], unexpected []`
- 原始数据：`outputs/said_exgap/f0_3epoch_canonical_retrieval.json`（sha256 见 §6）

---

## 1. 训练侧 trajectory

| metric | step 0 | step 500 | epoch1 (1216) | epoch2 (2432) | epoch3 (3648) |
| --- | ---: | ---: | ---: | ---: | ---: |
| （实际日志步） | 1 | 500 | 1201 | 2451 | 3648 |
| `loss_said` | 3.1810 | 0.1111 | 0.0589 | 0.0658 | **0.0260** |
| `route_top1` | 0.0078 | 0.9844 | 1.0000 | 0.9883 | **1.0000** |
| `evidence_top1` | 0.8633 | 0.9648 | 0.9727 | 0.9688 | **0.9883** |
| `S_gc` | 0.3326 | 0.2765 | 0.3043 | 0.3221 | 0.3306 |
| `S_sc` | 0.3141 | 0.3377 | 0.3451 | 0.3443 | **0.3563** |
| `S_sc − S_gc` | −0.0185 | **0.0612** | 0.0408 | 0.0221 | 0.0257 |
| `cos(z_S, g)` | 0.9312 | 0.7963 | 0.8335 | 0.8326 | 0.8313 |
| relevance mean | 0.3192 | 0.2519 | 0.2865 | 0.2975 | 0.2967 |
| relevance std | 0.0856 | 0.1772 | 0.1939 | 0.1943 | **0.1964** |
| relevance p10 / p50 / p90 | 0.213 / — / 0.434 | 0.051 / — / 0.512 | 0.059 / — / 0.574 | 0.071 / — / 0.586 | 0.066 / — / **0.586** |
| relevance > 0.6（**仅诊断**，非训练 mask） | 0.0016 | 0.0488 | 0.0810 | 0.0895 | 0.0909 |

## 2. 固定 64-image cohort 的原生 CLS 几何

| metric | step 0 | step 500 | epoch1 | epoch2 | epoch3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `global_cls_pairwise_cos` | 0.4961 | 0.4258 | 0.3926 | 0.3691 | **0.3672** |
| `global_cls_std` | 0.0294 | 0.0318 | 0.0329 | 0.0335 | **0.0336** |
| `64way_i2t@1` | 0.9375 | **0.9531** | 0.9219 | 0.8594 | 0.8750 |
| `64way_t2i@1` | 0.9219 | 0.9688 | 0.9688 | 0.9688 | 0.9688 |

`global_cls_pairwise_cos` 全程**下降**（0.4961 → 0.3672），从未接近 0.9 的塌缩阈值；
`64way_i2t@1` 在 step500 达到最高 0.9531，之后回落到 0.8750（64 类随机基线 0.0156），
`64way_t2i@1` 保持 0.9688。

## 3. Canonical retrieval trajectory（PRIMARY = legacy_cls）

| metric | Initial | step500 | epoch1 | epoch2 | epoch3 | 最优 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| COCO I2T R@1 | **0.5170** | 0.4630 | 0.4744 | 0.4878 | 0.4844 | Initial |
| COCO T2I R@1 | 0.3269 | 0.3627 | 0.3697 | **0.3764** | 0.3759 | epoch2 |
| COCO I2T R@5 | **0.7662** | 0.7194 | 0.7246 | 0.7306 | 0.7282 | Initial |
| COCO T2I R@5 | 0.5776 | 0.6145 | 0.6266 | 0.6316 | **0.6325** | epoch3 |
| COCO I2T R@10 | **0.8428** | 0.8122 | 0.8136 | 0.8190 | 0.8166 | Initial |
| COCO T2I R@10 | 0.6823 | 0.7210 | 0.7303 | **0.7359** | 0.7354 | epoch2 |
| 1K first I2T R@1 | **0.5440** | 0.4930 | 0.5300 | 0.5380 | 0.5300 | Initial |
| 1K first T2I R@1 | 0.5140 | 0.5850 | 0.6190 | 0.6260 | **0.6290** | epoch3 |
| 1K sparse I2T R@1 | 0.7460 | **0.7670** | 0.7390 | 0.7230 | 0.7260 | step500 |
| 1K sparse T2I R@1 | 0.7400 | 0.8410 | 0.8640 | 0.8700 | **0.8720** | epoch3 |
| 1K full I2T R@1 | 0.7580 | 0.8850 | 0.9010 | 0.9180 | **0.9200** | epoch3 |
| 1K full T2I R@1 | 0.7760 | 0.9270 | 0.9450 | 0.9490 | **0.9500** | epoch3 |

诊断用的复合量（**定义在此**：8 个 R@1 单元的算术平均，只是本报告的自定义汇总，不是协议指标）：

| checkpoint | Initial | step500 | epoch1 | epoch2 | epoch3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8-cell mean R@1 | 0.61524 | 0.66546 | 0.68027 | **0.68603** | 0.68591 |
| 相对 Initial | — | +0.0502 | +0.0650 | **+0.0708** | +0.0707 |

## 4. 三个问题的直接回答

### 4.1 COCO I2T 在 500 步的下降是否会在 epoch1/2/3 恢复？

**部分恢复（46%），但不会回到 Initial。**

```
0.5170 (Initial) → 0.4630 (step500, −0.0540 = −10.4%)
                 → 0.4744 (epoch1, 已收回 +0.0114)
                 → 0.4878 (epoch2, 最佳，已收回 +0.0248 = step500 跌幅的 45.9%)
                 → 0.4844 (epoch3, 与 epoch2 基本持平)
剩余与 Initial 的差距：−0.0292
```
同类部分恢复也出现在 1K `first_sentence` I2T（0.5440 → 0.4930 → **0.5380** @epoch2，差距仅 −0.0060）
与 COCO I2T R@5/R@10（best 都在 epoch2）。**结论：下降是“早期下冲后部分回弹”，不是一次性损伤，也不是完全恢复。**

### 4.2 T2I / sparse / full_dense 的提升是否继续增长？

**T2I 与 full_dense 是持续增长并在 epoch3 达到最高的；sparse I2T 相反。**

* T2I 四个数据集**全部**单调上升，且 epoch3 最高：COCO 0.3269 → 0.3759（+15%）、
  1K first 0.5140 → 0.6290（+22%）、1K sparse 0.7400 → 0.8720（+18%）、1K full 0.7760 → 0.9500（+22%）。
  其中 COCO T2I R@1/R@10 在 epoch2 见顶（0.3764 / 0.7359），epoch3 持平或 −0.0005。
* 1K `full_dense` I2T **持续增长到底**：0.7580 → 0.8850 → 0.9010 → 0.9180 → **0.9200**（+21.4%）。
* 1K `sparse` I2T **相反**：0.7460 → **0.7670（step500 见顶）** → 0.7390 → 0.7230 → 0.7260，
  epoch3 反而**低于** Initial（−0.0200）。这是唯一一个“早期见顶后跌破起点”的单元。

### 4.3 是否出现 representation collapse？

**没有。** 三项独立证据：

1. `global_cls_pairwise_cos`：0.4961（step0）→ 0.4258 → 0.3926 → 0.3691 → **0.3672**，
   全程**单调下降**，最大值为初始的 0.4961，**从未 ≥ 0.9**（对照：已塌缩的 Full-Base
   gap_completion 臂是 0.977）。
2. `global_cls_std` 单调上升 0.0294 → **0.0336**（表示更分散而非收缩）。
3. 下游 `64way_i2t@1` 最低 0.8594、`64way_t2i@1` 恒为 0.9688，远高于 1/64 = 0.0156 的随机水平。

### 4.4 F0 最佳 checkpoint 在哪里？

**按 8-cell 复合量：epoch2（step 2432，0.68603），epoch3（0.68591）几乎并列为第二。**
分单元看（8 个 R@1 单元的最佳 checkpoint 分布）：epoch3 占 4 个（全部 T2I 与 1K full I2T）、
Initial 占 3 个（COCO I2T、COCO I2T R@5/R@10、1K first I2T）、epoch2 占 1 个（COCO T2I R@1）、
step500 占 1 个（1K sparse I2T）。
**如果只看“相对 Initial 不亏”的部署用途，epoch2/epoch3 是同一个平台期；若要单一 checkpoint，
取 epoch2（复合最高），但 epoch3 在 T2I 与 full_dense 上更好——即“最佳”取决于用哪个单元衡量。**

## 5. 终局判定

| 候选模式 | 是否成立 | 依据 |
| --- | --- | --- |
| **RECOVERS** | **部分 / 否** | COCO I2T 只收回 45.9% 且最终仍 −0.0292；1K first I2T 差 −0.0060；没有任何一个 I2T 单元超过 Initial |
| **PERSISTENT TRADE-OFF** | **是（主判定）** | 3/4 数据集的 I2T 在 epoch3 仍低于 Initial（COCO −0.0326、first −0.0140、sparse −0.0200），而 **8/8 单元中的 5 个显著更好**（T2I ×4 与 1K full I2T +0.1620），trade-off 在整个 3 epoch 内**稳定存在、不随时间消失** |
| **PEAKS EARLY** | 局部成立 | 1K sparse I2T 在 step500 见顶后跌破 Initial；64way_i2t@1 也在 step500 见顶（0.9531 → 0.8750）。但复合量在 epoch2 见顶、1K full 与全部 T2I 到 epoch3 仍在涨，所以**不是整体 early peak** |
| **COLLAPSES** | **否** | `global_cls_pairwise_cos` 单调下降至 0.3672（< 0.9），`global_cls_std` 上升，64-way 检索远高于随机 |

```text
VERDICT: PERSISTENT TRADE-OFF
  （无 collapse；COCO I2T 早期下冲后部分回弹至 epoch2 平台，仍低于 Initial；
    T2I 与 ShareGPT4V-1K full_dense 在整个 3 epoch 内持续改善；
    最佳 checkpoint：epoch2（复合最高），epoch3 在 T2I / full_dense 上更好）
```

**STOP —— 未启动 F1 3 epoch，未修改方法。**

---

## 6. Provenance

| checkpoint | step | sha256 |
| --- | ---: | --- |
| `salu_exgap_step000000.pt` (Initial) | 0 | `26a793bc82b5c413db1fffd14a3e8601d4ee00cc8da820d87563ba9d3c824e1f` |
| `salu_exgap_step000500.pt` | 500 | `d2d1185554831677e0aaeba7d91caa908cd8f19a0af529316e09e3a7fad2680d` |
| `salu_exgap_step001216.pt` (epoch1) | 1216 | `44d0e26192183a83a57c0b37f89fa4633d86896e5daaf0a24ae07c108f0a3d9a` |
| `salu_exgap_step002432.pt` (epoch2) | 2432 | `179a8c71c297f8b1b931e9bb2542b1b88731fe8dcdd077ffacd07db62cec7a20` |
| `salu_exgap_step003648.pt` (epoch3) | 3648 | `0b0525e576f08a7b383071b59b71263e9f41d7bf63bec8a40044b5334790b85c` |

`outputs/said_exgap/f0_3epoch_canonical_retrieval.json` sha256 =
`3d662ea2cdd0f349b95be510510f9406a930d3633cfa13cb7086a32cb19bff21`；
5 个 checkpoint 全部 `loaded 326 tensors, missing [], unexpected []`。

- 训练：`ALL_F0_3EP_DONE`，`ARM_EXIT=0`，3648/3648 步，0.727–0.752 s/step，峰值显存 76 GB，0 报错
- 评测：`RETRIEVAL_EXIT=0`，5 个 checkpoint 全部 `loaded 326 tensors, missing [], unexpected []`
- 复现命令：`bash tools/exp_said_exgap_finalcls_f0_3epoch.sh` →
  `bash tools/exp_said_exgap_finalcls_f0_retrieval.sh` →
  `python tools/exp_said_exgap_finalcls_f0_trajectory.py`
