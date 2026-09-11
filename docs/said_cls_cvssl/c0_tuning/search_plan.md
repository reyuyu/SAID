# C0 限额调优 — 搜索计划（执行前固定）

本轮唯一研究目标：**在相同初始化、相同数据流、相同 500 optimizer steps 预算下，找到原生 CLS
检索优于已有 S0@500 的 C0 配置。**

固定基线提交：`beb8c7d7fcf3324260e5a9f5bf653f6df2a7e56e`
分支：`codex/said-cls-cvssl-v01`

## 预算

| 项 | 值 |
| --- | --- |
| 本轮新增训练 | 最多 **4** 个配置，每个 **500** optimizer steps |
| 总新增步数上限 | 2000 steps |
| 只训练 | `C0_complement_vssl` |
| 不重跑 | S0，既有 `lambda_U = 1.0` 的 C0（`C_existing_L10`），G0，R0，3 epoch |
| 达到晋级标准 | 立即停止剩余搜索 |

固定 S0 与既有 C0 的 checkpoint 和检索结果全部复用，不重新评估。

## 固定评测与候选排序指标

主选择指标在本文件写定，看到结果后不更换：

$$J = \frac{R_{\mathrm{COCO,I2T@1}} + R_{\mathrm{COCO,T2I@1}}}{2}$$

用原始 JSON 全精度计算（表格里的四位数只是显示）。晋级判定一律用全精度，不用四舍五入值。

$$J_{S0} = 0.5090800000 \qquad J_{\mathrm{C0\text{-}L10}} = 0.5057200000 \qquad \Delta J = -0.0033600000$$

## 晋级标准（硬门）

`CANDIDATE_PASSED_500STEP_SCREENING`，当且仅当在 step500 上同时满足：

$$R_{\mathrm{COCO,I2T@1}}^{C0} \ge R_{\mathrm{COCO,I2T@1}}^{S0} \quad\wedge\quad R_{\mathrm{COCO,T2I@1}}^{C0} \ge R_{\mathrm{COCO,T2I@1}}^{S0}$$

且**至少一项严格大于**。门槛值（全精度）：

| 指标 | S0@500 门槛 |
| --- | --- |
| COCO I2T R@1 | `0.6058000000` |
| COCO T2I R@1 | `0.4123600000` |

不允许改成"24 格任意一格上涨"，也不允许一个方向上涨掩盖另一个方向下降。

若 $J_{C0} - J_{S0} \ge 0.003$，可额外标记 `larger_screening_gain = true`。这只是"更值得投入完整训练"
的工程信号，不是统计显著性证明，也不是额外硬门。

## 候选顺序（按此顺序执行，达标记即停）

| 次序 | run 名称 | lambda_U target | U 权重安排 | tau_U | 说明 |
| --- | --- | --- | --- | --- | --- |
| — | `C_existing_L10` | 1.0 | 固定 | 0.1 | 历史结果，直接读取，不重跑 |
| 1 | `C_L03` | 0.3 | 固定 | 0.1 | 现有 CLI 即可 |
| 2 | `C_L01` | 0.1 | 固定 | 0.1 | 现有 CLI 即可 |
| 3 | `C_BEST_W200` | 三个固定权重 C0 中按 J 最佳者的 lambda_U | 前 200 步线性渐增（`w = 200`） | 0.1 | 需实现 `--u_weight_warmup_steps` |
| 4 | `C_BEST_T02` | 已评估 C0 中按 J 最佳者的 lambda_U | 沿用该最佳配置的安排 | 0.2 | 只改 tau_U |

排序细则：`J` 相同则优先较小 `lambda_U`；仍相同则优先固定权重配置。"最佳配置"只能从 C0 候选中选，
不能把 S0 的 `lambda_U = 0` 当作最佳 C0。

若四个新配置都未达标：

```
SEARCH_BUDGET_EXHAUSTED
NO_CANDIDATE_PASSED
```

如实报告最接近的配置与差距，不追加第五个配置，不自动改 `rho` / 增广 / mask 类型，不把选择规则改成
"挑一个上涨格子"。

## 所有候选的共同训练条件（逐个核实，不复制不同初始化）

| 项 | 值 |
| --- | --- |
| 共同初始化 | `runs_salu/said_cls_cvssl/shared_init/cvssl_initial.pt`，加载后摘要须为 `caf61198def1b78654d6b70ef16db9a3475ea81b3a16ab8a799989f574de2faf` |
| arm | `C0_complement_vssl`（**不因 run 名称改变**，否则会改动参与随机种子生成的算法标识） |
| 模型 | ViT-B/16，LongCLIP 248 tokens |
| 数据 | Full ShareGPT4V，seed 0 |
| batch | 256 pairs/GPU × 4 A800 = 1024 global pairs，2048 views |
| mask | hard-ST Said mask，`rho = 0`，`duplicate_policy = exclude` |
| 损失权重 | `lambda_align = 10`，`lambda_sparse = 2` |
| optimizer | backbone lr 1e-6 wd 1e-2；mask_net lr 1e-3 wd 0 |
| warmup | backbone 200，mask_net 0 |
| 精度/DDP | fp32 主参数 + bf16 autocast，真实 DDP，`ddp_gradient_averaging = 1`，non-reentrant activation checkpointing |
| 视图 | 保持 `shared_content_weak_v1`：同窗、[192,224] 双线性重采样、k=3 高斯模糊 σ~U[0.1,1.0]；不改裁剪/caption 采样 |
| horizon | `epochs = 3`，`lr_horizon = 3 × len(loader) = 3651`，`max_steps = 500` 只截断 |
| 保存点 | `--save_completed_steps 100,250,500` |

禁止：从 S0@500 继续训练再叫它 C0；从既有 C0@500 接着调参；不同候选使用不同 lr horizon。

默认只评估 **step500**；step100/250 只用于必要时回看，不用于事后挑一个"刚好超过 S0"的中间点。

## 评估协议（冻结，复用现有 canonical evaluator）

主表示固定 `normalize(clip.encode_image(image))` / `normalize(clip.encode_text(text))`；不使用 masked
U / masked Said 特征或重排序替代主检索。COCO val2017 主协议 I2T/T2I R@1/R@5/R@10；ShareGPT4V 固定 1K
`first_sentence` / `fixed_sparse` / `full_dense`。候选规模、预处理、截断长度、划分、frozen manifest、
评估精度与相似度计算方式全部冻结（沿用 `phase30a-1d-fixed-cohort`，`repr=legacy_cls pool=mean`）。

## 记录

结果写入 `docs/said_cls_cvssl/c0_tuning/{search_plan.md,results.csv,progress.json,c0_tuning_report.md}`。
`progress.json` 记录已完成配置、checkpoint / 评估路径、当前最佳、下一个候选；上下文中断后续做先读它。

## 声明

本轮评估集被反复用于超参数选择，属于**开发阶段结果**，不构成独立泛化验证；也不构成统计显著性、
SOTA、Unsaid 语义保留或"互补 mask 优于全局/随机 SSL"的证明。
