# C0 限额调优 — 单候选结果（lambda_U = 0.3）

本轮按指令**只执行一个实验**：`C_L03`，`arm = C0_complement_vssl`，`lambda_U = 0.3`，
`tau_U = 0.1`，固定 U 权重（`--u_weight_warmup_steps 0`）。`C_L01`、`C_BEST_W200`、`C_BEST_T02`
**未执行**（原因见 §6）。

工作分支 `codex/said-cls-cvssl-v01`；预算内代码提交 `f3d109a2e91f45d2f57a5dc033f297c7a9942a20`
（U 权重渐增开关 + 调优工具 + 前置搜索计划），训练与评测都在该提交上运行。
参考基线提交 `beb8c7d7fcf3324260e5a9f5bf653f6df2a7e56e`。

## 1. 冻结的排序指标与晋级门（使用全精度）

$$J = \frac{R_{\mathrm{COCO,I2T@1}} + R_{\mathrm{COCO,T2I@1}}}{2}$$

* $J_{S0} = 0.5090800000$
* 晋级门：COCO I2T R@1 $\ge 0.6058000000$ **且** COCO T2I R@1 $\ge 0.4123600000$，且至少一项严格大于
* `larger_screening_gain` 阈值：$J_{C0}-J_{S0} \ge 0.003$（工程信号，非硬门）

## 2. 训练条件核实（A/§4）

| 项 | 实测值 | 判定 |
| --- | --- | --- |
| 初始化 | `initial_state_sha256 = caf61198def1b78654d6b70ef16db9a3475ea81b3a16ab8a799989f574de2faf` | 与历史共享 init 一致 |
| arm | `C0_complement_vssl`（run 名称未参与算法标识） | 一致 |
| `lambda_U` / `tau_U` | 0.3 / 0.1 | 一致 |
| `lambda_U_target` / `lambda_U_effective` | 全程 `0.3` / `0.3` | 固定权重路径未被调度改动 |
| `u_weight_warmup_steps` | 0 | 默认值，历史行为 |
| `steps_per_epoch` / `lr_horizon_steps` | 1217 / 3651（= 3 × len(loader)） | 与 S0/C0 相同 |
| `completed_steps` | **500** | 达标 |
| `sample_stream_sha256` | `39c9885c41aaf11cd79a05b7828cf07b1cfa6af2d4ad507c978e870f3a422856` | 与 S0@500 / C0-L10@500 相同 |
| `caption_stream_sha256` | `799f8efae42541759ac92239643d2150dc8aa23d1da984676f029063c21c0fd4` | 相同 |
| `view_b_param_stream_sha256` | `cbfb7ffdfee3151a214beb57474a796adc9f5545702a73df4f8f1cb5fbbe7c19` | 相同 |
| `view_b_pixels_sha256_step0` | `81888072f9b73f6f` | 相同 |
| checkpoint | step100 / step250 / step500 均存在；step500 严格加载 `missing=[] unexpected=[]`，317/317 fp32 | 达标 |
| 同步/稳定性 | 21/21 个不同 `rank_param_digest`，0 非有限值，无 OOM/死锁 | 达标 |
| 峰值显存 / 步时 | 27.05 GB / 1.19 s（500 步 wall 642.9 s） | 正常 |

## 3. step500 检索结果（canonical，`phase30a-1d-fixed-cohort`）

| 指标 | S0@500 | C0-L10@500 | **C_L03@500** | C_L03 − S0 |
| --- | --- | --- | --- | --- |
| COCO I2T R@1 | 0.6058 | 0.6042 | **0.6046** | **−0.0012** |
| COCO I2T R@5 | 0.8220 | 0.8224 | **0.8268** | +0.0048 |
| COCO I2T R@10 | 0.8906 | 0.8896 | **0.8942** | +0.0036 |
| COCO T2I R@1 | 0.4124 | 0.4072 | **0.4128** | **+0.0004** |
| COCO T2I R@5 | 0.6709 | 0.6655 | **0.6710** | +0.0001 |
| COCO T2I R@10 | 0.7662 | 0.7636 | **0.7664** | +0.0002 |
| 1K first I2T R@1 | 0.6620 | 0.6670 | **0.6670** | +0.0050 |
| 1K first I2T R@5 | 0.8940 | 0.8960 | **0.8980** | +0.0040 |
| 1K first I2T R@10 | 0.9520 | 0.9550 | **0.9540** | +0.0020 |
| 1K first T2I R@1 | 0.6350 | 0.6220 | **0.6270** | −0.0080 |
| 1K first T2I R@5 | 0.8580 | 0.8610 | **0.8570** | −0.0010 |
| 1K first T2I R@10 | 0.9290 | 0.9310 | **0.9270** | −0.0020 |
| 1K sparse I2T R@1 | 0.9010 | 0.9030 | **0.9000** | −0.0010 |
| 1K sparse I2T R@5 | 0.9820 | 0.9850 | **0.9830** | +0.0010 |
| 1K sparse I2T R@10 | 0.9970 | 0.9970 | **0.9970** | +0.0000 |
| 1K sparse T2I R@1 | 0.8790 | 0.8840 | **0.8840** | +0.0050 |
| 1K sparse T2I R@5 | 0.9710 | 0.9670 | **0.9710** | +0.0000 |
| 1K sparse T2I R@10 | 0.9820 | 0.9820 | **0.9840** | +0.0020 |
| 1K full I2T R@1 | 0.9650 | 0.9640 | **0.9660** | +0.0010 |
| 1K full I2T R@5 | 0.9980 | 0.9980 | **0.9980** | +0.0000 |
| 1K full I2T R@10 | 0.9990 | 0.9990 | **0.9990** | +0.0000 |
| 1K full T2I R@1 | 0.9620 | 0.9620 | **0.9610** | −0.0010 |
| 1K full T2I R@5 | 0.9970 | 0.9980 | **0.9970** | +0.0000 |
| 1K full T2I R@10 | 0.9990 | 0.9990 | **0.9990** | +0.0000 |

| 汇总（全精度） | 值 |
| --- | --- |
| COCO I2T R@1 | `0.6046000000`（门 `0.6058000000`，**FAIL**） |
| COCO T2I R@1 | `0.4127600000`（门 `0.4123600000`，**PASS**） |
| 至少一项严格提升 | 是（T2I） |
| $J_{C\_L03}$ | `0.5086800000` |
| $\Delta J$ vs S0 | `−0.0004000000` |
| `larger_screening_gain` | `false`（需 ≥ 0.003） |
| **晋级** | **NO** |

C_L03 是三个固定权重 C0 中**最好**的一个（`J`: C_L03 `0.50868` > C_existing_L10 `0.50572`；
C_L01 未跑），但仍差 S0 一个方向：T2I R@1 首次超过 S0（+0.0004），而 I2T R@1 低 0.0012，
不满足"两个方向都不低于 S0"的硬门。

## 4. 训练诊断

| 量（steps 251–500 均值） | C_L03 | C0-L10@500（对照） |
| --- | ---: | ---: |
| `loss_smart_global_mean` | 6.335 | 5.745 |
| `loss_vssl_global_mean` | 2.196 | 0.896 |
| `lambda_U_effective` | 0.300 | —（1.0） |
| U 有效比例 | 1.000 | 1.000 |
| Said / U 坐标占比 | 0.812 / 0.188 | 0.787 / 0.213 |
| Said / U 保留能量 | 0.752 / 0.248 | 0.803 / 0.197 |
| 峰值显存 | 27.05 GB | 27.05 GB |
| 步时 | 1.19 s | 1.21 s |
| 500 步训练 wall | 642.9 s | 707.4 s |
| 评测 wall | ≈ 240 s | ≈ 660 s（3 个 checkpoint） |
| 非有限值 / rank digest | 0 / 21 个均不同 | 0 |

`lambda_U_effective` 全程等于 target，说明固定权重路径未被调度机制影响。
不同温度或不同权重之间**不按 U loss 大小选模型**；所有候选统一用 $J$ 排序。

## 5. 文件与 provenance

| 项 | 路径 |
| --- | --- |
| 训练目录 | `runs_salu/said_cls_cvssl/c0_tuning/C_L03/` |
| step500 checkpoint | `runs_salu/said_cls_cvssl/c0_tuning/C_L03/cvssl_C0_complement_vssl_step000500.pt` |
| checkpoint sha256 | `bae2f576ba9f3be90f440e0edf771bfe8969f9fc5d03fc35bc6223718b74247c` |
| 检索 JSON | `outputs/cvssl_screening/c0_tuning/C_L03_canonical.json` |
| 本次编排日志 | `/tmp/c0_search.log`、`/tmp/eval_C_L03.log` |
| 台账 | `docs/said_cls_cvssl/c0_tuning/{search_plan.md,results.csv,progress.json,c0_tuning_report.md}` |

## 6. 未执行候选及跳过原因

| 候选 | 状态 | 原因 |
| --- | --- | --- |
| `C_L01` | **未执行** | 本轮指令要求只执行一个实验（`lambda_U = 0.3`） |
| `C_BEST_W200` | **未执行** | 同上；其 `lambda_U` 需从三个固定权重 C0 中选最佳，而 C_L01 未跑 |
| `C_BEST_T02` | **未执行** | 同上 |

`--u_weight_warmup_steps` 已实现并有 7 个针对性测试（`w=0` 恒等于 target；`w=200` 时第 1 步
= target/200、第 100 步 = target/2、第 200 步及以后 = target；U 项只乘一次 effective weight；
SmartCLIP 项不受调度影响；CLI 默认 0）。本轮 `C_L03` 使用默认 `w = 0`，未走该路径。

## 7. 本轮结论与声明

* `C_L03` **未通过**预设晋级门：I2T R@1 低 0.0012，只有 T2I R@1 以 +0.0004 超过 S0。
* 它在 24 格中 17 格不低于 S0，是当前最好的 C0 固定权重配置，但硬门要求**两个方向都不低于**
  且至少一项严格提升，因此判定 NO。
* 本轮**不宣布算法失败**，也不据此调参；按指令停在此处，不自动启动 G0/R0、3 epoch、也不追加
  第 5 个配置。
* 本轮评估集被反复用于模型选择，属**开发阶段结果**，不构成独立泛化验证，也不构成统计显著性、
  正式 SOTA、Unsaid 语义保留或"互补 mask 优于全局/随机 SSL"的证明。
