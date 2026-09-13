# 均衡损失权重臂 `S0_TriMask_HS_BAL` @500 次更新 —— 结果

按你的要求新增一个权重平衡的 TriMask-HS 臂：**三项对齐权重都取 10**（原为 10/1/1），**两项稀疏权重都取 2**（原为 2/0.2）。从共享初始化重新训练**恰好 500 次 optimizer update**，与两个冻结对照臂在同预算下比较。

**结论一句话：均衡权重这一版明显更差，而且不是噪声。** COCO 图→文 R@1 相对原权重臂低 **1.14pp**（配对检验 95% CI [−1.74, −0.56]，McNemar p = 0.00024），相对门槛基准低 **1.66pp**（CI [−2.40, −0.92]，p = 1.2e−5），门槛判定 **FAIL**。

## 权重设定与实现方式

| 臂 | λ1 / λ2 / λ3 | λ_sparse_I / λ_sparse_T | arm / objective | profile |
|---|---|---|---|---|
| `S0_smartclip`（门槛基准） | 10 / 1 / 1 | 2 / 0（无文本稀疏项） | `S0_smartclip` | — |
| `S0_TriMask_HS`（冻结 v0.2 对照） | 10 / 1 / 1 | 2 / 0.2 | `S0_TriMask_HS` / `smartclip_trimask_hs` | `default` |
| **`S0_TriMask_HS_BAL`（本轮新臂）** | **10 / 10 / 10** | **2 / 2** | `S0_TriMask_HS_BAL` / `smartclip_trimask_hs_bal` | `balanced` |

实现方式（默认行为逐位不变）：新增"损失权重 profile"概念，profile 同时拥有**五个系数**与 **arm/objective/phase 名**。`default` profile 返回原值与原名字，因此所有既有 checkpoint 与启动脚本语义不变；`balanced` profile 只在 `hard_st` 门下定义，名称带 `_BAL` 后缀。训练器把 profile 写进 config 与 checkpoint，`check_checkpoint_compatibility` 现在会**逐项比对五个系数**（原来只查 λ_sparse_T），runner 的检查点校验也不再硬编码 10/1/1 —— 也就是说，一个权重不同的 checkpoint 不可能被当成另一版继续训练或导出。

## 训练

| 项目 | 均衡权重臂 | 原权重臂（对照） |
|---|---|---|
| 更新步数 / LR horizon | 500 / 3651 | 500 / 3651 |
| 墙钟 / 秒每步 | 516.8 s / 0.9055 | 520.6 s / 0.9085 |
| 峰值显存 | 24.48 GiB | 24.48 GiB |
| checkpoint | `d12179649a3a2dd2d4a303ab4410826d32845c0fb2471b3510a91a7a63016fb0` | `46f6d9c7e31ce3298d08e87a32b4228ca3bf5e216a674079deb5b0fa9fc0c4ef` |
| 裸学生 | `6d23dde353e1ab18c3de1e43793525c3139464f6fec44741441e6d802b186f17` | `c9ad482c3df3a2628eeb2985f259e09ec2efec065bd9b75ab050fca9ace40987` |

共享初始化、batch 256×4、LR 1e-6 / mask LR 1e-3、warmup 200、seed 0、三个 epoch 的 LR 计划均未变；两臂唯一差别就是上表的权重。

## 结果（标准评估器，native CLS/EOS，R@1/5/10）

| 臂 | COCO I2T | COCO T2I | Urban I2T | Urban T2I | 500 步门槛 |
|---|---|---|---|---|---|
| `S0_smartclip`（基准） | 0.6058/0.8220/0.8906 | 0.41236/0.6709/0.7662 | 0.8700/0.9710/0.9880 | 0.8420/0.9670/0.9810 | 基准（门槛定义） |
| `S0_TriMask_HS`（10/1/1+2/0.2） | 0.6006/0.8240/0.8870 | 0.41208/0.6710/0.7660 | 0.8740/0.9750/0.9890 | 0.8370/0.9630/0.9820 | 未通过 |
| **`S0_TriMask_HS_BAL`（10/10/10+2/2）** | **0.5892**/0.8184/0.8812 | **0.40828**/0.6692/0.7654 | **0.8650**/0.9720/0.9860 | **0.8280**/0.9620/0.9800 | **未通过（FAIL）** |

差值（百分点）：均衡臂相对原权重臂 图→文 **−1.14**、文→图 **−0.38**（Urban −0.90 / −0.90）；相对门槛基准 图→文 **−1.66**、文→图 **−0.41**（Urban −0.50 / −1.40）。逐指标表见 `weight_balance_tables.md`。

## 配对逐查询证据（COCO val2017，同一批查询）

0.5pp 量级的点估计单看没有意义（5000 个查询的二项标准误约 0.7pp），因此本轮同样复用了既有的逐查询配对工具（自算聚合值已断言等于标准评估器写出的数字）：

| 比较 | 方向 | R@1 左 | R@1 右 | Δ(pp) | 95% CI (pp) | 排除 0 | 仅左对/仅右对 | McNemar p |
|---|---|---|---|---|---|---|---|---|
| BAL vs HS(10/1/1+2/0.2) | 图→文 | 0.5892 | 0.6006 | **−1.14** | [−1.74, −0.56] | 是 | 89/146 | **0.00024** |
| BAL vs HS(10/1/1+2/0.2) | 文→图 | 0.4083 | 0.4121 | −0.38 | [−0.57, −0.19] | 是 | 244/339 | **9.6e−05** |
| BAL vs S0（门槛基准） | 图→文 | 0.5892 | 0.6058 | **−1.66** | [−2.40, −0.92] | 是 | 136/219 | **1.2e−05** |
| BAL vs S0（门槛基准） | 文→图 | 0.4083 | 0.4124 | −0.41 | [−0.62, −0.20] | 是 | 298/400 | **1.3e−04** |

四个比较的置信区间都排除 0，且"仅右对"的查询数明显多于"仅左对"（146 vs 89、339 vs 244 等），方向一致：**均衡权重这一版的落后是系统性的，不是几个查询的偶然**。R@5/R@10 也同向（如 vs 原权重臂 图→文 R@5 −0.56pp、R@10 −0.58pp，CI 均排除 0）。

## 训练侧的旁证与一处需要拆分的混淆

- **不是"只输在评估"**：两臂在 step 500 的**未加权**分项损失，均衡臂都更高（L1 0.3451 vs 0.3251、L2 0.3626 vs 0.3253、L3 0.3710 vs 0.3466）。也就是说把 L2/L3 的权重提高 10 倍，并没有把这两项压得更低，反而整体变差。加权总损失不可直接比较（权重不同），故不列出比较。
- **文本 mask 的关闭比例**（各自 `mask_snapshot.json`，step 500、训练 batch，单一 batch 快照）：均衡臂 keep 0.9304 / 关闭 6.96%，原权重臂 keep 0.9117 / 关闭 8.83%。注意这是**单 batch 快照**，且两臂的目标函数在两处同时不同，因此**不能**从中读出"λ_sparse_T 越大门关得越少"这类因果结论。（上一轮对冻结 checkpoint 的离线诊断给出的是另一批数据上的数字：COCO 256 图的 mT keep 0.9501、mI keep 0.8132，口径不同，不可混用。）
- **必须说明的混淆**：本轮同时改了两件事——对齐权重（1→10）与文本稀疏权重（0.2→2）。因此结果只能归因于**这个组合**，无法区分是"拉平对齐权重"还是"把文本稀疏压力提高 10 倍"造成的。要拆分需要再做两个单因素臂（例如只把对齐拉平、λ_sparse_T 仍取 0.2），本轮不做。

## 过程问题（都已在代码与测试层面修掉）

1. **导出工具的白名单是硬编码的**，只接受 `smartclip_trimask` / `smartclip_trimask_hs`，因此新 arm 的 objective 被拒，第一次运行在**导出阶段**失败（`export exited 1`）。训练本身已完成并保存了 step 100/250/500 检查点，检查点校验也通过；修复后以 `--phases export,coco,urban` 续跑剩余阶段完成，**没有重训**。现在白名单由 profile 表生成，并新增 `--expect-loss-profile`，新 arm 不会再因为"新"而被拒，而未知的 arm/objective 组合仍然会被拒。
2. **前端页面当时是坏的，这是我的 bug**：我在上一轮加"离线诊断"区域时写的 `obj.R@1_mean`、`paired.delta_R@1` 在 JavaScript 里非法（`@` 不能出现在标识符中），整个 `app.js` 解析失败，于是页面 HTML 正常加载但所有动态内容（运行列表、曲线、表格）都不渲染——表现就是"什么记录都看不到"。服务端 API 一直是正常的（`/health`、`/api/runs` 均 200）。修复后用 `node --check` 复验，并新增测试 `test_frontend_assets_parse_and_are_served`：包含对非法属性写法的扫描与（有 node 时）真实的语法检查——用带 bug 的副本验证过它确实会失败。

## 建议的下一步（本轮不执行）

1. 先做**单因素拆分**：把"三项对齐都取 10、λ_sparse_T 仍 0.2"与"对齐仍 10/1/1、两项稀疏都取 2"分别跑 500 步，才能知道是拉平对齐权重还是提高文本稀疏压力导致的下降。
2. 若只想改善"不平衡"而保留已验证有效的配置，目前证据支持**不要动 λ2/λ3 与 λ_sparse_T**：原权重臂在三个可比指标上都优于均衡臂，且它与门槛基准的差距（图→文 −0.52pp）在配对检验下并不显著。
3. 上一轮诊断指出的方向（文本门平均保留 95% 的坐标、NORMAL ≈ 全开）提示：文本门的收益本来就很小，**提高其稀疏压力**是一个风险较高的方向，与本轮结果一致。

## 复现与 SHA

```bash
cd /root/SAID-s0-trimask-hs-v02
python tools/trimask_hs_runner.py --run-id s0_trimask_hs_bal \
  --run-dir runs_salu/said_s0_trimask_hs_v02/bal500 \
  --loss-profile balanced --save-steps 100,250,500 --steps 500
# 若训练已完成、只补评估阶段：
python tools/trimask_hs_runner.py --run-id s0_trimask_hs_bal \
  --run-dir runs_salu/said_s0_trimask_hs_v02/bal500 \
  --loss-profile balanced --steps 500 --phases export,coco,urban
# 三臂对比：
python tools/diag/report_trimask_weights.py \
  --paired runs_salu/said_s0_trimask_hs_v02/bal500/evaluation/paired_BAL_vs_HS_vs_S0.json \
  --out docs/said_trimask_hs/weight_balance_results.json \
  --markdown docs/said_trimask_hs/weight_balance_tables.md
```

| 产物 | SHA256 |
|---|---|
| checkpoint `trimask_S0_TriMask_HS_BAL_step000500.pt` | `d12179649a3a2dd2d4a303ab4410826d32845c0fb2471b3510a91a7a63016fb0` |
| 裸学生 `student_000500.pt` | `6d23dde353e1ab18c3de1e43793525c3139464f6fec44741441e6d802b186f17` |
| `evaluation/S0_TriMask_HS_BAL_step000500_canonical.json` | `af51fd52f2b8abd4b192af843b306a3c9b518740ba7e7952abf3bfc9e25219f4` |
| `evaluation/S0_TriMask_HS_BAL_step000500_urban1k.json` | `caa834325fe0e48d308d7a40c715d4c702be9d5fcd87e24dea68ae549d10d7d3` |
| `evaluation/paired_BAL_vs_HS_vs_S0.json` | `d3d6ec65707b9ca62483050f308429c59c0cf839e178d784af9338b88581157a` |
| 逐查询命中 `query_hits_S0_TriMask_HS_BAL_step000500.json` | `42186bf936b5906a785e603d9d32a09be57cea3b889e15f00007673bb5408dcb` |

两个冻结臂（`S0_smartclip`@500、`S0_TriMask_HS`@500）的检查点、日志与评估文件均未被修改。
