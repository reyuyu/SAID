# S0 与 S0-TriMask-HS 续训至 1000 次更新 —— 结果

两臂各自从自己的 @500 checkpoint 诚实续训到**恰好 1000 次 optimizer update**：数据流经 digest 校验确认从 batch 500 继续（不是重训前 500 个 batch），LR horizon 仍是 3651，目标函数、超参、global batch 1024 全部未变。只评估了 COCO canonical 与 Urban-1k，全部交给标准评估器。

**结论一句话**：在 1000 步预算下，S0_TriMask_HS 与 S0_smartclip 在 COCO 图→文 R@1 上**无法区分**（配对检验 Δ = -0.24pp，95% CI [-0.80, +0.30]，p = 0.43）；当年 500 步门槛判定 HS 落败所依据的 -0.52pp 差异，在配对检验下也并不显著（p = 0.069），即**那次判负建立在一个噪声级差异上**。

## 训练

| 项目 | S0_smartclip | S0_TriMask_HS |
|---|---|---|
| 父 checkpoint（@500） | `758dcdd2a9112d54ccb8784e211d329297209c5d7be26cd7a89e68185867c743` | `46f6d9c7e31ce3298d08e87a32b4228ca3bf5e216a674079deb5b0fa9fc0c4ef` |
| 续训 checkpoint（@1000） | `0959b01b4445be14feaf1976d8986ec0697b9173d48b72ca84f5e4d75345658b` | `4848e9c574c6725eaf48bda7b450f62d220a78e915b6ae3189c02c3ffaafdcef` |
| 导出的裸学生 | `b7db22bc9d9ee0d315a1aea20101d6089167a755a882e357d623c46d57a5071d` | `a101afee7c3fc10437c255d234aa590c584fa3fae24d83d6973dfe07673227e8` |
| 完成步数 / LR horizon | 1000 / 3651 | 1000 / 3651 |
| 训练墙钟 / 秒每步 | 886.3 s / 0.9354 | 873.6 s / 0.9217 |
| 峰值显存 | 33.56 GiB | 24.48 GiB |
| `replay_verified` / `resume_start_step` | true / 500 | true / 500 |
| 代码 | `codex/s0-continue-1000`（运行记录 `git_head=392e9ec3`） | `codex/s0-trimask-hs-v02`（`git_head=cd4308e4`） |

数据流守卫两条都通过，且**两臂的累计 digest 完全相同**——续训读的确实是同一批数据：

```
REPLAY_VERIFIED skipped_batches=500
  caption_sha=799f8efae42541759ac92239643d2150dc8aa23d1da984676f029063c21c0fd4   (父 @500 前缀)
  sample_sha=39c9885c41aaf11cd79a05b7828cf07b1cfa6af2d4ad507c978e870f3a422856
  view_sha=cbfb7ffdfee3151a214beb57474a796adc9f5545702a73df4f8f1cb5fbbe7c19
@1000 累计流（两臂逐位相同）
  caption_stream_sha256=5a656f58de116c75a5025b131e8d244b7e928a748a16572f654100d6c7306150
  sample_stream_sha256=92cb40830292fd65ad0195135c9d33f5bd8f4e02e22ae6b75210322f985ee9d6
```

### 事故：梯度探针 OOM（S0 第 2 次尝试）

第 2 次尝试跑到 step 740、在 **step 750** 于四个 rank 上同时 OOM：

```
torch.OutOfMemoryError: Tried to allocate 592.00 MiB. GPU 0 ... of which 169.81 MiB is free.
Process 235259 has 79.13 GiB memory in use.
```

崩点在 `grad_probe -> norms -> torch.autograd.grad(..., retain_graph=True)`：探针在训练图仍存活时对 2 个 loss term × 3 个参数组反复保留反向图，稳态约 37 GiB 之上再要 ~42 GiB。查证后确认**这个探针是本续训 runner 自己多加的诊断**：冻结 S0@500 运行的 `config.json` 里没有 grad_probe 字段，训练器 `--grad_probe_steps` 默认为空。因此修复方式是**删掉它**（改为 `--grad-probe-steps` 可选、默认空），顺带让训练器调用与冻结运行完全一致；没有任何指标依赖探针。该次尝试未写出 checkpoint，故从 @500 重来（attempt 3 成功）。

## 检索结果（原生 CLS/EOS，R@1/5/10）

| 模型 | COCO I2T | COCO T2I | Urban I2T | Urban T2I |
|---|---|---|---|---|
| S0@500（门槛定义本身） | 0.6058/0.8220/0.8906 | 0.41236/0.6709/0.7662 | 0.8700/0.9710/0.9880 | 0.8420/0.9670/0.9810 |
| S0_TriMask_HS@500 | 0.6006/0.8240/0.8870 | 0.41208/0.6710/0.7660 | 0.8740/0.9750/0.9890 | 0.8370/0.9630/0.9820 |
| **S0@1000** | **0.6172**/0.8304/0.8946 | **0.41992**/0.6789/0.7760 | **0.8890**/0.9770/0.9930 | **0.8570**/0.9720/0.9850 |
| **S0_TriMask_HS@1000** | **0.61480**/0.8288/0.8936 | **0.41796**/0.6777/0.7748 | **0.8900**/0.9770/0.9900 | **0.8550**/0.9680/0.9830 |

## 差值（百分点）

同预算下的跨臂比较（HS − S0）：

| 预算 | COCO 图→文 R@1 | COCO 文→图 R@1 | Urban 图→文 R@1 | Urban 文→图 R@1 |
|---|---|---|---|---|
| @500 | -0.52 | -0.03 | +0.40 | -0.50 |
| @1000 | **-0.24** | **-0.20** | **+0.10** | **-0.20** |

同臂跨预算（@1000 − @500）：

| 臂 | COCO 图→文 R@1 | COCO 文→图 R@1 | Urban 图→文 R@1 | Urban 文→图 R@1 |
|---|---|---|---|---|
| S0_smartclip | +1.14 | +0.76 | +1.90 | +1.50 |
| S0_TriMask_HS | +1.42 | +0.59 | +1.60 | +1.80 |

两个臂都从多跑的 500 步里真实获益（COCO 图→文 +1.1 ~ +1.4pp）；HS 获益略多，于是两臂差距从 -0.52pp 收窄到 -0.24pp。完整逐指标差值见 `tables.md`。

## 配对逐查询证据（COCO val2017，同一批查询）

汇总数字（0.5pp 量级）单看没有意义：COCO 图→文有 5000 个查询、文→图有 25000 个，各自独立的二项标准误约 0.7pp。两臂评的是**同一批图与同一批文本**，所以正确的检验是配对检验。工具复用仓库自己的 `eval/paired_statistics.py`（10000 次配对 bootstrap + 精确 McNemar），逐查询命中向量由 `tools/diag/coco_query_hits.py` 采集，且**断言**「自算聚合值 == 标准评估器写下的数字」（四个 checkpoint 全部通过，例如 HS@1000 图→文 0.6148、文→图 0.41796）。

delta = 左 − 右，负值表示左侧更差：

| 比较 | 方向 | R@1 左 | R@1 右 | Δ(pp) | 95% CI (pp) | 排除 0 | 仅左对/仅右对 | McNemar p |
|---|---|---|---|---|---|---|---|---|
| HS@1000 vs S0@1000 | 图→文 | 0.61480 | 0.6172 | -0.24 | [-0.80, +0.30] | **否** | 93/105 | **0.434** |
| HS@1000 vs S0@1000 | 文→图 | 0.41796 | 0.41992 | -0.20 | [-0.38, -0.02] | 是 | 240/289 | 0.037 |
| S0@500 vs S0@1000 | 图→文 | 0.6058 | 0.6172 | -1.14 | [-1.82, -0.44] | 是 | 124/181 | 0.0013 |
| S0@500 vs S0@1000 | 文→图 | 0.41236 | 0.41992 | -0.76 | [-1.02, -0.49] | 是 | 479/668 | 2.6e-08 |
| HS@500 vs HS@1000 | 图→文 | 0.6006 | 0.61480 | -1.42 | [-2.12, -0.72] | 是 | 121/192 | 7.1e-05 |
| HS@500 vs HS@1000 | 文→图 | 0.41208 | 0.41796 | -0.59 | [-0.85, -0.33] | 是 | 503/650 | 1.7e-05 |
| **HS@500 vs S0@500** | 图→文 | 0.6006 | 0.6058 | -0.52 | [-1.06, **+0.02**] | **否** | 82/108 | **0.069** |
| **HS@500 vs S0@500** | 文→图 | 0.41208 | 0.41236 | -0.03 | [-0.20, +0.15] | **否** | 239/246 | **0.785** |

读法：

- **@1000 的跨臂差距不显著**：图→文 Δ=-0.24pp，CI 跨 0，p=0.43。文→图 Δ=-0.20pp 的 CI 勉强排除 0（[-0.38,-0.02]），p=0.037 —— 这张表里有 8 个检验且未做多重比较校正，这个边缘结果不足以支撑「HS 在文→图上更差」的结论，只能说「若真有差异，量级在 0.2pp 以下」。
- **多跑 500 步的效果是真的**：两臂四个方向全部排除 0，p 从 0.0013 到 2.6e-08。
- **当年 500 步门槛的判负值得重新审视**：HS 相对 S0 的图→文差异 -0.52pp 在配对检验下 CI 跨 0（[-1.06,+0.02]）、p=0.069，文→图 p=0.785。门槛规则按 raw precision 逐位比较，把一个噪声级的差异判成了「未通过」。

工具附注：仓库的 `eval.paired_statistics.mcnemar_exact` 用 `float(2 ** discordant)`，不一致对数超过约 1023 时整数溢出（COCO 文→图有 25000 个查询，必然触发）。工具里的对数空间回退实现了同一个定义，并已验证在库函数能运行的四个案例上**逐位一致**（如 n10=30/n01=10：0.00222143377323 vs 0.00222143377323）。

## 冻结的 500 步门槛（参考行，不适用于 1000 步）

门槛定义：COCO I2T R@1 ≥ 0.6058 且 COCO T2I R@1 ≥ 0.41236，且至少一个严格更高（raw 全精度）。

- S0@500 的两个值恰是门槛阈值本身，因此严格规则里的「至少一个严格更高」**对基准自身不成立**——它是门槛的参照物，不参与判定。
- S0_TriMask_HS@500：0.6006 / 0.41208 → 未通过（-0.52pp / -0.03pp）。结合上面的配对证据，该判定落在噪声内。

**本报告不把门槛套用到 @1000 行**：门槛只在 500 次更新处定义，1000 步是另一套预算。需要说明的是，HS runner 的 `conclusion` 字段在续训前写死了这道 500 步规则，于是给 @1000 的运行打上了 `PROMISING_AT_500`。已修复 runner（按预算判断，非 500 步时输出 `GATE_NOT_APPLICABLE_AT_<steps>` 并保留 `gate_would_say` 参考值；500 步语义逐位不变），并在该运行目录留下 `conclusion_budget_aware.json` 记录「原记录 vs 修正后」；**没有改动该运行的任何权重、指标或日志**。

## 结论与边界

可以说的：

1. 在 1000 步预算下，两臂的 COCO 图→文 R@1 无法区分；文→图最多相差 0.2pp 量级。
2. 硬文本门（HS）并没有在更长预算下「暴露问题」：它多跑 500 步的收益（图→文 +1.42pp）与 S0（+1.14pp）相当甚至略高。
3. 两个臂在 Urban-1k 上都涨得更多（+1.6 ~ +1.9pp 图→文），@1000 时 I2T 分别为 0.8890 与 0.8900。

不能说的：

- 这不是「HS 通过了门槛」：门槛只在 500 步定义，本报告的 @1000 数字不属于门槛判定。
- 只看 COCO canonical 与 Urban-1k 两个评估；配对检验只覆盖 COCO（Urban 每方向 1000 个查询，二项标准误约 1.0~1.1pp，且标准评估器不导出逐查询结果，故未做配对）。
- 没有做三 epoch 训练、没有调参、没有新增第四路、没有改 LR horizon；两臂的唯一差异仍是目标函数。

## 复现

```bash
# S0 臂（branch codex/s0-continue-1000）
python tools/s0_continue_runner.py --run-id s0_continue_1000 \
    --run-dir runs_salu/said_cls_cvssl/s0_continue_1000 --steps 1000

# HS 臂（branch codex/s0-trimask-hs-v02）
python tools/trimask_hs_runner.py --run-id s0_trimask_hs_v02 \
    --run-dir runs_salu/said_s0_trimask_hs_v02/cont1000 \
    --resume runs_salu/said_s0_trimask_hs_v02/step500/trimask_S0_TriMask_HS_step000500.pt \
    --steps 1000 --lambda-sparse-t 0.2

# 对比 + 配对证据（只读）
python tools/diag/coco_query_hits.py --label <arm@step> --checkpoint <student> --expect <i2t,t2i> --out <hits.json>
python tools/diag/coco_paired_report.py --hits A=a.json --hits B=b.json --pair A:B --out paired.json
python tools/diag/report_s0_continuation.py --paired <paired.json> \
    --out docs/s0_continuation/results.json --markdown docs/s0_continuation/tables.md
```

产物清单：本文件、`tables.md`（逐指标差值表）、`results.json`（全部原始数字与 SHA）、`implementation.md`（实现与守卫细节）；逐查询命中与配对结果在运行目录 `runs_salu/said_cls_cvssl/s0_continue_1000/evaluation/`（`query_hits_*.json`、`paired_all_coco.{json,md}`）。
