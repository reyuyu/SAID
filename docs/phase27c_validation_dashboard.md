# Phase 2.7C：训练中验证得分曲线（前端）

Phase 2.7B 让训练在每个验证点把结果写进 `<output_dir>/validation_history.jsonl`。本阶段把这个
文件接进既有的 Streamlit 看板：**训练过程中就能看到验证集得分随步数的变化**，不用等训练结束，
也不用单独跑评测脚本。

## 目标与边界

- **只读**：页面只读训练已经写入的 JSONL；不加载 checkpoint、不运行模型、不重算任何指标，
  因此切换实验/数据集/变体是瞬时的。
- **不改训练**：本阶段没有改动 Said、损失、数据集、caption 变体、检索协议、RNG 协议与
  `similarity_chunk`；曲线用的就是 2.7B 定稿的 protocol（`similarity_chunk = 512`）。
- **不引入新的评测路径**：页面展示的数字与 `validation_history.jsonl` 中的数字逐位相同。

## 组件

| 文件 | 作用 |
| --- | --- |
| `tools/said_dashboard/validation_curves_data.py` | 纯函数（不 import streamlit）：发现 run、容错读取 JSONL、拍平两种 metrics 形状、canonical chunk 过滤（`split_canonical`）、series 分组规则（`curve_series_mode`）、长表/宽表、Balancing Gain 文案常量 |
| `tools/said_dashboard/validation_curves_page.py` | 页面 `main()`：侧边栏筛选 + 每指标独立 y 轴曲线 + 数值表 + 原始 JSONL |
| `tools/said_dashboard/app.py` | 导航新增「验证得分曲线（训练中）」入口 |
| `tests/test_validation_curves_dashboard.py` | 17 个测试，其中 4 个用 `streamlit.testing.v1.AppTest` 真实渲染页面 |

## 比较规则（merge 前加固）

1. **canonical chunk 过滤**：主曲线、latest points、主数值表**只使用
   `similarity_chunk == 512`** 的记录（`split_canonical()`）。
   `similarity_chunk` 缺失或不是 512 的记录：
   - 继续显示警告（页面顶部）
   - 继续列在异常记录表里（实验 / 步数 / 数据集 / 变体 / 验证点 / chunk）
   - 可以在「原始 JSONL」展开区看到原始行（含无法解析的行）
   - **不进入**主曲线、latest points、主数值表
2. **一条 series 永不合并两个 run**：「只按文本变体」只在只选 1 个 run 时生效；
   选了 ≥ 2 个 run 时页面给出警告并自动回退为「实验 · 数据集 · 变体」
   （`curve_series_mode(n_runs, requested)`），避免两个 run 的同名 variant 被画成同一条线。

## 数据来源与字段

数据来源就是训练本身写的文件：`<run_dir>/validation_history.jsonl`，一行一个验证点：

```json
{"step": 680, "epoch": 0, "dataset": "sharegpt4v1k", "caption_variant": "full_dense",
 "reason": "interval", "protocol": "sharegpt4v1k-fixed-captions-v1", "similarity_chunk": 512,
 "wall_sec": 14.1, "metrics": {"retrieval": {...}, "diagnostics": {...}}}
```

- `sharegpt4v1k`：`metrics = {'retrieval': {...}, 'diagnostics': {...}}`
- `coco_val2017`：`metrics` 是扁平的检索字典
- `reason`：`initial`（step 0）/ `interval`（`--val_every`）/ `epoch_end` / `final`
- 页面把两种形状统一拍平，所以 COCO 与 ShareGPT4V-1K 的检索指标可以画在同一张图里；
  诊断指标（Pair Gap / RMG / Balancing Gain / Conditioning Margin）只有 ShareGPT4V-1K 有

`Balancing Gain = Full Pair Gap − Said Pair Gap`（正值 = Said 更好，负值 = Said 更差），与
`eval/validation_protocol.py` 的 `BALANCING_GAIN_DEFINITION` / `BALANCING_GAIN_SIGN` 同一句话；
看板不 import 那个模块（它会拉起 torch），改由测试断言两边字符串一致，防止措辞漂移。

## 页面交互

侧边栏：

- 运行目录：run 的父目录（默认 `runs_salu`，可用环境变量 `SAID_RUNS_ROOT` 覆盖），也可以直接指向单个 run 目录
- 实验：`runs_salu/*/validation_history.jsonl` 自动发现，可多选（多选即多 run 对比）
- 数据集：ShareGPT4V-1K / COCO val2017
- 文本变体：第一句 / 训练同规则稀疏 / 完整 caption / COCO 5 captions
- 指标：6 个检索指标 + 7 个诊断指标，默认选中 I2T/T2I R@1、R@5 与 Balancing Gain
- 曲线分组：`实验 · 数据集 · 变体`（跨 run 对比）或 `只按文本变体`（同一 run 内比较三种 caption；
  多选 run 时自动回退并给出警告，见上节比较规则）
- 显示数值表

主区：

- 非 canonical chunk 的记录先给警告，并列出异常记录表（这些点不会进入下面的主曲线与数值表）
- 各实验最新验证点表（含 chunk 与耗时；只用 canonical 记录）
- 每个指标一张曲线图（两列布局，各自独立 y 轴，Hover 显示实验/数据集/变体/验证点/步数/数值/耗时）
- 验证记录数值表（每个验证点一行，指标为列；只用 canonical 记录）
- 展开区列出记录字段、Balancing Gain 定义、每个 run 的文件路径与 canonical/总记录条数，
  以及被排除记录的原始 JSONL 行

## 实测：demo run 的曲线数据

配置：4 × A800-80GB，ViT-B/16，`--batch_size 256`（global 1,024）、bf16、从
`runs_salu/phase22/salu_said_only_last.pt`（step 659）续训 21 步到 step 680，
`--val_every 5 --val_sharegpt4v --eval_coco --val_batch_size 64`，输出 `runs_salu/phase27c_curve_demo`。

验证记录共 **16 条**：ShareGPT4V-1K 三变体 × 5 个验证点（step 660 / 665 / 670 / 675 / 680）
+ COCO val2017 1 条（step 680，`reason=final`）。step 680 同时是 interval 与 final，
按 `(step, dataset, caption_variant)` 去重后只写了一条。

| 文本变体 | I2T R@1（660 → 680） | T2I R@1（660 → 680） | Full Pair Gap | Said Pair Gap | Balancing Gain（660 → 680） | Conditioning Margin |
| --- | --- | --- | --- | --- | --- | --- |
| first_sentence | 0.6310 → 0.6310 | 0.6100 → 0.6100 | 0.6755 | 0.6675 | +0.0080 → +0.0079 | 0.0988 |
| fixed_sparse | 0.8780 → 0.8780 | 0.8520 → 0.8520 | 0.6460 | 0.6758 | −0.0297 → −0.0298 | 0.1640 |
| full_dense | 0.9470 → 0.9470 | 0.9310 → 0.9310 | 0.6378 | 0.6828 | −0.0449 → −0.0450 | 0.1835 |

- 中间三个验证点与端点只在小数第 3–4 位不同（例如 fixed_sparse I2T R@5 0.9730 → 0.9740），
  21 步、学习率已被 cosine 衰减到近 0，曲线基本平坦 —— 这是**接线与渲染的验证，不是模型质量结论**。
- 三变体的分层与 Phase 2.7B 定稿时的测量一致：越详细的 caption 检索越高，而 Balancing Gain 只有
  `first_sentence` 为正（Said 更好），另外两个变体为负（Said 更差）。
- COCO val2017（step 680，chunk 512）：I2T R@1 0.5840 / R@5 0.8130 / R@10 0.8850，
  T2I R@1 0.3997 / R@5 0.6588 / R@10 0.7572。

### 实测耗时拆解（全部读出，不做估算）

数据来源：`runs_salu/phase27c_curve_demo/` 下的 `salu_summary.json`（训练脚本自己写的 summary）、
`salu_log.jsonl`（训练 JSONL）、`validation_history.jsonl`（验证 JSONL），以及 torchrun 日志首行
时间戳（16:29:24.747）与 run 目录文件 mtime。

| 项目 | 数值 | 依据 |
| --- | --- | --- |
| optimizer steps | 21（step 659 → 679） | `salu_summary.json: steps = 21`；`salu_log.jsonl` 首/末记录 step 660 → 679（`log_every 5`，首步 659 不打印） |
| 单步 compute | 0.687466 s/step（21 步合计 14.44 s） | `salu_summary.json: compute_sec_per_step`（= `salu_log.jsonl` 末条同名字段） |
| 单步 wall（训练循环内，不含验证） | 0.870126 s/step（21 步合计 18.27 s） | `salu_summary.json: wall_sec_per_step`；`wall_times.append()` 在验证之前，故不含验证 |
| 验证记录 | 16 条 = 1K 15 条（5 个验证点 × 3 变体，`interval`）+ COCO 1 条（`final`） | `validation_history.jsonl` |
| 1K 验证耗时 | 219.205 s（15 条，单变体 13.6–16.3 s） | 同上，`reason = interval` 的 `wall_sec` 之和 |
| COCO 验证耗时 | 102.667 s（1 条，step 680） | 同上，`reason = final` |
| 验证合计 | 321.872 s | 16 条 `wall_sec` 之和 |
| `salu_summary.json: wall_sec_total` | 246.005 s | `t_start` → 写 summary；**含** 5 个 1K 验证点（219.2 s），**不含**之后的 COCO 验证（102.7 s） |
| 整个 demo | 395.535 s | torchrun 启动（16:29:24.747）→ `reproducibility_rank0.json` mtime（16:36:00.282） |

把它拆成互不重叠的几段（每段都能对上一个产物）：

| 阶段 | 时间 | 怎么得到 |
| --- | --- | --- |
| 启动与加载（CLIP 权重、1.2M 条 dataset JSON、DDP、resume checkpoint） | 46.862 s | `wall_sec_total` 之前的部分 = 292.867 s（启动→summary）− 246.005 s |
| 21 个 optimizer steps | compute 14.44 s / 循环 wall 18.27 s | 0.687466 / 0.870126 × 21 |
| 未计入 `wall_sec_per_step` 的循环前开销（首个 batch 前的 DataLoader worker 启动与预取等） | 7.993 s | 差分：246.005 − 18.272 − 219.205 − 0.534（未单独计时，只能按差分给出） |
| 1K 验证（5 个点，训练循环内） | 219.205 s | 验证 JSONL `reason = interval` |
| 最终 checkpoint 保存 | 0.534 s | `salu_said_only_last.pt` mtime → `salu_summary.json` mtime |
| COCO 验证（summary 之后） | 102.668 s | summary mtime → 最后一个文件 mtime |

校验：46.862 + 7.993 + 18.272 + 219.205 + 0.534 + 102.668 = 395.535 s（= 整个 demo）。

因此正确的说法是：**训练的 21 个 optimizer steps 只占 14.44 s compute（18.27 s 循环 wall），
验证占 321.87 s（整个 demo 的 81%）**；`salu_summary.json` 的 `wall_sec_total = 246.005 s`
既不是训练时间、也不是验证总时间——它是"`t_start` → summary"这一段，含 5 个 1K 验证点、
不含最终的 COCO。这也是 `--val_every` 只驱动 1K、COCO 只在 `--eval_coco_initial` / `final`
这类显式节奏上跑的原因。

## 运行方式

```bash
cd /root/SAID
streamlit run tools/said_dashboard/app.py --server.address 127.0.0.1 --server.port 8501
# 本机隧道： ssh -L 8501:127.0.0.1:8501 root@<server>
# 打开 http://127.0.0.1:8501，左侧「页面导航」选择「验证得分曲线（训练中）」
```

若 run 不在默认位置：在侧边栏填路径，或设 `SAID_RUNS_ROOT=/path/to/runs`。

测试：

```bash
python -m pytest tests/test_validation_curves_dashboard.py -q
```

## 已知限制

- 曲线只画训练已经写入的记录：训练没写验证就没有点；点数由 `--val_every` 决定。
- 验证会串行化训练（rank 0 评测，其余 rank 停在 barrier），曲线好看不代表验证免费；
  间隔设置见 Phase 2.7B 的开销说明。
- 页面不做统计检验、不做插值；主曲线只接受 canonical chunk 的记录，其它 chunk 的记录被排除在
  主曲线/latest/数值表之外，只在警告、异常记录表与原始 JSONL 展开区出现。
- 一条 series 永不跨 run 合并；多 run 时「只按文本变体」会被自动改写为「实验 · 数据集 · 变体」。
- 短程 run 的曲线近乎平坦，不能当作模型质量结论；它的用途是确认验证在跑、指标在记录、
  数值与协议一致。
