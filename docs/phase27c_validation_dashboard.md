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
| `tools/said_dashboard/validation_curves_data.py` | 纯函数（不 import streamlit）：发现 run、容错读取 JSONL、拍平两种 metrics 形状、长表/宽表、canonical chunk 检查、Balancing Gain 文案常量 |
| `tools/said_dashboard/validation_curves_page.py` | 页面 `main()`：侧边栏筛选 + 每指标独立 y 轴曲线 + 数值表 + 原始 JSONL |
| `tools/said_dashboard/app.py` | 导航新增「验证得分曲线（训练中）」入口 |
| `tests/test_validation_curves_dashboard.py` | 11 个测试，其中 3 个用 `streamlit.testing.v1.AppTest` 真实渲染页面 |

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
- 曲线分组：`实验 · 数据集 · 变体`（跨 run 对比）或 `只按文本变体`（同一 run 内比较三种 caption）
- 显示数值表

主区：

- 若非 canonical chunk 的记录（`similarity_chunk ≠ 512`）出现在所选实验里，页面先给出警告并列出这些点，
  避免把不同 chunk 的数字画进同一条曲线
- 各实验最新验证点表（含 chunk 与耗时）
- 每个指标一张曲线图（两列布局，各自独立 y 轴，Hover 显示实验/数据集/变体/验证点/步数/数值/耗时）
- 验证记录数值表（每个验证点一行，指标为列）
- 展开区列出记录字段、Balancing Gain 定义、每个 run 的文件路径与记录条数

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

验证开销（页面同时展示）：ShareGPT4V-1K 每个变体 13.6–16.3 s，5 个验证点共 15 条记录约 219 s；
COCO 一条 102.7 s；合计约 322 s，而同期 21 个训练步 wall 246 s（0.87 s/step）。这解释了为什么
`--val_every` 只驱动 1K、COCO 只在 initial / final 这类显式节奏上跑。

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
- 页面不做统计检验、不做插值，也不把不同 chunk / 不同 protocol 的数字合并成一条曲线
  （只会警告并原样列出）。
- 短程 run 的曲线近乎平坦，不能当作模型质量结论；它的用途是确认验证在跑、指标在记录、
  数值与协议一致。
