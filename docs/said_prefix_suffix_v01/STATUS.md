# S0-Suffix v0.1 — 当前状态（**未完成，不可当作结果**）

> 本文件与同分支的代码是**调试存档**，不是交付物。两个固定配置各 500 步**尚未运行**，没有任何
> 训练/评测数字，前端未登记。分支名 `codex/s0-suffix-v01-wip-unfinished` 即为标记。

- **base**：`5676666`（分支 `codex/s0-suffix-v01` 的起点，未改动）
- **工作树**：`/root/SAID-s0-suffix-v01`
- **共同初始化**：`/root/SAID-gap-completion/runs_salu/said_cls_cvssl/shared_init/cvssl_initial.pt`
  sha256 `c1a4a2be1b212f38677f729a4f03160d788a6d6db79ca5021a43022af0f5cba8`（317 张量，含 `mask_net.*`，248 token）
- **S0 血统**：`arm=S0_smartclip`，objective `said_cls_cvssl`，`lambda_align=10.0`、`lambda_sparse=2.0`、`lambda_U=0.0`

## 1. 本分支包含的文件

| 文件 | 作用 |
| --- | --- |
| `docs/said_prefix_suffix_v01/spec.md` | 绑定规格（13 节 + 冻结 API 名单） |
| `model/said_prefix_suffix.py` | 新分支：P/R 切分、新掩码 F、有效子集 DDP 缩放、分块读出、检查点身份 |
| `train/train_said_prefix_suffix.py` | 两臂训练器（复用 S0 目标，新增后缀分支与第三个优化器组） |
| `train/said_cvssl_data.py` | **唯一一处对既有管线文件的改动**：新增 `caption_full` 字段（严格增量：不动 RNG、顺序、任何既有字段、任何摘要） |
| `tests/test_said_prefix_suffix.py`、`tests/_suffix_ddp_worker.py` | 验收测试 A–I 与两 rank worker |
| `tools/prefix_suffix_runner.py` | 六阶段 runner（train→verify→export→coco→urban→report，两臂顺序执行） |
| `tools/diag/export_suffix_student.py` | 裸学生导出（只含原生 CLIP state，新 F 不进入） |
| `tools/diag/suffix_mechanism_diagnostic.py` | 只读机制诊断（NORMAL/U_ALL_ONES/PREFIX_SHUFFLED/IMAGE_SHUFFLED） |
| `docs/said_prefix_suffix_v01/debug/pytest_first_run_full.log` | 第一次完整 pytest 输出（29 failed / 10 passed，含每条断言原文） |
| `docs/said_prefix_suffix_v01/debug/smoke_cuda_assert_excerpt.log` | 真实入口冒烟崩溃处的 CUDA 断言原文 |

## 2. 已验证通过的部分

- **P/R 切分与规格工作例逐条一致**（真实入口内运行的检查）：
  `'A. B. C. D'` k=1→`'B. C'`、k=2→`'C'`、k=3→`''`、k=4→`''`、`'A. B. C. D. '` k=2→`'C'`。
- **两个臂的真实入口冒烟曾经全绿**（各 2 次优化更新，4 卡 DDP）：
  `global_presentations = 2048`（2 步 × 1024，全局口径）、`rank_local_presentations = 512`、
  检查点 14 个冻结键齐全、`suffix_mask_state` 的四个张量形状精确为
  `layer1.weight [512,1024]`、`layer1.bias [512]`、`layer2.weight [512,512]`、`layer2.bias [512]`。
- **A 组测试（切分）全部通过**；规格工作例、末尾空片段、`sample_k` 双端包含等断言均绿。

## 3. 已发现并修掉的真实缺陷（冒烟与实现期）

1. **调度器调用崩溃**：`run_schedulers` 把 `cosine_lr` 返回的**可调用对象**当作 `(scheduler, optimizer)`
   元组解包 → 四卡立即 `TypeError`。已改为按可调用对象调用，并从优化器回读真实 lr。
2. **后缀文本前向 OOM**：S0 那次只有一次文本前向，新增的第二次没有梯度检查点，80 GiB 卡打满
   （79.25 GiB）。已改为使用模块自带的 checkpointed 文本编码器 + 按行分块（默认 128 行，规格允许的旋钮）。
3. **日志键 KeyError**：`loss_smart_unweighted` 只由 S0 目标的**高层** `forward` 返回，低层 helper 不返回；
   已加语义等价的回退（`loss_sidm + loss_dism`）并记录实际键集。
4. **设计级 bug（最严重）**：数据集返回的 `caption_said` **已经是按 K 截断的前缀**
   （`'. '.join(caption.split('. ')[:prefix_k])`），拿它再切后缀必然恒空 —— 实测 `global_valid_V = 0`、
   `loss_suffix = 0`，等于这一步只跑了 S0。已修：严格增量新增 `caption_full`，训练器用**完整 caption**
   切分 P/R，并把"由完整 caption 重建的前缀"与 `caption_said` **逐字符断言相等**（把"S0 前缀流不变"
   变成运行时硬检查）；同时拒绝"传入截断文本"这种会静默清零后缀损失的输入形状。
5. **presentations 口径不一致**：`run_summary.json` 里 `global_presentations` 曾是 rank 局部值（128000），
   与日志口径冲突。已统一为全局值（500×1024=512000），局部值另立 `rank_local_presentations`。

模型/训练器代理另自报用本地 harness 抓到并修掉 5 个缺陷（后缀 off-by-one、`_global_index_of` 的 rank 偏移错、
`[Bi,1,1024]` 与 `[Bi,Bj,1024]` 广播错、重日志统计函数重复参数、检查点把 `layer1.weight` 剥成 `weight`）。

## 4. 当前阻塞点（从这里接手）

第一个训练步在**写出任何日志记录之前**崩溃（rank 2 先 `SIGABRT`，其余 rank 被 NCCL watchdog 带走）：

```
../aten/src/ATen/cuda/Indexing.cu:1308 indexSelectLargeIndex:
Assertion `srcIndex < srcSelectDimSize` failed.
```

定位在 `model/said_prefix_suffix.py:1402-1404`：

```python
j_all = validity['global_valid_indices']
g_sub      = g_all.index_select(0, j_all)
mask_s_sub = mask_s_all.index_select(0, j_all)
t_r_sub    = t_r_all.index_select(0, j_all)
```

`j_all` 的取值空间是 `[0, world*B)`（全局有效索引），但 `g_all` / `mask_s_all` / `t_r_all` 三者中至少
有一个并不是同一"已 gather 的全局"空间（很可能 `t_r_all` 仍是 rank 局部 `B` 行，或反之），于是越界。
**这与模型代理自报修过的 `_global_index_of` 索引空间错误同类**，说明该调用点仍存在空间混用。
修法：让三个 `*_all` 统一为 gather 后的 `world*B` 空间，或把 `j_all` 换成局部行索引后再 select；
并用测试 G 组（有效数不等 / 某 rank 为 0 / V<2）验证。

## 5. 测试侧的剩余缺陷（不是实现问题）

第一次完整 pytest：**29 failed / 10 passed**，失败全部在测试自身的 helper：
- `call_readout` 用**猜的**位置参数顺序与返回形状调实现（实现返回 `dict`，不是 `(scores, mU)` 元组）；
- `randomise_suffix_mask` 在 CPU 上造噪声却加到 CUDA 参数上（设备不匹配）。
已把实现的精确签名与两类报错原文发给测试代理修，**尚未回传**。完整日志见
`debug/pytest_first_run_full.log`。

## 6. NOT RUN（不得当成已完成）

- 两个固定配置各 **500 步**训练：未运行（冒烟未过，故未启动正式 run）。
- **COCO canonical（5000×25000）与 Urban-1k（1000×1000）原生评估**：未运行。
- **只读机制诊断四模式**：工具已实现，未运行（队列尚未生成）。
- **前端登记两个新 run**：未做。
- **报告 / results.json / 正式提交与 push**：未做。
- 未做浏览器目视验收；未做超参搜索、未跑第三臂、未续训到 1000 步。

资源：4 张 A800 空闲，无训练/评测进程；未 force push；未改动任何其他工作树；未提交检查点/数据/大日志。

## 7. 恢复顺序

1. 修 `model/said_prefix_suffix.py:1402-1404` 的索引空间，在 2 步冒烟上确认
   `global_valid_V > 0`、`loss_suffix > 0`、`loss_total ≠ loss_s0`。
2. 测试代理回传后跑全绿（`tests/test_said_prefix_suffix.py`，含两 rank 的 G 组五情形）。
3. `tools/prefix_suffix_runner.py` 顺序跑 `S0_SUFFIX_NATIVE` → `S0_SUFFIX_MASK` 各 500 步，
   每臂 train → 校验 step500 → export → COCO → Urban → report。
4. 生成诊断队列并跑四模式；前端登记两臂；写报告与 results.json；普通 push（不 force）。
