# NEST JointMask 性能诊断报告

日期：2026-09-29（UTC）

审查基准：`codex/nest-jointmask-probe-v1` / `82e06b2e5b0a76e51a924928a036be447e8fce02`

诊断分支：`codex/nest-jointmask-perf-diagnosis`

## 运行状态与边界

开始诊断前核查到正式 T 组仍在基准提交上运行。T 已正常完成 500 次四卡同步更新，四个 rank 的训练时间均约 2282.4 秒，峰值 allocated 显存均为 17.268 GiB，最终参数相对 rank0 的最大差异为 0。随后停止了串行调度进程，TI 和 TI-Shuffle 均未启动。旧检查点、日志和结果未被修改。

性能实验位于独立工作区 `/root/lk_projects/SAID-jointmask-perf`，不会在正式训练旁替换源码。测试结束时四张卡均空闲：4 张不同 UUID 的 NVIDIA A100 80GB PCIe，无 MIG。

本轮未启动新的 500-step 实验，也未修改 batch、候选池、精度、loss、RandomK、shuffle、pair mask、detach、包含项或正式训练预算。

## 测试协议

所有稳定计时均使用同一 step-0 状态和同一固定真实训练 batch：单机四卡、每 rank 256、全局候选 1024、encoder checkpoint 保持开启。每项先预热 2 次，再记录 8 次完整同步更新；结果取每个 step 四个 rank 中的最大值，再报告这些最大值的中位数和 P90。最终计时关闭 profiler。

为了把模型计算和输入流水线分开，稳定模型计时固定复用已取出的 batch，并单独记录 H2D；生产式 8-worker DataLoader 另做等待时间探针。Profiler 每项只捕获 1 个 active step，数值仅用于定位 kernel、同步和 GPU 间隙，不用于最终性能比较。

## 稳定端到端结果

`step`、`pair`、`backward` 均为毫秒。`backward` 包含 checkpoint 重算和 DDP 梯度通信。加速比以各组当前 32×64、pair checkpoint 开启的基线为 1.00×。

| 组别 | 实现 | 分块 | Pair checkpoint | Step median / P90 | Pair median | Backward median | 峰值 allocated / rank | 吞吐倍数 |
|---|---|---:|:---:|---:|---:|---:|---:|---:|
| T | 当前实现 | 32×64 | 开 | 4200.1 / 4389.8 | 1171.5 | 2526.7 | 17.270 GiB | 1.00× |
| T | 仅统计改写 | 32×64 | 开 | 4071.3 / 4135.4 | 971.0 | 2555.8 | 17.270 GiB | 1.03× |
| T | 统计改写 | 64×128 | 开 | 2411.5 / 2505.1 | 322.2 | 1586.9 | 17.270 GiB | 1.74× |
| T | 统计改写 | 128×128 | 开 | 1990.7 / 2096.0 | 106.6 | 1406.7 | 17.270 GiB | 2.11× |
| T | 统计改写 | 128×128 | 关 | **1891.3 / 1944.2** | **76.7** | **1333.0** | 28.463 GiB | **2.22×** |
| T | 原矩阵路径参考 | — | — | 1749.1 / 1794.0 | — | 1266.0 | 17.353 GiB | 2.40× |
| TI | 当前实现 | 32×64 | 开 | 4639.3 / 4792.2 | 1263.6 | 2866.5 | 17.275 GiB | 1.00× |
| TI | 仅统计改写 | 32×64 | 开 | 4460.8 / 4575.3 | 1057.3 | 2850.9 | 17.274 GiB | 1.04× |
| TI | 统计改写 | 64×128 | 开 | 2450.0 / 2493.3 | 249.0 | 1634.2 | 17.274 GiB | 1.89× |
| TI | 统计改写 | 128×128 | 开 | 2088.7 / 2173.2 | 117.6 | 1458.2 | 17.274 GiB | 2.22× |
| TI | 统计改写 | 128×128 | 关 | **1929.6 / 2027.2** | **86.4** | **1353.9** | 28.843 GiB | **2.40×** |

仅改统计且保持 32×64/checkpoint 开启时，T 和 TI 的端到端中位数分别下降 3.07% 和 3.85%，8 个测量 step 的 loss 序列与基线逐项完全一致。把分块增至 128×128 后，checkpoint 开启时 T/TI 已达到 2.11×/2.22×；关闭 pair checkpoint 后进一步达到 2.22×/2.40×，代价是每卡约增加 11.2–11.6 GiB allocated 显存。

同一 128×128 分块下，checkpoint 开关的 8-step loss 序列在 T 和 TI 中都逐项一致。不同分块经过多次参数更新后会因浮点归约差异和 hard mask 阈值产生不同训练轨迹，因此跨分块验收采用固定状态下的 loss、梯度和一步更新容差，而不要求多步轨迹逐位一致。

关键配置的逐卡峰值 allocated 显存如下；其余配置的四卡原值保存在机器可读结果中。

| 配置 | rank0 | rank1 | rank2 | rank3 |
|---|---:|---:|---:|---:|
| T 当前 32×64 cp-on | 17.270 | 17.270 | 17.270 | 17.270 GiB |
| T 128×128 cp-off | 28.463 | 28.463 | 28.463 | 28.463 GiB |
| TI 当前 32×64 cp-on | 17.273 | 17.273 | 17.273 | 17.275 GiB |
| TI 128×128 cp-off | 28.843 | 28.843 | 28.843 | 28.843 GiB |

T 的矩阵快速路径仍是最快参考，比最快 pair 实现再快约 7.5%。它已通过零 delta 条件下与 pair 路径的数值测试，但本补丁没有启用该路径，也没有让 TI 使用局部 `qi.T`；TI 的全局候选语义保持不变。

## 分项时间

以下为 profiler 关闭后的四 rank 最大值中位数，单位毫秒。

| 组别/实现 | H2D | Encoder | Pair forward | 前向 collective | Backward/重算/DDP | Optimizer | 日志 |
|---|---:|---:|---:|---:|---:|---:|---:|
| T 当前 32×64 cp-on | 23.9 | 411.4 | 1171.5 | 138.2 | 2526.7 | 15.7 | 10.2 |
| T 最快安全候选 128×128 cp-off | 24.0 | 417.1 | 76.7 | 13.8 | 1333.0 | 16.0 | 7.9 |
| TI 当前 32×64 cp-on | 23.9 | 412.7 | 1263.6 | 85.4 | 2866.5 | 17.5 | 20.6 |
| TI 最快安全候选 128×128 cp-off | 23.8 | 418.2 | 86.4 | 22.3 | 1353.9 | 15.9 | 10.6 |

前向 collective 计时覆盖显式 gather/global sum。DDP backward collective 与反向计算重叠，无法从 event 计时中可靠拆开，因此保留在 `Backward/重算/DDP`。Profiler 的 NCCL kernel 中位总时长从 T/TI 基线的 229.9/221.1 ms 降至候选的 98.9/121.1 ms；collective 次数没有被删减，下降主要来自更少的同步干扰与更连续的 GPU 提交。

生产式 8-worker DataLoader 探针的四 rank 最大首次等待为 10.65 秒，之后 5 次最大等待依次为 2478.9、634.2、725.9、0.73、0.74 ms。前三次反映 worker 启动和预取队列填充，队列稳定后等待低于 1 ms。该探针退出时触发过 PyTorch worker 清理告警，因此稳定模型矩阵改用固定 batch；此告警不影响已记录 batch，也未出现在正式 T 训练中。

## Profiler 摘要

每项为四 rank 的中位数；原始 trace 位于服务器结果目录，不进入 Git。

| 组别/实现 | Pair block 调用 | CUDA kernel | <10µs kernel | `aten::nonzero` | `aten::index` | `cudaStreamSynchronize` | GPU 空闲比例 | 最大空闲间隙 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T 当前 | 1536 | 83057 | 76211 | 1573 | 1569 | 3153 | 62.3% | 83.1 ms |
| T 候选 | 96 | 13937 | 6019 | 37 | 33 | 81 | 8.1% | 10.5 ms |
| TI 当前 | 1536 | 95797 | 85860 | 1573 | 1569 | 3153 | 66.0% | 106.0 ms |
| TI 候选 | 96 | 15397 | 7216 | 37 | 33 | 81 | 10.7% | 11.0 ms |

基线的 1536 次 block 调用由 768 次前向和 768 次 non-reentrant checkpoint 重算组成。当前实现每个 block 的两次动态布尔索引触发 `nonzero/index` 和流同步，并产生大量短 kernel 与 CPU 提交间隙。固定形状 reduction 去掉这些动态选择，关闭被丢弃的 T2I summary 后不再建立对应统计路径；128×128 将前向块数从 768 降到 96；关闭 pair checkpoint 后不再重算，因此候选总调用数为 96。

Profiler 下的绝对时间受采样开销明显放大。最终端到端数字以关闭 profiler 的 8-step 结果为准。

## 数值与分布式验收

最小生产补丁只修改 `model/nested_semantic_mask.py`：

1. 把 active pair 统计改为固定形状 masked reduction；
2. 为 `pair_scores` 增加 `collect_stats`，T2I 路径关闭最终被丢弃的 summary；
3. 不改变 score、loss、梯度路径、collective 顺序和全局候选定义。

验收结果：

- 统计改写：score 逐位一致；计数统计逐位一致；delta 统计误差不超过 `2e-6`，覆盖 mixed、全无和全 active pair。
- 分块/checkpoint/统计关闭：固定状态下 loss 和 score 容差 `atol=2e-5, rtol=2e-6`；梯度 `atol=3e-4, rtol=3e-5`；一步更新 `atol=5e-7, rtol=5e-6`。
- 实际尾批几何：本地 180、全局候选 720，32×64 与 128×128 的 score/stat 测试通过。
- 两 rank NCCL 单进程参考：最大 loss 误差 `1.22e-4`，最大梯度误差 `2.44e-4`，最大一步更新误差 `5.96e-8`。
- 分布式用例覆盖全有效、局部零有效 rank、全局 V=0、V=1、shuffle、pair checkpoint 开/关，并保留可导 gather。
- 相关测试共 40 项通过；无失败。

正式配置文件仍保持 32×64 和 pair checkpoint 开启。本轮没有把 128×128 或 checkpoint-off 写入正式配置，因为用户要求后续采用哪个实现依据报告另行决定。

## 建议与限制

固定形状统计和关闭无用 T2I summary 是低风险最小补丁，建议作为独立代码优化采用。它不改变正式配置即可获得约 3–4% 的端到端收益。

在本机 80GB A100 上，未来若以吞吐为优先，可选择 128×128、pair checkpoint 关闭：T/TI 峰值分别为 28.463/28.843 GiB，留有充足显存余量。若需要兼容显存更小的设备，128×128、checkpoint 开启仍保持约 17.27 GiB，并已获得 2.11×/2.22× 吞吐。正式训练切换前应按目标机器重新做一次显存和稳定性核验。

T 的矩阵路径可作为后续独立选择；本轮只提供数值和性能参考。未强制 T 永久走低效 pair 实现，也未把该特例加入最小补丁。

本结论来自一台 4×A100 80GB PCIe 服务器、固定 batch 和 8 个稳定 step。Profiler 只捕获一个 active step；DataLoader 启动抖动与长期数据吞吐需要在下一次正式任务中继续观察。没有据此启动新训练或改变现有实验结果。

## 产物

- 稳定计时机器可读结果：`performance/performance_results.json`
- Profiler 摘要：`performance/profiler_summary.json`
- 数值验收摘要：`performance/validation_summary.json`
- 原始 benchmark、console 与 trace：`/root/lk_projects/SAID-nest-clip-v1/jointmask_perf_diagnosis/`
- 正式 T 验收：`/root/lk_projects/SAID-nest-clip-v1/jointmask_probe_v1/formal/T/acceptance.json`
