# NEST-HybridF-E11：实现完成，四卡smoke通信验收阻塞

**未启动正式3651步训练或六项检索评测。** 本轮smoke完成5次参数更新，但末尾记录NCCL异步系统错误，不满足用户要求的“无worker/NCCL异常”。按完整方案的失败停止要求保留现场，没有自动修改通信环境、降低batch、重跑或换方案。

实现提交 `9327ea90611f85122b0788420513f04ea8e24977`，独立分支 `codex/nest-hybridf-v1`。工作区原始提交为 `37c05f1`，NEST相关模型/数据/trainer与指定参考 `25a5d12` 一致。3/0参考直接从固定ref `8aaad8820b62d3e87285efc2d14fb389f880fd9d` 读取，未切换main或混入其训练逻辑。本轮最新指令明确包含DCI/Long-DCI和J_long，覆盖此前跳过DCI的范围要求。

已实现且测试通过：

- 新增 `full_native_mix`，默认0，E11固定0.25，内部arm=A3。全文混合两项分别计算的CE：0.75 masked + 0.25 native；P/R、稀疏及包含权重保持不变，F-only回退也使用同一混合。
- native复用已有z/tF与可导全局候选，FP32、scale100、eps1e-6；不增加编码、可训练参数或mask。默认0不调用native分支、无新增通信或计算图。
- 增加全文mask/native双向日志、hybrid值、eta及总loss逐项重构；恢复校验包含eta，不能用旧A3目标恢复E11。
- RandomK数据源码未修改。专用脚本首次smoke/正式均从共同step0启动，不带旧A3的resume。

20项单元/恢复测试通过。默认eta=0与固定ref旧NEST在三视图及V=0/1情况下的loss、日志、梯度、一步AdamW更新逐项相同，RNG不变。native显式评分/双向CE、全1mask对照、native直接梯度不进入mask、包含直接梯度只进pF、CE混合总公式及F-only、编码调用次数、state_dict/optimizer序列化和eta恢复拒绝均通过。两rank NCCL测试覆盖有效数不均、局部零有效、全局V=0/1，真实结果见 [ddp-results.json](evidence/ddp-results.json)。沿用原FP32/AdamW零梯度方向容差，不声称任意浮点输入逐位等价。

共同初始化SHA256为 `54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`。完整训练集1,245,901条，1217 updates/epoch、H=3651；四张A100 80GB，WORLD_SIZE=4，每rank256，accumulation1。资源、数据索引、初始化以及六项协议资产检查通过。3/0本机3651权重/原始结果与固定ref一致；Clean/Full/A3参考保存在证据中，仅作为未来正式比较来源，未冒充E11成绩。

真实smoke观察值：

| 更新 | 全局总loss | native F候选数 | 全局局部有效数 | 包含权重 |
|---|---:|---:|---:|---:|
| 1 | 57.187019 | 1024 | 1024 | 0.000 |
| 2 | 31.850668 | 1024 | 1024 | 0.005 |
| 3 | 25.551161 | 1024 | 1024 | 0.010 |
| 4 | 19.097485 | 1024 | 1024 | 0.015 |
| 5 | 19.362375 | 1024 | 1024 | 0.020 |

四rank记录均完成5更新，loss/梯度有限，20个step/rank的sample/F/P/R/K与原A3完全一致。step0模型/optimizer与共同初始化相同，step5 optimizer计数正确。trainer的末尾参数比较记录差异为0，CUDA峰值allocated每卡17.310 GiB；这些观察值不构成通过通信验收的依据。

真实阻塞：日志在第5步后记录：

```text
[Rank 3] Collective WorkNCCL(SeqNum=410, OpType=BROADCAST,
NumelIn=1769472, NumelOut=1769472, Timeout(ms)=600000)
NCCL error: unhandled system error
ncclSystemError
```

错误关联末尾逐参数一致性检查的broadcast，在完成/清理区域被watchdog异步报告；精确根因未确定。torchrun返回0，`acceptance.json`写入`passed=true`，但[独立验收](evidence/blocked-audit.json)明确为 **BLOCKED_SMOKE_NCCL_ERROR / smoke_passed=false**。原始[控制台](evidence/smoke.console.txt)完整保留，不因退出码0忽略NCCL错误，也不修改原始acceptance文件。

本次未执行正式训练、严格最终导出或六项评测，因此无E11@3651、J_long或相对A3/3-0/Clean/Full的效果结论。后续需要先处理并重新通过真实四卡通信验收，才能按授权的固定方案进入正式训练；本报告不自动触发该操作。

代码、配置、[复现脚本](run.sh)、[机器可读状态](results.json)、原始测试/错误证据均保留。大检查点位于服务器 `/root/lk_projects/SAID-nest-clip-v1/hybridf_v1/E11/smoke`，哈希见results.json，不上传Git。所有GPU任务已经退出。
