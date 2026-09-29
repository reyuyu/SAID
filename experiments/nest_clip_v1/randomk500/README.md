# A3-RandomK @500

本实验只改变原A3的局部文本切分：先用旧 `text_views` 确定F，再在F的n个实际可见句段中均匀抽取 `K in {1,...,n-1}`，得到连续前缀P和全部剩余描述R。R保留F末句。`tokens_o/tokens_e` 与日志中的 `O/E` 在本实验分别表示 **P/R**，不再固定代表首句概述/展开。

模型内部仍为 `arm="A3"`；包含权重 `min(1, completed_updates/200)`、网络、评分、稀疏、可导gather及归约完全复用旧实现。`fixed_first` 是缺省模式，保持旧数据输出。首句超长与单句继续F-only；随机切分遇到独立token长度超限直接报出sample_id/epoch/n/K及文本，不重抽、不截断、不改变F。

采样种子为字符串 `"{sampling_seed}:{epoch}:{original_sample_id}"` 的UTF-8字节，经SHA256后按大端整数转换，再调用独立的 `random.Random(local_seed).randrange(1,n)`。sampling_seed=0，原sample_id=index+1000。它不依赖rank/worker/PID，不修改全局Python/NumPy/Torch RNG。每个epoch在创建DataLoader迭代器前同时设置sampler和dataset epoch；spawn worker不持久化。

训练配置与固定A3一致：4×A100 80GB，4×256，无梯度累积，seed=0，完整1,245,901样本；每epoch1217步，epochs=3，完整H=3651。smoke停止点5，正式停止点500，二者都从共同step0独立开始。

每rank日志另记录sample_ids、n/K（无效样本K=0；首句超长n=0是不可用计数标记）、按n分组K直方图、有效样本P/R句段数与未截断token长度直方图、K=1计数，以及sample_id/F/P-R/split独立SHA256。`fixed_first_reference_stream_sha256`只用于CPU日志：对同批重新使用原固定切分的视图/token，按旧日志的规范序列化，直接与原A3每step/rank的摘要对照。该参考视图不进入训练前向或loss。

以下命令需要在服务器现有环境中逐条执行并检查退出码；脚本拒绝覆盖已有输出。正式训练不读取smoke权重。

```bash
cd /root/lk_projects/SAID
OMP_NUM_THREADS=2 /root/miniconda3/envs/said-repro/bin/python -m pytest -q tests/test_nested_randomk.py tests/test_nested_semantic_mask.py
bash experiments/nest_clip_v1/randomk500/run.sh smoke
bash experiments/nest_clip_v1/randomk500/run.sh train500
bash experiments/nest_clip_v1/randomk500/run.sh export500
bash experiments/nest_clip_v1/randomk500/run.sh verify-export
bash experiments/nest_clip_v1/randomk500/run.sh coco
bash experiments/nest_clip_v1/randomk500/run.sh urban
```

正式输出：`/root/lk_projects/SAID-nest-clip-v1/randomk500/A3-RandomK`。smoke输出：同级 `smoke/A3-RandomK`。共同初始化沿用 `shared/step000000.pt`，SHA256为 `54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`。

主比较是随机K与固定K=1的A3，各500步、同原生评测协议；Clean/Full只引用已有500步结果作横向参考。本实验不能单独证明包含项有效，单seed的小幅差值不代表统计显著。
