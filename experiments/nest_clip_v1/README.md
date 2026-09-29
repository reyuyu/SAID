# NEST-CLIP v1

后续任务遵循[最新评测范围](EVALUATION_POLICY.md)，不再运行DCI及Long-DCI。下文保留第一轮5步冒烟归档。

共享 CLIP/LongCLIP backbone 和原 MaskNetwork，独立编码全文 F、概述 O、展开 E。A2 使用三视图 masked 对齐和稀疏；A3 仅增加局部到全文的软包含项。

本轮只完成实现、必要测试及 A2/A3 各 5 步真实四卡冒烟。**未启动正式 500 步训练或检索评测**，这些结果不能用于判断方法效果。

| 实验 | 每 rank 更新数 | 第1步 loss | 第5步 loss | 每卡峰值 allocated |
|---|---:|---:|---:|---:|
| A2 | 5 | 56.216709 | 18.103582 | 17.301 GiB |
| A3 | 5 | 56.216709 | 18.107388 | 17.301 GiB |

两组均为单机 4 张 A100 80GB、每 rank batch=256。全部步骤 F/O/E 各有 1024 个全局候选，四 rank 最终参数差异为0。完整数据 1,245,901 条，每 epoch 1217 步，scheduler horizon=3651。

- [完整报告与正式训练、导出、评测命令](REPORT.md)
- [A2 配置与逐步结果](smoke/A2)、[A3 配置与逐步结果](smoke/A3)
- [两卡参考测试](evidence/ddp-results.json)、[最终四卡审计](evidence/final-audit.json)
- [共同初始化来源](manifests/initialization.json)、[训练数据元信息](manifests/training_data.json)
- [服务器 checkpoint 路径及 SHA256](manifests/checkpoints.json)
- [发布证据校验清单](manifests/published_evidence.json)

原始证据中的绝对路径是运行时服务器路径，保留用于溯源。数据集、完整数据索引与 `.pt` 文件未放入 Git；指标、配置、验收和调试记录已包含在此目录。

在现有服务器环境运行单元测试和两卡参考测试：

```bash
cd /root/lk_projects/SAID
OMP_NUM_THREADS=2 /root/miniconda3/envs/said-repro/bin/python -m pytest -q tests/test_nested_semantic_mask.py
OMP_NUM_THREADS=2 /root/miniconda3/envs/said-repro/bin/torchrun \
  --standalone --nnodes=1 --nproc-per-node=2 --max-restarts=0 \
  -m tests.nest_ddp_worker
```

真实四卡冒烟命令（每次必须使用新输出目录；两组顺序执行）：

```bash
cd /root/lk_projects/SAID
for arm in A2 A3; do
  OMP_NUM_THREADS=4 /root/miniconda3/envs/said-repro/bin/torchrun \
    --standalone --nnodes=1 --nproc-per-node=4 --max-restarts=0 \
    -m train.train_nested_semantic_mask \
    --config "configs/nest_clip_${arm,,}.json" \
    --init-state /root/lk_projects/SAID-nest-clip-v1/shared/step000000.pt \
    --index-dir /root/lk_projects/SAID-nest-clip-v1/data_index \
    --image-root /root/lk_projects/SAID-assets/training/ShareGPT4V \
    --output-dir "/root/lk_projects/SAID-nest-clip-v1/smoke-rerun/$arm" \
    --run-type smoke --max-updates 5 || exit
done
```

[run.sh](run.sh) 保留本服务器的完整正式命令，需要显式传入动作及 A2/A3，不会默认启动训练。其他服务器需调整资源和解释器路径，保留方法、batch、全局候选池和完整 scheduler horizon。
