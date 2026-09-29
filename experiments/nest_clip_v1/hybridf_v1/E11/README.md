# NEST-HybridF-E11

本轮实际交付：[step500四集原生评测](STEP500_REPORT.md)、[机器可读结果](step500_results.json)。用户在正式训练超过500步后指示停止，已中断于step1123，仅评先前保存的step500检查点。没有完成3651步，也未评测DCI/Long-DCI。[原始失败smoke报告](REPORT.md)与[历史阻塞状态](results.json)保留为当时记录：最初的真实四卡smoke末尾有NCCL异步错误；用户指示继续后，在独立目录重跑相同5步，仅打开`NCCL_DEBUG=INFO`诊断，[严格复验](evidence/smoke-retry-1-audit.json)通过。

最新用户指令取消了本次的DCI与Long-DCI评测，因此没有计算需要DCI的`J_long`。只训练E11，不运行E00/E01或扫描eta。

模型内部 `arm=A3`，`full_native_mix=0.25` 从首步固定；先计算两项独立双向CE，再混合 `L_F_hybrid=.75 L_F_mask+.25 L_F_native`。正常三视图目标为 `(10/3)(L_F_hybrid+L_P_mask+L_R_mask)+(Omega_F+2 Omega_P+2 Omega_R)/3+lambda_inc L_inc`。全局V<2时为 `10 L_F_hybrid+Omega_F`。没有把稀疏或包含乘0.75，没有混合logits，没有额外编码或参数。

native项复用本次已有z/tF和全局图像bank，对全文文本另做可导gather；每rank本地双向CE sum按W/N归约。FP32评分、固定scale100、eps1e-6，无mask及可学习温度。默认eta=0完全跳过native分支。恢复时校验eta、arm、RandomK、horizon及原有代码指纹，旧A3检查点不能冒充E11恢复。

数据/RandomK代码保持原样。4×A100 80GB、256/rank、accumulation1、seed0、workers8、epochs3、1217更新/epoch、H=3651；原AdamW、LR与warmup、非重入checkpoint和score_chunk64不变。smoke与正式均从共同step0独立启动，首次正式启动不传resume。共同初始化SHA256：`54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`。

```bash
cd /root/lk_projects/SAID
OMP_NUM_THREADS=2 /root/miniconda3/envs/said-repro/bin/python -m pytest -q tests/test_nested_hybridf.py tests/test_nested_resume.py
OMP_NUM_THREADS=2 /root/miniconda3/envs/said-repro/bin/torchrun --standalone --nnodes=1 --nproc-per-node=2 --max-restarts=0 -m tests.hybridf_ddp_worker
bash experiments/nest_clip_v1/hybridf_v1/E11/run500.sh coco
for dataset in urban flickr_test1k docci; do
  bash experiments/nest_clip_v1/hybridf_v1/E11/run500.sh "$dataset" || exit
done
```

上述评测命令为历史执行记录，已生成的JSON与产物保留服务器；重新执行时会因目标文件存在而拒绝覆盖。原计划的完整3651步脚本仍在[run.sh](run.sh)，本轮已由用户终止，不将其终点命令当成本次复现流程。step500同预算对照为A3-RandomK，COCO/Urban另列固定A3、S0、Clean和Full历史参考。
