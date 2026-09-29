# NEST-HybridF-E11

当前状态：实现和单元/两rank测试通过，但真实四卡smoke末尾出现NCCL异步错误，正式训练尚未启动。见[阻塞报告](REPORT.md)与[机器可读状态](results.json)。以下命令仅作复现记录，不能将原始acceptance.json的passed字段当作完整smoke验收通过。

本轮最新完整指令明确授权六项评测和J_long，因此本实验恢复DCI/Long-DCI；它覆盖此前“后续跳过DCI”的范围要求。只训练E11，不运行E00/E01或扫描eta。

模型内部 `arm=A3`，`full_native_mix=0.25` 从首步固定；先计算两项独立双向CE，再混合 `L_F_hybrid=.75 L_F_mask+.25 L_F_native`。正常三视图目标为 `(10/3)(L_F_hybrid+L_P_mask+L_R_mask)+(Omega_F+2 Omega_P+2 Omega_R)/3+lambda_inc L_inc`。全局V<2时为 `10 L_F_hybrid+Omega_F`。没有把稀疏或包含乘0.75，没有混合logits，没有额外编码或参数。

native项复用本次已有z/tF和全局图像bank，对全文文本另做可导gather；每rank本地双向CE sum按W/N归约。FP32评分、固定scale100、eps1e-6，无mask及可学习温度。默认eta=0完全跳过native分支。恢复时校验eta、arm、RandomK、horizon及原有代码指纹，旧A3检查点不能冒充E11恢复。

数据/RandomK代码保持原样。4×A100 80GB、256/rank、accumulation1、seed0、workers8、epochs3、1217更新/epoch、H=3651；原AdamW、LR与warmup、非重入checkpoint和score_chunk64不变。smoke与正式均从共同step0独立启动，首次正式启动不传resume。共同初始化SHA256：`54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`。

```bash
cd /root/lk_projects/SAID
OMP_NUM_THREADS=2 /root/miniconda3/envs/said-repro/bin/python -m pytest -q tests/test_nested_hybridf.py tests/test_nested_resume.py
OMP_NUM_THREADS=2 /root/miniconda3/envs/said-repro/bin/torchrun --standalone --nnodes=1 --nproc-per-node=2 --max-restarts=0 -m tests.hybridf_ddp_worker
bash experiments/nest_clip_v1/hybridf_v1/E11/run.sh smoke
bash experiments/nest_clip_v1/hybridf_v1/E11/run.sh train
bash experiments/nest_clip_v1/hybridf_v1/E11/run.sh export
bash experiments/nest_clip_v1/hybridf_v1/E11/run.sh verify-export
for dataset in coco urban flickr_test1k docci dci long_dci; do
  bash experiments/nest_clip_v1/hybridf_v1/E11/run.sh "$dataset" || exit
done
```

输出独立保存至 `/root/lk_projects/SAID-nest-clip-v1/hybridf_v1/E11/{smoke,formal,evaluation}`，脚本拒绝覆盖。每100步及终点保存；正式只评step3651。主比较为E11−原A3-RandomK@3651，另引用固定ref `8aaad8820b62d3e87285efc2d14fb389f880fd9d` 的3/0，以及已核验Clean/Full@3651本机结果。预先固定 `J_long=mean(Urban,DOCCI,DCI 的双向R@1)`，Long-DCI单独报告，不重复加权。
