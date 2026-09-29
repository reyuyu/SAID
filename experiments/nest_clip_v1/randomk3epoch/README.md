# A3-RandomK：完整3 epoch与扩展检索

从已完成的A3-RandomK step500完整检查点继续到 **累计3651更新**，新增3151更新。原本scheduler horizon就是3651，因此这次只延后停止点，不延长日程、不重启warmup。保持模型、RandomK采样、包含项最高权重1、优化器及精度配置。

父检查点：`/root/lk_projects/SAID-nest-clip-v1/randomk500/A3-RandomK/step000500.pt`，SHA256 `1fb8f630b181a4b93c4b303f5844e43bcf8fd6aff3b0a73d872bbc0c42ed4d40`。父trainer SHA256为 `66da82ca76945eb3b7803767996c0216bc354974d204361740aee85c22cabc31`，由续训命令显式固定。仅允许已知trainer迁移，模型/目标/数据源码哈希及方法配置必须一致。

加载完整模型、AdamW状态和每rank Python/NumPy/CPU/CUDA RNG；从epoch0、batch500接续，第一个新增更新为step501。现有入口会顺序读取但跳过已消费批次；图像预处理确定、K由(seed,epoch,sample_id)局部派生，读取跳过部分不消耗模型RNG。epoch1/2分别set_epoch并重建非持久spawn worker；每epoch1217更新，尾批每rank180。完整epoch停止时让DataLoader自然耗尽，避免仅依赖迭代器析构关闭。

输出目录独立为 `/root/lk_projects/SAID-nest-clip-v1/randomk3epoch/A3-RandomK`，保留续训起点step500、每100步和最终step3651。原500步与smoke产物不覆盖。

最终step3651评测COCO、Urban、Flickr30k test1k、DOCCI test、DCI full、Long-DCI重建版。另补step500的四项扩展评测，与已有step500 COCO/Urban一起形成同模型轨迹比较；不重训其他模型、不挑中间最好检查点。可引用已有Clean/Full@3651作为方法横向参考。

全部使用原生归一化图像/文本embedding内积，无mask/融合/rerank。COCO/Urban沿用现有入口；扩展协议直接运行已冻结 `eval_extended_real.py`，batch64、FP32、原tokenizer/context248与排序，扩展评测CPU线程数沿用历史OMP/MKL=4。Long-DCI为7602图/文的重建版，不声称官方长描述CSV协议。

现有服务器上的执行命令（输出必须全新，逐条检查退出码）：

```bash
cd /root/lk_projects/SAID
OMP_NUM_THREADS=2 /root/miniconda3/envs/said-repro/bin/python -m pytest -q tests/test_nested_resume.py tests/test_nested_randomk.py tests/test_nested_semantic_mask.py
bash experiments/nest_clip_v1/randomk3epoch/run.sh train
bash experiments/nest_clip_v1/randomk3epoch/run.sh export
bash experiments/nest_clip_v1/randomk3epoch/run.sh verify-export
bash experiments/nest_clip_v1/randomk3epoch/run.sh coco
bash experiments/nest_clip_v1/randomk3epoch/run.sh urban
for dataset in flickr_test1k docci dci long_dci; do
  bash experiments/nest_clip_v1/randomk3epoch/run.sh "$dataset" 3651 || exit
done
for dataset in flickr_test1k docci dci long_dci; do
  bash experiments/nest_clip_v1/randomk3epoch/run.sh "$dataset" 500 || exit
done
```
