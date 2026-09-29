# A3-RandomK step2000评测

对同一A3-RandomK三epoch训练轨迹的step2000检查点做原生评测，不重新训练。按[最新评测范围](../EVALUATION_POLICY.md)，只运行COCO、Urban、Flickr30k test1k和DOCCI，跳过DCI及Long-DCI。

```bash
cd /root/lk_projects/SAID
bash experiments/nest_clip_v1/randomk2000/run.sh export
bash experiments/nest_clip_v1/randomk2000/run.sh verify-export
bash experiments/nest_clip_v1/randomk2000/run.sh coco
bash experiments/nest_clip_v1/randomk2000/run.sh urban
bash experiments/nest_clip_v1/randomk2000/run.sh flickr_test1k
bash experiments/nest_clip_v1/randomk2000/run.sh docci
```

结果放入独立目录 `/root/lk_projects/SAID-nest-clip-v1/randomk2000`，不覆盖500/3651结果。现有导出器严格加载，并核对optimizer更新数为2000及native图文embedding一致。评测协议、文本编码与排序保持不变；只评用户指定的step2000，不自动增加其他检查点。
