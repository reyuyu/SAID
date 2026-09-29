# 3/0：从500步接续至完整3 epoch

先按 [500步实验](../s0_dualmask_suffix3_500/REPRODUCE.md) 准备同一数据、环境、重建初态及3/0的完整step500检查点。必须保留优化器状态；裸学生不用于续训。

本次父checkpoint SHA256为 `4e905bf1160f640da70de6045729cf27bf6f8ca67834198f2e029e030ec5ed21`。原500步训练SHA为 `dcd33877f1f77a901d292190834f1ffd83a049a8`，续训SHA为 `873b43a5bc000528311e22aac01e92045ac898e0`；两者目标函数代码SHA相同。实际大文件身份见 [assets.json](manifests/assets.json)。自行重训得到的新容器应记录实际SHA，不能冒用本次文件身份。

## 固定代码与路径

```bash
# SAID_DOCS_REPO 为包含本说明的仓库克隆，不能是正在训练的目录。
export SAID_DOCS_REPO=/your/SAID-docs
export SAID_BUNDLE="$SAID_DOCS_REPO/experiments/s0_dualmask_suffix3_3epoch"
git -C "$SAID_DOCS_REPO" worktree add --detach /your/SAID-suffix3-resume 873b43a5bc000528311e22aac01e92045ac898e0
export SAID_RESUME_REPO=/your/SAID-suffix3-resume
export SAID_PYTHON=/your/conda/envs/said-suffix3/bin/python
export SAID_PARENT=/your/step500/s0_dual_mask_suffix_masked_step000500.pt
export SAID_INIT=/your/assets/common_init_reconstructed.pt
export SHARE4V_DATA_ROOT=/your/datasets/ShareGPT4V
export SHARE4V_JSON=share-captioner_coco_lcs_sam_1246k_1107.json
export SHARE4V_FULL_AUDIT=/your/audit/sharegpt4v_full_audit.json
export SAID_RUN=/your/new-runs/suffix3_u0_3epoch
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

环境锁及模型构造缓存要求沿用 [500步环境](../s0_dualmask_suffix3_500/environment/requirements.txt) 和 [500步复现步骤](../s0_dualmask_suffix3_500/REPRODUCE.md)。本次同一重建初态在两段运行间保持同一绝对路径，张量摘要为 `caf61198def1b78654d6b70ef16db9a3475ea81b3a16ab8a799989f574de2faf`。迁移时保存原文件或重建后验证相同张量身份，并如实记录路径与新容器SHA。

## 续训

```bash
bash "$SAID_BUNDLE/tools/resume3651.sh"
```

入口验证固定源码、初态张量、父checkpoint实际500更新、系数、world size、batch、loader和optimizer计数，再调用原始续训代码。保持：

- `lambda_suffix=3`、`lambda_u_sparse=0`，原S0目标仍为`10L_S+2S_S`。
- 4卡×256、workers8/rank、seed0、FP32主参数与BF16 autocast。
- 三组LR为1e-6 / 1e-3 / 1e-4；CLIP warmup200，完整cosine horizon3651。
- `epochs=3`、累计`max_steps=3651`，新增3151次更新；不扩展horizon，不重新warmup。
- 每500步保存，并保存最终3651；输出日志只含501–3651。

成功验收应为：退出码0，3651更新、epoch=2、step_in_epoch=1216，三组optimizer计数3651，新增日志3151条。原训练脚本按已完成更新数推导跳过范围，可兼容旧500最终checkpoint多前移一批的游标；第一条新更新应为501、epoch=0、step_in_epoch=500。

原500步日志已在 [父实验](../s0_dualmask_suffix3_500/evidence/training/training_steps_000001_000500.jsonl) 保存。该日志与本次 [501–3651日志](evidence/training/training_steps_000501_003651.jsonl) 按顺序合并即为完整3651更新历史，包内不再重复存放合并副本。

## 校验范围

本次CPU预检验证模型、U门和三个optimizer恢复后与父checkpoint逐项相同，[记录](validation/preflight.json)完整保留。

固定873b43a入口会重放/跳过已消费batch以恢复loader位置，但只累计新的续训数据摘要。**它没有独立比较被跳过500步的内容摘要。** 本次另外逐步对照了501–3651的四rank累计摘要和rank0批次摘要，均与此前同机Full复现的同一数据流一致；见 [continuation_data_stream.json](validation/continuation_data_stream.json)。这些摘要覆盖ID、prefix K与文本，不覆盖图片字节或具体计算算子。

## 导出与最终评测

```bash
"$SAID_PYTHON" "$SAID_BUNDLE/tools/export_student.py" \
  --repo "$SAID_RESUME_REPO" \
  --checkpoint "$SAID_RUN/s0_dual_mask_suffix_masked_step003651.pt" \
  --output "$SAID_RUN/bare_student_step3651.pt" --report "$SAID_RUN/export.json"

"$SAID_PYTHON" "$SAID_BUNDLE/tools/evaluate.py" \
  --repo "$SAID_RESUME_REPO" --checkpoint "$SAID_RUN/bare_student_step3651.pt" \
  --export-report "$SAID_RUN/export.json" \
  --coco-root /your/datasets/coco --urban-root /your/datasets/Urban1k/Urban1k \
  --benchmark-root /your/datasets/retrieval_benchmarks \
  --output-dir /your/new-evaluations/suffix3_u0_3651 --gpus 0,1,2,3
```

导出器验证3651更新、最终游标、系数3/0、原horizon、500步续训来源、317个学生张量与U门严格加载、三组optimizer计数及状态有限性，保存重载后逐张量相等。迁移本次完整checkpoint可额外给出 `--expected-sha256 e0c607a8e8001c597fffcc5046f8a2a77309845f46cd7fa48784c7ec9c0c753a`。

评测沿用相同冻结函数和FP32原生学生向量：COCO图像batch64/文本通常320、chunk512及逐行argsort；Urban文本一次1000；扩展图文各batch64、CPU完整相似度矩阵topk。Long-DCI沿用7602条重建版，清单及数据准备见 [500步记录](../s0_dualmask_suffix3_500/REPRODUCE.md)。

每项原始输出、实际命令、退出码在 [evidence/evaluation](evidence/evaluation/)。[comparison.json](comparison.json) 和 [comparison.csv](comparison.csv) 分别保留与自身500步、本机Clean/Full3651及原历史A800结果的差值；不要混淆不同时长、机器或权重身份。
