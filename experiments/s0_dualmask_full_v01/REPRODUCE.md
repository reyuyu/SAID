# Full v0.1 固定轨迹复现

本说明复现已报告的 `0 → 500 → 3651` 轨迹。主 README 的 main 代码仅作维护入口；历史数值对应下面两份固定提交。

## 代码与输入

```bash
git clone https://github.com/reyuyu/SAID.git SAID-docs
cd SAID-docs
git worktree add --detach ../SAID-full500 dcd33877f1f77a901d292190834f1ffd83a049a8
git worktree add --detach ../SAID-full3651 873b43a5bc000528311e22aac01e92045ac898e0
```

环境安装参考 [项目说明](../../README.md)与 [Clean 环境快照](../s0_dualmask_masked_3epoch/environment/runtime.json)。两项实验在同一训练配置家族中，但 Clean 环境快照不能冒充一次重新采集的 Full 环境；Full 的原始运行配置以 [500 步报告 configuration](../../docs/dual_mask_suffix_full_v01/masked_full_formal500_report.json) 为准。

准备 [资产清单](../ASSETS.md) 中的完整训练输入、共同初始化、OpenAI CLIP base、评测数据。用同一份 `cvssl_initial.pt`，SHA256 为 `c1a4a2be1b212f38677f729a4f03160d788a6d6db79ca5021a43022af0f5cba8`。两段运行之间保持初始化的绝对路径不变。

## 配置

ViT-B/16；4 GPUs × batch256；accumulation=1；seed=0；3 epoch = 3×1217 = 3651 updates。AdamW 三组 LR：CLIP `1e-6`、S0 门 `1e-3`、U 门 `1e-4`；CLIP weight decay `0.01`、两组门 weight decay `0`；CLIP warmup200；完整 cosine horizon3651。FP32 主参数 + BF16 autocast；image/text chunk16/32；workers8/rank；total_len1000。前缀 K 分布和后缀排除末句规则保持原训练数据实现。

在个人、不提交的 shell 中设置路径：

```bash
export FULL_PYTHON=/your/env/bin/python
export FULL_REPO500=/your/SAID-full500
export FULL_REPO3651=/your/SAID-full3651
export FULL_RUN500=/your/new-runs/full500
export FULL_RUN3651=/your/new-runs/full3651
export FULL_INIT=/your/assets/cvssl_initial.pt
export SHARE4V_DATA_ROOT=/your/datasets/ShareGPT4V
export SHARE4V_JSON=share-captioner_coco_lcs_sam_1246k_1107.json
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
```

上述 loopback NCCL 设置取自原始单机运行。开训前确认四卡空闲；模型不适配减小 global batch 的替代“复现”。常规输入身份检查可参考 [Clean 数据迁移说明](../s0_dualmask_masked_3epoch/TRANSFER_AND_UPLOAD.md)，但其中的 Clean checkpoint/代码哈希不能用来校验 Full checkpoint/代码。

## 0 → 500

```bash
cd "$FULL_REPO500"
test "$(git rev-parse HEAD)" = dcd33877f1f77a901d292190834f1ffd83a049a8
test ! -e "$FULL_RUN500"
"$FULL_PYTHON" -m torch.distributed.run --nproc_per_node=4 \
  --master_addr=127.0.0.1 --master_port=29610 train/train_dual_mask_suffix.py \
  --suffix-mode masked --run-type formal --base-model B16 \
  --batch-size 256 --epochs 3 --max-steps 500 --seed 0 --num-workers 8 \
  --lr 1e-6 --mask-lr 1e-3 --suffix-lr 1e-4 --weight-decay 0.01 --warmup 200 \
  --lambda-suffix 10 --lambda-u-sparse 2 --total-len 1000 --amp-dtype bf16 \
  --image-chunk 16 --text-chunk 32 --save-every 100 \
  --init-state "$FULL_INIT" --output-dir "$FULL_RUN500"
```

确认进程退出 0、完整 step500 checkpoint 存在且更新数为 500，再续训；失败轨迹保留并先定位，不混算更新数。

## 500 → 3651

原轨迹在新的输出目录里保留从 500 恢复的状态、优化器和流位置。使用累计上限3651，不是新增3651。

```bash
cd "$FULL_REPO3651"
test "$(git rev-parse HEAD)" = 873b43a5bc000528311e22aac01e92045ac898e0
test ! -e "$FULL_RUN3651"
"$FULL_PYTHON" -m torch.distributed.run --nproc_per_node=4 \
  --master_addr=127.0.0.1 --master_port=29612 train/train_dual_mask_suffix.py \
  --suffix-mode masked --run-type formal --base-model B16 \
  --batch-size 256 --epochs 3 --max-steps 3651 --seed 0 --num-workers 8 \
  --lr 1e-6 --mask-lr 1e-3 --suffix-lr 1e-4 --weight-decay 0.01 --warmup 200 \
  --lambda-suffix 10 --lambda-u-sparse 2 --total-len 1000 --amp-dtype bf16 \
  --image-chunk 16 --text-chunk 32 --save-every 500 \
  --init-state "$FULL_INIT" --output-dir "$FULL_RUN3651" \
  --resume "$FULL_RUN500/s0_dual_mask_suffix_masked_step000500.pt" \
  --expect-model-sha e92232b78511c53612a4eec9ebc7fea1302f60c397eba53c81fba0b9c12ff630
```

检查点包含1000/1500/2000/2500/3000/3500/3651。不要在这条历史轨迹上使用后来加入的调度扩展开关。内容级逐批 K 流未获外部独立重放的限制见 [原始续训报告](../../docs/dual_mask_suffix_full_v01/continuation_3epoch_report.md)。

## 导出和评测

完整 checkpoint 含 clip、suffix_gate 和优化器；bare student 只含317个 CLIP 张量。使用固定 Full 仓库的 `tools/diag/export_dualmask_full_student.py --help`，按其 `--expect_steps` 和参数要求严格导出；读取 [原始导出 metadata](evidence/step3651/bare_student_metadata.json)核对来源。

原生推理示例在 [主 README](../../README.md)。冻结 COCO/Urban、扩展评测的原始文件与命令见 [结果说明](../RESULTS.md)及 [step2000 执行记录](evidence/step2000/new_evaluations/run_extended.sh)。这些历史脚本中的 `/root/...` 是原机器路径，迁移时需映射到实际代码/数据位置；绝不能因为目录叫 Clean 就加载 Clean 权重代替 Full。
