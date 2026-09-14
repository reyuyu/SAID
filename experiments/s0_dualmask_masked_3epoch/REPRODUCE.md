# 在另一台服务器复现

以下命令针对 Linux x86_64、单机4张可运行 BF16 的 NVIDIA GPU。先完成 [输入文件迁移](TRANSFER_AND_UPLOAD.md)，再训练。命令用于你新机器上的新目录；不会触碰原服务器的在训目录。

## 1. 获取说明和两份固定代码

```bash
git clone --branch codex/s0-dualmask-clean-v01 https://github.com/reyuyu/SAID.git SAID-docs
cd SAID-docs
git worktree add --detach ../SAID-base500 ff5ad1d4b918d56c6bfa48a2870dc5223e757237
git worktree add --detach ../SAID-train 11af80b344c623b27b93069f9be526970c9c950c
```

`SAID-docs` 存放本包，`SAID-base500` 复现原0–500，`SAID-train` 负责后续恢复。不要把最新文档提交当训练 SHA；运行中不要在两份训练工作区切分支或 pull。checkpoint 保存时会读取工作区 HEAD。

## 2. 重建环境

优先使用采集的 Conda linux-64 包列表和 pip 版本锁，而不是仓库根目录中未完全锁版本的基础 requirements。

```bash
conda create -n said-repro --file experiments/s0_dualmask_masked_3epoch/environment/conda-explicit-linux-64.txt -y
conda activate said-repro
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
  -r experiments/s0_dualmask_masked_3epoch/environment/requirements-resolved.txt
python -m pip check
```

`pip-freeze.txt` 是原始快照，里面 pip 自身的 `file:///...` 是 Conda 构建路径，不是可跨机下载的源；`requirements-resolved.txt` 仅把这一项规范化成同版本 `pip==26.2.1`，其余版本保持原样。原环境 `pip check` 退出0。未在另一台机器实际重新安装整个环境，包源若已移除，需要从原环境/轮子缓存迁移；不要静默升级版本后声称环境相同。

检查 `environment/runtime.json` 与 `nvidia-smi.csv`。原驱动535.129.03、CUDA12.4 PyTorch轮子；系统/CPU/driver的实际情况都已记录。仅更改显卡型号仍可能改变浮点结果。

## 3. 设置路径并验证

复制 `commands/env.example.sh` 到个人不提交的文件，替换所有 CHANGE_ME，然后 `source /your/path/env.sh`。`REPRO_INIT` 在本次复现的所有阶段都必须保持同一绝对路径。缓存 CLIP base 放在当前运行用户的 `~/.cache/clip/ViT-B-16.pt`。

```bash
"$REPRO_PYTHON" "$REPRO_BUNDLE/tools/verify_inputs.py" \
  --repo "$REPRO_TRAIN_REPO" --init "$REPRO_INIT" \
  --train-json "$SHARE4V_DATA_ROOT/$SHARE4V_JSON" \
  --coco-root "$REPRO_COCO_ROOT" --urban-root "$REPRO_URBAN_ROOT" \
  --eval-files --check-training-paths --training-image-root "$SHARE4V_DATA_ROOT"
```

这会核对训练代码、初始化、base CLIP、两项标注的 SHA256，评估7000个文件（COCO5000图 + Urban1000图/1000文本），以及全部训练样本的图片路径是否存在。训练图片路径存在不等于图片字节一致，完整迁移校验方法见下一文档。输入缺失、摘要不符或环境不一致时先解决输入问题。

数学/续训检查可在 CPU 执行：

```bash
cd "$REPRO_TRAIN_REPO"
CUDA_VISIBLE_DEVICES='' GLOO_SOCKET_IFNAME=lo "$REPRO_PYTHON" \
  tests/_dual_mask_suffix_resume_worker.py
CUDA_VISIBLE_DEVICES='' GLOO_SOCKET_IFNAME=lo "$REPRO_PYTHON" -m torch.distributed.run \
  --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29630 \
  tests/_dual_mask_suffix_ddp_worker.py --backend gloo --out /your/new/path/ddp_gloo.json
```

必须使用全目标与非全开 gate 的已提交测试，报告容差内一致；不要仅以 loss 相同或梯度非 None 作为通过。参考实测证据在 `evidence/`。

## 4. 从共同初始化跑到累计500

先用 `nvidia-smi` 确认选中四卡无其他任务；本配置显存实际峰值约65.8 GiB allocated、68.7 GiB reserved/卡，nvidia-smi占用更高。输出目录必须全新。

```bash
set -eu
test ! -e "$REPRO_RUN"
mkdir -p "$REPRO_RUN"
cd "$REPRO_BASE_REPO"
test "$(git rev-parse HEAD)" = ff5ad1d4b918d56c6bfa48a2870dc5223e757237
set +e
"$REPRO_PYTHON" -m torch.distributed.run --nproc_per_node=4 \
  --master_addr=127.0.0.1 --master_port=29610 train/train_dual_mask_suffix.py \
  --suffix-mode masked --run-type formal --base-model B16 \
  --batch-size 256 --epochs 3 --max-steps 500 --seed 0 --num-workers 8 \
  --lr 1e-6 --mask-lr 1e-3 --suffix-lr 1e-4 --weight-decay 0.01 --warmup 200 \
  --total-len 1000 --amp-dtype bf16 --image-chunk 16 --text-chunk 32 --save-every 100 \
  --init-state "$REPRO_INIT" --output-dir "$REPRO_RUN" \
  > "$REPRO_RUN/train500.console.log" 2>&1
stage_code=$?
printf '%s\n' "$stage_code" > "$REPRO_RUN/train500.exitcode"
set -e
test "$stage_code" = 0
```

只有成功退出、checkpoint 完整后执行下一段。OOM/非有限值/资源冲突时停止；保留失败目录，不重命名更新数、不自动恢复失败轨迹。任何 traceback 都应先查具体原因。

## 5. 同一状态500→1000

保持相同 REPRO_RUN、REPRO_INIT、环境和参数，从下面的 checkpoint 恢复；max_steps 是累计1000，不是再加1000。

```bash
cd "$REPRO_TRAIN_REPO"
test "$(git rev-parse HEAD)" = 11af80b344c623b27b93069f9be526970c9c950c
set +e
"$REPRO_PYTHON" -m torch.distributed.run --nproc_per_node=4 \
  --master_addr=127.0.0.1 --master_port=29612 train/train_dual_mask_suffix.py \
  --suffix-mode masked --run-type formal --base-model B16 \
  --batch-size 256 --epochs 3 --max-steps 1000 --seed 0 --num-workers 8 \
  --lr 1e-6 --mask-lr 1e-3 --suffix-lr 1e-4 --weight-decay 0.01 --warmup 200 \
  --total-len 1000 --amp-dtype bf16 --image-chunk 16 --text-chunk 32 --save-every 100 \
  --init-state "$REPRO_INIT" --output-dir "$REPRO_RUN" \
  --resume "$REPRO_RUN/s0_dual_mask_suffix_masked_step000500.pt" \
  > "$REPRO_RUN/resume500.console.log" 2>&1
stage_code=$?
printf '%s\n' "$stage_code" > "$REPRO_RUN/resume500.exitcode"
set -e
test "$stage_code" = 0
```

看到四条 `RESUME_REPLAY_VERIFIED ... completed=500 next_batch=500` 才完成原数据流核对。回放可能需数分钟，GPU空闲而数据worker忙是正常的；没有第501步日志前不能说已新增训练。下一条应为 completed_steps=501 / step_in_epoch=500。

## 6. step1000严格导出与原生测评

本包 `tools/native_export_eval.py` 只替换路径和入口，不重写检索度量；它调用 `reference_eval/` 中未经修改的原函数。源码副本 SHA 与原评估器已核对。

```bash
set +e
"$REPRO_PYTHON" "$REPRO_BUNDLE/tools/native_export_eval.py" --action export \
  --repo "$REPRO_TRAIN_REPO" --step 1000 \
  --checkpoint "$REPRO_RUN/s0_dual_mask_suffix_masked_step001000.pt" \
  --bare-output "$REPRO_RUN/bare_student_step1000.pt" \
  --report "$REPRO_RUN/export1000.json" > "$REPRO_RUN/export1000.log" 2>&1
stage_code=$?; printf '%s\n' "$stage_code" > "$REPRO_RUN/export1000.exitcode"
set -e; test "$stage_code" = 0

set +e
CUDA_VISIBLE_DEVICES=0 "$REPRO_PYTHON" "$REPRO_BUNDLE/tools/native_export_eval.py" \
  --action coco --repo "$REPRO_TRAIN_REPO" --device cuda:0 \
  --checkpoint "$REPRO_RUN/bare_student_step1000.pt" \
  --coco-root "$REPRO_COCO_ROOT" --report "$REPRO_RUN/coco1000.json" \
  > "$REPRO_RUN/coco1000.log" 2>&1
stage_code=$?; printf '%s\n' "$stage_code" > "$REPRO_RUN/coco1000.exitcode"
set -e; test "$stage_code" = 0

set +e
CUDA_VISIBLE_DEVICES=0 "$REPRO_PYTHON" "$REPRO_BUNDLE/tools/native_export_eval.py" \
  --action urban --repo "$REPRO_TRAIN_REPO" --device cuda:0 \
  --checkpoint "$REPRO_RUN/bare_student_step1000.pt" \
  --urban-root "$REPRO_URBAN_ROOT" --report "$REPRO_RUN/urban1000.json" \
  > "$REPRO_RUN/urban1000.log" 2>&1
stage_code=$?; printf '%s\n' "$stage_code" > "$REPRO_RUN/urban1000.exitcode"
set -e; test "$stage_code" = 0
```

原始实际服务器导出还做过同一小batch原生/后缀score roundtrip；误差记录在 `evidence/step1000/export_report_step1000.json`。上述可移植导出验证完整严格加载和bare所有tensor相等，不声称已在另一张GPU做同一全量评估。

## 7. step1000→完整3epoch

测评退出码全部为0、四卡再次空闲后：

```bash
cd "$REPRO_TRAIN_REPO"
set +e
"$REPRO_PYTHON" -m torch.distributed.run --nproc_per_node=4 \
  --master_addr=127.0.0.1 --master_port=29612 train/train_dual_mask_suffix.py \
  --suffix-mode masked --run-type formal --base-model B16 \
  --batch-size 256 --epochs 3 --max-steps 3651 --seed 0 --num-workers 8 \
  --lr 1e-6 --mask-lr 1e-3 --suffix-lr 1e-4 --weight-decay 0.01 --warmup 200 \
  --total-len 1000 --amp-dtype bf16 --image-chunk 16 --text-chunk 32 --save-every 100 \
  --init-state "$REPRO_INIT" --output-dir "$REPRO_RUN" \
  --resume "$REPRO_RUN/s0_dual_mask_suffix_masked_step001000.pt" \
  > "$REPRO_RUN/resume1000.console.log" 2>&1
stage_code=$?
printf '%s\n' "$stage_code" > "$REPRO_RUN/resume1000.exitcode"
set -e
test "$stage_code" = 0
```

先核对前1000步累计数据摘要，再执行1001。最后3651、epoch=2、step_in_epoch=1216；保存step003651，不执行3652。按历史计划保留每100步与最终检查点。不启动native，也不把两段失败尝试混入计数。需要最终checkpoint原生评估时可按上面的导出/测评命令把step改为3651并用新文件名；这份复现包不自动启动任何评估/训练。

## 直接迁移本次原step500或step1000

无需重新训练前缀轨迹时，先迁移 `manifests/assets.json` 指定的**原完整checkpoint**，然后：

```bash
"$REPRO_PYTHON" "$REPRO_BUNDLE/tools/prepare_migrated_run.py" \
  --checkpoint /your/migrated/s0_dual_mask_suffix_masked_step001000.pt \
  --step 1000 --output "$REPRO_RUN"
```

工具验证原SHA256并复制未经改写的checkpoint，继承本包前1000条日志。500同理，换step与文件。只适用于原资产；自己重训得到的不同checkpoint不能强行配本包日志。

**原checkpoint的 config.init_state 是 `/root/SAID-gap-completion/runs_salu/said_cls_cvssl/shared_init/cvssl_initial.pt`。** 固定恢复代码检查这一字符串；需要在新服务器/容器中将同一初始文件挂载或放置到该路径，并把 REPRO_INIT 设为此原路径。普通无root用户可以用具备该路径的容器；如果做不到，选择从共同初始化开始的自定义路径路线，不修改checkpoint元数据绕过检查。REPRO_RUN 可以不同；数据根目录也可不同，只要内容和stream完全一致。

## 不同 GPU 的限制

优先仍为同一主机上的4卡、每卡能容纳本配置，BF16支持。可改 CUDA_VISIBLE_DEVICES 的物理索引，但保持逻辑rank0..3、4卡、batch256和原数据worker；保存硬件与软件差异。仅两张卡/单卡或通过accumulation凑1024会改变DDP拓扑、候选池或更新语义，本包未验证，不能称原配置复现。

小显存卡若OOM，先保留失败；本包不擅自降低global batch。等价chunk更改可能改变浮点数与严格resume配置，只能另行验证并记录新配置，不能在原checkpoint恢复时静默改参数。loopback NCCL是原单机成功配置，不适用于跨主机通信；本包没有把多机训练验证为等价配置。
