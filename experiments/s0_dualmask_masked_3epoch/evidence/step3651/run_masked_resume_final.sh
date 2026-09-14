#!/usr/bin/env bash
set -eu
cd /root/SAID-s0-dualmask-clean-v01 || exit 90
run_dir=/root/SAID-s0-dualmask-clean-v01/runs_salu/dual_mask_suffix_masked_from500_continuation_v02
resume_ckpt="$run_dir/s0_dual_mask_suffix_masked_step001000.pt"
expected_sha=11af80b344c623b27b93069f9be526970c9c950c
trap 'code=$?; printf "%s\n" "$code" > "$run_dir/resume1000.exitcode"' EXIT
test ! -e "$run_dir/resume1000_started.txt" || exit 96
test "$(git rev-parse HEAD)" = "$expected_sha" || exit 91
git diff --quiet || exit 92
test -f "$resume_ckpt" || exit 94
test -f "$run_dir/salu_log.jsonl" || exit 95
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || exit 86
export NCCL_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 GLOO_SOCKET_IFNAME=lo CUDA_VISIBLE_DEVICES=0,1,2,3
export SHARE4V_DATA_ROOT=/root/datasets/ShareGPT4V
export SHARE4V_JSON=share-captioner_coco_lcs_sam_1246k_1107.json
train_started=$(date +%s)
date --iso-8601=seconds > "$run_dir/resume1000_started.txt"
set +e
/root/miniconda3/envs/said-smartclip/bin/python -m torch.distributed.run \
  --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29612 \
  train/train_dual_mask_suffix.py --suffix-mode masked --run-type formal \
  --batch-size 256 --epochs 3 --max-steps 3651 --seed 0 --num-workers 8 \
  --amp-dtype bf16 --image-chunk 16 --text-chunk 32 --save-every 100 \
  --init-state /root/SAID-gap-completion/runs_salu/said_cls_cvssl/shared_init/cvssl_initial.pt \
  --resume "$resume_ckpt" --output-dir "$run_dir"
run_exit=$?
printf '%s\n' "$run_exit" > "$run_dir/resume1000.exitcode"
printf '%s\n' "$(( $(date +%s) - train_started ))" > "$run_dir/resume1000_wall_seconds.txt"
date --iso-8601=seconds > "$run_dir/resume1000_finished.txt"
exit "$run_exit"
