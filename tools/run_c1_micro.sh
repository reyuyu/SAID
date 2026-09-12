#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/SAID-c1-micro-ablation-v01
cd "$ROOT"
export SHARE4V_DATA_ROOT=/root/datasets/ShareGPT4V
export SHARE4V_JSON=share-captioner_coco_lcs_sam_1246k_1107.json
export SHARE4V_FULL_AUDIT=/root/SAID/outputs/data_audit/sharegpt4v_full_audit.json
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo OMP_NUM_THREADS=1
export PATH=/root/miniconda3/envs/said-smartclip/bin:$PATH
INIT=/root/SAID-gap-completion/runs_salu/said_cls_cvssl/shared_init/cvssl_initial.pt
VARIANT=${1:?C1-UN or C1-VWarm}; OUT=${2:?output directory}; MAX=${3:?250 or 500}; RESUME=${4:-}
ARGS=(--arm C1_text_conditional_reconstruction --variant "$VARIANT" --init_state "$INIT" --output_dir "$OUT" --batch-size 256 --epochs 3 --max_steps "$MAX" --save_completed_steps "0,100,250,500" --log_every 25 --num_workers 8 --grad_checkpoint_views 1)
if [ -n "$RESUME" ]; then ARGS+=(--resume "$RESUME"); fi
python -m torch.distributed.run --nproc_per_node=4 --master_port=34951 train/train_said_cls_c1.py "${ARGS[@]}"
