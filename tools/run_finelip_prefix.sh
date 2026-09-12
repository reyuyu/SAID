#!/bin/bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
OUT=${1:?new absolute output directory required}
if [ -e "$OUT" ]; then echo 'Refusing existing run directory'; exit 2; fi
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$busy" -eq 0 ] || { echo 'NOT RUN: GPUs busy'; exit 3; }
[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 4 ] || exit 4
export SHARE4V_DATA_ROOT=/root/datasets/ShareGPT4V
export SHARE4V_JSON=share-captioner_coco_lcs_sam_1246k_1107.json
export SHARE4V_FULL_AUDIT=/root/SAID/outputs/data_audit/sharegpt4v_full_audit.json
export COCO_DATA_ROOT=/root/datasets/coco
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo OMP_NUM_THREADS=1
export PATH=/root/miniconda3/envs/said-smartclip/bin:$PATH
python -m torch.distributed.run --nproc_per_node=4 --master_port=34941 train/train_finelip_prefix.py \
 --init_state /root/SAID-gap-completion/runs_salu/said_cls_cvssl/shared_init/cvssl_initial.pt \
 --output_dir "$OUT" --batch_size 32 --accumulation 4 --workers 8 \
 --schedule_epochs 6 --run_epochs 2 --seed 0
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$busy" -eq 0 ] || { echo 'Evaluation NOT RUN: GPUs busy after training'; exit 5; }
CUDA_VISIBLE_DEVICES=0 python tools/eval_finelip_prefix.py --run "$OUT"
echo FP0_COMPLETE_STOP
