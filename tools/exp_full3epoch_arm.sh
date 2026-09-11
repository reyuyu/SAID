#!/bin/bash
# Phase 3.0A.2: one matched full 3-epoch arm.
# Usage: bash exp_full3epoch_arm.sh <lambda_gap_discover> <lambda_global_absorb> <output_dir>
# Both arms start from the same frozen initial state (seed 0, no resume), so the initial
# state, the sampler order and the C_S prefix stream are matched by construction.
GAP="$1"
ABSORB="$2"
OUTDIR="$3"
export SHARE4V_DATA_ROOT=/root/datasets/ShareGPT4V
export SHARE4V_JSON=share-captioner_coco_lcs_sam_1246k_1107.json
export SHARE4V_FULL_AUDIT=/root/SAID/outputs/data_audit/sharegpt4v_full_audit.json
export COCO_DATA_ROOT=/root/datasets/coco
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
cd /root/SAID-gap-completion || exit 1
echo "=== ARM lambda_gap_discover=$GAP lambda_global_absorb=$ABSORB output=$OUTDIR ==="
date -Is
/root/miniconda3/bin/conda run --no-capture-output -n said-smartclip \
  torchrun --nproc_per_node=4 --master_port=25981 train/train_salu.py \
  --objective_mode gap_completion \
  --lambda_global 0 --lambda_unsaid 0 \
  --lambda_said 1.0 --lambda_gap_discover "$GAP" --lambda_global_absorb "$ABSORB" \
  --gap_anti_temperature 1.0 \
  --said_loss_mode identifiable --said_feature_source residual \
  --base_model B16 --batch_size 256 --epochs 3 --max_steps 3648 --lr_total_steps 3648 \
  --warmup_length 200 --backbone_lr 1e-6 --head_lr 1e-4 \
  --amp_dtype bf16 --seed 0 --num_workers 8 \
  --log_every 50 --save_every 0 \
  --save_initial --save_at 100,200,500,1000,1216,1800,2432,3000,3648 \
  --grad_contribution_steps 0,100,200,500,1216,2432,3648 \
  --strict_manifest \
  --output_dir "$OUTDIR"
echo "EXIT=$?"
date -Is
