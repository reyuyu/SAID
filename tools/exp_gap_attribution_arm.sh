#!/bin/bash
# Phase 3.0A.1d loss-attribution arm.
# Usage: bash exp_gap_attribution_arm.sh <lambda_gap_discover> <lambda_global_absorb> <output_dir>
# Everything except the two weights and the output directory is identical to the existing
# tau=1.0 full arm (C), so A / B / C share one initial state and one data stream.
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
echo "=== arm lambda_gap_discover=$GAP lambda_global_absorb=$ABSORB output=$OUTDIR ==="
/root/miniconda3/bin/conda run --no-capture-output -n said-smartclip \
  torchrun --nproc_per_node=4 --master_port=25973 train/train_salu.py \
  --objective_mode gap_completion \
  --lambda_global 0 --lambda_unsaid 0 \
  --lambda_said 1.0 --lambda_gap_discover "$GAP" --lambda_global_absorb "$ABSORB" \
  --gap_anti_temperature 1.0 \
  --said_loss_mode identifiable --said_feature_source residual \
  --base_model B16 --batch_size 256 --max_steps 100 --lr_total_steps 3648 \
  --warmup_length 200 --backbone_lr 1e-6 --head_lr 1e-4 \
  --amp_dtype bf16 --seed 0 --num_workers 8 \
  --log_every 10 --save_every 0 --save_initial \
  --save_at 20,50,100 \
  --grad_contribution_steps 0,20,50,100 \
  --strict_manifest \
  --output_dir "$OUTDIR"
echo "EXIT=$?"
