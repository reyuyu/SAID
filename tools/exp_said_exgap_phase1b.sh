#!/bin/bash
# Phase ExGAP-1B: 500-step matched causal control (M0 = Said + masking diagnostics only,
# M1 = full SAID-ExGAP). mean pooling, identical everything except --lambda-exgap.
#
#   bash tools/exp_said_exgap_phase1b.sh
#
# 4x A800, full ShareGPT4V (Full Data Gate), batch 256/GPU => global 1024, bf16, seed 0.
# Checkpoints at COMPLETED optimizer updates 0 / 100 / 500 (salu_exgap_step%06d.pt); gradient
# attribution (G_S, G_ExGAP, R_grad) on completed steps 20 / 100 / 500.
set -u
cd /root/SAID-gap-completion || exit 1
export SHARE4V_DATA_ROOT=/root/datasets/ShareGPT4V
export SHARE4V_JSON=share-captioner_coco_lcs_sam_1246k_1107.json
export SHARE4V_FULL_AUDIT=/root/SAID/outputs/data_audit/sharegpt4v_full_audit.json
export COCO_DATA_ROOT=/root/datasets/coco
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
OUT=runs_salu/said_exgap_phase1b
mkdir -p "$OUT"

run_arm () {
  local arm="$1"; local lambda_exgap="$2"
  # fresh rendezvous port per arm: a killed run can leave the old one in TIME_WAIT
  local port=$((27100 + RANDOM % 500))
  echo "=== ExGAP-1B arm=$arm lambda_exgap=$lambda_exgap port=$port ==="
  date -Is
  /root/miniconda3/bin/conda run --no-capture-output -n said-smartclip \
    torchrun --nproc_per_node=4 --master_port="$port" train/train_salu.py \
    --objective_mode said_exgap \
    --lambda_global 0 --lambda_unsaid 0 \
    --lambda_said 1.0 --lambda-exgap "$lambda_exgap" \
    --exgap-global-pool mean \
    --exgap-mask-threshold 0.6 \
    --exgap-temperature 0.05 \
    --exgap-collapse-every 20 \
    --grad-attribution-steps 20,100,500 \
    --save-completed-steps 0,100,500 \
    --said_loss_mode identifiable --said_feature_source residual \
    --base_model B16 --batch_size 256 --max_steps 500 --lr_total_steps 3648 \
    --warmup_length 200 --backbone_lr 1e-6 --head_lr 1e-4 \
    --amp_dtype bf16 --seed 0 --num_workers 8 \
    --log_every 10 --save_every 0 \
    --strict_manifest \
    --output_dir "$OUT/$arm"
  echo "ARM_EXIT arm=$arm lambda_exgap=$lambda_exgap exit=$?"
}

run_arm M0_said_only_masking 0.0
run_arm M1_full_exgap 1.0
echo "ALL_EXGAP_PHASE1B_DONE"
date -Is
