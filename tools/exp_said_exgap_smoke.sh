#!/bin/bash
# SAID-ExGAP v1 smoke runs: 20 and 100 steps, both global pooling modes.
# Full ShareGPT4V + Full Data Gate, 4x A800, batch 256/GPU (global 1024), bf16, seed 0.
# No global-text CLIP loss, no Unsaid branch, no reconstruction.
set -u
cd /root/SAID-gap-completion || exit 1
export SHARE4V_DATA_ROOT=/root/datasets/ShareGPT4V
export SHARE4V_JSON=share-captioner_coco_lcs_sam_1246k_1107.json
export SHARE4V_FULL_AUDIT=/root/SAID/outputs/data_audit/sharegpt4v_full_audit.json
export COCO_DATA_ROOT=/root/datasets/coco
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
OUT=runs_salu/said_exgap_smoke
mkdir -p "$OUT"

run_arm () {
  local pool="$1"; local steps="$2"
  # fresh port per arm: a killed run can leave the rendezvous port in TIME_WAIT
  local port=$((26100 + RANDOM % 500))
  echo "=== said_exgap pool=$pool steps=$steps port=$port ==="
  date -Is
  /root/miniconda3/bin/conda run --no-capture-output -n said-smartclip \
    torchrun --nproc_per_node=4 --master_port="$port" train/train_salu.py \
    --objective_mode said_exgap \
    --lambda_global 0 --lambda_unsaid 0 \
    --lambda_said 1.0 --lambda-exgap 1.0 \
    --exgap-global-pool "$pool" \
    --exgap-mask-threshold 0.6 \
    --exgap-temperature 0.05 \
    --exgap-collapse-every 20 \
    --said_loss_mode identifiable --said_feature_source residual \
    --base_model B16 --batch_size 256 --max_steps "$steps" --lr_total_steps 3648 \
    --warmup_length 200 --backbone_lr 1e-6 --head_lr 1e-4 \
    --amp_dtype bf16 --seed 0 --num_workers 8 \
    --log_every 10 --save_every 0 \
    --strict_manifest \
    --output_dir "$OUT/${pool}_${steps}"
  echo "ARM_EXIT pool=$pool steps=$steps exit=$?"
}
run_arm mean 20
run_arm mean 100
run_arm mean 500
run_arm attention 20
run_arm attention 100
run_arm attention 500
echo "ALL_EXGAP_SMOKE_DONE"
date -Is
