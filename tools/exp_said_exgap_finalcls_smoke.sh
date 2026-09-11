#!/bin/bash
# SAID-ExGAP v1.5 (said_exgap_finalcls): staged smoke.
#
#   bash tools/exp_said_exgap_finalcls_smoke.sh <steps> [tag]
#
#   F0 = L_S only            (lambda_exgap 0)  -- matched control, full geometry still computed
#   F1 = L_S + L_ExGAP       (lambda_exgap 1)
#
# 4x A800, full ShareGPT4V (Full Data Gate), batch 256/GPU => global 1024, bf16, seed 0.
# Checkpoints at COMPLETED steps 0 and <steps>; gradient attribution at min(<steps>, 20)/100/500.
set -u
STEPS=${1:-20}
TAG=${2:-step$STEPS}
cd /root/SAID-gap-completion || exit 1
export SHARE4V_DATA_ROOT=/root/datasets/ShareGPT4V
export SHARE4V_JSON=share-captioner_coco_lcs_sam_1246k_1107.json
export SHARE4V_FULL_AUDIT=/root/SAID/outputs/data_audit/sharegpt4v_full_audit.json
export COCO_DATA_ROOT=/root/datasets/coco
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
OUT=runs_salu/said_exgap_finalcls
mkdir -p "$OUT"

run_arm () {
  local arm="$1"; local lambda_exgap="$2"
  local port=$((29100 + RANDOM % 500))
  echo "=== v1.5 arm=$arm lambda_exgap=$lambda_exgap steps=$STEPS port=$port ==="
  date -Is
  /root/miniconda3/bin/conda run --no-capture-output -n said-smartclip \
    torchrun --nproc_per_node=4 --master_port="$port" train/train_salu.py \
    --objective_mode said_exgap_finalcls \
    --lambda_global 0 --lambda_unsaid 0 \
    --lambda_said 1.0 --lambda-exgap "$lambda_exgap" \
    --finalcls-pair-chunk-size 32 \
    --exgap-temperature 0.05 \
    --exgap-collapse-every 5 \
    --grad-attribution-steps "20,100,500" \
    --save-completed-steps "0,$STEPS" \
    --said_loss_mode identifiable --said_feature_source residual \
    --base_model B16 --batch_size 256 --max_steps "$STEPS" --lr_total_steps 3648 \
    --warmup_length 200 --backbone_lr 1e-6 --head_lr 1e-4 \
    --amp_dtype bf16 --seed 0 --num_workers 8 \
    --log_every 5 --save_every 0 \
    --strict_manifest \
    --output_dir "$OUT/${TAG}_${arm}"
  echo "ARM_EXIT tag=$TAG arm=$arm lambda_exgap=$lambda_exgap exit=$?"
}

run_arm F0_said_only 0.0
run_arm F1_full_exgap 1.0
echo "ALL_EXGAP_FINALCLS_SMOKE_DONE tag=$TAG"
date -Is
