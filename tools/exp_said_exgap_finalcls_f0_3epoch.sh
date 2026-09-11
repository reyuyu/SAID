#!/bin/bash
# SAID-ExGAP v1.5 -- F0 (Said-only) FULL 3-EPOCH diagnostic run.
#
#   bash tools/exp_said_exgap_finalcls_f0_3epoch.sh
#
# objective_mode = said_exgap_finalcls, lambda_said = 1.0, lambda_exgap = 0.0
# (F1 is NOT started). Exactly the same initialisation / data stream / hyper-parameters as the
# matched F0/F1 smoke: ViT-B/16, LongCLIP 248, full ShareGPT4V, 4x A800, batch 256/GPU
# (global 1024), bf16, seed 0, backbone lr 1e-6, head lr 1e-4, warmup 200, 3 epochs
# (steps_per_epoch 1216 => 3648 completed steps).
#
# Checkpoints at completed steps 0 / 500 / 1216 / 2432 / 3648.
set -u
cd /root/SAID-gap-completion || exit 1
export SHARE4V_DATA_ROOT=/root/datasets/ShareGPT4V
export SHARE4V_JSON=share-captioner_coco_lcs_sam_1246k_1107.json
export SHARE4V_FULL_AUDIT=/root/SAID/outputs/data_audit/sharegpt4v_full_audit.json
export COCO_DATA_ROOT=/root/datasets/coco
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
OUT=runs_salu/said_exgap_finalcls_3ep
mkdir -p "$OUT"

PORT=$((30100 + RANDOM % 500))
echo "=== F0 Said-only full 3 epochs (lambda_exgap=0.0) port=$PORT ==="
date -Is
/root/miniconda3/bin/conda run --no-capture-output -n said-smartclip \
  torchrun --nproc_per_node=4 --master_port="$PORT" train/train_salu.py \
  --objective_mode said_exgap_finalcls \
  --lambda_global 0 --lambda_unsaid 0 \
  --lambda_said 1.0 --lambda-exgap 0.0 \
  --finalcls-pair-chunk-size 32 \
  --exgap-temperature 0.05 \
  --exgap-collapse-every 50 \
  --grad-attribution-steps 20,100,500 \
  --save-completed-steps 0,500,1216,2432,3648 \
  --said_loss_mode identifiable --said_feature_source residual \
  --base_model B16 --batch_size 256 --epochs 3 --lr_total_steps 3648 \
  --warmup_length 200 --backbone_lr 1e-6 --head_lr 1e-4 \
  --amp_dtype bf16 --seed 0 --num_workers 8 \
  --log_every 50 --save_every 0 \
  --strict_manifest \
  --output_dir "$OUT/F0_said_only_3ep"
echo "ARM_EXIT F0_said_only_3ep exit=$?"
echo "ALL_F0_3EP_DONE"
date -Is
