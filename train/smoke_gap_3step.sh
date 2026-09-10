#!/bin/bash
# Phase 3.0A.1b base gap completion smoke: 4x A800, 3 steps, full ShareGPT4V + Full Data
# Gate. Reproduces the acceptance run exactly (bf16, seed 0, identifiable Said, residual
# feature source, global batch 1024). Only (I, C_S) reaches the model.
export SHARE4V_DATA_ROOT=/root/datasets/ShareGPT4V
export SHARE4V_JSON=share-captioner_coco_lcs_sam_1246k_1107.json
export SHARE4V_FULL_AUDIT=/root/SAID/outputs/data_audit/sharegpt4v_full_audit.json
export COCO_DATA_ROOT=/root/datasets/coco
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
cd /root/SAID-gap-completion || exit 1
echo "=== launching 3-step gap_completion smoke ==="
/root/miniconda3/bin/conda run --no-capture-output -n said-smartclip \
  torchrun --nproc_per_node=4 --master_port=25971 train/train_salu.py \
  --objective_mode gap_completion \
  --lambda_global 0 --lambda_unsaid 0 \
  --lambda_said 1.0 --lambda_gap_discover 1.0 --lambda_global_absorb 1.0 \
  --gap_anti_temperature 1.0 \
  --said_loss_mode identifiable --said_feature_source residual \
  --base_model B16 --batch_size 256 --max_steps 3 --lr_total_steps 3648 \
  --warmup_length 200 --backbone_lr 1e-6 --head_lr 1e-4 \
  --amp_dtype bf16 --seed 0 --num_workers 8 \
  --log_every 1 --save_every 0 --save_at 3 \
  --strict_manifest \
  --output_dir runs_salu/phase30a_1b_smoke3
echo "EXIT=$?"
