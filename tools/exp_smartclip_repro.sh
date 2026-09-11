#!/bin/bash
# Phase 3.0A baseline validation: independent reproduction of the ORIGINAL SmartCLIP
# training path (train/train.py, class CLIP_Clean_Train), 3 epochs on full ShareGPT4V.
#
# Official-faithful configuration; nothing about the method is changed:
#   loss = lambda_sparse * loss_sparsity + lambda_align * (loss_sidm + loss_dism)
#   base_model B16 (ViT-B/16), batch 256/GPU x 4 GPU -> accumulation 1 -> global batch 1024
#   lr 1e-6, mask_lr 1e-3, weight_decay 1e-2, lambda_sparse 2, lambda_align 10
#   warmup_length 200, epochs 3, soft_mask 0 (the repo default)
# The only additions are the opt-in --output_dir and --seed flags plus a read-only
# initial-state digest / data-scale record. No SALU, Gap or Unsaid loss is involved.
set -u
cd /root/SAID-gap-completion || exit 1
export SHARE4V_DATA_ROOT=/root/datasets/ShareGPT4V
export SHARE4V_JSON=share-captioner_coco_lcs_sam_1246k_1107.json
export SHARE4V_FULL_AUDIT=/root/SAID/outputs/data_audit/sharegpt4v_full_audit.json
export COCO_DATA_ROOT=/root/datasets/coco
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

OUT=runs_smartclip/smartclip_B16_full3ep_repro_20260911_initfrozen
mkdir -p "$OUT"
# Frozen reference initial state (the weights the Comparison-CLIP "Initial" row is measured on).
# train.py builds its model from an unseeded RNG, so without this the reproduction would start
# from DIFFERENT pretrained weights and could not be compared with the Said-only / Full-Base
# arms. --init_state loads exactly the CLIP part of this state dict before training starts.
INIT_STATE=/root/SAID-gap-completion/runs_salu/phase30a_2_A_said_only/salu_initial.pt
echo "=== SmartCLIP reproduction ==="
date -Is
echo "init_state: $INIT_STATE"
echo "dataset json: $SHARE4V_DATA_ROOT/$SHARE4V_JSON"
python3 - <<'PY'
import hashlib
import os

path = os.path.join(os.environ['SHARE4V_DATA_ROOT'], os.environ['SHARE4V_JSON'])
digest = hashlib.sha256()
with open(path, 'rb') as handle:
    for block in iter(lambda: handle.read(1 << 22), b''):
        digest.update(block)
print('DATASET_JSON_SHA256', digest.hexdigest())
print('DATASET_JSON_SIZE_BYTES', os.path.getsize(path))
PY

/root/miniconda3/bin/conda run --no-capture-output -n said-smartclip \
  torchrun --nproc_per_node=4 --master_port=25991 train/train.py \
  --base_model B16 \
  --batch-size 256 \
  --epochs 3 \
  --lr 1e-6 --mask_lr 1e-3 --weight_decay 1e-2 \
  --lambda_sparse 2 --lambda_align 10 \
  --soft_mask 0 \
  --warmup_length 200 \
  --seed 0 \
  --init_state "$INIT_STATE" \
  --output_dir "$OUT"
echo "SMARTCLIP_EXIT=$?"
date -Is
