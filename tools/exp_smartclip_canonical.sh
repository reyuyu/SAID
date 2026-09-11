#!/bin/bash
# SmartCLIP reproduction: canonical evaluation with the SAME evaluator used for A / C.
# initial = the fresh pretrained ViT-B/16 (identical to the A/C "initial" state), then the
# three epoch checkpoints of the reproduction. Read-only, no optimizer step.
set -u
cd /root/SAID || exit 1
export SHARE4V_DATA_ROOT=/root/datasets/ShareGPT4V
export SHARE4V_JSON=share-captioner_coco_lcs_sam_1246k_1107.json
export COCO_DATA_ROOT=/root/datasets/coco
export CUDA_VISIBLE_DEVICES=0
W=/root/SAID-gap-completion
R=$W/runs_smartclip/smartclip_B16_full3ep_repro_20260911
A=$W/runs_salu/phase30a_2_A_said_only
M=outputs/validation/sharegpt4v1k_usr_manifest.json
S=outputs/validation/sharegpt4v1k_manifest.json
SG=outputs/validation/sharegpt4v1k_manifest.json
OUT=$W/outputs/smartclip_reproduction
mkdir -p "$OUT"

SPECS="initial:$A/salu_initial.pt"
SPECS="$SPECS,SCepoch1:$R/smartclip_epoch00.pt"
SPECS="$SPECS,SCepoch2:$R/smartclip_epoch01.pt"
SPECS="$SPECS,SCepoch3:$R/smartclip_epoch02.pt"

/root/miniconda3/bin/conda run --no-capture-output -n said-smartclip \
  python "$W/tools/phase30a_fixed_cohort_eval.py" \
  --label smartclip --gap_anti_temperature 1.0 \
  --usr_manifest "$M" --source_manifest "$S" \
  --sharegpt4v_manifest "$SG" \
  --data_root "$SHARE4V_DATA_ROOT" --image_root "$SHARE4V_DATA_ROOT" \
  --image_batch_size 64 --canonical --canonical_only --coco \
  --canonical_tags "initial,SCepoch1,SCepoch2,SCepoch3" \
  --canonical_names "SmartCLIP_initial,SmartCLIP_epoch1,SmartCLIP_epoch2,SmartCLIP_epoch3" \
  --checkpoints "$SPECS" \
  --output "$OUT/smartclip_3epoch_canonical_raw.json"
echo "SMARTCLIP_CANONICAL_DONE exit=$?"
date -Is
