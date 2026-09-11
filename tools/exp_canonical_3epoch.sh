#!/bin/bash
# Phase 3.0A.2 (canonical first): canonical CLIP CLS image-text retrieval.
# Stage 1 = the priority set: initial, A step3648, C step3648.
# Frozen protocol: eval/validation_protocol.py (ShareGPT4V-1K, 3 variants) and
# eval/retrieval/coco_retrieval.py (COCO val2017, 5-caption). Nothing in either is modified.
# Read-only: no optimizer step, no training.
set -u
cd /root/SAID || exit 1
export SHARE4V_DATA_ROOT=/root/datasets/ShareGPT4V
export SHARE4V_JSON=share-captioner_coco_lcs_sam_1246k_1107.json
export COCO_DATA_ROOT=/root/datasets/coco
export CUDA_VISIBLE_DEVICES=0
W=/root/SAID-gap-completion
R=$W/runs_salu
M=outputs/validation/sharegpt4v1k_usr_manifest.json
S=outputs/validation/sharegpt4v1k_manifest.json
SG=outputs/validation/sharegpt4v1k_manifest.json
OUT=$W/outputs/phase30a_full3epoch
mkdir -p "$OUT"

A=$R/phase30a_2_A_said_only
C=$R/phase30a_2_C_full_base

STAGE="${1:-1}"
if [ "$STAGE" = "1" ]; then
  SPECS="initial:$A/salu_initial.pt"
  SPECS="$SPECS,Aend:$A/salu_said_only_step003648.pt"
  SPECS="$SPECS,Cend:$C/salu_said_only_step003648.pt"
  TAGS="initial,Aend,Cend"
  RESULT="$OUT/canonical_stage1_priority.json"
else
  SPECS="initial:$A/salu_initial.pt"
  SPECS="$SPECS,A1216:$A/salu_said_only_step001216.pt"
  SPECS="$SPECS,C1216:$C/salu_said_only_step001216.pt"
  SPECS="$SPECS,A2432:$A/salu_said_only_step002432.pt"
  SPECS="$SPECS,C2432:$C/salu_said_only_step002432.pt"
  TAGS="initial,A1216,C1216,A2432,C2432"
  RESULT="$OUT/canonical_stage2_epochs.json"
fi

/root/miniconda3/bin/conda run --no-capture-output -n said-smartclip \
  python "$W/tools/phase30a_fixed_cohort_eval.py" \
  --label "canonical_stage${STAGE}" --gap_anti_temperature 1.0 \
  --usr_manifest "$M" --source_manifest "$S" \
  --sharegpt4v_manifest "$SG" \
  --data_root "$SHARE4V_DATA_ROOT" --image_root "$SHARE4V_DATA_ROOT" \
  --image_batch_size 64 --canonical --canonical_only --coco \
  --canonical_tags "$TAGS" \
  --checkpoints "$SPECS" \
  --output "$RESULT"
echo "CANONICAL_STAGE${STAGE}_DONE exit=$?"
date -Is
