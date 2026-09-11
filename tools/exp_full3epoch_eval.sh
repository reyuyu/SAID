#!/bin/bash
# Phase 3.0A.2: fixed-cohort evaluation of both full 3-epoch arms.
# One combined run per arm pair so the paired bootstrap can see both endpoints of every
# comparison for each matched step. Read-only: optimizer_steps = 0.
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
OUT=$W/outputs/phase30a_full3epoch
mkdir -p "$OUT"

A=$R/phase30a_2_A_said_only
C=$R/phase30a_2_C_full_base

# matched checkpoints: initial, 100, 200, 500, 1216 (epoch 1), 2432 (epoch 2),
# and "last" (step 3647, i.e. the end of epoch 3)
SPECS="initial:$A/salu_initial.pt"
SPECS="$SPECS,A100:$A/salu_said_only_step000100.pt,C100:$C/salu_said_only_step000100.pt"
SPECS="$SPECS,A200:$A/salu_said_only_step000200.pt,C200:$C/salu_said_only_step000200.pt"
SPECS="$SPECS,A500:$A/salu_said_only_step000500.pt,C500:$C/salu_said_only_step000500.pt"
SPECS="$SPECS,A1216:$A/salu_said_only_step001216.pt,C1216:$C/salu_said_only_step001216.pt"
SPECS="$SPECS,A2432:$A/salu_said_only_step002432.pt,C2432:$C/salu_said_only_step002432.pt"
SPECS="$SPECS,A3000:$A/salu_said_only_step003000.pt,C3000:$C/salu_said_only_step003000.pt"
SPECS="$SPECS,Aend:$A/salu_said_only_last.pt,Cend:$C/salu_said_only_last.pt"

/root/miniconda3/bin/conda run --no-capture-output -n said-smartclip \
  python "$W/tools/phase30a_fixed_cohort_eval.py" \
  --label full3epoch --gap_anti_temperature 1.0 \
  --usr_manifest "$M" --source_manifest "$S" \
  --image_batch_size 64 --decomposition \
  --bootstrap_replicates 10000 --bootstrap_seed 20260911 \
  --checkpoints "$SPECS" \
  --output "$OUT/full3epoch_fixed_cohort_raw.json"
echo "FULL3EPOCH_EVAL_DONE exit=$?"
date -Is
