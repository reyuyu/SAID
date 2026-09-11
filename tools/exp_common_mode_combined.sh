#!/bin/bash
# Phase 3.0A.1e: one combined decomposition run over all five checkpoints.
# The paired bootstrap is inherently per-run (it needs both endpoints' per-query outcomes at
# the same time), so the comparisons are computed here in a single process.
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
OUT=$W/outputs/phase30a_common_mode_semantics
mkdir -p "$OUT"

SPECTS="initial:$R/phase30a_1c_tau1/salu_initial.pt"
SPECTS="$SPECTS,A100:$R/phase30a_1d_A_said_only/salu_said_only_step000100.pt"
SPECTS="$SPECTS,B100:$R/phase30a_1d_B_said_discover/salu_said_only_step000100.pt"
SPECTS="$SPECTS,C100:$R/phase30a_1c_tau1/salu_said_only_step000100.pt"
SPECTS="$SPECTS,D100:$R/phase30a_1c_tau05/salu_said_only_step000100.pt"

/root/miniconda3/bin/conda run --no-capture-output -n said-smartclip \
  python "$W/tools/phase30a_fixed_cohort_eval.py" \
  --label combined --gap_anti_temperature 1.0 \
  --usr_manifest "$M" --source_manifest "$S" \
  --image_batch_size 64 --decomposition \
  --bootstrap_replicates 10000 --bootstrap_seed 20260911 \
  --checkpoints "$SPECTS" \
  --output "$OUT/combined_all_checkpoints.json"
echo "COMBINED_DONE exit=$?"
