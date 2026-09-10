#!/bin/bash
# Phase 3.0A.1d: fixed-cohort evaluation of every attribution / temperature checkpoint.
# Same frozen cohort (sharegpt4v1k-usr-v1, Q=868, candidate pool 868) for every checkpoint.
set -u
cd /root/SAID || exit 1
export SHARE4V_DATA_ROOT=/root/datasets/ShareGPT4V
export SHARE4V_JSON=share-captioner_coco_lcs_sam_1246k_1107.json
export COCO_DATA_ROOT=/root/datasets/coco
export CUDA_VISIBLE_DEVICES=0
W=/root/SAID-gap-completion
R=$W/runs_salu
# relative paths, exactly as the frozen manifest records them, so
# load_or_create_usr_manifest verifies the cohort instead of rebuilding it
M=outputs/validation/sharegpt4v1k_usr_manifest.json
S=outputs/validation/sharegpt4v1k_manifest.json
OUT=$W/outputs/phase30a_fixed_semantic_eval
mkdir -p "$OUT"

run_arm () {
  local label="$1"; local tau="$2"; local dir="$3"
  /root/miniconda3/bin/conda run --no-capture-output -n said-smartclip \
    python "$W/tools/phase30a_fixed_cohort_eval.py" \
    --label "$label" --gap_anti_temperature "$tau" \
    --usr_manifest "$M" --source_manifest "$S" \
    --image_batch_size 64 \
    --checkpoints "initial:$dir/salu_initial.pt,step20:$dir/salu_said_only_step000020.pt,step50:$dir/salu_said_only_step000050.pt,step100:$dir/salu_said_only_step000100.pt" \
    --output "$OUT/$label.json"
  echo "ARM_DONE $label exit=$?"
}

run_arm C_tau1.0_S+D+A 1.0 "$R/phase30a_1c_tau1"
run_arm A_tau1.0_S_only 1.0 "$R/phase30a_1d_A_said_only"
run_arm B_tau1.0_S+D 1.0 "$R/phase30a_1d_B_said_discover"
run_arm D_tau0.5_S+D+A 0.5 "$R/phase30a_1c_tau05"
echo "ALL_EVAL_DONE"
