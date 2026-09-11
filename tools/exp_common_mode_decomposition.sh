#!/bin/bash
# Phase 3.0A.1e: common-mode semantic decomposition on the frozen USR cohort.
# Read-only over existing checkpoints: optimizer_steps = 0, no training, no new loss.
# Every scorer is built from (I, C_S) before the candidate pool is touched, and all eight
# scorers share one precomputed query feature set.
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

run_one () {
  local label="$1"; local tau="$2"; local ckpt="$3"
  /root/miniconda3/bin/conda run --no-capture-output -n said-smartclip \
    python "$W/tools/phase30a_fixed_cohort_eval.py" \
    --label "$label" --gap_anti_temperature "$tau" \
    --usr_manifest "$M" --source_manifest "$S" \
    --image_batch_size 64 --decomposition \
    --bootstrap_replicates 10000 --bootstrap_seed 20260911 \
    --checkpoints "$ckpt" \
    --output "$OUT/$label.json"
  echo "DECOMP_DONE $label exit=$?"
}

# the initial checkpoint is shared by every arm (identical initial_state_sha256)
run_one init_initial 1.0 \
  "initial:$R/phase30a_1c_tau1/salu_initial.pt"
run_one A_said100 1.0 \
  "A100:$R/phase30a_1d_A_said_only/salu_said_only_step000100.pt"
run_one B_discover100 1.0 \
  "B100:$R/phase30a_1d_B_said_discover/salu_said_only_step000100.pt"
run_one C_full100 1.0 \
  "C100:$R/phase30a_1c_tau1/salu_said_only_step000100.pt"
run_one D_tau05_100 0.5 \
  "D100:$R/phase30a_1c_tau05/salu_said_only_step000100.pt"
echo "ALL_DECOMP_DONE"
