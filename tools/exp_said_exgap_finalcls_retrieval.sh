#!/bin/bash
# SAID-ExGAP v1.5 canonical retrieval gate.
#
#   Initial (== F0/F1 completed step 0), F0 step 500, F1 step 500
#   PRIMARY image representation : legacy_cls == native clip.encode_image (v1.5's own z_G path)
#   (patch_global is still computed and stored, as a diagnostic only)
# Datasets: COCO val2017 + ShareGPT4V-1K (first_sentence / fixed_sparse / full_dense).
#
#   bash tools/exp_said_exgap_finalcls_retrieval.sh
set -u
cd /root/SAID-gap-completion || exit 1
export SHARE4V_DATA_ROOT=/root/datasets/ShareGPT4V
export SHARE4V_JSON=share-captioner_coco_lcs_sam_1246k_1107.json
export SHARE4V_FULL_AUDIT=/root/SAID/outputs/data_audit/sharegpt4v_full_audit.json
export COCO_DATA_ROOT=/root/datasets/coco
OUT=runs_salu/said_exgap_finalcls
mkdir -p outputs/said_exgap

/root/miniconda3/envs/said-smartclip/bin/python tools/phase30a_fixed_cohort_eval.py \
  --checkpoints "initial:${OUT}/step500_F0_said_only/salu_exgap_step000000.pt,F0_500:${OUT}/step500_F0_said_only/salu_exgap_step000500.pt,F1_500:${OUT}/step500_F1_full_exgap/salu_exgap_step000500.pt" \
  --label said_exgap_finalcls_v15 \
  --gap_anti_temperature 1.0 \
  --canonical --canonical_only \
  --canonical_tags initial,F0_500,F1_500 \
  --canonical_names initial,F0_500,F1_500 \
  --image-representation legacy_cls \
  --global-pool mean \
  --coco \
  --image_batch_size 64 \
  --output outputs/said_exgap/finalcls_v15_canonical_retrieval.json
echo "RETRIEVAL_EXIT=$?"
