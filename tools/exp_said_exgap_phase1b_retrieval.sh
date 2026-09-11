#!/bin/bash
# Phase ExGAP-1B canonical retrieval gate.
#
#   Initial (== M0/M1 completed step 0), M0 step 500, M1 step 500
#   PRIMARY  image representation : patch_global (z_G = Pool(H), mean)
#   DIAGNOSTIC                    : legacy_cls (native CLIP encode_image)
# Datasets: COCO val2017 + ShareGPT4V-1K (first_sentence / fixed_sparse / full_dense).
#
#   bash tools/exp_said_exgap_phase1b_retrieval.sh
set -u
cd /root/SAID-gap-completion || exit 1
export SHARE4V_DATA_ROOT=/root/datasets/ShareGPT4V
export SHARE4V_JSON=share-captioner_coco_lcs_sam_1246k_1107.json
export SHARE4V_FULL_AUDIT=/root/SAID/outputs/data_audit/sharegpt4v_full_audit.json
export COCO_DATA_ROOT=/root/datasets/coco
OUT=runs_salu/said_exgap_phase1b
mkdir -p outputs/said_exgap

/root/miniconda3/envs/said-smartclip/bin/python tools/phase30a_fixed_cohort_eval.py \
  --checkpoints "initial:${OUT}/M0_said_only_masking/salu_exgap_step000000.pt,M0_500:${OUT}/M0_said_only_masking/salu_exgap_step000500.pt,M1_500:${OUT}/M1_full_exgap/salu_exgap_step000500.pt" \
  --label said_exgap_phase1b \
  --gap_anti_temperature 1.0 \
  --canonical --canonical_only \
  --canonical_tags initial,M0_500,M1_500 \
  --canonical_names initial,M0_500,M1_500 \
  --image-representation patch_global \
  --global-pool mean \
  --coco \
  --image_batch_size 64 \
  --output outputs/said_exgap/phase1b_canonical_retrieval.json
echo "RETRIEVAL_EXIT=$?"
