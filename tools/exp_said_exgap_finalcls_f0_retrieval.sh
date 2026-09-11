#!/bin/bash
# SAID-ExGAP v1.5 F0 3-epoch checkpoints -> canonical retrieval (PRIMARY = native legacy_cls).
#
#   Initial / step500 / epoch1(1216) / epoch2(2432) / epoch3(3648)
# Same canonical evaluator for every checkpoint; COCO val2017 + ShareGPT4V-1K (3 variants).
# Only `legacy_cls` (= clip.encode_image) is used as the PRIMARY representation.
#
#   bash tools/exp_said_exgap_finalcls_f0_retrieval.sh
set -u
cd /root/SAID-gap-completion || exit 1
export SHARE4V_DATA_ROOT=/root/datasets/ShareGPT4V
export SHARE4V_JSON=share-captioner_coco_lcs_sam_1246k_1107.json
export SHARE4V_FULL_AUDIT=/root/SAID/outputs/data_audit/sharegpt4v_full_audit.json
export COCO_DATA_ROOT=/root/datasets/coco
OUT=runs_salu/said_exgap_finalcls_3ep/F0_said_only_3ep
mkdir -p outputs/said_exgap

CKPTS="${OUT}/salu_exgap_step000000.pt ${OUT}/salu_exgap_step000500.pt ${OUT}/salu_exgap_step001216.pt ${OUT}/salu_exgap_step002432.pt ${OUT}/salu_exgap_step003648.pt"
for path in $CKPTS; do
  if [ ! -f "$path" ]; then
    echo "MISSING checkpoint $path -- refusing to evaluate an incomplete trajectory"
    exit 1
  fi
done

SPEC="initial:${OUT}/salu_exgap_step000000.pt,step500:${OUT}/salu_exgap_step000500.pt,epoch1:${OUT}/salu_exgap_step001216.pt,epoch2:${OUT}/salu_exgap_step002432.pt,epoch3:${OUT}/salu_exgap_step003648.pt"

/root/miniconda3/envs/said-smartclip/bin/python tools/phase30a_fixed_cohort_eval.py \
  --checkpoints "$SPEC" \
  --label said_exgap_finalcls_f0_3epoch \
  --gap_anti_temperature 1.0 \
  --canonical --canonical_only \
  --canonical_tags initial,step500,epoch1,epoch2,epoch3 \
  --canonical_names initial,step500,epoch1,epoch2,epoch3 \
  --image-representation legacy_cls \
  --global-pool mean \
  --coco \
  --image_batch_size 64 \
  --output outputs/said_exgap/f0_3epoch_canonical_retrieval.json
echo "RETRIEVAL_EXIT=$?"
