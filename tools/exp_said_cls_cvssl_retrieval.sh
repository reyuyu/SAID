#!/bin/bash
# Canonical CLS retrieval for the SAID-CLS-CVSSL S0 / C0 screening.
#
#   bash tools/exp_said_cls_cvssl_retrieval.sh S0
#   bash tools/exp_said_cls_cvssl_retrieval.sh C0
#
# Protocol is the frozen historical one (tools/exp_smartclip_canonical.sh): the same evaluator, the
# same manifests, the same COCO split, the same ShareGPT4V fixed-1K variants. The PRIMARY
# representation is normalize(clip.encode_image(image)) / normalize(clip.encode_text(text)) -- no
# masked Said/U features, no patch mean, no pair-conditioned reranking.
#
# Checkpoints per arm: shared_init (Initial), step100, step500.
set -u
cd /root/SAID || exit 1

ARM=${1:-}
case "$ARM" in
  S0) DIR=runs_salu/said_cls_cvssl/ddpfix_step500_S0_smartclip
      PREFIX=cvssl_S0_smartclip ;;
  C0) DIR=runs_salu/said_cls_cvssl/ddpfix_step500_C0_complement_vssl
      PREFIX=cvssl_C0_complement_vssl ;;
  *) echo "usage: $0 S0|C0"; exit 1 ;;
esac

export SHARE4V_DATA_ROOT=/root/datasets/ShareGPT4V
export SHARE4V_JSON=share-captioner_coco_lcs_sam_1246k_1107.json
export COCO_DATA_ROOT=/root/datasets/coco
export CUDA_VISIBLE_DEVICES=0

W=/root/SAID-gap-completion
INIT=$W/runs_salu/said_cls_cvssl/shared_init/cvssl_initial.pt
S100=$W/$DIR/${PREFIX}_step000100.pt
S500=$W/$DIR/${PREFIX}_step000500.pt

for f in "$INIT" "$S100" "$S500"; do
  if [ ! -f "$f" ]; then echo "MISSING $f"; exit 1; fi
done

M=outputs/validation/sharegpt4v1k_usr_manifest.json
S=outputs/validation/sharegpt4v1k_manifest.json
SG=outputs/validation/sharegpt4v1k_manifest.json
OUT=$W/outputs/cvssl_screening
mkdir -p "$OUT"

SPECS="initial:$INIT,step100:$S100,step500:$S500"
echo "=== canonical retrieval arm=$ARM ==="
echo "    checkpoints: $SPECS"
date -Is

/root/miniconda3/bin/conda run --no-capture-output -n said-smartclip \
  python "$W/tools/phase30a_fixed_cohort_eval.py" \
  --label "cvssl_$ARM" --gap_anti_temperature 1.0 \
  --usr_manifest "$M" --source_manifest "$S" \
  --sharegpt4v_manifest "$SG" \
  --data_root "$SHARE4V_DATA_ROOT" --image_root "$SHARE4V_DATA_ROOT" \
  --image_batch_size 64 --canonical --canonical_only --coco \
  --canonical_tags "initial,step100,step500" \
  --canonical_names "Initial,${ARM}_step100,${ARM}_step500" \
  --checkpoints "$SPECS" \
  --output "$OUT/${ARM}_canonical.json"
status=$?
echo "CANONICAL_EXIT arm=$ARM exit=$status"
date -Is
exit $status
