#!/usr/bin/env bash
set -euo pipefail
cd /root/SAID-c1-micro-ablation-v01
export PATH=/root/miniconda3/envs/said-smartclip/bin:$PATH
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0
export COCO_DATA_ROOT=/root/datasets/coco
CK=$PWD/runs_salu/c1_micro_B500/c1_C1_text_conditional_reconstruction_step000500.pt
OUT=$PWD/runs_salu/c1_micro_B500/evaluation
mkdir -p "$OUT"
python -c 'import sys,torch; p=torch.load(sys.argv[1],map_location="cpu",weights_only=False); assert p["completed_steps"]==500 and p["config"]["variant"]=="C1-VWarm" and p["next_batch_index"]==500; print("EXACT_B500_CHECK_PASS")' "$CK"
if [ ! -e "$OUT/B500_coco.json" ]; then
  python tools/phase30a_fixed_cohort_eval.py --label c1_vwarm_final --gap_anti_temperature 1.0 --sharegpt4v_manifest '' --image_batch_size 64 --canonical --canonical_only --coco --canonical_tags 500 --canonical_names C1_VWarm@500 --checkpoints 500:"$CK" --output "$OUT/B500_coco.json"
fi
if [ ! -e "$OUT/B500_urban1k.json" ]; then
  python /root/SAID-gap-completion/tools/eval_urban1k_cls.py --checkpoint "$CK" --label C1_VWarm@500 --expect-steps 500 --base_model ViT-B/16 --device cuda --batch_size 64 --urban_root /root/datasets/Urban1k/Urban1k --out "$OUT/B500_urban1k.json"
fi
echo B500_EVAL_COMPLETE_STOP
