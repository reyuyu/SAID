#!/usr/bin/env bash
# Explicit actions only; this script never starts a formal run by default.
set -euo pipefail
cd /root/lk_projects/SAID
action=${1:?Usage: run.sh train500\|export500\|coco\|urban A2\|A3}
arm=${2:?Specify A2 or A3}
[[ "$arm" == A2 || "$arm" == A3 ]] || exit 2
nest_root=/root/lk_projects/SAID-nest-clip-v1
nest_python=/root/miniconda3/envs/said-repro/bin/python
nest_torchrun=/root/miniconda3/envs/said-repro/bin/torchrun
case "$action" in
  train500)
    OMP_NUM_THREADS=4 "$nest_torchrun" --standalone --nnodes=1 --nproc-per-node=4 --max-restarts=0 \
      -m train.train_nested_semantic_mask \
      --config "configs/nest_clip_${arm,,}.json" \
      --init-state "$nest_root/shared/step000000.pt" \
      --index-dir "$nest_root/data_index" \
      --image-root /root/lk_projects/SAID-assets/training/ShareGPT4V \
      --output-dir "$nest_root/formal/$arm" --run-type formal --max-updates 500
    ;;
  export500)
    "$nest_python" -m tools.nest_clip export \
      --checkpoint "$nest_root/formal/$arm/step000500.pt" --expect-updates 500 \
      --output "$nest_root/formal/$arm/student_step500.pt"
    ;;
  coco|urban)
    if [[ "$action" == coco ]]; then
      eval_root=/root/lk_projects/SAID-assets/evaluation/coco/val2017
    else
      eval_root=/root/lk_projects/SAID-assets/evaluation/Urban1k/Urban1k
    fi
    "$nest_python" -m tools.eval_nest_native \
      --checkpoint "$nest_root/formal/$arm/student_step500.pt" \
      --dataset "$action" --root "$eval_root" --device cuda:0 --batch-size 64 \
      --output "$nest_root/formal/$arm/${action}_native.json"
    ;;
  *) exit 2 ;;
esac
