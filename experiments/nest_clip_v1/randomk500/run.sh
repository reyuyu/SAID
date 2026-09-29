#!/usr/bin/env bash
set -euo pipefail
cd /root/lk_projects/SAID
action=${1:?Usage: run.sh smoke\|train500\|export500\|verify-export\|coco\|urban}
nest_root=/root/lk_projects/SAID-nest-clip-v1
nest_run="$nest_root/randomk500/A3-RandomK"
nest_python=/root/miniconda3/envs/said-repro/bin/python
case "$action" in
  smoke|train500)
    if [[ "$action" == smoke ]]; then
      run_type=smoke; updates=5; destination="$nest_root/randomk500/smoke/A3-RandomK"
    else
      run_type=formal; updates=500; destination="$nest_run"
    fi
    OMP_NUM_THREADS=4 /root/miniconda3/envs/said-repro/bin/torchrun \
      --standalone --nnodes=1 --nproc-per-node=4 --max-restarts=0 \
      -m train.train_nested_semantic_mask --config configs/nest_clip_a3_randomk.json \
      --init-state "$nest_root/shared/step000000.pt" --index-dir "$nest_root/data_index" \
      --image-root /root/lk_projects/SAID-assets/training/ShareGPT4V \
      --output-dir "$destination" --run-type "$run_type" --max-updates "$updates"
    ;;
  export500)
    "$nest_python" -m tools.nest_clip export --checkpoint "$nest_run/step000500.pt" \
      --expect-updates 500 --output "$nest_run/student_step500.pt"
    ;;
  verify-export)
    "$nest_python" -m tools.nest_clip verify-export --checkpoint "$nest_run/step000500.pt" \
      --bare "$nest_run/student_step500.pt" --output "$nest_run/export-check.json" \
      --index-dir "$nest_root/data_index" --image-root /root/lk_projects/SAID-assets/training/ShareGPT4V
    ;;
  coco|urban)
    if [[ "$action" == coco ]]; then
      eval_root=/root/lk_projects/SAID-assets/evaluation/coco/val2017
    else
      eval_root=/root/lk_projects/SAID-assets/evaluation/Urban1k/Urban1k
    fi
    "$nest_python" -m tools.eval_nest_native --checkpoint "$nest_run/student_step500.pt" \
      --dataset "$action" --root "$eval_root" --device cuda:0 --batch-size 64 \
      --output "$nest_run/${action}_native.json"
    ;;
  *) exit 2 ;;
esac
