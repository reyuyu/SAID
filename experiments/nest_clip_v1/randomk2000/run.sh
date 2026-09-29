#!/usr/bin/env bash
set -euo pipefail
cd /root/lk_projects/SAID
action=${1:?Usage: run.sh export\|verify-export\|coco\|urban\|flickr_test1k\|docci}
nest_root=/root/lk_projects/SAID-nest-clip-v1
nest_run="$nest_root/randomk2000"
nest_checkpoint="$nest_root/randomk3epoch/A3-RandomK/step002000.pt"
nest_student="$nest_run/student_step2000.pt"
nest_python=/root/miniconda3/envs/said-repro/bin/python
case "$action" in
  export)
    "$nest_python" -m tools.nest_clip export --checkpoint "$nest_checkpoint" \
      --expect-updates 2000 --output "$nest_student"
    ;;
  verify-export)
    test ! -e "$nest_run/export-check.json"
    "$nest_python" -m tools.nest_clip verify-export --checkpoint "$nest_checkpoint" \
      --bare "$nest_student" --output "$nest_run/export-check.json" \
      --index-dir "$nest_root/data_index" --image-root /root/lk_projects/SAID-assets/training/ShareGPT4V
    ;;
  coco|urban)
    if [[ "$action" == coco ]]; then
      eval_root=/root/lk_projects/SAID-assets/evaluation/coco/val2017
    else
      eval_root=/root/lk_projects/SAID-assets/evaluation/Urban1k/Urban1k
    fi
    "$nest_python" -m tools.eval_nest_native --checkpoint "$nest_student" --dataset "$action" \
      --root "$eval_root" --device cuda:0 --batch-size 64 --output "$nest_run/evaluation/${action}_native.json"
    ;;
  flickr_test1k|docci)
    bench=/root/lk_projects/SAID-assets/retrieval_benchmarks
    if [[ "$action" == flickr_test1k ]]; then
      manifest=flickr30k_test1k.jsonl; images=flickr30k/images
    else
      manifest=docci_test.jsonl; images=docci/images
    fi
    test ! -e "$nest_run/evaluation/$action/$action.json"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$nest_python" \
      -m experiments.s0_dualmask_full_v01.evidence.step2000.new_evaluations.eval_extended_real \
      --checkpoint "$nest_student" --device cuda:0 --batch-size 64 \
      --output-dir "$nest_run/evaluation/$action" "$action:$bench/manifests/$manifest:$bench/$images"
    ;;
  *) echo 'Only export, verify-export, coco, urban, flickr_test1k and docci are enabled.' >&2; exit 2 ;;
esac
