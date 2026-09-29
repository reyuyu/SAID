#!/usr/bin/env bash
set -euo pipefail
cd /root/lk_projects/SAID
action=${1:?Usage: run500.sh coco\|urban\|flickr_test1k\|docci}
root=/root/lk_projects/SAID-nest-clip-v1/hybridf_v1/E11/step500
bare="$root/student_step500.pt"
python=/root/miniconda3/envs/said-repro/bin/python
test -s "$bare"
case "$action" in
  coco|urban)
    if [[ "$action" == coco ]]; then
      dataset_root=/root/lk_projects/SAID-assets/evaluation/coco/val2017
    else
      dataset_root=/root/lk_projects/SAID-assets/evaluation/Urban1k/Urban1k
    fi
    "$python" -m tools.eval_nest_native --checkpoint "$bare" \
      --dataset "$action" --root "$dataset_root" --device cuda:0 \
      --batch-size 64 --output "$root/evaluation/${action}_native.json"
    ;;
  flickr_test1k|docci)
    bench=/root/lk_projects/SAID-assets/retrieval_benchmarks
    case "$action" in
      flickr_test1k) manifest=flickr30k_test1k.jsonl; images=flickr30k/images ;;
      docci) manifest=docci_test.jsonl; images=docci/images ;;
    esac
    test ! -e "$root/evaluation/$action/$action.json"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$python" \
      -m experiments.s0_dualmask_full_v01.evidence.step2000.new_evaluations.eval_extended_real \
      --checkpoint "$bare" --device cuda:0 --batch-size 64 \
      --output-dir "$root/evaluation/$action" \
      "$action:$bench/manifests/$manifest:$bench/$images"
    ;;
  *) exit 2 ;;
esac
