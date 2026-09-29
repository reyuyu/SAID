#!/usr/bin/env bash
set -euo pipefail
cd /root/lk_projects/SAID
action=${1:?Usage: run.sh train\|export\|verify-export\|coco\|urban\|flickr_test1k\|docci\|dci\|long_dci [500|3651]}
nest_root=/root/lk_projects/SAID-nest-clip-v1
nest_run="$nest_root/randomk3epoch/A3-RandomK"
nest_python=/root/miniconda3/envs/said-repro/bin/python
eval_step=${2:-3651}
[[ "$eval_step" == 500 || "$eval_step" == 3651 ]] || exit 2
if [[ "$eval_step" == 500 ]]; then
  student="$nest_root/randomk500/A3-RandomK/student_step500.pt"
else
  student="$nest_run/student_step3651.pt"
fi
eval_dir="$nest_root/randomk3epoch/evaluation/step$eval_step"
case "$action" in
  train)
    OMP_NUM_THREADS=4 /root/miniconda3/envs/said-repro/bin/torchrun \
      --standalone --nnodes=1 --nproc-per-node=4 --max-restarts=0 \
      -m train.train_nested_semantic_mask --config configs/nest_clip_a3_randomk.json \
      --init-state "$nest_root/shared/step000000.pt" --index-dir "$nest_root/data_index" \
      --image-root /root/lk_projects/SAID-assets/training/ShareGPT4V \
      --output-dir "$nest_run" --run-type formal --max-updates 3651 \
      --resume "$nest_root/randomk500/A3-RandomK/step000500.pt" \
      --expected-parent-trainer-sha256 66da82ca76945eb3b7803767996c0216bc354974d204361740aee85c22cabc31
    ;;
  export)
    "$nest_python" -m tools.nest_clip export --checkpoint "$nest_run/step003651.pt" \
      --expect-updates 3651 --output "$nest_run/student_step3651.pt"
    ;;
  verify-export)
    "$nest_python" -m tools.nest_clip verify-export --checkpoint "$nest_run/step003651.pt" \
      --bare "$nest_run/student_step3651.pt" --output "$nest_run/export-check.json" \
      --index-dir "$nest_root/data_index" --image-root /root/lk_projects/SAID-assets/training/ShareGPT4V
    ;;
  coco|urban)
    if [[ "$action" == coco ]]; then
      eval_root=/root/lk_projects/SAID-assets/evaluation/coco/val2017
    else
      eval_root=/root/lk_projects/SAID-assets/evaluation/Urban1k/Urban1k
    fi
    "$nest_python" -m tools.eval_nest_native --checkpoint "$student" --dataset "$action" \
      --root "$eval_root" --device cuda:0 --batch-size 64 --output "$eval_dir/${action}_native.json"
    ;;
  flickr_test1k|docci|dci|long_dci)
    bench=/root/lk_projects/SAID-assets/retrieval_benchmarks
    case "$action" in
      flickr_test1k) manifest=flickr30k_test1k.jsonl; images=flickr30k/images ;;
      docci) manifest=docci_test.jsonl; images=docci/images ;;
      dci) manifest=dci_full.jsonl; images=dci/images ;;
      long_dci) manifest=long_dci_reconstructed.jsonl; images=dci/images ;;
    esac
    test ! -e "$eval_dir/$action/$action.json"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$nest_python" \
      -m experiments.s0_dualmask_full_v01.evidence.step2000.new_evaluations.eval_extended_real \
      --checkpoint "$student" --device cuda:0 --batch-size 64 --output-dir "$eval_dir/$action" \
      "$action:$bench/manifests/$manifest:$bench/$images"
    ;;
  *) exit 2 ;;
esac
