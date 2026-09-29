#!/usr/bin/env bash
set -euo pipefail
cd /root/lk_projects/SAID
action=${1:?Usage: run.sh smoke\|train\|export\|verify-export\|coco\|urban\|flickr_test1k\|docci\|dci\|long_dci}
nest_root=/root/lk_projects/SAID-nest-clip-v1
e11_root="$nest_root/hybridf_v1/E11"
e11_run="$e11_root/formal"
nest_python=/root/miniconda3/envs/said-repro/bin/python
case "$action" in
  smoke|train)
    if [[ "$action" == smoke ]]; then
      run_type=smoke; stop=5; destination="$e11_root/smoke"
    else
      run_type=formal; stop=3651; destination="$e11_run"
    fi
    OMP_NUM_THREADS=4 /root/miniconda3/envs/said-repro/bin/torchrun \
      --standalone --nnodes=1 --nproc-per-node=4 --max-restarts=0 \
      -m train.train_nested_semantic_mask --config configs/nest_hybridf_e11.json \
      --init-state "$nest_root/shared/step000000.pt" --index-dir "$nest_root/data_index" \
      --image-root /root/lk_projects/SAID-assets/training/ShareGPT4V \
      --output-dir "$destination" --run-type "$run_type" --max-updates "$stop"
    ;;
  export)
    "$nest_python" -m tools.nest_clip export --checkpoint "$e11_run/step003651.pt" \
      --expect-updates 3651 --output "$e11_run/student_step3651.pt"
    ;;
  verify-export)
    test ! -e "$e11_run/export-check.json"
    "$nest_python" -m tools.nest_clip verify-export --checkpoint "$e11_run/step003651.pt" \
      --bare "$e11_run/student_step3651.pt" --output "$e11_run/export-check.json" \
      --index-dir "$nest_root/data_index" --image-root /root/lk_projects/SAID-assets/training/ShareGPT4V
    ;;
  coco|urban)
    if [[ "$action" == coco ]]; then
      eval_root=/root/lk_projects/SAID-assets/evaluation/coco/val2017
    else
      eval_root=/root/lk_projects/SAID-assets/evaluation/Urban1k/Urban1k
    fi
    "$nest_python" -m tools.eval_nest_native --checkpoint "$e11_run/student_step3651.pt" \
      --dataset "$action" --root "$eval_root" --device cuda:0 --batch-size 64 \
      --output "$e11_root/evaluation/${action}_native.json"
    ;;
  flickr_test1k|docci|dci|long_dci)
    bench=/root/lk_projects/SAID-assets/retrieval_benchmarks
    case "$action" in
      flickr_test1k) manifest=flickr30k_test1k.jsonl; images=flickr30k/images ;;
      docci) manifest=docci_test.jsonl; images=docci/images ;;
      dci) manifest=dci_full.jsonl; images=dci/images ;;
      long_dci) manifest=long_dci_reconstructed.jsonl; images=dci/images ;;
    esac
    test ! -e "$e11_root/evaluation/$action/$action.json"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$nest_python" \
      -m experiments.s0_dualmask_full_v01.evidence.step2000.new_evaluations.eval_extended_real \
      --checkpoint "$e11_run/student_step3651.pt" --device cuda:0 --batch-size 64 \
      --output-dir "$e11_root/evaluation/$action" "$action:$bench/manifests/$manifest:$bench/$images"
    ;;
  *) exit 2 ;;
esac
