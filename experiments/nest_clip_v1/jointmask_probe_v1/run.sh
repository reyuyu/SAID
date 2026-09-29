#!/usr/bin/env bash
set -euo pipefail
cd /root/lk_projects/SAID
action=${1:?Usage: run.sh smoke|formal|export500|verify-export|coco|urban|flickr_test1k|docci|long_dci T|TI|TI-Shuffle}
group=${2:?Missing group: T|TI|TI-Shuffle}
case "$group" in
  T) config=configs/nest_jointmask_t.json ;;
  TI) config=configs/nest_jointmask_ti.json ;;
  TI-Shuffle) config=configs/nest_jointmask_ti_shuffle.json ;;
  *) exit 2 ;;
esac
root=/root/lk_projects/SAID-nest-clip-v1
probe="$root/jointmask_probe_v1"
formal="$probe/formal/$group"
python=/root/miniconda3/envs/said-repro/bin/python
case "$action" in
  smoke|formal)
    if [[ "$action" == smoke ]]; then
      run_type=smoke; updates=5; destination="$probe/smoke/$group"
    else
      run_type=formal; updates=500; destination="$formal"
    fi
    test ! -e "$destination"
    OMP_NUM_THREADS=4 /root/miniconda3/envs/said-repro/bin/torchrun \
      --standalone --nnodes=1 --nproc-per-node=4 --max-restarts=0 \
      -m train.train_nested_semantic_mask --config "$config" \
      --init-state "$root/shared/step000000.pt" --index-dir "$root/data_index" \
      --image-root /root/lk_projects/SAID-assets/training/ShareGPT4V \
      --output-dir "$destination" --run-type "$run_type" --max-updates "$updates"
    ;;
  export500)
    "$python" -m tools.nest_clip export --checkpoint "$formal/step000500.pt" \
      --expect-updates 500 --output "$formal/student_step500.pt"
    ;;
  verify-export)
    "$python" -m tools.nest_clip verify-export --checkpoint "$formal/step000500.pt" \
      --bare "$formal/student_step500.pt" --output "$formal/export-check.json" \
      --index-dir "$root/data_index" \
      --image-root /root/lk_projects/SAID-assets/training/ShareGPT4V
    ;;
  coco|urban)
    if [[ "$action" == coco ]]; then
      eval_root=/root/lk_projects/SAID-assets/evaluation/coco/val2017
    else
      eval_root=/root/lk_projects/SAID-assets/evaluation/Urban1k/Urban1k
    fi
    "$python" -m tools.eval_nest_native --checkpoint "$formal/student_step500.pt" \
      --dataset "$action" --root "$eval_root" --device cuda:0 --batch-size 64 \
      --output "$formal/${action}_native.json"
    ;;
  flickr_test1k|docci|long_dci)
    bench=/root/lk_projects/SAID-assets/retrieval_benchmarks
    case "$action" in
      flickr_test1k) manifest=flickr30k_test1k.jsonl; images=flickr30k/images ;;
      docci) manifest=docci_test.jsonl; images=docci/images ;;
      long_dci) manifest=long_dci_reconstructed.jsonl; images=dci/images ;;
    esac
    test ! -e "$formal/$action/$action.json"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$python" \
      -m experiments.s0_dualmask_full_v01.evidence.step2000.new_evaluations.eval_extended_real \
      --checkpoint "$formal/student_step500.pt" --device cuda:0 --batch-size 64 \
      --output-dir "$formal/$action" \
      "$action:$bench/manifests/$manifest:$bench/$images"
    ;;
  *) exit 2 ;;
esac
