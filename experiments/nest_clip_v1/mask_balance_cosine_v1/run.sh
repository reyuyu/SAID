#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$script_dir/../../.." && pwd)
cd "$repo"
action=${1:?Missing action}
group=${2:?Missing candidate}
case "$group" in
  Balanced-Stack-Patch) config="$repo/configs/nest_balanced_stack_patch.json" ;;
  Cosine-CrossScore-CLS) config="$repo/configs/nest_cosine_crossscore_cls.json" ;;
  *) exit 2 ;;
esac
root=/root/lk_projects/SAID-nest-clip-v1
exp="$root/mask_balance_cosine_v1"
formal="$exp/formal/$group"
python=/root/miniconda3/envs/said-repro/bin/python
case "$action" in
  probe|smoke|formal)
    case "$action" in
      probe) updates=35 ;;
      smoke) updates=5 ;;
      formal) updates=500 ;;
    esac
    destination="$exp/$action/$group"
    test ! -e "$destination"
    OMP_NUM_THREADS=4 /root/miniconda3/envs/said-repro/bin/torchrun \
      --standalone --nnodes=1 --nproc-per-node=4 --max-restarts=0 \
      -m train.train_nested_semantic_mask --config "$config" \
      --init-state "$root/shared/step000000.pt" --index-dir "$root/data_index" \
      --image-root /root/lk_projects/SAID-assets/training/ShareGPT4V \
      --output-dir "$destination" --run-type "$action" --max-updates "$updates"
    ;;
  export)
    test ! -e "$formal/student_step500.pt"
    "$python" -m tools.nest_clip export --checkpoint "$formal/step000500.pt" \
      --expect-updates 500 --output "$formal/student_step500.pt"
    ;;
  verify-export)
    test ! -e "$formal/export-check.json"
    "$python" -m tools.nest_clip verify-export --checkpoint "$formal/step000500.pt" \
      --bare "$formal/student_step500.pt" --output "$formal/export-check.json" \
      --index-dir "$root/data_index" --image-root /root/lk_projects/SAID-assets/training/ShareGPT4V
    ;;
  coco|urban)
    case "$action" in
      coco) eval_root=/root/lk_projects/SAID-assets/evaluation/coco/val2017 ;;
      urban) eval_root=/root/lk_projects/SAID-assets/evaluation/Urban1k/Urban1k ;;
    esac
    test ! -e "$formal/${action}_native.json"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$python" -m tools.eval_nest_native \
      --checkpoint "$formal/student_step500.pt" --dataset "$action" --root "$eval_root" \
      --device cuda:0 --batch-size 64 --output "$formal/${action}_native.json"
    ;;
  flickr_test1k|docci)
    bench=/root/lk_projects/SAID-assets/retrieval_benchmarks
    case "$action" in
      flickr_test1k) manifest=flickr30k_test1k.jsonl; images=flickr30k/images ;;
      docci) manifest=docci_test.jsonl; images=docci/images ;;
    esac
    test ! -e "$formal/$action/$action.json"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$python" \
      -m experiments.s0_dualmask_full_v01.evidence.step2000.new_evaluations.eval_extended_real \
      --checkpoint "$formal/student_step500.pt" --device cuda:0 --batch-size 64 \
      --output-dir "$formal/$action" "$action:$bench/manifests/$manifest:$bench/$images"
    ;;
  *) exit 2 ;;
esac
