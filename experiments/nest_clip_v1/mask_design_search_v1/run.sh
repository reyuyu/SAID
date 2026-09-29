#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$script_dir/../../.." && pwd)
cd "$repo"
action=${1:?Usage: run.sh tests|validate|resource C1|C2|P1|P2}
python=/root/miniconda3/envs/said-repro/bin/python
root=/root/lk_projects/SAID-nest-clip-v1
case "$action" in
  tests)
    "$python" -m pytest tests/test_nested_joint_input.py tests/test_nested_jointmask.py tests/test_nested_semantic_mask.py -q
    CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 PYTHONPATH=. \
      /root/miniconda3/envs/said-repro/bin/torchrun --standalone --nnodes=1 \
      --nproc-per-node=2 --max-restarts=0 -m tests.joint_input_ddp_worker
    ;;
  validate)
    CUDA_VISIBLE_DEVICES=0 "$python" -m experiments.nest_clip_v1.mask_design_search_v1.validate_actual \
      --init-state "$root/shared/step000000.pt" --index-dir "$root/data_index" \
      --image-root /root/lk_projects/SAID-assets/training/ShareGPT4V \
      --output "$script_dir/evidence/actual-validation-reproduced.json"
    ;;
  resource)
    candidate=${2:?Missing C1|C2|P1|P2}
    case "$candidate" in
      C1) visual=cls; readout=all; pairs=512; label=cls-all ;;
      C2) visual=cls; readout=text; pairs=512; label=cls-text ;;
      P1) visual=patch; readout=all; pairs=512; label=patch-all ;;
      P2) visual=patch; readout=text; pairs=512; label=patch-text ;;
      *) exit 2 ;;
    esac
    destination="$root/mask_design_search_v1/resource/reproduced-$label"
    test ! -e "$destination"
    OMP_NUM_THREADS=4 /root/miniconda3/envs/said-repro/bin/torchrun --standalone \
      --nnodes=1 --nproc-per-node=4 --max-restarts=0 \
      -m experiments.nest_clip_v1.mask_design_search_v1.benchmark_pairs \
      --visual-mode "$visual" --readout-mode "$readout" --pair-batch "$pairs" \
      --warmup 2 --steps 5 --checkpoint-block 1 \
      --init-state "$root/shared/step000000.pt" --output "$destination"
    ;;
  *) exit 2 ;;
esac
