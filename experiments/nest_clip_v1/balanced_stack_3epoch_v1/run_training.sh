#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$script_dir/../../.." && pwd)
cd "$repo"
asset_root=/root/lk_projects/SAID-nest-clip-v1
parent="$asset_root/mask_balance_cosine_v1/formal/Balanced-Stack-Patch/step000500.pt"
out="$asset_root/balanced_stack_3epoch_v1/formal/Balanced-Stack-Patch"
if [[ -e "$out" ]]; then
  echo "Refusing to overwrite $out" >&2
  exit 3
fi
mkdir -p "$(dirname "$out")"
OMP_NUM_THREADS=4 /root/miniconda3/envs/said-repro/bin/torchrun \
  --standalone --nnodes=1 --nproc-per-node=4 --max-restarts=0 \
  -m train.train_nested_semantic_mask \
  --config "$repo/configs/nest_balanced_stack_patch_3epoch.json" \
  --init-state "$asset_root/shared/step000000.pt" --index-dir "$asset_root/data_index" \
  --image-root /root/lk_projects/SAID-assets/training/ShareGPT4V \
  --output-dir "$out" --run-type formal --max-updates 3651 --resume "$parent"
