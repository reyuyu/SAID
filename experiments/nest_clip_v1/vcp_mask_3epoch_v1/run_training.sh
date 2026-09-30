#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$script_dir/../../.." && pwd)
cd "$repo"
asset_root=/root/lk_projects/SAID-nest-clip-v1
parent="$asset_root/vcp_mask_v1/formal/VCP-Mask/step000500.pt"
out="$asset_root/vcp_mask_3epoch_v1/formal/VCP-Mask"
trainer_parent_sha=09c007f88935ee1a5fc300342bc2382c6bc5f06da53fb252369dca2c108543b2
if [[ -e "$out" ]]; then
  echo "Refusing to overwrite $out" >&2
  exit 3
fi
mkdir -p "$(dirname "$out")"
OMP_NUM_THREADS=4 /root/miniconda3/envs/said-repro/bin/torchrun \
  --standalone --nnodes=1 --nproc-per-node=4 --max-restarts=0 \
  -m train.train_nested_semantic_mask \
  --config "$repo/configs/nest_vcp_mask_3epoch.json" \
  --init-state "$asset_root/shared/step000000.pt" --index-dir "$asset_root/data_index" \
  --image-root /root/lk_projects/SAID-assets/training/ShareGPT4V \
  --output-dir "$out" --run-type formal --max-updates 3651 \
  --resume "$parent" --expected-parent-trainer-sha256 "$trainer_parent_sha"
