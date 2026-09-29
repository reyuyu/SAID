#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$script_dir/../../.." && pwd)
cd "$repo"
python=/root/miniconda3/envs/said-repro/bin/python
torchrun=/root/miniconda3/envs/said-repro/bin/torchrun
asset_root=/root/lk_projects/SAID-nest-clip-v1
parent="$asset_root/jointmask_fast_v1/formal/TI-fast/step000500.pt"
out="$asset_root/jointmask_fast_v1/full3epoch/TI-fast"
config="$repo/configs/nest_jointmask_ti_fast_3epoch.json"
trainer_parent_sha=d94aec667618110bca233dad8bd377e8cbde08c3da5599be5e3e2e22a92659ba
if [[ -e "$out" ]]; then
  echo "Refusing to overwrite $out" >&2
  exit 3
fi
mkdir -p "$(dirname "$out")"
OMP_NUM_THREADS=4 "$torchrun" --standalone --nnodes=1 --nproc-per-node=4 --max-restarts=0 \
  -m train.train_nested_semantic_mask --config "$config" \
  --init-state "$asset_root/shared/step000000.pt" --index-dir "$asset_root/data_index" \
  --image-root /root/lk_projects/SAID-assets/training/ShareGPT4V \
  --output-dir "$out" --run-type formal --max-updates 3651 \
  --resume "$parent" --expected-parent-trainer-sha256 "$trainer_parent_sha"
"$python" -m tools.nest_clip export --checkpoint "$out/step003651.pt" \
  --expect-updates 3651 --output "$out/student_step3651.pt"
"$python" -m tools.nest_clip verify-export --checkpoint "$out/step003651.pt" \
  --bare "$out/student_step3651.pt" --output "$out/export-check.json" \
  --index-dir "$asset_root/data_index" \
  --image-root /root/lk_projects/SAID-assets/training/ShareGPT4V
"$python" -m tools.eval_nest_native --checkpoint "$out/student_step3651.pt" \
  --dataset coco --root /root/lk_projects/SAID-assets/evaluation/coco/val2017 \
  --device cuda:0 --batch-size 64 --output "$out/coco_native.json"
"$python" -m tools.eval_nest_native --checkpoint "$out/student_step3651.pt" \
  --dataset urban --root /root/lk_projects/SAID-assets/evaluation/Urban1k/Urban1k \
  --device cuda:0 --batch-size 64 --output "$out/urban_native.json"
bench=/root/lk_projects/SAID-assets/retrieval_benchmarks
for dataset in flickr_test1k docci; do
  if [[ "$dataset" == flickr_test1k ]]; then
    manifest=flickr30k_test1k.jsonl; images=flickr30k/images
  else
    manifest=docci_test.jsonl; images=docci/images
  fi
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$python" \
    -m experiments.s0_dualmask_full_v01.evidence.step2000.new_evaluations.eval_extended_real \
    --checkpoint "$out/student_step3651.pt" --device cuda:0 --batch-size 64 \
    --output-dir "$out/$dataset" \
    "$dataset:$bench/manifests/$manifest:$bench/$images"
done
"$python" "$script_dir/summarize.py" --output "$out"
