#!/usr/bin/env bash
set -euo pipefail
cd /root/lk_projects/SAID-jointmask-perf
root=/root/lk_projects/SAID-nest-clip-v1/jointmask_perf_diagnosis
worker=experiments.nest_clip_v1.jointmask_probe_v1.performance.benchmark_worker
torchrun=/root/miniconda3/envs/said-repro/bin/torchrun
common=(--init-state /root/lk_projects/SAID-nest-clip-v1/shared/step000000.pt
        --index-dir /root/lk_projects/SAID-nest-clip-v1/data_index
        --image-root /root/lk_projects/SAID-assets/training/ShareGPT4V
        --warmup 2 --steps 8)

run_one() {
  local name=$1; shift
  test ! -e "$root/$name"
  OMP_NUM_THREADS=4 "$torchrun" --standalone --nnodes=1 --nproc-per-node=4 \
    --max-restarts=0 -m "$worker" "$@" --output "$root/$name" "${common[@]}" \
    > "$root/$name.console.txt" 2>&1
  cat "$root/$name/result.json" | /root/miniconda3/envs/said-repro/bin/python -c \
    'import json,sys; x=json.load(sys.stdin); print(x["args"]["mode"], x["args"]["implementation"], x["args"]["image_chunk"], x["args"]["text_chunk"], x["args"]["pair_checkpoint"], x["max_rank_summary"]["step_ms"]["median_ms"], x["max_rank_summary"]["step_ms"]["p90_ms"])'
}

run_one baseline-T-32x64-cp1-r2 --mode T --implementation baseline --image-chunk 32 --text-chunk 64 --pair-checkpoint 1
run_one baseline-TI-32x64-cp1-r2 --mode TI --implementation baseline --image-chunk 32 --text-chunk 64 --pair-checkpoint 1
run_one optimized-T-32x64-cp1-r2 --mode T --implementation optimized --image-chunk 32 --text-chunk 64 --pair-checkpoint 1
run_one optimized-TI-32x64-cp1-r2 --mode TI --implementation optimized --image-chunk 32 --text-chunk 64 --pair-checkpoint 1
run_one optimized-T-64x128-cp1-r2 --mode T --implementation optimized --image-chunk 64 --text-chunk 128 --pair-checkpoint 1
run_one optimized-TI-64x128-cp1-r2 --mode TI --implementation optimized --image-chunk 64 --text-chunk 128 --pair-checkpoint 1
run_one optimized-T-128x128-cp1-r2 --mode T --implementation optimized --image-chunk 128 --text-chunk 128 --pair-checkpoint 1
run_one optimized-TI-128x128-cp1-r2 --mode TI --implementation optimized --image-chunk 128 --text-chunk 128 --pair-checkpoint 1
run_one optimized-T-128x128-cp0 --mode T --implementation optimized --image-chunk 128 --text-chunk 128 --pair-checkpoint 0
run_one optimized-TI-128x128-cp0 --mode TI --implementation optimized --image-chunk 128 --text-chunk 128 --pair-checkpoint 0
run_one matrix-T-r2 --mode T --implementation matrix --image-chunk 128 --text-chunk 64 --pair-checkpoint 0
