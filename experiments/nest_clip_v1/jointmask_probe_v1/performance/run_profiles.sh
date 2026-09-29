#!/usr/bin/env bash
set -euo pipefail
cd /root/lk_projects/SAID-jointmask-perf
root=/root/lk_projects/SAID-nest-clip-v1/jointmask_perf_diagnosis
worker=experiments.nest_clip_v1.jointmask_probe_v1.performance.benchmark_worker
torchrun=/root/miniconda3/envs/said-repro/bin/torchrun
common=(--init-state /root/lk_projects/SAID-nest-clip-v1/shared/step000000.pt
        --index-dir /root/lk_projects/SAID-nest-clip-v1/data_index
        --image-root /root/lk_projects/SAID-assets/training/ShareGPT4V
        --warmup 1 --steps 1 --profile)

run_one() {
  local name=$1; shift
  test ! -e "$root/$name"
  OMP_NUM_THREADS=4 "$torchrun" --standalone --nnodes=1 --nproc-per-node=4 \
    --max-restarts=0 -m "$worker" "$@" --output "$root/$name" "${common[@]}" \
    > "$root/$name.console.txt" 2>&1
}

run_one profile-baseline-T --mode T --implementation baseline --image-chunk 32 --text-chunk 64 --pair-checkpoint 1
run_one profile-baseline-TI --mode TI --implementation baseline --image-chunk 32 --text-chunk 64 --pair-checkpoint 1
run_one profile-final-T --mode T --implementation optimized --image-chunk 128 --text-chunk 128 --pair-checkpoint 0
run_one profile-final-TI --mode TI --implementation optimized --image-chunk 128 --text-chunk 128 --pair-checkpoint 0
