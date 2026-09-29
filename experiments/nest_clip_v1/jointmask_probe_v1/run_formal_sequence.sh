#!/usr/bin/env bash
set -euo pipefail
cd /root/lk_projects/SAID
runner=/root/miniconda3/envs/said-repro/bin/python
stage=experiments/nest_clip_v1/jointmask_probe_v1/evidence/run_stage.py
evidence=experiments/nest_clip_v1/jointmask_probe_v1/evidence
for group in T TI TI-Shuffle; do
  name="formal-$group"
  if [[ -e "$evidence/$name.execution.json" || -e "$evidence/$name.console.txt" ]]; then
    attempt=1
    while [[ -e "$evidence/$name-restart$attempt.execution.json" ||
             -e "$evidence/$name-restart$attempt.console.txt" ]]; do
      attempt=$((attempt + 1))
    done
    name="$name-restart$attempt"
  fi
  "$runner" "$stage" "$name" \
    bash experiments/nest_clip_v1/jointmask_probe_v1/run.sh formal "$group"
done
