#!/usr/bin/env bash
set -euo pipefail
cd /root/lk_projects/SAID
runner=/root/miniconda3/envs/said-repro/bin/python
stage=experiments/nest_clip_v1/jointmask_probe_v1/evidence/run_stage.py
for group in T TI TI-Shuffle; do
  "$runner" "$stage" "formal-$group" \
    bash experiments/nest_clip_v1/jointmask_probe_v1/run.sh formal "$group"
done
