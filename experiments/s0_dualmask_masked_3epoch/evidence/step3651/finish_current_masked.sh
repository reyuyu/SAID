#!/usr/bin/env bash
# One-off sequential execution for the already authorized 500 -> 1000 -> 3-epoch trajectory.
set -eu
cd /root/SAID-s0-dualmask-clean-v01
run_dir=/root/SAID-s0-dualmask-clean-v01/runs_salu/dual_mask_suffix_masked_from500_continuation_v02
runtime=/root/miniconda3/envs/said-smartclip/bin/python
trap 'code=$?; printf "%s\n" "$code" > "$run_dir/continuation_sequence.exitcode"' EXIT
test ! -e "$run_dir/continuation_sequence_started.txt"
date --iso-8601=seconds > "$run_dir/continuation_sequence_started.txt"
while ps -p "$1" -o args= | grep -q '^bash /tmp/run_masked_resume_3epoch.sh$'; do
    sleep 30
done
test "$(cat "$run_dir/resume.exitcode")" = 0
test -f "$run_dir/s0_dual_mask_suffix_masked_step001000.pt"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || exit 86
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv > "$run_dir/gpu_before_eval1000.csv"
bash /tmp/run_eval_1000.sh
test "$(cat "$run_dir/export1000.exitcode")" = 0
test "$(cat "$run_dir/coco1000.exitcode")" = 0
test "$(cat "$run_dir/urban1000.exitcode")" = 0
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || exit 86
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv > "$run_dir/gpu_before_resume1000.csv"
bash /tmp/run_masked_resume_final.sh > "$run_dir/resume1000.console.log" 2>&1
test "$(cat "$run_dir/resume1000.exitcode")" = 0
test -f "$run_dir/s0_dual_mask_suffix_masked_step003651.pt"
date --iso-8601=seconds > "$run_dir/continuation_sequence_finished.txt"
