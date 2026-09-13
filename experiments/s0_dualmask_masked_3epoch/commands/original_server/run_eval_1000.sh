#!/usr/bin/env bash
set -u
cd /root/SAID-s0-dualmask-clean-v01 || exit 90
run_dir=/root/SAID-s0-dualmask-clean-v01/runs_salu/dual_mask_suffix_masked_from500_continuation_v02
runtime=/root/miniconda3/envs/said-smartclip/bin/python
test "$(cat "$run_dir/resume.exitcode")" = 0 || exit 91
export CUDA_VISIBLE_DEVICES=0
eval_started=$(date +%s)
"$runtime" /tmp/export_continuation.py --run "$run_dir" --step 1000 > "$run_dir/export1000.console.log" 2>&1
code=$?
printf '%s\n' "$code" > "$run_dir/export1000.exitcode"
test "$code" = 0 || exit "$code"
"$runtime" /tmp/eval_continuation_coco.py --run "$run_dir" --step 1000 > "$run_dir/coco1000.console.log" 2>&1
code=$?
printf '%s\n' "$code" > "$run_dir/coco1000.exitcode"
test "$code" = 0 || exit "$code"
"$runtime" /root/SAID-gap-completion/tools/eval_urban1k_cls.py \
  --checkpoint "$run_dir/bare_student_step1000.pt" --label masked_dualmask_continuation1000 \
  --batch_size 64 --device cuda:0 --out "$run_dir/urban1k1000.json" > "$run_dir/urban1000.console.log" 2>&1
code=$?
printf '%s\n' "$code" > "$run_dir/urban1000.exitcode"
printf '%s\n' "$(( $(date +%s) - eval_started ))" > "$run_dir/eval1000_wall_seconds.txt"
exit "$code"
