#!/bin/bash
# Monitor a CVSSL run's progress.
#   bash tools/diag/watch_cvssl_run.sh <run_dir> [samples] [interval_sec]
cd /root/SAID-gap-completion || exit 1
D=$1
N=${2:-6}
IV=${3:-20}
for i in $(seq 1 "$N"); do
  logs=$(grep -c '^LOG' "$D/train.log" 2>/dev/null || echo 0)
  saved=$(grep -c '^SAVED' "$D/train.log" 2>/dev/null || echo 0)
  files=$(ls "$D" 2>/dev/null | tr '\n' ' ')
  last=$(grep '^LOG' "$D/train.log" 2>/dev/null | tail -1)
  step=$(echo "$last" | sed -n 's/.*"completed_steps": \([0-9]*\).*/\1/p')
  smart=$(echo "$last" | sed -n 's/.*"loss_smart_global_mean": \([0-9.eE+-]*\).*/\1/p')
  vssl=$(echo "$last" | sed -n 's/.*"loss_vssl_global_mean": \([0-9.eE+-]*\).*/\1/p')
  sec=$(echo "$last" | sed -n 's/.*"sec_per_step": \([0-9.eE+-]*\).*/\1/p')
  mem=$(echo "$last" | sed -n 's/.*"peak_memory_gb": \([0-9.eE+-]*\).*/\1/p')
  echo "poll=$i step=${step:-?} smart=${smart:-?} u=${vssl:-?} sec=${sec:-?} mem=${mem:-?} logs=$logs saved=$saved files=[$files]"
  if grep -q 'RUN_SUMMARY' "$D/train.log" 2>/dev/null; then
    echo "RUN_FINISHED"
    grep 'RUN_SUMMARY' "$D/train.log"
    exit 0
  fi
  if grep -qE 'Traceback|ChildFailedError' "$D/train.log" 2>/dev/null; then
    echo "RUN_FAILED"
    grep -nE 'Error|Traceback' "$D/train.log" | head -10
    exit 1
  fi
  sleep "$IV"
done
echo "STILL_RUNNING"
