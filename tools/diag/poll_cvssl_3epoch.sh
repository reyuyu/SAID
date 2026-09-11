#!/bin/bash
# Poll a 3-epoch CVSSL run: step, epoch, checkpoints, throughput, memory, errors.
#   bash tools/diag/poll_cvssl_3epoch.sh <run_dir> [polls] [interval_sec]
cd /root/SAID-gap-completion || exit 1
D=${1:-runs_salu/said_cls_cvssl/c0_3epoch/L03}
N=${2:-12}
IV=${3:-280}
LOG="$D/train.log"
for i in $(seq 1 "$N"); do
  step=$(grep -o 'completed_steps": [0-9]*' "$LOG" 2>/dev/null | tail -1 | grep -o '[0-9]*')
  epoch=$(grep -o '"epoch": [0-9]*' "$LOG" 2>/dev/null | tail -1 | grep -o '[0-9]*')
  sec=$(grep -o '"sec_per_step": [0-9.]*' "$LOG" 2>/dev/null | tail -1 | grep -o '[0-9.]*')
  mem=$(grep -o '"peak_memory_gb": [0-9.]*' "$LOG" 2>/dev/null | tail -1 | grep -o '[0-9.]*')
  smart=$(grep -o '"loss_smart_global_mean": [0-9.]*' "$LOG" 2>/dev/null | tail -1 | grep -o '[0-9.]*')
  u=$(grep -o '"loss_vssl_global_mean": [0-9.]*' "$LOG" 2>/dev/null | tail -1 | grep -o '[0-9.]*')
  ckpts=$(ls "$D"/cvssl_*.pt 2>/dev/null | wc -l)
  echo "poll=$i step=${step:-?} epoch=${epoch:-?} smart=${smart:-?} u=${u:-?} sec=${sec:-?} mem=${mem:-?} ckpts=$ckpts"
  if grep -q 'RUN_SUMMARY' "$LOG" 2>/dev/null; then
    echo RUN_FINISHED
    grep 'RUN_SUMMARY' "$LOG"
    exit 0
  fi
  if grep -qE 'Traceback|ChildFailedError|CUDA out of memory' "$LOG" 2>/dev/null; then
    echo RUN_FAILED
    grep -nE 'Error|Traceback|out of memory' "$LOG" | head -12
    exit 1
  fi
  sleep "$IV"
done
echo STILL_RUNNING
