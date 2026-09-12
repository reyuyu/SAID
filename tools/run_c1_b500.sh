#!/usr/bin/env bash
set -euo pipefail
cd /root/SAID-c1-micro-ablation-v01
OUT=$PWD/runs_salu/c1_micro_B500
PARENT=$PWD/runs_salu/c1_micro_B250/c1_C1_text_conditional_reconstruction_step000250.pt
[ ! -e "$OUT" ] || { echo 'Refusing existing B500 output'; exit 2; }
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { echo 'NOT RUN GPUs busy'; exit 3; }
bash tools/run_c1_micro.sh C1-VWarm "$OUT" 500 "$PARENT"
echo B500_TRAIN_COMPLETE_STOP
