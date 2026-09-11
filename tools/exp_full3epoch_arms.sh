#!/bin/bash
# Phase 3.0A.2: the two matched full 3-epoch arms, sequentially.
#   A: Said-only control   lambda_said=1, lambda_gap_discover=0, lambda_global_absorb=0
#   C: Full Base           lambda_said=1, lambda_gap_discover=1, lambda_global_absorb=1
set -u
cd /root/SAID-gap-completion || exit 1
bash tools/exp_full3epoch_arm.sh 0 0 runs_salu/phase30a_2_A_said_only
bash tools/exp_full3epoch_arm.sh 1 1 runs_salu/phase30a_2_C_full_base
echo "ALL_FULL3EPOCH_ARMS_DONE"
date -Is
