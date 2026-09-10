#!/bin/bash
# Phase 3.0A.1c: run the two matched 100-step arms sequentially.
#   A: tau = 1.0  (current default baseline)
#   B: tau = 0.5  (selected stronger complement from the forward-only sweep)
set -u
cd /root/SAID-gap-completion || exit 1
bash tools/exp_gap_100step.sh 1.0 runs_salu/phase30a_1c_tau1
bash tools/exp_gap_100step.sh 0.5 runs_salu/phase30a_1c_tau05
echo "ALL_ARMS_DONE"
