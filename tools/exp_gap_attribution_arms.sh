#!/bin/bash
# Phase 3.0A.1d: the two new matched loss-attribution arms (tau=1.0).
#   A: Said-only control      lambda_said=1, lambda_gap_discover=0, lambda_global_absorb=0
#   B: Said + Discovery       lambda_said=1, lambda_gap_discover=1, lambda_global_absorb=0
# C (Said + Discovery + Absorption) already exists as runs_salu/phase30a_1c_tau1 and is
# reused, not re-run.
set -u
cd /root/SAID-gap-completion || exit 1
bash tools/exp_gap_attribution_arm.sh 0 0 runs_salu/phase30a_1d_A_said_only
bash tools/exp_gap_attribution_arm.sh 1 0 runs_salu/phase30a_1d_B_said_discover
echo "ALL_ATTRIBUTION_ARMS_DONE"
