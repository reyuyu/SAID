#!/usr/bin/env bash
set -euo pipefail
: "${SAID_PYTHON:?Set the reproduction environment Python}"
: "${SAID_TRAIN_REPO:?Set the fixed dcd33877 training checkout}"
: "${SAID_INIT:?Set the validated reconstructed initial state}"
: "${SAID_RUN:?Set a new output directory}"
: "${SHARE4V_DATA_ROOT:?Set the complete ShareGPT4V root}"
: "${SHARE4V_FULL_AUDIT:?Set the successful full-data audit JSON}"
export SHARE4V_JSON=${SHARE4V_JSON:-share-captioner_coco_lcs_sam_1246k_1107.json}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
bundle_tools=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
"$SAID_PYTHON" "$bundle_tools/common.py" --repo "$SAID_TRAIN_REPO" --init "$SAID_INIT"
test "$(git -C "$SAID_TRAIN_REPO" rev-parse HEAD)" = dcd33877f1f77a901d292190834f1ffd83a049a8
test ! -e "$SAID_RUN"
mkdir -p "$SAID_RUN"
cd "$SAID_TRAIN_REPO"
set +e
"$SAID_PYTHON" -m torch.distributed.run --nproc_per_node=4 --master_addr=127.0.0.1 --master_port="${SAID_MASTER_PORT:-29642}" train/train_dual_mask_suffix.py \
  --suffix-mode masked --run-type formal --base-model B16 --batch-size 256 --epochs 3 --max-steps 500 \
  --seed 0 --num-workers 8 --lr 1e-6 --mask-lr 1e-3 --suffix-lr 1e-4 --weight-decay 0.01 --warmup 200 \
  --total-len 1000 --amp-dtype bf16 --image-chunk 16 --text-chunk 32 --save-every 500 \
  --lambda-suffix 3 --lambda-u-sparse 0 --init-state "$SAID_INIT" --output-dir "$SAID_RUN" \
  > "$SAID_RUN/train.console.log" 2>&1
code=$?
printf '%s\n' "$code" > "$SAID_RUN/train.exitcode"
exit "$code"
