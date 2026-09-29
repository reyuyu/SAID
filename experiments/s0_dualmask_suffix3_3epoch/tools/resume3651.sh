#!/usr/bin/env bash
set -euo pipefail
: "${SAID_PYTHON:?Set the reproduction Python}"
: "${SAID_RESUME_REPO:?Set the fixed 873b43a checkout}"
: "${SAID_PARENT:?Set the complete 3/0 step500 checkpoint}"
: "${SAID_INIT:?Set the matching common initial state}"
: "${SAID_RUN:?Set a fresh output directory}"
: "${SHARE4V_DATA_ROOT:?Set the complete ShareGPT4V root}"
: "${SHARE4V_FULL_AUDIT:?Set the successful full-data audit}"
export SHARE4V_JSON=${SHARE4V_JSON:-share-captioner_coco_lcs_sam_1246k_1107.json}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
unset PYTORCH_CUDA_ALLOC_CONF
bundle_tools=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
"$SAID_PYTHON" "$bundle_tools/common.py" --repo "$SAID_RESUME_REPO" --init "$SAID_INIT"
test "$(git -C "$SAID_RESUME_REPO" rev-parse HEAD)" = 873b43a5bc000528311e22aac01e92045ac898e0
"$SAID_PYTHON" - "$SAID_PARENT" <<'PY'
import sys
import torch
p = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
c = p['config']
assert p['completed_steps'] == c['formal_optimizer_updates'] == 500
for key, value in {'suffix_lambda': 3, 'u_sparsity': 0, 'world_size': 4,
                   'batch_size_per_gpu': 256, 'epochs': 3, 'loader_batches': 1217,
                   'lr_horizon_steps': 3651, 'seed': 0}.items():
    assert c[key] == value, key
assert c['arguments']['num_workers'] == 8
for name in ['clip', 'mask', 'suffix']:
    steps = [int(v['step']) for v in p['optimizer_states'][name]['state'].values() if 'step' in v]
    assert steps and min(steps) == max(steps) == 500, name
print('PARENT_500_CONFIGURATION_AND_OPTIMIZER_STEPS_VERIFIED')
PY
test ! -e "$SAID_RUN"
mkdir -p "$SAID_RUN"
cd "$SAID_RESUME_REPO"
set +e
"$SAID_PYTHON" -m torch.distributed.run --nproc_per_node=4 --master_addr=127.0.0.1 --master_port="${SAID_MASTER_PORT:-29645}" train/train_dual_mask_suffix.py \
  --suffix-mode masked --run-type formal --base-model B16 --batch-size 256 --epochs 3 --max-steps 3651 \
  --seed 0 --num-workers 8 --lr 1e-6 --mask-lr 1e-3 --suffix-lr 1e-4 --weight-decay 0.01 --warmup 200 \
  --total-len 1000 --amp-dtype bf16 --image-chunk 16 --text-chunk 32 --save-every 500 \
  --lambda-suffix 3 --lambda-u-sparse 0 --init-state "$SAID_INIT" --output-dir "$SAID_RUN" --resume "$SAID_PARENT" \
  --expect-model-sha e92232b78511c53612a4eec9ebc7fea1302f60c397eba53c81fba0b9c12ff630 > "$SAID_RUN/train.console.log" 2>&1
code=$?
printf '%s\n' "$code" > "$SAID_RUN/train.exitcode"
exit "$code"
