#!/usr/bin/env bash
set -euo pipefail
source /root/lk_projects/SAID-reproduction/run_env.sh
export MKL_NUM_THREADS=1
unset PYTORCH_CUDA_ALLOC_CONF
cd /root/lk_projects/SAID-reproduction/full-resume
run_dir=/root/lk_projects/SAID-suffix3-3epoch/run
parent=/root/lk_projects/SAID-tuning-500/runs/suffix3_sparse0/s0_dual_mask_suffix_masked_step000500.pt
test ! -e "$run_dir"
mkdir -p "$run_dir"
set +e
"$REPRO_PYTHON" -m torch.distributed.run --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29645 train/train_dual_mask_suffix.py \
 --suffix-mode masked --run-type formal --base-model B16 --batch-size 256 --epochs 3 --max-steps 3651 \
 --seed 0 --num-workers 8 --lr 1e-6 --mask-lr 1e-3 --suffix-lr 1e-4 --weight-decay 0.01 --warmup 200 \
 --total-len 1000 --amp-dtype bf16 --image-chunk 16 --text-chunk 32 --save-every 500 \
 --lambda-suffix 3 --lambda-u-sparse 0 --init-state "$REPRO_INIT" --output-dir "$run_dir" --resume "$parent" \
 --expect-model-sha e92232b78511c53612a4eec9ebc7fea1302f60c397eba53c81fba0b9c12ff630 > "$run_dir/console.log" 2>&1
code=$?
printf '%s\n' "$code" > "$run_dir/exitcode"
exit "$code"
