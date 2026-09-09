#!/usr/bin/env bash
set -euo pipefail
: "${SHARE4V_DATA_ROOT:?Set the existing ShareGPT4V root}"
: "${SHARE4V_JSON:?Set the unchanged no-SAM JSON}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo OMP_NUM_THREADS=4
experiment_root=${LOCAL_ROUTER_RUN_ROOT:-runs_salu/phase25}
for arm in residual attention_delta; do
    run_dir="$experiment_root/$arm"
    if [ -e "$run_dir/salu_log.jsonl" ]; then
        echo "Refusing to overwrite existing arm: $run_dir" >&2
        exit 1
    fi
    mkdir -p "$run_dir"
    torchrun --nproc_per_node=4 --master_port=25966 train/train_salu.py \
        --base_model B16 --seed 25 --said_feature_source "$arm" \
        --batch_size 256 --epochs 1 --backbone_lr 1e-6 --head_lr 1e-4 \
        --weight_decay 1e-2 --warmup_length 200 --lambda_global 1.0 \
        --lambda_said 1.0 --tau_said 0.07 --said_loss_mode identifiable \
        --pair_chunk_size 64 --amp_dtype bf16 --num_workers 8 \
        --save_initial --save_at 100,200,400 --save_every 0 --log_every 10 \
        --output_dir "$run_dir" > "$run_dir/train.log" 2>&1
done
