#!/bin/bash
# C0 limited tuning launcher -- one candidate per invocation, C0_complement_vssl only.
#
#   bash tools/exp_said_cls_cvssl_c0_tuning.sh train <run_tag> <lambda_U> <tau_U> [u_warmup]
#   bash tools/exp_said_cls_cvssl_c0_tuning.sh eval  <run_tag>
#
# Fixed conditions are identical to the S0/C0 screening runs (tools/exp_said_cls_cvssl_s0c0.sh):
# same shared init, same data, same views, same optimizers, same 3 x len(loader) LR horizon,
# real DDP with gradient averaging, fp32 master + bf16 autocast, non-reentrant checkpointing.
# Only lambda_U / tau_U / the U-weight schedule differ between candidates.
set -u
cd /root/SAID-gap-completion || exit 1

MODE=${1:-}
TAG=${2:-}
LAMBDA=${3:-}
TAU=${4:-0.1}
UWARM=${5:-0}

ARM=C0_complement_vssl
OUT=runs_salu/said_cls_cvssl/c0_tuning
INIT=${INIT_STATE:-runs_salu/said_cls_cvssl/shared_init/cvssl_initial.pt}
STEPS=500

export SHARE4V_DATA_ROOT=/root/datasets/ShareGPT4V
export SHARE4V_JSON=share-captioner_coco_lcs_sam_1246k_1107.json
export SHARE4V_FULL_AUDIT=/root/SAID/outputs/data_audit/sharegpt4v_full_audit.json
export COCO_DATA_ROOT=/root/datasets/coco
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

case "$MODE" in
  train)
    [ -n "$TAG" ] && [ -n "$LAMBDA" ] || { echo "usage: train <run_tag> <lambda_U> [tau_U] [u_warmup]"; exit 1; }
    DIR="$OUT/$TAG"
    if [ -d "$DIR" ]; then echo "REFUSING to reuse existing dir $DIR"; exit 1; fi
    mkdir -p "$DIR"
    port=$(( 32200 + RANDOM % 600 ))
    echo "=== TRAIN tag=$TAG arm=$ARM lambda_U=$LAMBDA tau_U=$TAU u_weight_warmup=$UWARM ==="
    echo "    init=$INIT  dir=$DIR  port=$port  steps=$STEPS"
    date -Is
    nohup /root/miniconda3/bin/conda run --no-capture-output -n said-smartclip \
      torchrun --nproc_per_node=4 --master_port="$port" train/train_said_cls_cvssl.py \
      --arm "$ARM" --lambda_U "$LAMBDA" --tau_U "$TAU" --rho 0.0 --soft_mask 0 \
      --base_model B16 --batch-size 256 --epochs 3 \
      --lr 1e-6 --mask_lr 1e-3 --weight_decay 1e-2 --warmup_length 200 \
      --lambda_align 10 --lambda_sparse 2 \
      --seed 0 --init_state "$INIT" \
      --output_dir "$DIR" \
      --max_steps "$STEPS" --save_completed_steps "100,250,$STEPS" \
      --u_weight_warmup_steps "$UWARM" \
      --log_every 25 --grad_probe_steps "" --num_workers 8 --amp_dtype bf16 \
      > "$DIR/train.log" 2>&1 &
    echo "PID $! log=$DIR/train.log"
    ;;
  eval)
    [ -n "$TAG" ] || { echo "usage: eval <run_tag>"; exit 1; }
    DIR="$OUT/$TAG"
    CKPT="$DIR/cvssl_${ARM}_step$(printf '%06d' "$STEPS").pt"
    [ -f "$CKPT" ] || { echo "MISSING $CKPT"; exit 1; }
    export CUDA_VISIBLE_DEVICES=0
    W=/root/SAID-gap-completion
    # the evaluator runs from /root/SAID (it resolves the frozen manifests relatively), so every
    # checkpoint path handed to it must be absolute
    CKPT=$W/$CKPT
    EOUT=$W/outputs/cvssl_screening/c0_tuning
    mkdir -p "$EOUT"
    cd /root/SAID || exit 1
    echo "=== EVAL tag=$TAG ckpt=$CKPT ==="
    date -Is
    /root/miniconda3/bin/conda run --no-capture-output -n said-smartclip \
      python "$W/tools/phase30a_fixed_cohort_eval.py" \
      --label "c0tune_$TAG" --gap_anti_temperature 1.0 \
      --usr_manifest outputs/validation/sharegpt4v1k_usr_manifest.json \
      --source_manifest outputs/validation/sharegpt4v1k_manifest.json \
      --sharegpt4v_manifest outputs/validation/sharegpt4v1k_manifest.json \
      --data_root "$SHARE4V_DATA_ROOT" --image_root "$SHARE4V_DATA_ROOT" \
      --image_batch_size 64 --canonical --canonical_only --coco \
      --canonical_tags "step500" --canonical_names "${TAG}_step500" \
      --checkpoints "step500:$CKPT" \
      --output "$EOUT/${TAG}_canonical.json"
    status=$?
    echo "EVAL_EXIT tag=$TAG exit=$status"
    date -Is
    exit $status
    ;;
  *)
    echo "usage: $0 train <run_tag> <lambda_U> [tau_U] [u_warmup] | eval <run_tag>"
    exit 1
    ;;
esac
