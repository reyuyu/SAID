#!/bin/bash
# SAID-CLS-CVSSL v0.1 -- S0 / C0 500-step screening on the shared init.
#
#   bash tools/exp_said_cls_cvssl_s0c0.sh preflight        # init digest + dataset length + horizon
#   bash tools/exp_said_cls_cvssl_s0c0.sh check <arm>      # 20-step online check (run 1)
#   bash tools/exp_said_cls_cvssl_s0c0.sh full <arm>       # 500-step run (run 2)
#
# Only S0_smartclip (lambda_U=0) and C0_complement_vssl (lambda_U=1) are ever started here.
# G0/R0 are deliberately absent from this script.
set -u
cd /root/SAID-gap-completion || exit 1

ARM=${2:-}
TAG_PREFIX=ddpfix_step
STEPS_FULL=500
STEPS_CHECK=20
OUT=runs_salu/said_cls_cvssl
INIT=${INIT_STATE:-$OUT/shared_init/cvssl_initial.pt}

export SHARE4V_DATA_ROOT=/root/datasets/ShareGPT4V
export SHARE4V_JSON=share-captioner_coco_lcs_sam_1246k_1107.json
export SHARE4V_FULL_AUDIT=/root/SAID/outputs/data_audit/sharegpt4v_full_audit.json
export COCO_DATA_ROOT=/root/datasets/coco
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/root/miniconda3/envs/said-smartclip/bin/python

arm_config () {
  case "$1" in
    S0_smartclip)       LAMBDA=0.0 ;;
    C0_complement_vssl) LAMBDA=1.0 ;;
    *) echo "unknown arm $1"; return 1 ;;
  esac
  return 0
}

launch () {                                # $1 arm, $2 steps, $3 tag, $4 save steps
  local arm="$1" steps="$2" tag="$3" saves="$4"
  local port=$(( 31300 + RANDOM % 600 ))
  local dir="$OUT/${tag}_${arm}"
  mkdir -p "$dir"
  echo "=== LAUNCH arm=$arm lambda_U=$LAMBDA steps=$steps tag=$tag port=$port dir=$dir ==="
  date -Is
  nohup /root/miniconda3/bin/conda run --no-capture-output -n said-smartclip \
    torchrun --nproc_per_node=4 --master_port="$port" train/train_said_cls_cvssl.py \
    --arm "$arm" --lambda_U "$LAMBDA" --tau_U 0.1 --rho 0.0 --soft_mask 0 \
    --base_model B16 --batch-size 256 --epochs 3 \
    --lr 1e-6 --mask_lr 1e-3 --weight_decay 1e-2 --warmup_length 200 \
    --lambda_align 10 --lambda_sparse 2 \
    --seed 0 --init_state "$INIT" \
    --output_dir "$dir" \
    --max_steps "$steps" --save_completed_steps "$saves" \
    --log_every 5 --grad_probe_steps "" --num_workers 8 --amp_dtype bf16 \
    > "$dir/train.log" 2>&1 &
  echo "PID $! log=$dir/train.log"
}

case "$1" in
  preflight)
    arm_config S0_smartclip
    echo "init=$INIT"
    ls -l "$INIT"
    sha256sum "$INIT"
    $PY tools/diag/check_cvssl_init.py "$INIT" || exit 1
    $PY tools/diag/check_cvssl_horizon.py || exit 1
    ;;
  check)
    arm_config "$ARM" || exit 1
    launch "$ARM" "$STEPS_CHECK" "ddpcheck_step20" "0,$STEPS_CHECK"
    ;;
  full)
    arm_config "$ARM" || exit 1
    launch "$ARM" "$STEPS_FULL" "$TAG_PREFIX$STEPS_FULL" "0,20,100,$STEPS_FULL"
    ;;
  *)
    echo "usage: $0 preflight | check <arm> | full <arm>"
    exit 1
    ;;
esac
