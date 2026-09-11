#!/bin/bash
# SAID-CLS-CVSSL v0.1 -- 20-step engineering smoke for all four pre-registered arms.
#
#   bash tools/exp_said_cls_cvssl_smoke.sh [steps] [tag]
#
#   S0_smartclip        lambda_U = 0   (complement computed for diagnostics only)
#   G0_global_vssl      lambda_U = 1   (all-ones mask)
#   R0_random_vssl      lambda_U = 1   (coordinate-permuted complement)
#   C0_complement_vssl  lambda_U = 1   (detached complement of the Said mask)
#
# ViT-B/16, LongCLIP 248, full ShareGPT4V, 4x A800, 256 pairs/GPU (global 1024 pairs, 2048 image
# views), seed 0, fp32 master + bf16. The LR horizon is the FULL 3-epoch horizon
# (3 * len(loader)); --max_steps only truncates this run.
set -u
STEPS=${1:-20}
TAG=${2:-smoke}
cd /root/SAID-gap-completion || exit 1
export SHARE4V_DATA_ROOT=/root/datasets/ShareGPT4V
export SHARE4V_JSON=share-captioner_coco_lcs_sam_1246k_1107.json
export SHARE4V_FULL_AUDIT=/root/SAID/outputs/data_audit/sharegpt4v_full_audit.json
export COCO_DATA_ROOT=/root/datasets/coco
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
OUT=runs_salu/said_cls_cvssl
mkdir -p "$OUT"

INIT=${INIT_STATE:-$OUT/shared_init/cvssl_initial.pt}
if [ ! -f "$INIT" ]; then
  echo "MISSING shared init $INIT -- run tools/exp_said_cls_cvssl_init.sh first"
  exit 1
fi

run_arm () {
  local arm="$1"; local lambda_u="$2"
  local port=$((31100 + RANDOM % 500))
  echo "=== arm=$arm lambda_U=$lambda_u steps=$STEPS tag=$TAG port=$port ==="
  date -Is
  /root/miniconda3/bin/conda run --no-capture-output -n said-smartclip \
    torchrun --nproc_per_node=4 --master_port="$port" train/train_said_cls_cvssl.py \
    --arm "$arm" --lambda_U "$lambda_u" --tau_U 0.1 --rho 0.0 --soft_mask 0 \
    --base_model B16 --batch-size 256 --epochs 3 \
    --lr 1e-6 --mask_lr 1e-3 --weight_decay 1e-2 --warmup_length 200 \
    --lambda_align 10 --lambda_sparse 2 \
    --seed 0 --init_state "$INIT" \
    --output_dir "$OUT/${TAG}_${arm}" \
    --max_steps "$STEPS" --save_completed_steps "0" \
    --log_every 5 --grad_probe_steps "" --num_workers 8 --amp_dtype bf16
  echo "ARM_EXIT arm=$arm lambda_U=$lambda_u exit=$?"
}

run_arm S0_smartclip 0.0
run_arm G0_global_vssl 1.0
run_arm R0_random_vssl 1.0
run_arm C0_complement_vssl 1.0
echo "ALL_CVSSL_SMOKE_DONE tag=$TAG"
date -Is
