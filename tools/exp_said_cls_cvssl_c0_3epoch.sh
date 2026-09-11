#!/bin/bash
# SAID-CLS-CVSSL -- 3-epoch run of the current best C0 (lambda_U = 0.3), then canonical retrieval.
#
#   bash tools/exp_said_cls_cvssl_c0_3epoch.sh check   # 20-step online check (short, separate dir)
#   bash tools/exp_said_cls_cvssl_c0_3epoch.sh train   # full 3 epochs = 3 * 1217 = 3651... (see note)
#   bash tools/exp_said_cls_cvssl_c0_3epoch.sh eval <which>   # which = ep1|ep2|ep3
#
# Total steps: measured from the 500-step run's own log, len(loader) = steps_per_epoch = 1217 is the
# FULL epoch -- the DistributedSampler already divides the 1,245,901 samples across the 4 ranks, so
# one epoch is 1217 optimizer steps of 1024 global pairs. The loop is range(epochs) x enumerate(
# loader), therefore 3 epochs = 3 * 1217 = 3651 steps, which is exactly lr_horizon_steps = 3 *
# len(loader) = 3651: the cosine schedule spans the whole run. The last batch of each epoch is ragged
# (ragged-size batches share the same size across ranks, which the trainer supports).
set -u
cd /root/SAID-gap-completion || exit 1

MODE=${1:-}
WHICH=${2:-}

ARM=C0_complement_vssl
LAMBDA=0.3
TAU=0.1
EPOCHS=3
INIT=${INIT_STATE:-runs_salu/said_cls_cvssl/shared_init/cvssl_initial.pt}
OUT=runs_salu/said_cls_cvssl/c0_3epoch/$( [ "$MODE" = check ] && echo check_L03 || echo L03 )

export SHARE4V_DATA_ROOT=/root/datasets/ShareGPT4V
export SHARE4V_JSON=share-captioner_coco_lcs_sam_1246k_1107.json
export SHARE4V_FULL_AUDIT=/root/SAID/outputs/data_audit/sharegpt4v_full_audit.json
export COCO_DATA_ROOT=/root/datasets/coco
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

case "$MODE" in
  check)
    mkdir -p "$OUT"
    port=$(( 33400 + RANDOM % 400 ))
    echo "=== CHECK 20 steps arm=$ARM lambda_U=$LAMBDA dir=$OUT ==="
    date -Is
    nohup /root/miniconda3/bin/conda run --no-capture-output -n said-smartclip \
      torchrun --nproc_per_node=4 --master_port="$port" train/train_said_cls_cvssl.py \
      --arm "$ARM" --lambda_U "$LAMBDA" --tau_U "$TAU" --rho 0.0 --soft_mask 0 \
      --base_model B16 --batch-size 256 --epochs "$EPOCHS" \
      --lr 1e-6 --mask_lr 1e-3 --weight_decay 1e-2 --warmup_length 200 \
      --lambda_align 10 --lambda_sparse 2 \
      --seed 0 --init_state "$INIT" --output_dir "$OUT" \
      --max_steps 20 --save_completed_steps 20 \
      --u_weight_warmup_steps 0 \
      --log_every 5 --grad_probe_steps "" --num_workers 8 --amp_dtype bf16 \
      > "$OUT/train.log" 2>&1 &
    echo "PID $! log=$OUT/train.log"
    ;;
  train)
    if [ -d "$OUT" ]; then echo "REFUSING to reuse existing dir $OUT"; exit 1; fi
    mkdir -p "$OUT"
    port=$(( 33400 + RANDOM % 400 ))
    echo "=== TRAIN 3 epochs arm=$ARM lambda_U=$LAMBDA tau_U=$TAU epochs=$EPOCHS ==="
    echo "    dir=$OUT port=$port  (no --max_steps: the full 3-epoch pass runs to the end)"
    date -Is
    nohup /root/miniconda3/bin/conda run --no-capture-output -n said-smartclip \
      torchrun --nproc_per_node=4 --master_port="$port" train/train_said_cls_cvssl.py \
      --arm "$ARM" --lambda_U "$LAMBDA" --tau_U "$TAU" --rho 0.0 --soft_mask 0 \
      --base_model B16 --batch-size 256 --epochs "$EPOCHS" \
      --lr 1e-6 --mask_lr 1e-3 --weight_decay 1e-2 --warmup_length 200 \
      --lambda_align 10 --lambda_sparse 2 \
      --seed 0 --init_state "$INIT" --output_dir "$OUT" \
      --save_completed_steps "1217,2434,3651" \
      --u_weight_warmup_steps 0 \
      --log_every 200 --grad_probe_steps "" --num_workers 8 --amp_dtype bf16 \
      > "$OUT/train.log" 2>&1 &
    echo "PID $! log=$OUT/train.log"
    ;;
  eval)
    case "$WHICH" in
      ep1) STEP=1217 ;; ep2) STEP=2434 ;; ep3) STEP=3651 ;;
      *) echo "usage: eval ep1|ep2|ep3"; exit 1 ;;
    esac
    W=/root/SAID-gap-completion
    BASE=$W/runs_salu/said_cls_cvssl/c0_3epoch/L03
    CKPT=$(ls "$BASE"/cvssl_${ARM}_step*.pt 2>/dev/null | grep -E "step0*${STEP}\.pt$" | head -1)
    if [ -z "$CKPT" ]; then
      CKPT=$(ls "$BASE"/cvssl_${ARM}_step*.pt 2>/dev/null | tail -1)
      echo "WARN: exact step $STEP not found; using last available checkpoint $CKPT"
    fi
    [ -n "$CKPT" ] || { echo "no checkpoint in $BASE"; exit 1; }
    EOUT=$W/outputs/cvssl_screening/c0_3epoch
    mkdir -p "$EOUT"
    export CUDA_VISIBLE_DEVICES=0
    cd /root/SAID || exit 1
    echo "=== EVAL $WHICH ckpt=$CKPT ==="
    date -Is
    /root/miniconda3/bin/conda run --no-capture-output -n said-smartclip \
      python "$W/tools/phase30a_fixed_cohort_eval.py" \
      --label "c0_3ep_$WHICH" --gap_anti_temperature 1.0 \
      --usr_manifest outputs/validation/sharegpt4v1k_usr_manifest.json \
      --source_manifest outputs/validation/sharegpt4v1k_manifest.json \
      --sharegpt4v_manifest outputs/validation/sharegpt4v1k_manifest.json \
      --data_root "$SHARE4V_DATA_ROOT" --image_root "$SHARE4V_DATA_ROOT" \
      --image_batch_size 64 --canonical --canonical_only --coco \
      --canonical_tags "$WHICH" --canonical_names "C0_L03_3ep_$WHICH" \
      --checkpoints "$WHICH:$CKPT" \
      --output "$EOUT/L03_${WHICH}_canonical.json"
    echo "EVAL_EXIT $WHICH exit=$?"
    date -Is
    ;;
  *)
    echo "usage: $0 check | train | eval ep1|ep2|ep3"
    exit 1
    ;;
esac
