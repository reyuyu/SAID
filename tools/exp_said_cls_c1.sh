#!/bin/bash
# SAID-C1-TCR v0.1 -- one 500-step run, then canonical retrieval on the student CLS.
#
#   bash tools/exp_said_cls_c1.sh train              # the single 500-step C1 run
#   bash tools/exp_said_cls_c1.sh eval               # canonical retrieval of C1@500
#   bash tools/exp_said_cls_c1.sh diag <step>        # re-print the input-dependency diagnostics
#
# Worktree: /root/SAID-c1-tcr   Branch: codex/said-c1-tcr-v01
# Shared assets are referenced by ABSOLUTE path: this worktree does not contain the untracked
# runs_salu/ tree or the frozen evaluation manifests.
set -u
# this launcher lives in a single-purpose worktree and already uses absolute paths for the shared
# init, so the repository root is pinned rather than derived from the caller's cwd
REPO=/root/SAID-c1-tcr
cd "$REPO" || exit 1
if [ ! -f "$REPO/train/train_said_cls_c1.py" ]; then
  echo "FATAL: unexpected worktree $REPO"; exit 1
fi

ARM=C1_text_conditional_reconstruction
INIT=/root/SAID-gap-completion/runs_salu/said_cls_cvssl/shared_init/cvssl_initial.pt
OUT=${OUT_DIR:-$REPO/runs_salu/said_cls_c1}
STEPS=${STEPS:-500}

export SHARE4V_DATA_ROOT=/root/datasets/ShareGPT4V
export SHARE4V_JSON=share-captioner_coco_lcs_sam_1246k_1107.json
export SHARE4V_FULL_AUDIT=/root/SAID/outputs/data_audit/sharegpt4v_full_audit.json
export COCO_DATA_ROOT=/root/datasets/coco
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

case "${1:-train}" in
  train)
    if [ ! -f "$INIT" ]; then echo "MISSING shared init $INIT"; exit 1; fi
    if [ -d "$OUT" ]; then echo "REFUSING to reuse existing dir $OUT"; exit 1; fi
    mkdir -p "$OUT"
    port=$(( 34100 + RANDOM % 500 ))
    echo "=== TRAIN C1 arm=$ARM steps=$STEPS dir=$OUT port=$port ==="
    echo "    init=$INIT"
    date -Is
    nohup /root/miniconda3/bin/conda run --no-capture-output -n said-smartclip \
      torchrun --nproc_per_node=4 --master_port="$port" "$REPO/train/train_said_cls_c1.py" \
      --arm "$ARM" --objective said_cls_tcr \
      --lambda_rec 1.0 --lambda_align 10 --lambda_sparse 2 \
      --decoder_hidden 512 --decoder_lr 1e-4 --decoder_wd 0 --decoder_seed 0 \
      --duplicate_policy exclude \
      --base_model B16 --batch-size 256 --epochs 3 \
      --lr 1e-6 --mask_lr 1e-3 --weight_decay 1e-2 --warmup_length 200 \
      --seed 0 --init_state "$INIT" --output_dir "$OUT" \
      --max_steps "$STEPS" --save_completed_steps "0,20,100,$STEPS" \
      --log_every 25 --diag_every 100 \
      --grad_checkpoint_views 1 --num_workers 8 --amp_dtype bf16 \
      > "$OUT/train.log" 2>&1 &
    echo "PID $! log=$OUT/train.log"
    ;;
  eval)
    W=$REPO
    CKPT=$OUT/c1_${ARM}_step$(printf '%06d' "$STEPS").pt
    [ -f "$CKPT" ] || { echo "MISSING $CKPT"; exit 1; }
    EOUT=$W/outputs/cvssl_screening/c1_tcr
    mkdir -p "$EOUT"
    export CUDA_VISIBLE_DEVICES=0
    cd /root/SAID || exit 1
    echo "=== EVAL C1@$STEPS ckpt=$CKPT ==="
    date -Is
    /root/miniconda3/bin/conda run --no-capture-output -n said-smartclip \
      python "$W/tools/phase30a_fixed_cohort_eval.py" \
      --label c1_tcr --gap_anti_temperature 1.0 \
      --usr_manifest outputs/validation/sharegpt4v1k_usr_manifest.json \
      --source_manifest outputs/validation/sharegpt4v1k_manifest.json \
      --sharegpt4v_manifest outputs/validation/sharegpt4v1k_manifest.json \
      --data_root "$SHARE4V_DATA_ROOT" --image_root "$SHARE4V_DATA_ROOT" \
      --image_batch_size 64 --canonical --canonical_only --coco \
      --canonical_tags "step$STEPS" --canonical_names "C1_step$STEPS" \
      --checkpoints "step$STEPS:$CKPT" \
      --output "$EOUT/C1_step${STEPS}_canonical.json"
    status=$?
    echo "EVAL_EXIT exit=$status"
    date -Is
    exit $status
    ;;
  diag)
    echo "input-dependency diagnostics are logged at every --diag_every and at the saved steps;"
    echo "read them from $OUT/salu_log.jsonl (keys diag_*)"
    grep -o '"diag_[a-z_0-9]*": [0-9.eE+-]*' "$OUT/salu_log.jsonl" 2>/dev/null | tail -32
    ;;
  *)
    echo "usage: $0 train | eval | diag"; exit 1 ;;
esac
