#!/bin/bash
# S0-TriMask v0.1 -- one 500-step run, then the frozen COCO canonical and Urban-1k protocols on the
# bare student's native CLS/EOS.
#
#   bash tools/exp_said_trimask.sh acceptance        # real-model pre-flight (no training)
#   bash tools/exp_said_trimask.sh train             # the single authorised 500-step run
#   bash tools/exp_said_trimask.sh export [STEP]     # bare student + strict load
#   bash tools/exp_said_trimask.sh eval   [STEP]     # COCO canonical
#   bash tools/exp_said_trimask.sh urban  [STEP]     # Urban-1k
#   bash tools/exp_said_trimask.sh status
#
# Every shared asset is referenced by ABSOLUTE path: this single-purpose worktree does not contain
# the untracked runs_salu/ tree, the frozen manifests or the shared initialisation.
set -u
REPO=/root/SAID-s0-trimask-v01
cd "$REPO" || exit 1
if [ ! -f "$REPO/train/train_said_trimask.py" ]; then
  echo "FATAL: unexpected worktree $REPO"; exit 1
fi

ARM=S0_TriMask
OBJECTIVE=smartclip_trimask
INIT=/root/SAID-gap-completion/runs_salu/said_cls_cvssl/shared_init/cvssl_initial.pt
OUT=${OUT_DIR:-$REPO/runs_salu/said_s0_trimask_v01/step500}
PY=/root/miniconda3/envs/said-smartclip/bin/python
CONDA=/root/miniconda3/bin/conda

export SHARE4V_DATA_ROOT=/root/datasets/ShareGPT4V
export SHARE4V_JSON=share-captioner_coco_lcs_sam_1246k_1107.json
export SHARE4V_FULL_AUDIT=/root/SAID/outputs/data_audit/sharegpt4v_full_audit.json
export COCO_DATA_ROOT=/root/datasets/coco
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STEP=${2:-500}
TAG=$(printf '%06d' "$STEP")

case "${1:-status}" in
  acceptance)
    "$PY" "$REPO/tools/diag/trimask_acceptance.py" \
      --init_state "$INIT" --out "$REPO/runs_salu/acceptance.json" \
      --checkpoint_dir "$REPO/runs_salu/acceptance"
    ;;

  train)
    if [ ! -f "$INIT" ]; then echo "MISSING shared init $INIT"; exit 1; fi
    if [ -d "$OUT" ]; then echo "REFUSING to reuse existing dir $OUT"; exit 1; fi
    busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
    if [ -n "$busy" ]; then echo "REFUSING: GPUs are not idle: $busy"; exit 3; fi
    mkdir -p "$OUT"
    port=$(( 35200 + RANDOM % 500 ))
    echo "=== TRAIN $ARM objective=$OBJECTIVE steps=$STEP dir=$OUT port=$port ==="
    echo "    init=$INIT"
    date -Is
    # setsid + stdin/stdout/stderr all detached from the caller: the launch returns immediately and
    # the run survives the ssh channel closing (the bridge that drives this worktree is not an
    # interactive login shell)
    setsid nohup $CONDA run --no-capture-output -n said-smartclip \
      torchrun --nproc_per_node=4 --master_port="$port" "$REPO/train/train_said_trimask.py" \
      --arm "$ARM" --objective "$OBJECTIVE" \
      --lambda_1 10 --lambda_2 1 --lambda_3 1 --lambda_sparse_i 2 \
      --text_mask_width 512 --text_mask_layers 1 --text_mask_heads 8 --text_mask_seed 0 \
      --base_model B16 --batch-size 256 --epochs 3 \
      --lr 1e-6 --mask_lr 1e-3 --weight_decay 1e-2 --warmup_length 200 \
      --seed 0 --init_state "$INIT" --output_dir "$OUT" \
      --max_steps "$STEP" --save_completed_steps "0,20,100,250,$STEP" \
      --log_every 25 --grad_health_steps "1,20,100,250,$STEP" \
      --grad_checkpoint_views 1 --num_workers 8 --amp_dtype bf16 \
      > "$OUT/train.log" 2>&1 < /dev/null &
    echo "PID $! log=$OUT/train.log"
    ;;

  export)
    CK=$OUT/trimask_${ARM}_step${TAG}.pt
    [ -f "$CK" ] || { echo "MISSING $CK"; exit 1; }
    "$PY" "$REPO/tools/diag/export_trimask_student.py" \
      --checkpoint "$CK" --out "$OUT/student_${TAG}.pt" --expect-steps "$STEP"
    ;;

  eval)
    ST=$OUT/student_${TAG}.pt
    [ -f "$ST" ] || { echo "MISSING $ST (run export first)"; exit 1; }
    mkdir -p "$OUT/evaluation"
    export CUDA_VISIBLE_DEVICES=0
    echo "=== EVAL COCO canonical $ARM@$STEP ==="; date -Is
    "$PY" "$REPO/tools/phase30a_fixed_cohort_eval.py" \
      --label trimask --gap_anti_temperature 1.0 \
      --sharegpt4v_manifest '' \
      --data_root "$SHARE4V_DATA_ROOT" --image_root "$SHARE4V_DATA_ROOT" \
      --image_batch_size 64 --canonical --canonical_only --coco \
      --canonical_tags "$STEP" --canonical_names "$ARM@$STEP" \
      --checkpoints "$STEP:$ST" \
      --output "$OUT/evaluation/${ARM}_step${TAG}_canonical.json"
    status=$?
    echo "EVAL_EXIT exit=$status"; date -Is
    exit $status
    ;;

  urban)
    ST=$OUT/student_${TAG}.pt
    [ -f "$ST" ] || { echo "MISSING $ST (run export first)"; exit 1; }
    mkdir -p "$OUT/evaluation"
    export CUDA_VISIBLE_DEVICES=0
    echo "=== EVAL Urban-1k $ARM@$STEP ==="; date -Is
    "$PY" /root/SAID-gap-completion/tools/eval_urban1k_cls.py \
      --checkpoint "$ST" --label "$ARM@$STEP" --expect-steps "$STEP" \
      --base_model ViT-B/16 --device cuda --batch_size 64 \
      --urban_root /root/datasets/Urban1k/Urban1k \
      --out "$OUT/evaluation/${ARM}_step${TAG}_urban1k.json"
    status=$?
    echo "URBAN_EXIT exit=$status"; date -Is
    exit $status
    ;;

  status)
    echo "== GPU =="; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
    echo "== procs =="; ps -eo pid,etime,args --width 3000 | grep -E '[t]rain_said_trimask|[t]orchrun' | head -8
    echo "== log tail =="; tail -3 "$OUT/train.log" 2>/dev/null
    echo "== summary =="; cat "$OUT/run_summary.json" 2>/dev/null
    ;;

  *)
    echo "usage: $0 acceptance | train | export [step] | eval [step] | urban [step] | status"
    exit 1 ;;
esac
