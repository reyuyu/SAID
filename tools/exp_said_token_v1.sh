#!/bin/bash
# SAID-Token v1 (experiment T1) -- one 500-step run, then canonical native-CLS retrieval.
#
#   bash tools/exp_said_token_v1.sh check             # are the four A800s idle?
#   bash tools/exp_said_token_v1.sh train             # the single 500-step T1 run
#   bash tools/exp_said_token_v1.sh eval              # canonical COCO retrieval of T1@500
#   bash tools/exp_said_token_v1.sh urban             # Urban-1k retrieval of T1@500
#   bash tools/exp_said_token_v1.sh diag <step>       # re-print the fixed-cohort diagnostics
#
# Worktree: /root/SAID-token-v1   Branch: codex/said-token-v1
# Shared assets are referenced by ABSOLUTE path: this worktree does not contain the untracked
# runs_salu/ tree, the frozen evaluation manifests, or the committed Urban-1k tooling of the C0
# worktree (single source of truth for that dataset, so it is called there instead of copied here).
set -u
REPO=/root/SAID-token-v1
cd "$REPO" || exit 1
if [ ! -f "$REPO/train/train_said_token_v1.py" ]; then
  echo "FATAL: unexpected worktree $REPO"; exit 1
fi

ARM=T1_said_token_reconstruction
INIT=/root/SAID-gap-completion/runs_salu/said_cls_cvssl/shared_init/cvssl_initial.pt
OUT=${OUT_DIR:-$REPO/runs_salu/said_token_v1/t1_500step}
STEPS=${STEPS:-500}
C0=/root/SAID-gap-completion
MAIN=/root/SAID

export SHARE4V_DATA_ROOT=/root/datasets/ShareGPT4V
export SHARE4V_JSON=share-captioner_coco_lcs_sam_1246k_1107.json
export SHARE4V_FULL_AUDIT=/root/SAID/outputs/data_audit/sharegpt4v_full_audit.json
export COCO_DATA_ROOT=/root/datasets/coco
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

case "${1:-check}" in
  check)
    echo "=== GPU state (the T1 run needs four idle A800s) ==="
    nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
    busy=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | wc -l)
    echo "compute processes: $busy"
    [ "$busy" -eq 0 ] && echo "IDLE_OK" || echo "NOT_IDLE"
    ;;
  train)
    if [ ! -f "$INIT" ]; then echo "MISSING shared init $INIT"; exit 1; fi
    if [ -d "$OUT" ]; then echo "REFUSING to reuse existing dir $OUT"; exit 1; fi
    busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | wc -l)
    if [ "$busy" -ne 0 ]; then echo "REFUSING to launch: $busy compute process(es) already on the GPUs"; exit 1; fi
    mkdir -p "$OUT"
    port=$(( 34600 + RANDOM % 400 ))
    echo "=== TRAIN T1 arm=$ARM steps=$STEPS dir=$OUT port=$port ==="
    echo "    init=$INIT"
    date -Is
    nohup /root/miniconda3/bin/conda run --no-capture-output -n said-smartclip \
      torchrun --nproc_per_node=4 --master_port="$port" "$REPO/train/train_said_token_v1.py" \
      --arm "$ARM" --objective said_token --lambda_rec 0.1 \
      --n_slots 32 --k_said 16 --router_tau 0.07 --router_dim 128 --surrogate_eta 0.1 \
      --margin 0.2 --chunk_image 8 --chunk_text 32 --module_seed 0 \
      --base_model B16 --batch-size 256 --epochs 3 \
      --lr 1e-6 --module_lr 2e-4 --module_wd 1e-2 --decoder_lr 1e-4 --decoder_wd 0 \
      --weight_decay 1e-2 --warmup_length 200 \
      --seed 0 --init_state "$INIT" --output_dir "$OUT" \
      --max_steps "$STEPS" --save_completed_steps "0,20,100,$STEPS" \
      --log_every 25 --online_check_steps 20 --diag_every 0 --cohort_size 64 \
      --grad_checkpoint_views 0 --num_workers 8 --amp_dtype bf16 \
      > "$OUT/train.log" 2>&1 &
    echo "PID $! log=$OUT/train.log"
    ;;
  smoke)
    SOUT=$REPO/runs_salu/said_token_v1/cli_smoke
    rm -rf "$SOUT"; mkdir -p "$SOUT"
    export CUDA_VISIBLE_DEVICES=0
    port=$(( 35200 + RANDOM % 200 ))
    echo "=== CLI SMOKE: 2 real steps, one GPU, real data, real save ==="
    date -Is
    /root/miniconda3/bin/conda run --no-capture-output -n said-smartclip \
      torchrun --nproc_per_node=1 --master_port="$port" "$REPO/train/train_said_token_v1.py" \
      --arm "$ARM" --objective said_token --lambda_rec 0.1 \
      --n_slots 32 --k_said 16 --router_tau 0.07 --router_dim 128 --surrogate_eta 0.1 \
      --margin 0.2 --chunk_image 8 --chunk_text 32 --module_seed 0 \
      --base_model B16 --batch-size 8 --epochs 3 \
      --lr 1e-6 --module_lr 2e-4 --module_wd 1e-2 --decoder_lr 1e-4 --decoder_wd 0 \
      --weight_decay 1e-2 --warmup_length 200 \
      --seed 0 --init_state "$INIT" --output_dir "$SOUT" \
      --max_steps 2 --save_completed_steps "0,1,2" \
      --log_every 1 --online_check_steps 2 --diag_every 0 --cohort_size 8 \
      --grad_checkpoint_views 0 --num_workers 2 --amp_dtype bf16 2>&1 | tail -30
    echo "CLI_SMOKE_EXIT ${PIPESTATUS[0]}"
    ls -la "$SOUT"
    ;;
  export)
    CKPT=$OUT/t1_${ARM}_step$(printf '%06d' "$STEPS").pt
    [ -f "$CKPT" ] || { echo "MISSING $CKPT"; exit 1; }
    mkdir -p "$OUT/export"
    /root/miniconda3/bin/conda run --no-capture-output -n said-smartclip \
      python "$REPO/tools/diag/export_t1_student.py" \
      --checkpoint "$CKPT" --expect-steps "$STEPS" \
      --out "$OUT/export/t1_student_step$(printf '%06d' "$STEPS").pt"
    ;;
  eval)
    CKPT=$OUT/export/t1_student_step$(printf '%06d' "$STEPS").pt
    [ -f "$CKPT" ] || { echo "MISSING $CKPT (run the export step first)"; exit 1; }
    EOUT=$REPO/outputs/said_token_v1
    mkdir -p "$EOUT"
    export CUDA_VISIBLE_DEVICES=0
    # the canonical protocol tool travels with this lineage: use this worktree's copy and pass the
    # frozen manifests by ABSOLUTE path (outputs/ is untracked, so they only exist in /root/SAID)
    cd "$REPO" || exit 1
    echo "=== EVAL canonical COCO T1@$STEPS ckpt=$CKPT ==="
    date -Is
    /root/miniconda3/bin/conda run --no-capture-output -n said-smartclip \
      python "$REPO/tools/phase30a_fixed_cohort_eval.py" \
      --label said_token_v1 --gap_anti_temperature 1.0 \
      --usr_manifest "$MAIN/outputs/validation/sharegpt4v1k_usr_manifest.json" \
      --source_manifest "$MAIN/outputs/validation/sharegpt4v1k_manifest.json" \
      --sharegpt4v_manifest "$MAIN/outputs/validation/sharegpt4v1k_manifest.json" \
      --data_root "$SHARE4V_DATA_ROOT" --image_root "$SHARE4V_DATA_ROOT" \
      --image_batch_size 64 --canonical --canonical_only --coco \
      --canonical_tags "step$STEPS" --canonical_names "T1_step$STEPS" \
      --checkpoints "step$STEPS:$CKPT" \
      --output "$EOUT/T1_step${STEPS}_canonical.json"
    status=$?
    echo "EVAL_EXIT exit=$status"
    date -Is
    exit $status
    ;;
  urban)
    CKPT=$OUT/export/t1_student_step$(printf '%06d' "$STEPS").pt
    [ -f "$CKPT" ] || { echo "MISSING $CKPT (run the export step first)"; exit 1; }
    EOUT=$C0/outputs/cvssl_screening/baseline_urban1k
    [ -d "$EOUT" ] || { echo "MISSING the shared Urban-1k result dir $EOUT"; exit 1; }
    export CUDA_VISIBLE_DEVICES=0
    cd "$C0" || exit 1
    echo "=== EVAL Urban-1k T1@$STEPS ckpt=$CKPT ==="
    date -Is
    /root/miniconda3/bin/conda run --no-capture-output -n said-smartclip \
      python "$C0/tools/eval_urban1k_cls.py" \
      --checkpoint "$CKPT" --label "T1_step$STEPS" --expect-steps "$STEPS" \
      --base_model 'ViT-B/16' --device cuda --batch_size 64 \
      --urban_root /root/datasets/Urban1k/Urban1k \
      --out "$EOUT/T1_step${STEPS}_urban1k.json"
    status=$?
    echo "URBAN_EXIT exit=$status"
    date -Is
    exit $status
    ;;
  post)
    # wait for the in-flight run, then do the whole post-processing chain in one background session:
    # export -> canonical COCO -> Urban-1k -> summary table. Each step is skipped loudly on failure.
    echo "=== waiting for the training process to exit ==="
    date -Is
    while pgrep -f train_said_token_v1.py > /dev/null; do sleep 60; done
    date -Is
    echo "log lines: $(wc -l < "$OUT/salu_log.jsonl")"
    tail -1 "$OUT/run_summary.json"
    STEPS=$STEPS bash "$REPO/tools/exp_said_token_v1.sh" export || { echo POST_FAILED_EXPORT; exit 1; }
    STEPS=$STEPS bash "$REPO/tools/exp_said_token_v1.sh" eval || { echo POST_FAILED_EVAL; exit 1; }
    STEPS=$STEPS bash "$REPO/tools/exp_said_token_v1.sh" urban || { echo POST_FAILED_URBAN; exit 1; }
    /root/miniconda3/bin/conda run --no-capture-output -n said-smartclip \
      python "$REPO/tools/diag/t1_summary.py" --retrieval \
      > "$REPO/outputs/said_token_v1/t1_summary_table.txt" 2>&1
    tail -22 "$REPO/outputs/said_token_v1/t1_summary_table.txt"
    echo POST_DONE
    ;;
  diag)
    echo "fixed-cohort diagnostics: keys diag_* at every saved step (0, 20, 100, $STEPS)"
    grep -o '"diag_[a-z_0-9]*": [-0-9.eE+]*' "$OUT/salu_log.jsonl" 2>/dev/null | tail -40
    ;;
  *)
    echo "usage: $0 check | train | export | eval | urban | diag"; exit 1 ;;
esac
