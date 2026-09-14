set -euo pipefail
run_dir=/root/SAID-s0-dualmask-clean-v01/runs_salu/dualmask_full_local_step2000_eval_20260914
py=/root/miniconda3/envs/said-smartclip/bin/python
cd "$run_dir"
test ! -e started_at.txt
printf '%s  %s\n' '54f5c8b601c237e883a79cb82cb85865e3e0ff27e7399a189eb95c21361faf15' 'bare_student_step2000.pt' | sha256sum -c -
for gpu in 0 1 2 3; do
  memory=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "$memory" -gt 100 ]; then
    printf 'RESOURCE_CONFLICT gpu=%s memory=%s MiB\n' "$gpu" "$memory"
    exit 40
  fi
done
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv > gpu_before.csv
sha256sum eval_extended_real.py /root/SAID-gap-completion/tools/eval_urban1k_cls.py /root/SAID-gap-completion/model/longclip.py /root/datasets/retrieval_benchmarks/manifests/*.jsonl > evaluation_files.sha256
date -Is > started_at.txt
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
"$py" "$run_dir/verify_checkpoint.py" > checkpoint_verification.log 2>&1
eval_one() {
  name=$1
  gpu=$2
  spec=$3
  start=$(date +%s)
  mkdir -p "$run_dir/$name"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" "$py" "$run_dir/eval_extended_real.py" --checkpoint "$run_dir/bare_student_step2000.pt" --device cuda:0 --batch-size 64 --output-dir "$run_dir/$name" "$spec" > "$run_dir/$name/console.log" 2>&1
  code=$?
  set -e
  printf '%s\n' "$code" > "$run_dir/$name/exitcode.txt"
  printf '%s\n' "$(( $(date +%s) - start ))" > "$run_dir/$name/wall_seconds.txt"
  return "$code"
}
eval_one flickr_test1k 0 'flickr_test1k:/root/datasets/retrieval_benchmarks/manifests/flickr30k_karpathy_test1k_native248_v1.jsonl:/root/datasets/retrieval_benchmarks/flickr30k/images_test1k/images_flickr_1k_test' & p0=$!
eval_one docci 1 'docci:/root/datasets/retrieval_benchmarks/manifests/docci_test.jsonl:/root/datasets/retrieval_benchmarks/docci/images' & p1=$!
eval_one dci 2 'dci:/root/datasets/retrieval_benchmarks/manifests/dci_full.jsonl:/root/datasets/retrieval_benchmarks/dci/images' & p2=$!
eval_one long_dci 3 'long_dci:/root/datasets/retrieval_benchmarks/manifests/long_dci_reconstructed.jsonl:/root/datasets/retrieval_benchmarks/dci/images' & p3=$!
result=0
for p in "$p0" "$p1" "$p2" "$p3"; do
  wait "$p" || result=1
done
date -Is > finished_at.txt
printf '%s\n' "$result" > exitcode.txt
exit "$result"
