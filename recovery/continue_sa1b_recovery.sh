#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.." || exit 1
export HF_HOME="$PWD/local_assets/hf_cache"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_ENDPOINT="$(.download-venv/bin/python -c 'import json; print(json.load(open("recovery/SA1B_SHARDS.json"))["endpoint"])')"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT=90
export HF_HUB_ETAG_TIMEOUT=30
export HF_XET_NUM_CONCURRENT_RANGE_GETS=4
export HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY=1
export SA1B_DOWNLOAD_CONCURRENCY="${SA1B_DOWNLOAD_CONCURRENCY:-$(.download-venv/bin/python -c 'import json; print(json.load(open("recovery/SA1B_SHARDS.json")).get("download_concurrency", 2))')}"
.download-venv/bin/python -u recovery/sa1b_recovery.py run --downloads-only --concurrency "${SA1B_DOWNLOAD_CONCURRENCY:-2}"
recovery_result=$?
.download-venv/bin/python recovery/sa1b_recovery.py report
.venv/bin/python recovery/report.py
exit "$recovery_result"
