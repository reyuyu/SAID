#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/local_assets/hf_cache"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_HUB_DISABLE_XET=0
export HF_HUB_DOWNLOAD_TIMEOUT=90
export HF_HUB_ETAG_TIMEOUT=30
export HF_XET_NUM_CONCURRENT_RANGE_GETS=4
export HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY=1
exec .download-venv/bin/python -u recovery/hf_training_recovery.py supervise
