#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.." || exit 1
.venv/bin/python -u recovery/fetch_assets.py --jobs 2 coco_train2017 llava_images
download_result=$?
.venv/bin/python -u recovery/prepare_assets.py --training || exit 1
.venv/bin/python -u recovery/audit.py training || exit 1
.venv/bin/python -u recovery/audit.py scan || exit 1
.venv/bin/python -u recovery/report.py || exit 1
exit "$download_result"
