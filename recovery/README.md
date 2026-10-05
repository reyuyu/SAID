# SAID recovery workspace

## Current status: 2026-10-05

All referenced images are installed: SAM569486, COCO118287, LLaVA558128;
the immutable1245901-record index has zero missing paths in the verified
installation inventories. All four container-invalid shards have their required
original JPEGs recovered. The separate final whole-index physical existence and
full decode audit is still running; it is not yet a passed gate.
The old NFS directory scanner has been terminated and replaced by
`final_manifest_audit.py`: frozen exact-path manifest, local-SSD SQLite restart
checkpoint,8/16/32-worker120-second benchmarks and visible five-second progress.
Current status: `FINAL_IMAGE_AUDIT.json`; procedure/results:
`FINAL_MANIFEST_AUDIT.md`. No directory inventory is needed for this final gate.
No smoke or formal training has been started. See `RECOVERY_STATUS_2026-10-05.md`
for the timestamped status; commands below are recovery references, not permission
to launch a second supervisor or a smoke run.

The later exact-path-audit instruction conditionally permits only the existing
four-A100 five-update smoke after `DATA_AUDIT_PASS`; the data auditor never
launches that trainer itself and never permits500/4868-step training.

The main checkout is the pinned S=0.2 preflight source. Canonical RandomK must use
`worktrees/randomk`, detached at `14653c92c6da9d552a2b624ab169eaaa275cdde8`.
Neither checkout's model, sampler, image preprocessing or native evaluator is
modified by recovery. Original experimental orchestration scripts contain old
absolute paths and can launch full training: do not execute their `run.py`.

## Re-audit and recover

Run from `/opt/data/private/lklk/SAID`:

```bash
.venv/bin/python recovery/audit.py scan
.venv/bin/python recovery/audit.py configs
.venv/bin/python recovery/audit.py environment
.venv/bin/python recovery/validate.py prepare
.venv/bin/python recovery/validate.py step0
.venv/bin/python recovery/validate.py sampling
.venv/bin/python recovery/validate.py cpu
.venv/bin/python recovery/validate.py construction
.venv/bin/python recovery/audit.py training
.venv/bin/python recovery/audit.py evaluators
.venv/bin/python recovery/validate.py subset
.venv/bin/python recovery/validate.py smoke
.venv/bin/python recovery/audit.py status
```

`prepare` creates only a seed0 initializer and the original skip-1000 data index;
it does not perform optimizer updates. It refuses partial/existing assets rather
than deleting them. `smoke` is fail-closed: all asset/environment/CPU/sampling
audits must pass, and its only trainer command uses `--run-type smoke` and
`--max-updates 5`. The scheduler horizon remains 4868, never 5.

Download helper, with resumable `.part` files and per-asset logs:

```bash
.venv/bin/python recovery/fetch_assets.py --jobs 2 flickr_images flickr_captions urban1k
.venv/bin/python recovery/fetch_assets.py --jobs 2 docci_captions docci_images dci_annotations dci_images_1 dci_images_2
.venv/bin/python recovery/fetch_assets.py --jobs 2 coco_annotations coco_val2017 coco_train2017 llava_images
```

Do not start two downloads of the same asset concurrently. SHA mismatches do not
produce verified final files. COCO downloads use the official bucket's path-style
S3 HTTPS endpoint because the legacy image hostname failed TLS certificate
validation here; certificate checking is never disabled.

`continue_public_downloads.sh` is the bounded public-archive continuation job:
each download has at most three one-hour attempts, retaining its current partial
offset between attempts. It then extracts completed public training archives and
refreshes reports, even if another archive failed. It never starts smoke or formal
training. Current job PIDs and logs are in `evidence/public-download-job.json`;
check that job before starting another download of the same archives.

SAM recovery needs publisher-authorized URLs or existing original copies for
`sa_000000.tar` through `sa_000050.tar`. Extract their original JPEGs without
recompression under `local_assets/training/ShareGPT4V/sam/images`. The annotation
requires 569486 distinct SAM paths after skip1000, IDs between 1 and 570590; do
not replace this with the 9K SFT subset or another dataset.

The subsequently verified `Aber-r/SA-1B_backup` source and immutable revision are
recorded in `SA1B_SHARDS.json`. Use `.download-venv/bin/python recovery/sa1b_recovery.py
report` to refresh its report. The locked `continue_sa1b_recovery.sh` job transfers
two original shards at a time, verifies actual MD5/LFS SHA256/tar integrity, and
extracts unchanged image bytes only after structure audit. All 1245901 referenced
images must pass existence and complete PIL decoding before its optional five-step
smoke. It has no formal-training command. Check `evidence/sa1b-job.json` and the
existing download lock before starting a second instance.

Use `tools.prepare_retrieval_benchmarks` for extraction and manifest construction.
The safe wrapper is `.venv/bin/python recovery/prepare_assets.py`; use
`.venv/bin/python recovery/prepare_assets.py --training` after both public training
archives finish. Neither command trains a model. Existing partial archives are not extracted.
Flickr is `parse_flickr(..., test1k=True)`, DOCCI is `parse_docci`, and Long-DCI is
`reconstruct_long_dci` on original DCI annotations. Publish manifests only after
matching their frozen SHA256. Long-DCI is 7602, not the 7805-item DCI Full protocol.

## Original COCO/LLaVA HF mirror continuation

The old public curl job has ended. Do not restart it for these training assets.
`HF_TRAINING_ASSETS.json` pins `pcuenq/coco-2017-mirror/train2017.zip` and
`liuhaotian/LLaVA-Pretrain/images.zip` to immutable revisions and LFS SHA256s.
The locked `continue_hf_training_recovery.sh` runs one official Hub SDK download
per family in `.download-venv`, preserving old curl partials and SDK resume files.
hf_xet is installed and enabled; this endpoint currently lacks Xet HEAD metadata,
so the SDK falls back to resumable HTTP. No SDK or training source files are edited.

Rates are SDK payload callbacks, not sparse-file sizes, sampled every 30 seconds.
The supervisor stops only its own source after ten continuous minutes below
1MiB/s, or a complete ten-minute payload average below that threshold. The
average includes stalls so brief buffered bursts cannot evade the limit, and
source history survives supervisor restarts. A bounded alternative-repository search accepts only the identical
original ZIP filename, size and SHA256; it never substitutes repacked images.
Archives must pass SHA256 and `unzip -t`, then preserve original image paths and
bytes during extraction. Each completion refreshes `TRAIN_IMAGE_COMPLETENESS.json`.
This supervisor never starts training or smoke; the SA1B supervisor is unaffected.

Inspect progress without launching another downloader:

```bash
tail -f recovery/evidence/hf-training-throughput.jsonl
cat recovery/HF_TRAINING_RECOVERY.md
cat recovery/TRAIN_IMAGE_COMPLETENESS.json
```

## Future training, not authorized during recovery

After `READY_TO_RESUME_RESEARCH` and a separate explicit training decision, use
the original trainer. The common initializer is
`runtime/SAID-nest-clip-v1/shared/step000000.pt`; the indexed data is `data_index`
beside `shared`. Configs are `recovery/configs/randomk.json` and `summary02.json`.
The original CLI supports stops of 500 or 4868 with H4868. Do not shorten the
scheduler horizon or silently resume a newly generated checkpoint as the lost
historical step500 parent.

```bash
# DOCUMENTATION ONLY: do not execute while recovery is NOT_READY.
# Canonical RandomK: cd worktrees/randomk first, use absolute NEW_ROOT paths.
# Summary0.2: use the main pinned checkout and summary02.json instead.
NEW_ROOT=/opt/data/private/lklk/SAID
"$NEW_ROOT/.venv/bin/torchrun" --standalone --nnodes=1 --nproc-per-node=4 --max-restarts=0 \
  -m train.train_nested_semantic_mask --config "$NEW_ROOT/recovery/configs/randomk.json" \
  --init-state "$NEW_ROOT/runtime/SAID-nest-clip-v1/shared/step000000.pt" \
  --index-dir "$NEW_ROOT/runtime/SAID-nest-clip-v1/data_index" \
  --image-root "$NEW_ROOT/local_assets/training/ShareGPT4V" \
  --output-dir "$NEW_ROOT/runtime/SAID-nest-clip-v1/new-randomk-500" \
  --run-type formal --max-updates 500
```

For 4868 from scratch change only the output directory and stop value. Continuing
a newly generated 500-step checkpoint requires the original `--resume` validation
with the same configuration/source/data and H4868; it is a new trajectory, not
restoration of lost optimizer/RNG files. Historical reference scores remain in
Git and are not recovery preconditions to reproduce.
