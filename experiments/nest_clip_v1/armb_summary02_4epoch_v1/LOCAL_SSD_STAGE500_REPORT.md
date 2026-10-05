# Local SSD first500 staging and reproduction

Updated UTC: 2026-10-05T23:36:18.173196+00:00

Trajectory: `/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/armb_summary02_500gate_localssd_v3`; status: `LOCAL_SSD_COPY_IN_PROGRESS`.

IO-only override; frozen sampling/preprocessing/model/loss/optimizer and horizon4868 unchanged. Native3 consecutive full cycles>3s protection unchanged. Fresh common step0; no resume of diagnostic or stopped runs.

Local block-backed root-overlay/NVMe mirror: `/root/SAID_train_cache/ShareGPT4V`.

## Manifest and copy

512000 records requested (500 x1024), de-duplicated images: `512000`.
Family counts: `{"sam": 234294, "coco": 48588, "llava": 229118}`.
Payload bytes: `250901003778`.
Payload GB / GiB: `250.901 / 233.67`.
Full historical500 sample-ID stream / first5 FSD+tokens: `{"historical_sample_ids_matched": 512000, "historical_FSD_string_token_samples_matched": 5120, "offline_global_RNG_unchanged": true}`.
Copy: `{"status": "COPYING", "checked": 46080, "total": 512000, "copied": 32565, "families": {"sam": 46080}, "source_destination_SHA256_matches": 46080, "payload_bytes": 45644899377, "newly_written_bytes": 32309844423, "elapsed_s": 314.44239714741707, "average_payload_MiB_s": 138.4367013026701, "workers": 16, "source_read": "NFS O_DIRECT; empirically byte-verified", "global_drop_caches": false, "private_local_cache_advice": "fsync + file-scoped DONTNEED on our new mirror only; no source-cache advice"}`.
Every required first500 image source SHA256 compared with SSD reread SHA256; atomic raw-byte copy. O_DIRECT source reads; private destination-file cache advice only. No global drop_caches, no source-cache eviction.

1000-example byte/RGB/native preprocess proof and cgroup/process admission: `{}`.

## Training and retrieval

Actual native worker resolved-path proofs: 0 samples, ranks: [].
Phase telemetry: `/tmp/said-s02-full-phase-localssd-v3`; data_wait/H2D/forward/backward/DDP/optimizer/cgroup/PSI recorded.
Retrieval status: `PENDING`; evaluated: `False`.
Scores percent: `null`.

Remaining images are NOT copied during500 training/evaluation. Only after REPRODUCTION_PASS: paused training, manifest-driven full local coverage+SHA verification, unchanged same500 checkpoint, then exact-state continuation. Old stall root cause is not claimed repaired; resource guard retained.
