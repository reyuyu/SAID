# S=0.2 local-only step500 reproduction

Status: `REPRODUCTION_PASS`; completed 500/500, horizon4868.

Fresh common step0 SHA256: `54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`. No resume.
Frozen Summary+RandomDetail [1.4,0.2,1.4], ViT-B/16 Balanced-Stack-Patch;4 A10080GB,256/rank,global1024,accum1,seed0; optimizer/LR/sparsity/inclusion/preprocessing unchanged.
Local-only image root: `/root/said_s02_stage500/ShareGPT4V`. Missing/escaped/symlink image fails before native decode, no NFS fallback.
Prelaunch random5000 frozen sampler positions all local; runtime path proofs: `{"count": 128, "ranks": [0, 1, 2, 3], "passed": true, "NFS_fallback": false}`.
Cache is ephemeral Docker overlay; NFS originals remain the persistent source of truth.

Full-cycle seconds: `{"count": 500, "median": 2.109856128692627, "p95": 2.2836993336677547, "p99": 2.350566895008087, "max": 56.95749282836914}`.
Slowest-rank data_wait seconds: `{"count": 500, "median": 0.0007102638483047485, "p95": 0.0012529470026493071, "p99": 0.0015942723304033278, "max": 30.68432257324457}`.
>3s:2; >10s:2. I/O errors:0; oom_kill:0.
Resources and all-rank distributions: LOCAL_RUNTIME_STATS.json. Slow-step phases: LOCAL_PER_STEP_SUMMARY.json.
No speed-driven changes to batch/workers/mathematics; no sample skipping/substitution.

| Dataset | I2T R@1/5/10 (%) | T2I R@1/5/10 (%) | R1 delta I2T/T2I pp |
|---|---|---|---|
|COCO|60.680000 / 82.700000 / 89.480000|41.900000 / 67.288000 / 77.152000|+0.440000 / +0.056000|
|Urban-1k|89.800006 / 98.100007 / 99.600005|88.200003 / 98.300004 / 99.300003|-0.299996 / +0.199997|
|Flickr30k-test1k|86.800000 / 97.700000 / 99.200000|71.560000 / 91.380000 / 95.420000|-0.700000 / -0.140000|
|DOCCI|76.300000 / 94.860000 / 97.580000|76.760000 / 94.820000 / 97.520000|-0.240000 / +0.340000|
|Long-DCI|54.919758 / 74.809261 / 81.057616|57.050776 / 76.506183 / 82.004736|+0.210471 / +0.460405|

Scores percent: `{"Score5_R1": 70.39705431298876, "J_long3": 73.83842385498127, "J_long": 82.76500226497649, "Short4_R1": 65.235, "Score5": 70.39705431298876, "Short4": 65.235}`.
Historical score deltas pp: `{"Score5_R1": 0.032687312988755934, "J_long3": 0.1118128549812667, "J_long": 2.649764923035036e-07, "Short4_R1": -0.08599999999999852}`.
Gate checks: `{"Score5_R1": true, "J_long3": true, "Urban_I2T": true, "Urban_T2I": true, "no_obvious_dataset_collapse": true}`.
Collapse screen fixed before evaluation: invalid/nonfinite R1 or any direction below50% of historical R1.
Checkpoint stays local: `/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/s02-local500-20261006/step500/step000500.pt`. Strict export/evaluation must leave it immutable.
PASS/FAIL both stop500; no automatic continuation. CPU/unit checks and GitHub receipt follow.
Raw logs and checkpoints remain local; path/size/SHA/UTC inventory in LOCAL_RUNTIME_STATS.json.
CPU/unit validation before launch: 20 passed, 0 failed (`recovery/test_local500_policy.py`, `recovery/test_s02_full_stage.py`, `recovery/test_nfs500_policy.py`). Final evidence code compiled and reviewed against the completed native run. Training/evaluation workers have exited.

GitHub synchronization: evidence `55def57680f409f1987811cb0689eb07666495e8` pushed/fetched, remote HEAD matched on `recovery/s02-local500`. Receipt commit is pushed/fetched and checked separately.
Error: None

Reviewed evidence:
Prelaunch frozen random sample paths:5000 local-only. Full sample stream records verified:512000.
Steady steps7+ full-cycle seconds: `{"count": 494, "median": 2.109856128692627, "p95": 2.278489887714386, "p99": 2.3282373642921446, "max": 2.40238356590271}`.
Steady steps7+ slowest-rank wait seconds: `{"count": 494, "median": 0.0007102638483047485, "p95": 0.0012512531131505964, "p99": 0.0015904037654399872, "max": 0.03131948411464691}`.
Peak cgroup:302.236458 GiB; GPU peaks:`{'0': {'allocated_GiB': 27.75317144393921, 'reserved_GiB': 28.369140625}, '1': {'allocated_GiB': 27.7598934173584, 'reserved_GiB': 28.447265625}, '2': {'allocated_GiB': 27.75901460647583, 'reserved_GiB': 28.416015625}, '3': {'allocated_GiB': 27.75901460647583, 'reserved_GiB': 28.416015625}}`.
PSI summaries are host-scoped on cgroup v1. Kernel summary: `{"available": true, "scope": "Host ringbuffer during run; host-wide, attribution not inferred", "time_range_utc": ["2026-10-06T09:52:43.959579+00:00", "2026-10-06T10:30:52.406408+00:00"], "line_count": 0, "keyword_counts": {"IO_FILESYSTEM": 0, "GPU_DRIVER": 0, "OOM_HANG": 0, "NFS_RPC": 0}, "raw_path": "/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/s02-local500-20261006/kernel-during-run.raw.log", "bytes": 1, "sha256": "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b", "uploaded": false}`.
Excluded assets:`["/root/said_s02_stage500", "/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/s02-local500-20261006/step500/step000005.pt", "/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/s02-local500-20261006/step500/step000500.pt", "/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/s02-local500-20261006/step500/student_step500.pt", "/opt/data/private/lklk/SAID/local_assets", "/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/shared", "/opt/data/private/lklk/SAID/recovery/evidence/s02-local-full-data-local"]`. No checkpoint/dataset/cache/raw log uploaded.
