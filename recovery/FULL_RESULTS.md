# S=0.2 full trajectory continuation

Status: `FULL_POSITIVE`; completed4868: `True`.
Resumed exact evaluated checkpoint500 SHA256 `10d5c156dbef97a53cdac97c8eee7ad1879615e39a32208a1c04489e2f5f496d`; continuation501..4868, horizon4868.
Full model/adapter/AdamW, per-rank Python/NumPy/CPU/CUDA RNG and loader generator restored. Native replay of consumed batches skips optimizer updates; step501..505 ordered IDs/FSD/token/LR gates pass.
Frozen Summary+RandomDetail [1.4,0.2,1.4], ViT-B/16 Balanced-Stack-Patch,4 A10080GB,256/rank,global1024,accum1,seed0; native optimizer/sparsity/inclusion/ramp/preprocess unchanged.
Local-only training root `/root/said_s02_stage500/ShareGPT4V`; missing/escaped/symlink paths fail-fast, no NFS fallback. No copy or full audit launched.
Cache is ephemeral Docker overlay; persistent NFS original images remain source of truth.

Continuation full-cycle seconds: `{"count": 4368, "median": 2.108485221862793, "p95": 2.2689209461212156, "p99": 2.346855676174164, "max": 33.73188781738281}`.
Slowest-rank data_wait seconds: `{"count": 4368, "median": 0.0007028132677078247, "p95": 0.0011115722358226775, "p99": 0.0015905800461769102, "max": 31.302938051521778}`.
Steps >3s: 4; >10s: 3. I/O errors: 0; oom_kill: 0.
Peak cgroup: 500.000GiB; peak file cache: 468.473GiB. GPU/rank and PSI distributions: FULL_RUNTIME_STATS.json.
Ordinary>3s warnings continue; hard stops only actual image/CUDA/DDP/finite-state/OOM failure, active step>60s or supervisor anomaly.
Checkpoints saved persistently at1217/2434/3651/4868, runtime checkpoint timing retained.

|Dataset|I2T R@1/5/10 (%)|T2I R@1/5/10 (%)|R1 delta vs RandomK pp|
|---|---|---|---|
|COCO|62.160000 / 84.340000 / 90.240000|43.324000 / 68.960000 / 78.436000|+0.460000 / +1.104000|
|Urban-1k|92.800003 / 98.800004 / 99.400002|91.800004 / 98.900002 / 99.400002|+0.700003 / +0.700004|
|Flickr30k-test1k|89.800000 / 98.500000 / 99.600000|73.980000 / 92.140000 / 95.900000|+0.900000 / +1.540000|
|DOCCI|79.200000 / 95.860000 / 98.340000|79.980000 / 95.860000 / 98.220000|+0.440000 / -0.000000|
|Long-DCI|58.510918 / 77.361221 / 83.162326|60.536701 / 78.413575 / 83.556959|-1.157590 / -0.276243|

Scores (%): `{"Score5_R1": 73.20916265816462, "J_long3": 77.1379377636077, "J_long": 85.94500188350678, "Short4_R1": 67.316, "Score5": 73.20916265816462, "Short4": 67.316}`.
RandomK deltas (pp): `{"Score5": 0.4410156581646163, "J_long3": 0.06769376360769286, "J_long": 0.45999888350677054, "Short4": 1.0010000000000048}`.
Classification fixed before evaluation: Score5>72.768147 and J_long3>=76.870244 => FULL_POSITIVE; improved Score5 with failed long guard => SHORT_LONG_TRADEOFF; otherwise EARLY_SIGNAL_DID_NOT_SCALE.
Strict bare native evaluation: normalized native image embedding @ normalized native full-caption text embedding.T; no mask/gate/rerank/ensemble/Summary/Detail inference.
Final complete checkpoint `/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/s02-local500-20261006/step4868/step004868.pt` SHA256 `c76557576f2bb7da3fe9e98b7dbf5d0b130215ca7004b0eb93bbeb1e9370b72d`.
Bare student `/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/s02-local500-20261006/step4868/student_step4868.pt` SHA256 `38594ccebc39bf6b96dc04d3f7adad5ad16bf417e21df6bab86d603c9b71e180`.
Raw logs path/size/SHA/UTC inventory: FULL_RUNTIME_STATS.json; raw logs, checkpoints, mirror and full hash ledgers stay local.
CPU checks:23 initial tests plus dataset spawn regression passed before corrected launch. Initial launch had a spawn-pickling error before any update; corrected and evidence retained, parent checkpoint unchanged. Native epoch tail remains180/rank (drop_last=False), nominal256/rank; no sampler/batch math changed. GitHub synchronization receipt follows. No automatic further training or experiments.
Error: None

Final CPU/unit verification before publication:24 passed,0 failed. Command: `.venv/bin/python -m pytest -q recovery/test_s02_local_full.py recovery/test_local500_policy.py recovery/test_s02_full_stage.py recovery/test_nfs500_policy.py`.

Long-DCI alone loses1.157590pp I2T and0.276243pp T2I relative to RandomK. The aggregate long guard still passes; FULL_POSITIVE uses the user's fixed Score5/J_long3 conditions.

GitHub branch: `recovery/s02-local-full-training`. Evidence commit `7bc1ddaf3f79fca56e2dbfafa46430a5a9aa5103` pushed successfully, fetched remote HEAD equal, verified UTC2026-10-06T14:01:29Z. Sync receipt: `FULL_GITHUB_RECEIPT.json`; receipt commit is also pushed/fetched and compared before final handoff.

Local assets excluded from Git: full `/root` training mirror and phase logs; persistent runtime complete checkpoints, bare students and raw logs; `local_assets` training/evaluation data; full staging hash ledgers under `recovery/evidence/s02-local-full-data-local`. Raw log path/size/SHA/UTC inventory remains in `FULL_RUNTIME_STATS.json`.
