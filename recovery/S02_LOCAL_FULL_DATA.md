# S=0.2 full local training cache

Status: `LOCAL_FULL_TRAINING_DATA_READY`. Training started: false.

Local images: `/root/said_s02_stage500/ShareGPT4V`. Ephemeral Docker overlay; the user explicitly accepts cache loss after pod rebuild.
NFS source of truth: `/opt/data/private/lklk/SAID/local_assets/training/ShareGPT4V`. Original images remain intact. Never treat `/root` as the only copy.
GitHub backs up reproducible code/config/reports. After cache loss, reconstruct/reuse the persistent manifest and stage again.

Frozen manifest SHA256: `60106915fb9ba11ba4686081d7bddb42536d9779b0d32ae3df83166d1e456c2f`; family counts: `{'coco': 118287, 'llava': 558128, 'sam': 569486}`.
Payload: 610063394179 bytes = 610.063394 GB = 568.165811 GiB. Earlier568 referred to GiB, not decimalGB.
Initial verified local images: 116817 (11519230546 bytes). Exact initial missing: 1129084 / 598544163633 bytes.
Reused old images: 116817; newly copied: 1129084; elapsed seconds: 12047.746439889073.
Copy elapsed seconds / mean MiB/s: 9231.586 / 61.833. Validation elapsed seconds: 2816.160. Preflight time is separate.
Queue: family COCO/LLaVA/SAM, source parent, filename. Each new file has one streamed NFS read with SHA256, preserved partial prefix, atomic rename; destination filesystem flush and persistent ledger fsync are batched every2000 files.

| Workers | Duration s | MiB/s | Files/s | Files | Healthy |
|---:|---:|---:|---:|---:|---|
| 2 | 180.029 | 14.722 | 94.352 | 16986 | True |
| 4 | 180.158 | 31.952 | 205.470 | 37017 | True |
| 6 | 180.031 | 27.172 | 337.276 | 60720 | True |
| 8 | not run | N/A | N/A | N/A | Conditional6->8 promotion not met, or earlier pause |

Selected workers: 4. Sustained copy: `{'workers': 4, 'requested_seconds': None, 'elapsed_s': 8691.369100481272, 'files': 1014361, 'bytes': 584599538470, 'MiB_s': 64.14611539060225, 'files_s': 116.70900042017905, 'family_counts': {'llava': 444875, 'sam': 569486}, 'healthy': True, 'source_passes_per_file': 1, 'benchmark_scope': 'Different real remaining sorted ranges; family composition recorded', 'resources': {'samples': 1736, 'peak_memory_current': 30806900736, 'peak_file_cache': 17077551104, 'last_memory_current': 29017411584, 'last_file_cache': 15180734464, 'oom': 0, 'oom_kill': 0, 'NFS_errors': 0, 'IO_PSI_full_avg10_max': 22.32, 'memory_PSI_full_avg10_max': 0.0, 'PSI_scope': 'host /proc/pressure; cgroup v1 does not expose cgroup PSI'}}`.
Benchmark ranges are different real remaining files; family mix is recorded in JSON. A higher concurrency is selected only with at least15% throughput improvement and healthy resources.

Peak cgroup memory: 33.746 GiB. oom_kill: 0. Pod anomaly: False.
Host IO/memory PSI summaries: `{'count': 206, 'min': 0.0, 'max': 22.32, 'median': 14.715, 'p95': 18.7175}` / `{'count': 206, 'min': 0.0, 'max': 0.0, 'median': 0.0, 'p95': 0.0}`. File cache is not RSS.
Integrity: `{'passed': True, 'exact_paths': 1245901, 'all_sizes_matched': True, 'symlink_to_NFS': 0, 'partials_remaining': 0, 'random_SHA_RGB_preprocess_exact_samples': 2000, 'sample_family_counts': {'sam': 668, 'coco': 666, 'llava': 666}, 'local_only_decode_checked': 1245901, 'local_only_decode_passed': 1245901, 'NFS_audit_fallback': False, 'resolved_sample_paths_checked': 5000, 'resolved_all_local': True}`. Full decode audit reads local images only. `.part` is never a training input; missing/symlink paths fail-fast with no NFS fallback.
Local free bytes after this stage: 655129276416.
Local-only formal data-path config: `recovery/configs/s02_local_full_paths.json`; native Dataset semantics and relative paths remain unchanged. No500/4868 training is launched.

Raw logs/manifests/full hash ledger/path proofs remain local. Reviewed inventory follows; GitHub contains only summaries.

- `/opt/data/private/lklk/SAID/recovery/evidence/s02-local-full-data-local/copy-hashes.jsonl`: 326129236 bytes; SHA256 `c3038974176c61e9a600d37fb9c34b62f8fc7496d4ee0b96e034232010149094`; UTC scope `['2026-10-06T06:12:50.303806+00:00', '2026-10-06T09:33:38.050725+00:00']`.
- `/opt/data/private/lklk/SAID/recovery/evidence/s02-local-full-data-local/equivalence-2000.json`: 416063 bytes; SHA256 `92702d4a7c3dbdb902a2553c90eed0cffc952025a779061b6efabae9ea9f1489`; UTC scope `['2026-10-06T06:12:50.303806+00:00', '2026-10-06T09:33:38.050725+00:00']`.
- `/opt/data/private/lklk/SAID/recovery/evidence/s02-local-full-data-local/local-cache-advice-result.json`: 324 bytes; SHA256 `d03c29df70117c2ec0dbb94aebcf01dd6f92f724e2c0d81e96e4bea6ef3e3aa3`; UTC scope `['2026-10-06T06:17:28.198492+00:00', '2026-10-06T09:33:46.889305+00:00']`.
- `/opt/data/private/lklk/SAID/recovery/evidence/s02-local-full-data-local/local-cache-advice.jsonl`: 38924 bytes; SHA256 `2b4e0ec1e256158c923bb7dae989a646036357fbee2951508c0f8682b27ceb31`; UTC scope `['2026-10-06T06:18:29.101785+00:00', '2026-10-06T09:33:22.569565+00:00']`.
- `/opt/data/private/lklk/SAID/recovery/evidence/s02-local-full-data-local/local-path-proof-5000.json`: 762873 bytes; SHA256 `be7c4fce4bdac0e0f9ecc0e19320bbed307c8d45bd6042c2f7232fad9d13d9d4`; UTC scope `['2026-10-06T06:12:50.303806+00:00', '2026-10-06T09:33:38.050725+00:00']`.
- `/opt/data/private/lklk/SAID/recovery/evidence/s02-local-full-data-local/preflight.json`: 1369 bytes; SHA256 `9d2d77ae21f57b8c40cb5e541092c095f73dfac6db8af95afede751df5bc02de`; UTC scope `['2026-10-06T06:12:50.303806+00:00', '2026-10-06T09:33:38.050725+00:00']`.
- `/opt/data/private/lklk/SAID/recovery/evidence/s02-local-full-data-local/progress.json`: 2631 bytes; SHA256 `b82875c5cb48600a9d311eeca94e9078feb7c19884c5348ed8c4d7aa499a3a55`; UTC scope `['2026-10-06T06:12:50.303806+00:00', '2026-10-06T09:33:38.050725+00:00']`.
- `/opt/data/private/lklk/SAID/recovery/evidence/s02-local-full-data-local/required_training_images.jsonl`: 316768447 bytes; SHA256 `60106915fb9ba11ba4686081d7bddb42536d9779b0d32ae3df83166d1e456c2f`; UTC scope `[None, '2026-10-06T06:12:18.328728+00:00']`.
- `/opt/data/private/lklk/SAID/recovery/evidence/s02-local-full-data-local/resource-progress.jsonl`: 433285 bytes; SHA256 `83e553dfc7576352c20801cb3b09ff629722e2f26b40c7e8e425284b6f7b981b`; UTC scope `['2026-10-06T06:12:50.408386+00:00', '2026-10-06T09:32:56.412768+00:00']`.
- `/opt/data/private/lklk/SAID/recovery/evidence/s02-stage500-local/hash-ledger.jsonl`: 32091301 bytes; SHA256 `bedc7435873a410cd983d8eb9080abad7dac70dde36c5a7b1e43ff0e3cd0caec`; UTC scope `existing retained artifact`.
- `/root/said_s02_stage500/full-stage-progress.sqlite`: 394178560 bytes; SHA256 `bee4433b507d04229154f2560a3360f7533ba095c150a9648ffd1712047553af`; UTC scope `existing retained artifact`.

Post-stage CPU checks: 29 passed (`recovery/test_s02_full_stage.py`, `recovery/test_nfs500_policy.py`, `recovery/test_s02_stage500.py`). Local records/offsets/metadata SHA256 reverified; formal local-only admission passed without launching training. Copy, audit and cache-advisor processes have exited.

GitHub sync: pending final commit/push/fetch verification; final phase completion remains false until synchronization succeeds.
