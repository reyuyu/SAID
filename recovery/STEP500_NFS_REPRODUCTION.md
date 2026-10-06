# S=0.2 canonical NFS step500 reproduction

Status: `INCOMPLETE_HARD_STOP`. Completed updates: 0/500; scheduler horizon4868.

Fresh common step0 SHA256: `54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`.
Frozen Summary+RandomDetail [1.4,0.2,1.4], ViT-B/16 Balanced-Stack-Patch, four A10080GB, batch256/rank, seed0, workers8; native mathematics/preprocessing unchanged.

SSD staging was operator-stopped before this run; retained images, partials and hash ledger. No copy workers or local mirror used during this run.
Canonical images: `/opt/data/private/lklk/SAID/local_assets/training/ShareGPT4V`. Actual worker path proof: `{'count': 128, 'ranks': [0, 1, 2, 3], 'valid': True}`.

Full-cycle seconds (all steps, including startup): `{"count": 0, "median": null, "p95": null, "p99": null, "max": null}`.
Slowest-rank data-wait seconds: `{"count": 0, "median": null, "p95": null, "p99": null, "max": null}`.
>3s steps: 0; >10s: 0. Logged I/O failures: 0.
Peak sampled cgroup bytes: 49051205632. File cache is reported separately and is not RSS.
Slow steps only warn; no samples skipped/replaced, no workers/batch/math changed. Resource telemetry includes cgroup memory, file cache, host PSI and GPU utilization. No forward/backward/optimizer phase completed.

Five-set results and all historical deltas: NOT EVALUATED.
STEP500_RESULTS.json records unavailable metrics and the historical reference; all-step and rank statistics in NFS_RUNTIME_STATS.json and NFS_PER_STEP_SUMMARY.json.

No step500 checkpoint exists. Training stopped before its first optimizer update.
Strict bare export and five-set evaluation were not run. No continuation is authorized; PASS and FAIL both stop500.

Raw logs remain local; reviewed inventory (paths, SHA256, size and UTC scope) is in NFS_RUNTIME_STATS.json. Checkpoints/datasets/images/caches are excluded from GitHub.
GitHub synchronization: PENDING; phase is not marked finally complete.

Error: RuntimeError: train500 process failed, returncode=1; see /opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/s02-nfs500-20261006/train500.log

## Reviewed run evidence

No optimizer step completed. First-five gate and full500 stream checks were not reached. Do not label this REPRODUCTION_PASS or REPRODUCTION_FAIL_AT_500.
NFS completion: This attempt could not proceed: all four ranks timed out on the first batch at60s; 500-step completion is unproven. Observed actual I/O failure: False.
Kernel evidence: `{"available": true, "scope": "Host kernel ringbuffer entries since run launch; host-wide, no attribution by proximity", "boot_seconds_start": 45671426.860601045, "boot_seconds_end": 45671607.542719044, "time_range_utc": ["2026-10-06T05:30:26.089933+00:00", "2026-10-06T05:33:26.772051+00:00"], "line_count": 0, "keyword_counts": {"NFS_RPC": 0, "IO_FILESYSTEM": 0, "GPU_DRIVER": 0, "OOM_HANG": 0}, "raw_path": "/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/s02-nfs500-20261006/kernel-during-run.raw.log", "raw_sha256": "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b", "raw_bytes": 1, "uploaded": false}`; old host-wide kernel entries outside this run are excluded.
Memory/IO PSI is host-scoped on this cgroup v1 system; no actual oom_kill observed: True.

| Dataset | I2T R@1/5/10 (%) | T2I R@1/5/10 (%) | R1 deltas I2T/T2I (pp) |
|---|---|---|---|
| COCO | NOT EVALUATED | NOT EVALUATED | N/A |
| Urban-1k | NOT EVALUATED | NOT EVALUATED | N/A |
| Flickr30k-test1k | NOT EVALUATED | NOT EVALUATED | N/A |
| DOCCI | NOT EVALUATED | NOT EVALUATED | N/A |
| Long-DCI | NOT EVALUATED | NOT EVALUATED | N/A |

Four native DataLoader exceptions confirm first-batch wait reached60s without a batch; this is the requested hard-stop condition. No physical EIO, image read failure, NFS-server-not-responding, CUDA/OOM or kernel-hang evidence was observed during this attempt.
Completed-step median/p95/p99/max are unavailable (zero completed steps); >3s and >10s **completed step counts** are both0. Four unreturned batches are separately reported as right-censored waits>=60s, not zero-duration batches.
Cold worker startup, decode/text/preprocess and NFS reads all contribute to initial wait. Evidence establishes this attempt cannot pass the60s gate; it does not isolate NFS as the sole cause.
The original heartbeat finally-block marked exceptions BATCH_RETURNED; native tracebacks demonstrate these were failures. Raw evidence is preserved, future instrumentation is corrected, and no automatic retry occurred.

Scores (%): `{"Score5": null, "J_long3": null, "J_long": null, "Short4": null}`.
Historical score deltas (pp): `null`.
Checks: `null`.
Dataset collapse definition: invalid/nonfinite R1 or any direction below50% of historical R1. Thresholds are fixed before evaluation; no weight tuning.

Run is stopped before step1; no step500 checkpoint exists. No continuation/retry is authorized by this result.
