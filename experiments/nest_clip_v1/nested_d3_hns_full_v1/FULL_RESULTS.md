# HNS-v1 full4epoch

Status: `FULL_POSITIVE`. Exactly4868 updates, continued pinned/evaluated500 without resetting state.

Native bare CLIP, normalized image/full-caption embeddings. Local-only images; no mask/gate/rerank inference. No additional experiment starts.

| Dataset | I2T R@1 | R@5 | R@10 | T2I R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|---:|---:|
| COCO | 61.600000 | 83.580000 | 89.780000 | 42.972000 | 68.648000 | 78.052000 |
| Urban-1k | 93.800002 | 99.100006 | 99.600005 | 92.700005 | 99.200004 | 99.400002 |
| Flickr30k-test1k | 89.800000 | 98.600000 | 99.100000 | 73.920000 | 91.960000 | 95.820000 |
| DOCCI | 80.160000 | 96.420000 | 98.540000 | 81.000000 | 96.080000 | 98.360000 |
| Long-DCI | 60.010524 | 78.203104 | 84.056827 | 60.418311 | 78.584583 | 83.478032 |

Scores (%): `{"Score5": 73.6380841431371, "J_long3": 78.01480690522851, "J_long": 86.9150017285347, "Short4": 67.07300000000001}`.

All available baseline score/recall deltas are in FULL_RESULTS.json. HNS-v1@500 is a learning-curve comparison; full baselines compare different completed training methods.

Checkpoint SHA256: `ced222987743c126ca784bc2d0dd0dd199790929a60ab9a37b7f7706c840536c`; bare SHA256: `20137df2637fb9ea33c3ae616cf575acde0ba805cb47317f2f11929096b88027`.

Runtime: `{"full_cycle_seconds": {"count": 4368, "median": 2.127390503883362, "p95": 2.300022292137146, "p99": 2.3712581491470335, "max": 30.559024810791016}, "data_wait_seconds_slowest_rank": {"count": 4368, "median": 0.0007300116121768951, "p95": 0.0010276168584823606, "p99": 0.0016113079339265817, "max": 28.26587053388357}, "GPU_peak_memory_GiB": {"0": 27.758373737335205, "1": 27.758373737335205, "2": 27.758373737335205, "3": 27.758219242095947}, "peak_cgroup_memory_bytes": 536870912000, "oom_kill": 0, "real_image_IO_errors": 0}`.

Code migration: explicitly pinned default-beta2/2 equivalent predecessor sources; fetched original v1 loss/every parameter gradient exact CPU test. RESUME_GATE.json verifies actual model/adapter/AdamW/RNG/loader restoration and501–505 IDs/text/tokens/indices/LR.

Checkpoints, bare weights, raw logs and stream/path proofs remain local. FULL_RUNTIME_STATS.json lists paths/sizes/SHA256/time ranges. `/root` images are disposable Docker overlay; NFS originals remain the durable source of truth.

One seed only. Full outcome does not establish statistical significance. No automatic hyperparameter search or new experiment.
