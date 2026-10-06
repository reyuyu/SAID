# S=0.2 AllDetail single-variable 500-step experiment

Status: `INCONCLUSIVE`. Exactly500 updates; no continuation or new experiments.

Only method change: RandomDetail → all ordered non-summary detail sentences from the unchanged visible F sentence pool. User confirmed this pool. F/S strings/tokens, fallback, sampler IDs/order, preprocessing, weights[1.4,0.2,1.4], seed0, global1024,4 A10080GB, horizon4868, model/optimizer/LR/sparsity/inclusion/ramp unchanged.

| Dataset | I2T R@1 / R@5 / R@10 (%) | T2I R@1 / R@5 / R@10 (%) | R1 delta vs historical S0.2 (I2T / T2I pp) |
|---|---|---|---|
| COCO | 60.780000 / 82.640000 / 89.120000 | 41.580000 / 67.156000 / 76.956000 | +0.540000 / -0.264000 |
| Urban-1k | 91.000003 / 98.500007 / 99.800003 | 88.100004 / 98.500007 / 99.400002 | +0.900003 / +0.100004 |
| Flickr30k-test1k | 87.700000 / 97.500000 / 99.300000 | 71.180000 / 91.380000 / 95.000000 | +0.200000 / -0.520000 |
| DOCCI | 77.380000 / 95.280000 / 98.160000 | 77.020000 / 94.820000 / 97.780000 | +0.840000 / +0.600000 |
| Long-DCI | 56.458827 / 74.861878 / 80.939227 | 55.643252 / 75.190739 / 81.070771 | +1.749540 / -0.947119 |

| Score (%) | AllDetail | Delta vs historical S0.2 (pp) | Delta vs current local S0.2 (pp) |
|---|---:|---:|---:|
| Score5 | 70.684209 | +0.319842 | +0.287154 |
| J_long3 | 74.267014 | +0.540403 | +0.428590 |
| J_long | 83.375002 | +0.610000 | +0.609999 |
| Short4 | 65.310000 | -0.011000 | +0.075000 |

Decision (predeclared user thresholds): `{"status": "INCONCLUSIVE", "Urban_T2I_percent": 88.1, "Urban_T2I_raw_float32_percent": 88.10000419616699, "Urban_T2I_hits": 881, "strong_positive": false, "positive": false, "thresholds": {"Urban_T2I_positive": 88.5, "Urban_T2I_strong": 89, "Score5_positive": 70.164367, "Score5_strong": 70.264367, "J_long3_min": 73.403324}, "automatic_continuation": false, "automatic_new_experiments": false}`.
Urban I2T delta: `0.9000026226043758`pp. Long-DCI per-direction deltas are shown above; no full-run claim.

| View | Last50 CE I2T | CE T2I | Directional mean CE | Weighted alignment contribution | Keep ratio |
|---|---:|---:|---:|---:|---:|
| F | 0.048398 | 0.058024 | 0.053211 | 0.496637 | 0.859680 |
| S | 0.851305 | 1.021958 | 0.936632 | 1.248842 | 0.713689 |
| D | 0.099692 | 0.114757 | 0.107225 | 1.000762 | 0.845474 |

D mean sentences 7.053141; mean tokens including SOT/EOT 153.890541; D/F content-token coverage 0.896273 (pooled), 0.888307 (mean per valid sample).
RandomDetail matched baseline: D mean sentences 4.027683, mean tokens 88.425860.
Last50 inclusion 0.013977, inclusion weight 1.000000, S/D mask IoU 0.767772. F/D mask IoU at diagnostic updates in ALL_DETAIL500_DIAGNOSTICS.json.
Weighted contribution=10/sum(weights) × weight × (CE I2T+CE T2I). Token coverage excludes SOT/EOT and uses the same valid samples. Optional F-D native text cosine omitted: no existing observer.

Prelaunch5000 local-only sample paths; runtime128 local-only reads. Local root `/root/said_s02_stage500/ShareGPT4V`; missing/escape/symlink fail-fast, no NFS fallback.
All512000 sample IDs/F/S match baseline; D ordered and complete; LR exact. First5 gate before update6 passed, AdamW counters5, four-rank parameter difference0.
Steady full-cycle seconds: `{"count": 494, "median": 2.11916720867157, "p95": 2.2775980472564696, "p99": 2.3654012727737426, "max": 2.4764301776885986}`.
Slowest-rank data_wait seconds: `{"count": 500, "median": 0.000699896365404129, "p95": 0.001183198019862175, "p99": 0.0015878723561763752, "max": 25.45247744768858}`.
>3s steps 2; >10s steps 2; actual I/O failures 0; oom_kill 0.
GPU/cgroup peaks, PSI and complete raw-log inventory: ALL_DETAIL500_RUNTIME_STATS.json.

Checkpoint `/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/s02-all-detail500-20261006-r2/step500/step000500.pt`; SHA256 `a55820de3eb43e7409af72fced8ec568c21e88b8929014a62c81ef83cef2de92`.
Bare student SHA256 `f5af960e770d418c664956088f480b499f2445afcd40ee7cdd8c321927803285`. Strict native embeddings exact; checkpoint SHA unchanged across eval; five sets share bare SHA; Long has7602 items and frozen manifest.
Inference: normalize(native image embedding) @ normalize(native full-caption text embedding).T. No mask/gate/rerank/ensemble/Summary/Detail inference.
Launch code snapshot `b45fa7cd8879dd4da39637526547eda18caf35f3` matches every launch source SHA. Post-training fixes only diagnostic log routing and float32 Urban boundary classification.
Initial attempt failed after its first optimizer update at diagnostic CUDA/CPU comparison. Preserved locally; corrected run starts fresh common0 and never resumes it. Per-worker proof logs were relocated after worker exit; future logger uses the per-run environment directory.
Checkpoints, bare student, datasets, local images/index/cache and raw logs remain local. NFS originals retained; /root is disposable overlay cache.
Report/code/config/tests only are eligible for GitHub. Fetch/HEAD verification follows publication.
