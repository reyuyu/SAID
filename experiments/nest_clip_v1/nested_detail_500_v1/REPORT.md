# F / All Detail / Atomic Detail matched500 experiment

Status: `NESTED_DETAIL_POSITIVE`. Completed 500/500. No automatic full run or new experiment.

Fresh common step0 SHA256: `54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`; horizon4868, global1024, seed0, workers8.
Local-only image root: `/root/said_s02_stage500/ShareGPT4V`. Missing/symlink/escape fails; no NFS fallback. Persistent NFS originals retained.
Visible F packing unchanged. Dall is ordered s2..sn; Ds is one uniformly sampled whole visible detail, using private seed/epoch/sample RNG. Single-detail Ds may equal Dall.
Alignment weights[1.4,1.4,0.2], directional CE summed. Chain detached-child edges only; native200-step inclusion ramp/max1 and original sparsity retained.

| Dataset | I2T R@1 / R@5 / R@10 (%) | T2I R@1 / R@5 / R@10 (%) |
|---|---|---|
| COCO | 60.240000 / 82.600000 / 89.040000 | 41.616000 / 67.256000 / 76.824000 |
| Urban-1k | 90.900004 / 98.700005 / 99.600005 | 88.500005 / 98.400003 / 99.400002 |
| Flickr30k-test1k | 88.000000 / 97.600000 / 99.200000 | 71.680000 / 91.500000 / 95.140000 |
| DOCCI | 76.840000 / 95.160000 / 98.060000 | 77.020000 / 94.920000 / 97.700000 |
| Long-DCI | 55.695870 / 74.348856 / 80.741910 | 57.011313 / 76.269403 / 82.096817 |

| Model | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I |
|---|---:|---:|---:|---:|---:|---:|
| S0.2 RandomDetail | 70.397054 | 73.838424 | 82.765002 | 65.235 | 89.800 | 88.200 |
| S0.2 AllDetail | 70.684209 | 74.267014 | 83.375002 | 65.310 | 91.000 | 88.100 |
| Nested Detail | 70.750319 | 74.327865 | 83.315002 | 65.384000 | 90.900 | 88.500 |

Aggregate delta vs both baselines (pp): `{"RandomDetail": {"Score5": 0.35326512372968466, "J_long3": 0.48944120621612797, "J_long": 0.5500002292137083, "Short4": 0.1490000000000009}, "AllDetail": {"Score5": 0.06611012372968617, "J_long3": 0.06085120621612816, "J_long": -0.05999977078629115, "Short4": 0.07399999999999807}}`.
All dataset/direction/R@k deltas: `RESULTS.json`.
Decision: `{"status": "NESTED_DETAIL_POSITIVE", "Urban_R1_percent": {"I2T": 90.9, "T2I": 88.5}, "positive_guard": true, "strong_guard": true, "atomic_anomaly": true, "thresholds": {"positive_Urban_T2I_gt": 88.2, "Score5_min": 70.48, "J_long3_min": 74.0, "strong_Urban_T2I_min": 88.7, "strong_Score5_min": 70.684209, "strong_J_long3_min": 74.267014}, "strong_guard_interpretation": "No regression against the stronger AllDetail matched500 baseline; tolerance1e-6pp for rounding", "automatic_continuation": false, "automatic_new_experiments": false}`.
Mask hierarchy telemetry: `MASK_HIERARCHY_AUDIT.json`; soft penalty is not an exact hard nesting guarantee.
CE/gate/keep/weighted contributions and sampling: `TRAINING_DIAGNOSTICS.json`; native-backbone gradient norms/cosines: `GRADIENT_SPOTCHECK.json`.
This joint view/weight/hierarchy comparison cannot isolate an independent causal benefit of atomic supervision; no automatic ablation is started.
Strict native inference: normalized image/full-caption embeddings, plain inner product; no mask/gate/detail/rerank/ensemble.
Checkpoint/bare SHA and immutable-evaluation proof: `{"passed": true, "strict_load": true, "optimizer_steps": [500], "image_max_abs": 0.0, "text_max_abs": 0.0, "checkpoint_sha256": "62b40994214fc7cf9a513d72fa71250e7a15be1cc88d206dc55fd62314ea399f", "bare_sha256": "5d050fda9bb0c80fd147fd9efa598a07f640a43ea7f3a95493093cfd7da795bd"}`.
Full raw evidence paths/SHA256/bytes/time ranges: `RUNTIME_STATS.json` and `SAMPLING_AUDIT.json`; retained locally, not uploaded.
Ephemeral Docker overlay cache; NFS source of truth retained. Checkpoints/bare remain under persistent project runtime, excluded from Git.
Git publication uses explicit small-file allowlist and credential/binary/size sanity check; remote HEAD must match final commit.
Error: None

## Evidence review

Launch code snapshot: `bacd6513d704658b68c05fdce4e4f3be9b6b6067`; every recorded launch source SHA256 matches its Git blob.
1000 text/token records and5000 frozen local paths passed before launch. All512000 training records matched baseline sample IDs and F strings/tokens. First5 structural gate passed before update6; final four-rank parameter difference0.
Final full checkpoint SHA256: `62b40994214fc7cf9a513d72fa71250e7a15be1cc88d206dc55fd62314ea399f`; size1884188218bytes. Model/adapter/AdamW/scheduler, four RNG states, sampler and cursor0:500 verified.
Bare student SHA256: `5d050fda9bb0c80fd147fd9efa598a07f640a43ea7f3a95493093cfd7da795bd`. Native image/text export equality exact; checkpoint unchanged by gradient diagnosis and five-set evaluation.

| Baseline | ΔScore5 | ΔJ_long3 | ΔJ_long | ΔShort4 | ΔUrban I2T | ΔUrban T2I |
|---|---:|---:|---:|---:|---:|---:|
| RandomDetail | +0.353265 | +0.489441 | +0.550000 | +0.149000 | +1.100 | +0.300 |
| AllDetail | +0.066110 | +0.060851 | -0.060000 | +0.074000 | -0.100 | +0.400 |

Last50 mask IoU F–Dall=0.967378, Dall–Ds=0.784682; hard violation Dall⊆F=0.011169, Ds⊆Dall=0.040424. The requested chain penalty and detached-child gradient routing are verified; empirical violations measure how closely learned hard masks follow it.
Last50 raw combined CE F=0.111501, Dall=0.210070, Ds=5.472811. Ds/Dall CE ratio=26.0523, mean backbone gradient norm ratio=3.2113. Descriptive atomic-pressure flag=True; all8 diagnostic batches finite.
Gradient norms and native-direction cosines: `{"mean_gradient_norms": {"F": 11.42037832736969, "Dall": 15.080010056495667, "Ds": 48.42653560638428}, "mean_cosine_to_native": {"F": 0.7849307283759117, "Dall": 0.4844956286251545, "Ds": 0.03205208025610773}, "mean_weighted_alignment_cosine_to_native": 0.6589201763272285}`.
After applying the actual1.4/1.4/0.2 coefficients, Ds/Dall per-view gradient norm ratio=0.4588. Weighted alignment cosine uses the linear combination of separately recomputed view gradients; no optimizer updates or combined-backward equivalence assertion.
Full-cycle distribution(s): `{"count": 500, "median": 2.1267170906066895, "p95": 2.2941793799400325, "p99": 2.366960253715515, "max": 40.94863510131836}`; local data_wait(s): `{"count": 500, "median": 0.000707726925611496, "p95": 0.0012348726391792292, "p99": 0.0015905339270830133, "max": 27.316400043666363}`.
Warnings >3s=2, >10s=2; true training I/O errors=0, oom_kill=0; no observed Pod/supervisor anomaly.
Training wall seconds=1185.808; GPU peaks(GiB)={'0': 27.760382175445557, '1': 27.760382175445557, '2': 27.760382175445557, '3': 27.760382175445557}. Cgroup peak=500.000GiB, file-cache peak=470.838GiB, anon peak=28.265GiB. Cache usage is not RSS or OOM evidence.
For the strong-positive gate, “Score5/J_long3 not worse” is interpreted against the higher AllDetail baseline. Detailed thresholds and fallback classifications are recorded in RESULTS.json.
Conclusion is the measured aggregate/dataset comparison; this one run cannot separate atomic supervision from removal of Summary and the changed hierarchy. Training remains stopped at500.
