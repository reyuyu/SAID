# Parallel five native evaluation validation

Same pre-existing strict bare HNS-SG@500, serial GPU0 in historical order vs four independent evaluator processes. No training or gradient audit.
All 30 directional R@1/5/10 values and all four existing aggregates are exact. Every semantic raw JSON field is exact; only device/elapsed_seconds differ. COCO JSON omits counts; the canonical 5000/25000 count basis and annotation SHA are recorded.

| Aggregate (percent) | Serial | Parallel | Delta |
|---|---:|---:|---:|
| Score5 | 71.12482436906207 | 71.12482436906207 | 0 |
| J_long3 | 74.96070728177012 | 74.96070728177012 | 0 |
| J_long | 83.96500199079513 | 83.96500199079513 | 0 |
| Short4 | 65.371 | 65.371 | 0 |

Every dataset has maximum absolute recall error 0. Full unrounded serial/parallel recalls and counts are in COMPARISON.json and results/. No embedding or similarity arrays are emitted by the existing evaluators; this validation compares every semantic field they actually emit.

Serial wall:973.398s; parallel wall:493.991s; speedup:1.970x.

| Dataset | GPU | Serial duration(s) | Parallel duration(s) |
|---|---:|---:|---:|
| coco | 0 | 131.643 | 132.675 |
| urban | 3 | 27.208 | 22.827 |
| flickr | 3 | 28.516 | 33.189 |
| docci | 1 | 293.030 | 298.299 |
| long_dci | 2 | 492.965 | 493.989 |

COCO GPU0; DOCCI GPU1; Long-DCI GPU2; Flickr then Urban on GPU3. No concurrent tasks share a GPU.
Timing includes each original Python process model load. One serial-then-parallel pair; filesystem cache/order can influence measured speedup. No claim that individual benchmarks speed up.
Evaluator/training/export sources unchanged. Formal shared scheduling integration occurs only after this numerical gate passes. Metrics/aggregation are reused unchanged.
Artifacts:SERIAL_RUN.json, PARALLEL_RUN.json (exact commands/UTC/returncodes/source hashes), COMPARISON.json, RUNTIME_COMPARISON.json, results/{serial,parallel}/*.json.
Checkpoint/bare/datasets and raw logs remain local; paths/sizes/SHA/time ranges are in COMPARISON.json.

Default integration: numerical gate passed before replacing the two shared single-model controller evaluation loops. Export, verify-export, training, evaluator sources and existing aggregates are unchanged. CPU AST regression tests compare all other controller code with the pre-task commit. Specialized three-model HNS-Half scheduling remains unchanged.

All 18 CPU tests passed: historical command parity, mapping, per-GPU sequencing, launch/return-code/JSON failures, duplicate outputs, occupied GPUs and visibility ambiguity, plus unchanged controller/export/aggregate source structure and evaluator source hashes. CPU_TESTS.json records the local JUnit path and SHA.
