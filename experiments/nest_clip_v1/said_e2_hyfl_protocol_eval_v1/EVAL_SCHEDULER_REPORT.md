# Resource-aware evaluator

Dynamic arbitrary-length dataset queue on explicit cuda:0–3 (no CUDA_VISIBLE_DEVICES remapping), one owned worker per GPU. Frozen estimated cost sorts largest jobs first; observed seconds/work-unit informs remaining priorities. Available RAM/disk checked before launch; GPU/process/CPU/disk evidence recorded. One GPU lock per lane, one output lock per queue. No external process is killed. Each job has an exclusive attempt directory and log.

On failure, successful peers are retained; only owned worker process groups are cancelled, with a bounded TERM/KILL cleanup. Pending jobs are cancelled, evidence saved. Explicit --resume verifies frozen plan identity and completed-result SHA, refuses live recorded workers, and creates new attempt directories. Verified feature shards resume without repeated feature rows/counts. Any orphan/corrupt shard is rejected.

Full Flickr uses the same general worker and verified explicit-positive parser once a complete manifest/root is supplied; missing full images currently block that job. No cross-GPU feature sharding was enabled. Data-level concurrency was sufficient; multiGPU sharding is not claimed validated.

Features are encoded in frozen batches64, loaders4 workers, original native FP32 backend settings. Scoring query chunk256/gallery chunk4096 bounds each similarity block to4MiB (plus merge buffers), rather than a20.2GB Full Flickr similarity matrix. Only feature banks and global top11 are retained. Exact ties use ascending candidate index; near ties retain unrounded scores. Feature cache binds checkpoint/manifest/record/image-content/tokenizer/context/preprocess/precision/backend/batch and encoder-version hashes, each tensor shard has SHA/IDs/shape checks.

Scheduler wall: 144.672 seconds, including loading/encoding/scoring/legacy regression. GPU small-batch tests are reported separately in EVAL_TESTS.json.

All queue attempts summed:324.182s; includes both retained incompatible runs. Independent frozen DOCCI reproduction:281.188s. First queue start to final completion:1401.881s (includes diagnosis, code correction, preflight and gaps). EVAL_TIMING.json keeps these denominators separate.

| Job | Physical GPU | Seconds | Peak allocated GiB | Peak reserved GiB | Status |
| --- | --- | --- | --- | --- | --- |
| COCO | 0 | 86.775 | 1.197 | 1.822 | COMPLETED |
| DCI | 1 | 141.262 | 1.197 | 1.822 | COMPLETED |
| Long-DCI | 2 | 138.457 | 1.197 | 1.822 | COMPLETED |
| DOCCI | 3 | 87.030 | 1.197 | 1.822 | COMPLETED |
| Flickr30k-Test1K | 0 | 22.900 | 1.197 | 1.822 | COMPLETED |
| Urban-1k | 3 | 17.634 | 1.197 | 1.822 | COMPLETED |

RUN_STATE.json includes start/end UTC, PIDs, commands, return codes, logs, physical GPU binding, CPU RAM/disk preflight and full worker durations. Per-worker CPU peak RSS, CUDA allocated/reserved peaks, counts/candidates and precision are in RESULTS_HYFL_PROTOCOL.json. No OOM was reported by completed jobs. Full-Flickr resource viability at its actual scale remains UNVERIFIED because images are unavailable.
