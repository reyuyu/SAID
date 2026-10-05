# ODL short concurrency benchmark

Original sample elapsed was read before slow NFS counter reads. Actual monotonic intervals reconstructed exactly from recorded byte deltas / interval rates; no rebenchmark or invented bytes.

| Concurrency | Total MiB/s | Avg per shard | Errors | Retries | Stable | HF MiB/s |
|---:|---:|---:|---:|---:|---|---:|
| 1 | 24.831 | 24.831 | 0 | 0 | True | 4.170 |
| 2 | 43.218 | 21.609 | 0 | 0 | True | 2.375 |
| 3 | 48.350 | 16.117 | 0 | 0 | True | 3.140 |

Selected ODL concurrency: 3. All benchmark SDK partials retained.
Eight Range threads per shard; requests used official OpenXLab SDK. Zero observed HTTP errors; retry count is the instrumented SDK counter, not a guarantee about unlogged internal retries.
Live combined rates and source-partitioned ETA: evidence/sa1b-fast-recovery/state.json.
