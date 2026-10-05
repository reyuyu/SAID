# Final manifest-driven image audit

Checked UTC: 2026-10-05T15:18:47.457854+00:00.
Status: **DATA_AUDIT_PASS**.

Frozen ShareGPT4V required paths only; no filesystem directory enumeration, recursive glob, find or os.walk.
Index SHA256: `0fed1fe12b625ba1f8e762b3545eb115ba084fb314743a74f9e90103e57a27c8`.
Manifest SHA256: `60106915fb9ba11ba4686081d7bddb42536d9779b0d32ae3df83166d1e456c2f`.
Manifest: `required_training_images.jsonl`; restart checkpoint: local-SSD SQLite, bound to index/manifest/root/counts.
Timeout/temporary IO errors receive at most3 short-backoff retries, not automatic corruption classification.
PIL full load and RGB conversion are in memory only; encoded image bytes are never changed.

| Family | Required | Existence checked | Exists | Missing | Decode checked | Decoded | Corrupt | IO errors | IO retries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sam | 569486 | 569486 | 569486 | 0 | 569486 | 569486 | 0 | 0 | 0 |
| coco | 118287 | 118287 | 118287 | 0 | 118287 | 118287 | 0 | 0 | 0 |
| llava | 558128 | 558128 | 558128 | 0 | 558128 | 558128 | 0 | 0 | 0 |

Required paths total: 1245901; existence checked: 1245901; missing: 0.
Decoded: 1245901; corrupt: 0; persistent IO errors: 0; IO retries: 0.
Elapsed seconds: 6761.1.

## Concurrency benchmarks

| Stage | Workers | Seconds | Paths/s | Mean latency ms | P95 latency ms | IO errors | Timeouts | Stable |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| existence | 8 | 120.0 | 1193.98 | 1.95 | 2.66 | 0 | 0 | True |
| existence | 16 | 120.0 | 2560.39 | 0.50 | 1.79 | 0 | 0 | True |
| existence | 32 | 120.0 | 4568.71 | 0.41 | 2.39 | 0 | 0 | True |
Selected existence workers: 32; choose lowest stable concurrency within5% of peak throughput.
| decode | 8 | 120.1 | 168.05 | 44.33 | 91.06 | 0 | 0 | True |
| decode | 16 | 120.1 | 234.44 | 64.47 | 122.14 | 0 | 0 | True |
| decode | 32 | 120.2 | 270.24 | 112.94 | 207.51 | 0 | 0 | True |
Selected decode workers: 32; choose lowest stable concurrency within5% of peak throughput.

Full decode starts only after exact-path existence reaches100% with zero missing/persistent IO errors.
DATA_AUDIT_PASS requires all three exact frozen family counts to exist and decode, zero missing/corrupt/IO failures.
No smoke,500-step or4868-step trainer is launched by this program.
If DATA_AUDIT_PASS, the next permitted training action is only the separately authorized four-A100 five-update smoke with common step0 and H4868.
