# SA-1B throughput benchmark

Started: 2026-10-05T02:15:07.638532+00:00; updated: 2026-10-05T02:37:48.065061+00:00.
Repository `Aber-r/SA-1B_backup`, immutable revision `140d15308aff47dae3b00214083838d56a4319c6`, endpoint `https://hf-mirror.com`.
No training/smoke. Existing supervisor process group stopped before transfers; no partial/cache/completed archive deleted.
Each tier uses disjoint unfinished shards, excluding 000014 and all checksum-matching tar-invalid quarantines.
Measured rates are net SDK HTTP payload persisted to archive/partial files, not total NIC traffic or sparse Xet allocation.
Every successful tier runs at least 300 seconds; start-up is included. Five-second raw samples retained.
Errors count HTTP >=400, request exceptions and worker failures; retries count max of SDK retry warnings and repeated GET attempts per shard.
Stable: full duration, nonzero rate, no 429/5xx/final failure, at most one retry per connection, no >=30-second aggregate stall.
Lowest stable concurrency within 5% of the fastest wins. Short-run results are provisional, not proof of sustained all-night performance.

| Concurrency | Total MiB/s | Avg per shard | Errors | Retries | Stable |
|---:|---:|---:|---:|---:|---|
| 2 | 4.866 | 2.433 | 0 | 0 | True |
| 4 | 5.300 | 1.325 | 0 | 0 | True |
| 6 | 5.566 | 0.928 | 0 | 0 | True |
| 8 | 4.233 | 0.529 | 0 | 0 | True |

## Per-shard and host metrics

CPU: host CPU busy percentage and benchmark-worker percentage of one core. Network: network-namespace interfaces except loopback.
Storage is NFS. File-growth/logical write rate is attributable to this benchmark; kernel physical disk write_bytes may be zero on NFS.
NFS mount write counters and NIC rates include other work/shared traffic and are not exclusive benchmark throughput.
File-growth rates have 10MiB SDK-buffer quantization; zero individual samples alone do not prove a connection dropped.

| Concurrency | Host CPU % | Workers CPU % of one core | NIC RX MiB/s | NIC TX MiB/s | Logical writes MiB/s | NFS writes MiB/s |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 4.50 | 5.04 | 5.073 | 0.120 | 4.867 | 4.868 |
| 4 | 4.37 | 6.09 | 5.599 | 0.139 | 5.300 | 5.299 |
| 6 | 4.31 | 9.04 | 5.893 | 0.181 | 5.567 | 5.566 |
| 8 | 4.46 | 8.26 | 4.602 | 0.160 | 4.228 | 4.228 |

### Concurrency 2
Duration: 300.01s; actual transport: SDK resumable HTTP; Xet metadata: False.
Longest aggregate stall: 5s; five-second rate CV: 0.32899473563469495.
Mean host/worker/network/storage metrics: `{"host_cpu_percent": 4.497675793546036, "logical_download_file_write_mib_s": 4.867363991544255, "mount_nfs_write_mib_s": 4.868126927270246, "network_rx_mib_s": 5.073348217288559, "network_tx_mib_s": 0.12015926506815362, "worker_cpu_percent_one_core": 5.04463095584301, "worker_physical_disk_write_mib_s": 4.8675458006389904}`.
Per-shard MiB/s: `{"sa_000018.tar": 2.8665587010946894, "sa_000019.tar": 1.9999246751823414}`.
Raw evidence: `/opt/data/private/lklk/SAID/recovery/evidence/sa1b-throughput-benchmark/concurrency-2`.

### Concurrency 4
Duration: 300.01s; actual transport: SDK resumable HTTP; Xet metadata: False.
Longest aggregate stall: 10s; five-second rate CV: 0.3924050744319295.
Mean host/worker/network/storage metrics: `{"host_cpu_percent": 4.365240457032099, "logical_download_file_write_mib_s": 5.299654037252519, "mount_nfs_write_mib_s": 5.299353694579479, "network_rx_mib_s": 5.5988698198278755, "network_tx_mib_s": 0.13890587220197384, "worker_cpu_percent_one_core": 6.089270316476504, "worker_physical_disk_write_mib_s": 5.299910056882293}`.
Per-shard MiB/s: `{"sa_000020.tar": 0.5666446205022554, "sa_000021.tar": 0.799968876003184, "sa_000022.tar": 2.8998871755115423, "sa_000023.tar": 1.0332931315041127}`.
Raw evidence: `/opt/data/private/lklk/SAID/recovery/evidence/sa1b-throughput-benchmark/concurrency-4`.

### Concurrency 6
Duration: 300.01s; actual transport: SDK resumable HTTP; Xet metadata: False.
Longest aggregate stall: 5s; five-second rate CV: 0.4569845996197215.
Mean host/worker/network/storage metrics: `{"host_cpu_percent": 4.312431126033634, "logical_download_file_write_mib_s": 5.567281450413119, "mount_nfs_write_mib_s": 5.565716255134742, "network_rx_mib_s": 5.8925827489183975, "network_tx_mib_s": 0.1808058478518956, "worker_cpu_percent_one_core": 9.038716590944114, "worker_physical_disk_write_mib_s": 5.566238781831891}`.
Per-shard MiB/s: `{"sa_000024.tar": 0.8666280510677868, "sa_000025.tar": 0.49997772176987704, "sa_000026.tar": 0.6999688104778278, "sa_000027.tar": 1.366605772837664, "sa_000028.tar": 0.6333051142418442, "sa_000029.tar": 1.499933165309631}`.
Raw evidence: `/opt/data/private/lklk/SAID/recovery/evidence/sa1b-throughput-benchmark/concurrency-6`.

### Concurrency 8
Duration: 300.01s; actual transport: SDK resumable HTTP; Xet metadata: False.
Longest aggregate stall: 5s; five-second rate CV: 0.5567441762944876.
Mean host/worker/network/storage metrics: `{"host_cpu_percent": 4.455536795998957, "logical_download_file_write_mib_s": 4.227552836038704, "mount_nfs_write_mib_s": 4.2280854556897545, "network_rx_mib_s": 4.602214638615729, "network_tx_mib_s": 0.16028242771654194, "worker_cpu_percent_one_core": 8.260086225009132, "worker_physical_disk_write_mib_s": 4.228326245190941}`.
Per-shard MiB/s: `{"sa_000030.tar": 1.2666110099572516, "sa_000031.tar": 0.8332967170771393, "sa_000032.tar": 0.09999560604925671, "sa_000033.tar": 0.19999121209851342, "sa_000034.tar": 0.4999780302462835, "sa_000035.tar": 0.13332747473234227, "sa_000036.tar": 0.999956060492567, "sa_000037.tar": 0.19999121209851342}`.
Raw evidence: `/opt/data/private/lklk/SAID/recovery/evidence/sa1b-throughput-benchmark/concurrency-8`.

## Selection and resumption

Best concurrency: 4; Lowest stable concurrency within 5% of highest aggregate throughput.
Best stable aggregate: 5.299793803521094 MiB/s. Baseline was approximate 4–5 MiB/s, not an identical-shard controlled test.
Improvement versus baseline range: 1.06–1.32x.
Remaining normal download bytes (excluding quarantine): 365107458357; pure download estimate: 18.24985077365449 hours.
Pure-download ETA excludes checksum/tar/extraction/existence/decode audits and unresolved invalid archives 000014/000016/000017.
Resumed supervisor: `{"alive_after_launch": true, "command": "flock -n recovery/evidence/sa1b-download.lock bash recovery/continue_sa1b_recovery.sh", "concurrency": 4, "log": "/opt/data/private/lklk/SAID/recovery/evidence/sa1b-download-benchmark-resumed.log", "pid": 24215, "smoke_allowed": false, "started_utc": "2026-10-05T02:35:09.538780+00:00", "training_allowed": false}`.
hf_xet is installed/enabled but HIGH_PERFORMANCE on/off is not compared when HEAD provides no Xet metadata.
Official HF speed test: `{"concurrency": 8, "connections": [{"bytes": 12976128, "connection": 0, "error": null, "status": 200}, {"bytes": 10682368, "connection": 1, "error": null, "status": 200}, {"bytes": 35127296, "connection": 2, "error": null, "status": 200}, {"bytes": 3670016, "connection": 3, "error": null, "status": 200}, {"bytes": 14286848, "connection": 4, "error": "ConnectionError: HTTPSConnectionPool(host='aws.cdn.hf.co', port=443): Read timed out.", "status": 200}, {"bytes": 20971520, "connection": 5, "error": null, "status": 200}, {"bytes": 13828096, "connection": 6, "error": null, "status": 200}, {"bytes": 24444928, "connection": 7, "error": null, "status": 200}], "discarded_payload": true, "elapsed_seconds": 20.547547109425068, "endpoint": "https://aws.cdn.hf.co/fast/5gb", "endpoint_discovered_from_official_frontend": "2.AhSll_gK.js", "method": "CLI streamed the official HuggingFast CDN object, matching frontend 8 streams / 20-second intent; not a browser LibreSpeed run", "not_concurrent_with_shard_tiers": true, "total_bytes": 135987200, "total_mib_s": 6.311580613946514, "utc": "2026-10-05T02:09:57.314836+00:00", "website": "https://fast.hf.co/", "website_available": true}`.

## Safety acceptance

Protected verified/quarantined archives unchanged: True.
All benchmark partial byte counts preserved or increased: True.
Instrumentation and cache-lock preflights are excluded from tier results; evidence and all downloaded bytes were retained.
Post-resume download sample: `{"active_shards": ["sa_000018.tar", "sa_000019.tar", "sa_000020.tar", "sa_000021.tar"], "caveat": "One-minute post-resume check; not a replacement for the five-minute benchmark", "concurrent_active_shards": 4, "configured_concurrency": 4, "per_shard_mib_s": {"sa_000018.tar": 1.8323993172056918, "sa_000019.tar": 1.3326540488768668, "sa_000020.tar": 2.1655628294249087, "sa_000021.tar": 0.16658175610960835}, "positive_growth": true, "processes": "  24215 Ss   flock -n recovery/evidence/sa1b-download.lock bash recovery/continue_sa1b_recovery.sh\n  24216 S    bash recovery/continue_sa1b_recovery.sh\n  24219 Sl   .download-venv/bin/python -u recovery/sa1b_recovery.py run --downloads-only --concurrency 4\n", "seconds": 60.03058338165283, "total_mib_s": 5.497197951617076, "utc": "2026-10-05T02:37:48.009797+00:00"}`.
No training/smoke started; invalid shards remain unresolved and are not counted as recovered.
