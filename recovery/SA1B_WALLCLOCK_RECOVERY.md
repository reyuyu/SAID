# SA-1B minimum-wall-clock recovery

Measured-rate snapshot: 2026-10-05T08:54:15.163554+00:00

HF aggregate: 4.314MiB/s; previous stable ODL3 aggregate: 48.350MiB/s.
B uses ODL aggregate divided by3, avoiding the false assumption that each of3 shards gets48MiB/s.
A uses the entire measured HF aggregate per shard, conservatively favoring retaining partials.
No new speed benchmark. All old HF partials are preserved. No confirmed ModelScope404 retries.

| Shard | Received bytes | Remaining bytes | Full bytes | HF A minutes | ODL B minutes | Saving | Assignment |
|---|---:|---:|---:|---:|---:|---:|---|
| sa_000021.tar | 10590617600 | 804190352 | 11394807952 | 2.96 | 11.24 | -279.3% | HF resume |
| sa_000028.tar | 587202560 | 10723490597 | 11310693157 | 39.51 | 11.15 | 71.8% | ODL |
| sa_000029.tar | 3764387840 | 7573211819 | 11337599659 | 27.90 | 11.18 | 59.9% | ODL |
| sa_000030.tar | 534773760 | 10730318284 | 11265092044 | 39.53 | 11.11 | 71.9% | ODL |
| sa_000031.tar | 272629760 | 11003433770 | 11276063530 | 40.54 | 11.12 | 72.6% | ODL |
| sa_000032.tar | 31457280 | 11281466461 | 11312923741 | 41.56 | 11.16 | 73.2% | ODL |
| sa_000033.tar | 62914560 | 11306602244 | 11369516804 | 41.66 | 11.21 | 73.1% | ODL |
| sa_000034.tar | 157286400 | 11116120944 | 11273407344 | 40.95 | 11.12 | 72.9% | ODL |
| sa_000035.tar | 41943040 | 11249933728 | 11291876768 | 41.45 | 11.14 | 73.1% | ODL |
| sa_000036.tar | 314572800 | 11034107265 | 11348680065 | 40.65 | 11.19 | 72.5% | ODL |
| sa_000037.tar | 62914560 | 11258756791 | 11321671351 | 41.48 | 11.17 | 73.1% | ODL |

Conservative pure-transfer projection including conditional17: 44.6minutes.
This is not completion ETA: checksum, gzip rescue, NAS extraction/installation and final audit are additional.
14 is an explicitly authorized independent full ODL download.17 is downloaded only if the local focused extraction fails.
16 is already complete and is never touched. ODL globally has at most3 downloads, including forensic jobs.
Latest live status: evidence/sa1b-fast-recovery/state.json. No smoke or training is authorized.

## Local SSD payload/cache

New normal ODL workers use `/root/.cache/said-recovery/odl-original-shards/` when at least128GiB is free.
Existing NAS ODL partials/caches are never moved or replaced. Forensic14/17 keep their explicitly independent forensic directories.
The normal canonical archive path is exposed by symlink only after actual pinned size, MD5 and SHA256 verification.
NAS cache links permit truthful remaining-byte telemetry and SDK resume. These SSD archives/cache targets must be retained.
Original training JPEGs are still installed into the project SAM namespace on NAS, with byte hashes and full PIL decode, without any image transformation.
Per-worker proof: `local-ssd-provenance.json`. No training environment dependency is changed.

## Verified handoff progress

- Independent14 archive downloaded successfully; actual size11213714588, MD5`45e15f9ff5ded968abe4eb01b95be29a`, SHA256`b9b1a1c9699c181e466e9f79302685208905137f67987c6526449768e5d7cdd3`.
- Its hash matches the historical HF/ODL object. This is expected and does not stop focused JPEG rescue.
-14 focused GNU extraction recovered all116 missing JPEGs; all were PIL-decoded and installed with unchanged bytes.14 now has11168/11168 required images, zero missing. Immutable inventory and per-image SHA256: `evidence/sa1b-wallclock/focused-rescue/sa_000014/odl/`.
-Independent17 ODL copy completed and matched the historical object. GNU tar,bsdtar,Python stream and ordinary512 scanner could not recover195618.
-Bounded read-only raw-deflate suffix recovery then recovered195618 in9 candidate attempts. Its checksum-valid tar member contains445138 original encoded JPEG bytes,2248x1500; SHA256`223d9ba1af7b3aab55458f9440a0f0b25eb1c1d5e386b2cdc598ad642301f07c`.
-Resync begins at compressed offset11293750687 with32929 literal stored bytes and an empty, never-guessed dictionary. The recovered suffix ends at the original eight-byte gzip footer. This is NOT a claim that the corrupt archive's whole-stream CRC/tar integrity is repaired.
-PIL full decode and byte-verified atomic installation pass.17 now has11168/11168 required images;14/16/17/19 all have zero missing required IDs. Original forensic archives are unchanged.
-16 is untouched and fully recovered. Confirmed ModelScope404 endpoints are not retried.
-109 CPU recovery tests pass; no smoke,500-step or4868-step process is launched.

## SDK stalled-range recovery

28/29 stopped receiving payload around09:10 while their SDK processes/counter publishers stayed alive.
At09:21 only those two worker processes were terminated; all10.5/11.0GB cached ranges were retained.
The unchanged normal supervisor resumes the missing byte ranges in bounded attempt2, without whole-shard redownload or a fourth ODL slot.
New workers have a watchdog:5minutes without any payload, or10minutes without initial payload, exits with a resumable result instead of waiting forever on a failed SDK range thread.
Recent archive assembly writes suppress the watchdog. No incomplete/cache file is deleted by it.

## Installation pipeline

All normal/forensic transfer queues are now empty. No further SA-1B download is needed.
NAS installation became the limiting stage. After download completion the coordinator changes to three disjoint original-image ingestion jobs; any live prior transfer subprocess is retained and its complete pinned-inventory/decode proof adopted.
A commit lock rejects cross-shard duplicate provenance without overwriting prior ownership. Old partials,completed archives and already-installed JPEGs are not deleted.
The final whole-index physical existence/full decode audit runs only after installed-inventory coverage reaches1245901/1245901.
No smoke or formal training runs, even when all recovery data gates pass.
Live state and current installed counts: `evidence/sa1b-fast-recovery/state.json` and `TRAIN_IMAGE_COMPLETENESS.json`.

## Installed coverage milestone

At2026-10-05T10:34:16UTC the byte-verified, fully decoded installation inventories reach:

-SAM569486/569486; COCO118287/118287; LLaVA558128/558128.
-Filtered records1245901; training_index_missing0; all quarantined required-ID lists empty.
-All download queues and ingestion queues completed without recorded job failures.
-The separate final whole-index physical directory scan and full required-image decode audit is running. It is NOT yet declared passed.
-No smoke or training is authorized. READY_TO_START_S02_FULL is not claimed.
