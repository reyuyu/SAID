# SA-1B mirror recovery

Repository: `Aber-r/SA-1B_backup`.
Fixed revision: `140d15308aff47dae3b00214083838d56a4319c6`.
Discovery: COMPLETE_REVISION_SELECTED; revisions listed: 1007; candidate file lists checked: 1.
Main was not assumed complete. HfApi.list_repo_commits and list_repo_files were used on immutable revisions.
API/download endpoint: `https://hf-mirror.com`. End-to-end TLS verification remains enabled.

Checklist entries: 1000.
Checklist SHA256: `4a214716b1777d1f612fa1dcb2073df938616838364e329c472d2308d88b7883`.
MD5 algorithm verified against an actual downloaded archive: True.
32-character checksums and matching examples alone are not reported as MD5 file verification.

## Capacity and transfer

Free space at planning: 625271383261184 bytes.
Total pinned tar sizes: 576155235091 bytes.
Reserved minimum free capacity: 1797185182009 bytes.
Official huggingface_hub0.36.0 / hf_xet1.1.10 in isolated `.download-venv`; training dependencies unchanged.
HF parallelism: 4; ODL parallelism: 3. SDK partials/cache support resumption.
Per-attempt provenance records whether server Xet metadata is present and which SDK transport is selected.
Mirror API pagination links pointing to unreachable huggingface.co are routed through the same verified-TLS mirror endpoint.
Normal shards require nonzero exact size, actual MD5, pinned LFS SHA256 and tar -tf before ingestion.
Quarantined shards use independently decoded/byte-hashed required JPEG rescue on copies; original containers are never declared valid by that exception.
Checksum mismatches delete only this job's corrupt tar and retry up to three times.
Checksum-matching tar failures are QUARANTINED_ARCHIVE_INVALID, preserved, skipped, and never retried from the same object.
Resumed recovery policy: Download-only; selected benchmark concurrency; quarantined shards skipped; no decode/smoke/training
Shard14 independent diagnostics/access status: SA1B_000014_FORENSICS.md and evidence/sa1b-000014-forensics.json.
Original image bytes/basenames are retained. Mask JSON is skipped only after full structure audit.
No resize, recompression, rename, format conversion or processed-dataset substitution.
COCO/LLaVA existing download jobs are retained; this job never redownloads those archives.

## Shards

| Filename | Checklist checksum | Actual MD5 | Download | Extraction | Images |
| --- | --- | --- | --- | --- | --- |
| sa_000000.tar | `78d0f487c735a3de86ae6e1b0ff24ea9` | `78d0f487c735a3de86ae6e1b0ff24ea9` | verified | complete | 11186 |
| sa_000001.tar | `c69be32127c9b4a6dc89ff95b036e44d` | `c69be32127c9b4a6dc89ff95b036e44d` | verified | complete | 11186 |
| sa_000002.tar | `771eb6a2e6337a6e8c724309faf6a4d7` | `771eb6a2e6337a6e8c724309faf6a4d7` | verified | complete | 11186 |
| sa_000003.tar | `caab7a8a85d31b1c21b726993a21e894` | `caab7a8a85d31b1c21b726993a21e894` | verified | complete | 11186 |
| sa_000004.tar | `7c54aa5af99e1f0aa0d86789e8b2301a` | `7c54aa5af99e1f0aa0d86789e8b2301a` | verified | complete | 11186 |
| sa_000005.tar | `39af5057caeb94a38b55279ba535a8bb` | `39af5057caeb94a38b55279ba535a8bb` | verified | complete | 11186 |
| sa_000006.tar | `2973abc96f1907d598140db307459307` | `2973abc96f1907d598140db307459307` | verified | complete | 11186 |
| sa_000007.tar | `f1e40a5f566a5d92bc478de6cb025112` | `f1e40a5f566a5d92bc478de6cb025112` | verified | complete | 11186 |
| sa_000008.tar | `0c74a035dba90aafb6355463c0a8e15b` | `0c74a035dba90aafb6355463c0a8e15b` | verified | complete | 11186 |
| sa_000009.tar | `50d3d8d8ad7dc1d04de73de1ce171a59` | `50d3d8d8ad7dc1d04de73de1ce171a59` | verified | complete | 11186 |
| sa_000010.tar | `a4d29f2f00c66722acf1c96dff8f1296` | `a4d29f2f00c66722acf1c96dff8f1296` | verified | complete | 11186 |
| sa_000011.tar | `481dba8aee556bd2f5832278437e2e74` | `481dba8aee556bd2f5832278437e2e74` | verified | complete | 11186 |
| sa_000012.tar | `0957445aa7bbe267bc151f612ac3e015` | `0957445aa7bbe267bc151f612ac3e015` | verified | complete | 11186 |
| sa_000013.tar | `68fab5d691aa53cb6b8080128971eddd` | `68fab5d691aa53cb6b8080128971eddd` | verified | complete | 11186 |
| sa_000014.tar | `45e15f9ff5ded968abe4eb01b95be29a` | `45e15f9ff5ded968abe4eb01b95be29a` | QUARANTINED_ARCHIVE_INVALID | pending | 0 |
| sa_000015.tar | `259118b32686de86fdce3b41e979e517` | `259118b32686de86fdce3b41e979e517` | verified | complete | 11186 |
| sa_000016.tar | `441129d8f0cbed260c8eafaac98b3965` | `441129d8f0cbed260c8eafaac98b3965` | QUARANTINED_ARCHIVE_INVALID | pending | 0 |
| sa_000017.tar | `dade5977ae2736a79dbbed4e9a550ec4` | `dade5977ae2736a79dbbed4e9a550ec4` | QUARANTINED_ARCHIVE_INVALID | pending | 0 |
| sa_000018.tar | `f2abe4b0001cb243e699125833df36a5` | `f2abe4b0001cb243e699125833df36a5` | verified | complete | 11186 |
| sa_000019.tar | `05f1a621b3f685e5213602b99613eaf6` | `05f1a621b3f685e5213602b99613eaf6` | QUARANTINED_ARCHIVE_INVALID | pending | 0 |
| sa_000020.tar | `9f82d3ffe778c007d2b1a0b24c96f4d1` | `9f82d3ffe778c007d2b1a0b24c96f4d1` | verified | complete | 11186 |
| sa_000021.tar | `fe55567bc5931782ca64a18a03ac47e7` | `fe55567bc5931782ca64a18a03ac47e7` | verified | complete | 11186 |
| sa_000022.tar | `c1d23eccda2ba87150de151333cfbc15` | `c1d23eccda2ba87150de151333cfbc15` | verified | complete | 11186 |
| sa_000023.tar | `4138699181656ccaa33c30cb551c5c5e` | `4138699181656ccaa33c30cb551c5c5e` | verified | complete | 11186 |
| sa_000024.tar | `d466ecb2e925b270d208b4256bcd9416` | `d466ecb2e925b270d208b4256bcd9416` | verified | complete | 11186 |
| sa_000025.tar | `5ddf27fcf77da3610045275c899d7c9a` | `5ddf27fcf77da3610045275c899d7c9a` | verified | complete | 11186 |
| sa_000026.tar | `d1c539e5bf719411c2eb10ba066628f3` | `d1c539e5bf719411c2eb10ba066628f3` | verified | complete | 11186 |
| sa_000027.tar | `6adc45f0ee85e85bca6b23fac8e26076` | `6adc45f0ee85e85bca6b23fac8e26076` | verified | complete | 11186 |
| sa_000028.tar | `65d885301017048cf3bf5bfe6a0f46d7` | `65d885301017048cf3bf5bfe6a0f46d7` | verified | complete | 11186 |
| sa_000029.tar | `a88b6270eff050995fa4084cb03bf090` | `a88b6270eff050995fa4084cb03bf090` | verified | complete | 11186 |
| sa_000030.tar | `367997a8328dcbc62701b54fc89dadb3` | `367997a8328dcbc62701b54fc89dadb3` | verified | complete | 11186 |
| sa_000031.tar | `d6b43dfa4435af14154098f91fe27283` | `d6b43dfa4435af14154098f91fe27283` | verified | complete | 11186 |
| sa_000032.tar | `91385a6c9af7cc1a4ed558135e1c356c` | `91385a6c9af7cc1a4ed558135e1c356c` | verified | complete | 11186 |
| sa_000033.tar | `4e4d6a1de9309f6d4054775184328010` | `4e4d6a1de9309f6d4054775184328010` | verified | complete | 11186 |
| sa_000034.tar | `e36d6a702603d597c5ae99c4d83f4e6f` | `e36d6a702603d597c5ae99c4d83f4e6f` | verified | complete | 11186 |
| sa_000035.tar | `de2e1beaaa1d910bb180400ee55b73a7` | `de2e1beaaa1d910bb180400ee55b73a7` | verified | complete | 11186 |
| sa_000036.tar | `ee3dfb1940e6c0ddf1353834ec46b76b` | `ee3dfb1940e6c0ddf1353834ec46b76b` | verified | complete | 11186 |
| sa_000037.tar | `d8ee57265342af18a795330521d960e0` | `d8ee57265342af18a795330521d960e0` | verified | complete | 11186 |
| sa_000038.tar | `4723cf9f132e2f1305bff596c32ae0f1` | `4723cf9f132e2f1305bff596c32ae0f1` | verified | complete | 11186 |
| sa_000039.tar | `5b5db9afeea3cff452df7ca85beaf0cf` | `5b5db9afeea3cff452df7ca85beaf0cf` | verified | complete | 11186 |
| sa_000040.tar | `c6840ba75711e1dc30c8ecb07f18c806` | `c6840ba75711e1dc30c8ecb07f18c806` | verified | complete | 11186 |
| sa_000041.tar | `aed0f35b8646fe0cae73a1727213d88e` | `aed0f35b8646fe0cae73a1727213d88e` | verified | complete | 11186 |
| sa_000042.tar | `f0d04fb9f49b2e6f586294906300711e` | `f0d04fb9f49b2e6f586294906300711e` | verified | complete | 11186 |
| sa_000043.tar | `042b180dab65402030841a25fbe162d2` | `042b180dab65402030841a25fbe162d2` | verified | complete | 11186 |
| sa_000044.tar | `7fa838be2fe6433638fcda889441a234` | `7fa838be2fe6433638fcda889441a234` | verified | complete | 11186 |
| sa_000045.tar | `c0d4505821f5e92ec26185d9e2b234f1` | `c0d4505821f5e92ec26185d9e2b234f1` | verified | complete | 11186 |
| sa_000046.tar | `bc3e27501767317705a7b554928e0337` | `bc3e27501767317705a7b554928e0337` | verified | complete | 11186 |
| sa_000047.tar | `d5e6dd27274a2ab3f7fd4f2de93f9763` | `d5e6dd27274a2ab3f7fd4f2de93f9763` | verified | complete | 11186 |
| sa_000048.tar | `d6b8d33269b0d0e07a307aee7ffc52b8` | `d6b8d33269b0d0e07a307aee7ffc52b8` | verified | complete | 11186 |
| sa_000049.tar | `f860a9ddcb973029e0a7e73c3a3b7ee6` | `f860a9ddcb973029e0a7e73c3a3b7ee6` | verified | complete | 11186 |
| sa_000050.tar | `5a2ea71c804dceed1b00a8ba6dc659e7` | `5a2ea71c804dceed1b00a8ba6dc659e7` | verified | complete | 11186 |

## Training index completeness

Required SAM paths: 569486.
Recovered required SAM paths: 569486.
Missing required SAM paths: 0.
Duplicate index paths: 0.
Duplicate source basenames: 0.
Extracted image inventory count: 167790.
Full training index missing paths: 0 /1245901.
Full original-image decode audit passed: False.
Missing SAM path list: `evidence/sa1b-missing-training-paths.txt`; per-shard raw-byte inventory/provenance: `evidence/sa1b-shards/`.
All51 shard training-image protocols and569486 required SAM images verified: True.
No training is started. The latest operation policy also forbids smoke until separately authorized.
Archive provenance, candidate revisions and failures are recorded even when recovery is incomplete.

## Independent shard14 repair

Status: QUARANTINED_ARCHIVE_INVALID; current rescue coverage is reported below.
Latest minimum-wall-clock instruction explicitly authorizes independent full ODL14 and conditional17 forensic downloads, even when the object hash matches.
Completed shards are reused; normal source queues continue independently. No training or smoke is authorized.
Complete original forensic archive present: False.
The prior supervisor deleted checksum-matching tar-failure copies; its historical GNU stderr was not saved.
Historical shard14 forensic snapshot required images: 11168; historical missing: 11168. Current JPEG rescue coverage is listed below.
Second source k-m-irfan/sa1b access status: NETWORK_OR_ACCESS_UNRESOLVED.
Official authenticated access timed out; anonymous relay401 does not prove the existing token is unauthorized.
Confirm repository access conditions/token read permission and official HF connectivity; no gate bypass or substituted image data is used.
Detailed tests, original evidence limitations, bounded range fragments and exact missing-ID manifest: SA1B_000014_FORENSICS.md.

## Fast disjoint source recovery

Official OpenDataLab/SA-1B dataset6248 is BITWISE_EQUIVALENT_MIRROR. Source-plan.json is authoritative: slow partial ownership may migrate to ODL under the minimum-wall-clock policy; old partials are preserved.
ODL has at most3 simultaneous downloads globally, including independent forensic14/17. A shard is never downloaded concurrently from both sources.
Short benchmark selected ODL concurrency: 3; stable aggregate MiB/s: 48.35003721794759.
Fixed source ownership: evidence/sa1b-fast-recovery/source-plan.json. Live actual combined speed and source-partitioned ETA: evidence/sa1b-fast-recovery/state.json.
Benchmark raw/corrected timing and all retained partials: evidence/sa1b-fast-recovery/; summary: SA1B_ODL_SHORT_BENCHMARK.md.
Incremental inventory coverage is not a completed whole-index physical/decode audit. Final physical existence and full decode remain mandatory before smoke.
### Required JPEG rescue

| Shard | Required | Recovered | Missing | Status |
|---|---:|---:|---:|---|
| 000014 | 11168 | 11168 | 0 | TRAINING_IMAGES_FULLY_RECOVERED |
| 000016 | 11164 | 11164 | 0 | TRAINING_IMAGES_FULLY_RECOVERED |
| 000017 | 11168 | 11168 | 0 | TRAINING_IMAGES_FULLY_RECOVERED |
| 000019 | 11164 | 11164 | 0 | TRAINING_IMAGES_FULLY_RECOVERED |

Detailed rescue: SA1B_CORRUPT_SHARD_RESCUE.md. Only truly missing quarantine JPEG IDs: MISSING_REQUIRED_SA_IMAGES.json.
Source plan created: 2026-10-05T04:29:23.355780+00:00. No500/4868 training is launched.
### Completed-cache and NAS recovery

Official OpenXLab0.1.3 start() can wait indefinitely when every resume range already exists and no worker triggers its assembler. Exact contiguous cached coverage is checked before invoking the SDK's own assembler; no archive bytes are downloaded again.
NAS canonical/alternate bulk assembly returned EIO. Shards41/42 were assembled on persistent local SSD, pinned size/MD5/SHA256 rechecked, and exposed through canonical project symlinks. All old NAS partials and completed-cache chunks remain preserved.
Assembly proof and original-worker handoff: evidence/sa1b-fast-recovery/odl-workers/sa_000041.tar/1/completed-cache-resume.json and corresponding42 proof.
Local storage: /root/.cache/said-recovery/odl-completed/. These original archives must remain until native ingestion finishes; JPEG training paths and bytes are unchanged.
NAS metadata I/O can delay telemetry; latest confirmed installed-image audit is evidence/sa1b-fast-recovery/latest-installed-image-audit.json. Do not infer image completeness from archive byte progress.
