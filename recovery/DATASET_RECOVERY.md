# Dataset recovery

Expected annotation SHA256: `5c5f0f4ee58d7b7467f9e49eb5b17f930890a8a0c18a4e2a5be6b15714ef8b3c`.
Observed annotation SHA256: `5c5f0f4ee58d7b7467f9e49eb5b17f930890a8a0c18a4e2a5be6b15714ef8b3c`.
Original records: 1246901; skip: 1000;
training records: 1245901 (required 1245901).

| Family | Required training paths after skip1000 |
| --- | --- |
| LLaVA LAION/CC/SBU | 558128 |
| COCO train2017 | 118287 |
| SAM | 569486 |

SAM publisher archive range: sa_000000.tar through sa_000050.tar inclusive.
The frozen annotation requires original SAM IDs 1..570590, not the 9K SFT subset.
Expected indexed records SHA256: `0fed1fe12b625ba1f8e762b3545eb115ba084fb314743a74f9e90103e57a27c8`.
Observed indexed records SHA256: `0fed1fe12b625ba1f8e762b3545eb115ba084fb314743a74f9e90103e57a27c8`.
Missing training image paths: 401962.
Training data fully verified: False.

No image preprocessing code was changed. Recover original JPEGs without recompression.
Original index construction and text rules are reused; no replacement subset or synthetic training data.
See per-asset `evidence/download-*.json` / `.log` and resumable `local_assets/downloads/*.part`.
Public training archive continuation job: 15667; bounded retries, no training.
Job output: `evidence/public-downloads-supervisor.log`. The job refreshes reports after extracting any completed public archives.
Original-shard mirror selected: Aber-r/SA-1B_backup at 140d15308aff47dae3b00214083838d56a4319c6; see SA1B_MIRROR_RECOVERY.md and SA1B_SHARDS.json for actual checksum/download/extraction status.

## Independent image recovery

Completeness snapshot UTC: 2026-10-05T01:06:10.034468+00:00.
| Family | Recovered | Required | Missing |
| --- | --- | --- | --- |
| sam | 167524 | 569486 | 401962 |
| coco | 118287 | 118287 | 0 |
| llava | 558128 | 558128 | 0 |

COCO uses Python standard-library ZipFile.testzip and safe original-path extraction; no system unzip dependency.
LLaVA identity/progress: LLAVA_COPY_IDENTITY.md, HF_LLAVA_ASSET.json and HF_LLAVA_RECOVERY.md.
The original archive has numeric subdirectories; extract them unchanged under llava/llava_pretrain/images/.
Other families retain their recorded snapshot counts until their own audits refresh them.
Full decode audit and any five-update smoke stay blocked until all required training paths exist.
