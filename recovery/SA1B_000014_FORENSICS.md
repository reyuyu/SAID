# SA-1B sa_000014 archive forensics

Updated UTC: 2026-10-05T01:10:37.191151+00:00
Status: QUARANTINED_ARCHIVE_INVALID. Other shards continue independently.
No smoke, 500/4868 training, full Aber-r shard14 redownload, or gate bypass is performed.

## Preserved identity and evidence loss

Original repository/revision: Aber-r/SA-1B_backup / 140d15308aff47dae3b00214083838d56a4319c6.
Pinned size: 11213714588; historical actual MD5: `45e15f9ff5ded968abe4eb01b95be29a`; historical actual SHA256: `b9b1a1c9699c181e466e9f79302685208905137f67987c6526449768e5d7cdd3`.
Known archive size modulo512: 156.
Original complete local archive present: False.
The previous supervisor deleted each checksum-matching copy after GNU tar failure.
No complete cache copy was found. Those deleted original bytes and discarded historical GNU stderr cannot be recreated from checksum records.
Existing checklist, whole-file checksum results, failed attempts and last GNU stdout listing remain preserved.
The resumed downloader now saves complete GNU stderr and preserves checksum-matching tar failures instead of deleting/retrying them.

## Full-archive diagnostics

| Tool | Status |
| --- | --- |
| file | NOT_RUN_ORIGINAL_ARCHIVE_MISSING |
| gnu_tar | NOT_RUN_ORIGINAL_ARCHIVE_MISSING |
| Python tarfile.open(...).getmembers() | NOT_RUN_ORIGINAL_ARCHIVE_MISSING |
| bsdtar | NOT_RUN_ORIGINAL_ARCHIVE_MISSING |
| 7z | NOT_RUN_ORIGINAL_ARCHIVE_MISSING |

Historical GNU listed entries: 22354; JPEG names: 11181.
Historical last listed member: `./sa_162313.jpg`.
First whole-archive failure offset/member: UNKNOWN; original file and original stderr are missing. Last listed member is not asserted to be the failed member.
Two 512-byte uncompressed tar zero end blocks: NOT VERIFIED. A compressed-file byte size modulo512 is not by itself a tar validity test.

## Bounded remote forensic evidence

Only up to1MiB head and1MiB tail of the immutable public Aber-r object may be fetched as forensic fragments; no complete original archive is downloaded.
Range result: saved.
Detected header signature: gzip.
Fragments/prefix samples are not full archive acceptance and are not installed in the training tree.

## Required shard14 paths

ID range inferred from verified neighboring shard inventories: {'minimum': 156625, 'maximum': 167810, 'inferred_from': 'verified original shards13/15', 'historical_names_within_gap': True}.
Required paths in frozen training index: 11168.
Current shard14 required paths missing: 11168.
Exact missing image IDs: `/opt/data/private/lklk/SAID/recovery/evidence/sa1b-000014-forensics/required-missing-image-ids.txt`.
The inferred neighbor gap is cross-checked against the surviving GNU name list; it is not a replacement split.

## Second original mirror

Candidate: k-m-irfan/sa1b; filename: dataset/sa_000014.tar.
Access probe status: NETWORK_OR_ACCESS_UNRESOLVED; existing HF token present: True.
Authenticated official-HF requests timed out; anonymous relay requests returned401. These do not establish whether the existing token has authorized access.
User action: confirm the Hugging Face access conditions have been accepted for this repository and the server's token has read access; official HF connectivity must also work.
No private HF token is sent to the third-party API relay. No gated file is downloaded without an authenticated official permission check.
No second archive has been obtained, so cases A/B and full second-file hash/member/JPEG/decode comparisons remain pending (case C: source unavailable).

## Acceptance remains fail-closed

An authorized independent archive must be stored separately, with size/MD5/SHA256, full tar validation, all members/image IDs, every JPEG decode, all required-path matches and available original-sample byte comparisons recorded.
No partial listing or non-GNU parser result alone is treated as complete recovery. No repacked/resized substitute is used.

## Available original-byte comparison samples

Samples remain forensic-only, outside the training tree.
| Image | Encoded-byte SHA256 | JPEG decode |
| --- | --- | --- |
| sa_160259.jpg | `32c81787454d177e38bb97c7091318eed4da46c040db6536e9242e6f45188cb0` | True |
