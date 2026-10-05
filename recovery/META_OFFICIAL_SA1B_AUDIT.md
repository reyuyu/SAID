# Meta official SA-1B audit

Checked UTC: 2026-10-05T06:59:02.399134+00:00.

## Documented official entry

SmartCLIP source: `https://raw.githubusercontent.com/Mid-Push/SmartCLIP/729c5a6acdaa0d51797c094095f5868cc5c4c7f8/train/train.md`.
Pinned source commit: `729c5a6acdaa0d51797c094095f5868cc5c4c7f8`.
SmartCLIP source SHA256: `714995a7d51981105d898ee4d888f6e38fffcc2eaf0d6b0c8d23fb3bd2b65c31`.
Documented Meta download page: `https://ai.meta.com/datasets/segment-anything-downloads/`.
SmartCLIP explicitly limits SAM training images to shards000000 through000050 and warns against resizing.
The official facebookresearch/segment-anything README independently points to the ai.facebook.com download-page alias.

## Access results

| Official route | HTTP status | Result |
|---|---|---|
| https://ai.meta.com/datasets/segment-anything-downloads/ | NOT RECEIVED | OSError |
| https://ai.meta.com/datasets/segment-anything/ | NOT RECEIVED | OSError |
| https://ai.facebook.com/datasets/segment-anything-downloads/ | NOT RECEIVED | OSError |

No Meta official shard URL was obtained; no Meta archive has been downloaded.
Connection failures do not establish whether login, agreement acceptance, or a signed URL is currently required.
No permission bypass, login automation, agreement submission, guessed CDN object fetch, or HF/ODL object download was attempted.

## Three-source comparison

| Shard | Meta SHA256 | HF/ODL recorded SHA256 | Bitwise same? | Meta tar valid? | Existing required JPEGs recovered |
|---|---|---|---|---|---|
| sa_000014.tar | NOT OBTAINED | `b9b1a1c9699c181e466e9f79302685208905137f67987c6526449768e5d7cdd3` | UNKNOWN | NOT TESTED | 1/11168 |
| sa_000016.tar | NOT OBTAINED | `fb8963c07d3808b229c2df44daca9079a9610322ff15a6e48a5d228928778889` | UNKNOWN | NOT TESTED | 11160/11164 |
| sa_000017.tar | NOT OBTAINED | `59337466a068271e966145fc13f8e61255484978dfa829eacad92cf5d979d479` | UNKNOWN | NOT TESTED | 11165/11168 |

## Historical size / checksum / archive failures

### sa_000014.tar

Size bytes: 11213714588.
Recorded MD5: `45e15f9ff5ded968abe4eb01b95be29a`; checklist MD5: `45e15f9ff5ded968abe4eb01b95be29a`.
Recorded SHA256: `b9b1a1c9699c181e466e9f79302685208905137f67987c6526449768e5d7cdd3`; LFS SHA256: `b9b1a1c9699c181e466e9f79302685208905137f67987c6526449768e5d7cdd3`.
Historical archive integrity passed: False.
Historical GNU stderr: `None`.
Original historical archive currently present: False.
Still missing required JPEGs: 11167.
Remaining decode failures among recovered JPEGs: 0.

### sa_000016.tar

Size bytes: 11403943964.
Recorded MD5: `441129d8f0cbed260c8eafaac98b3965`; checklist MD5: `441129d8f0cbed260c8eafaac98b3965`.
Recorded SHA256: `fb8963c07d3808b229c2df44daca9079a9610322ff15a6e48a5d228928778889`; LFS SHA256: `fb8963c07d3808b229c2df44daca9079a9610322ff15a6e48a5d228928778889`.
Historical archive integrity passed: False.
Historical GNU stderr: `/opt/data/private/lklk/SAID/recovery/evidence/sa1b-shards/sa_000016.tar.tar-stderr.txt`.
Original historical archive currently present: True.
Still missing required JPEGs: 4.
Remaining decode failures among recovered JPEGs: 0.

### sa_000017.tar

Size bytes: 11294495309.
Recorded MD5: `dade5977ae2736a79dbbed4e9a550ec4`; checklist MD5: `dade5977ae2736a79dbbed4e9a550ec4`.
Recorded SHA256: `59337466a068271e966145fc13f8e61255484978dfa829eacad92cf5d979d479`; LFS SHA256: `59337466a068271e966145fc13f8e61255484978dfa829eacad92cf5d979d479`.
Historical archive integrity passed: False.
Historical GNU stderr: `/opt/data/private/lklk/SAID/recovery/evidence/sa1b-shards/sa_000017.tar.tar-stderr.txt`.
Original historical archive currently present: True.
Still missing required JPEGs: 3.
Remaining decode failures among recovered JPEGs: 0.

## Decision

META_OFFICIAL_COMPARISON_BLOCKED. No caseA/B conclusion is possible without actual Meta bytes.
Neither MIRROR_OBJECT_DIVERGENCE_CONFIRMED nor UPSTREAM_ARCHIVE_ANOMALY_CONFIRMED is asserted.
All three requested shards still lack required JPEGs; the abnormal-shard blocker is not removed.
Exact remaining image IDs are in MISSING_REQUIRED_SA_IMAGES.json and this directory's audit.json.
Shard000014's full historical archive had already disappeared before the preceding rescue; only one surviving original JPEG was recovered.

## Required user action

Open `https://ai.meta.com/datasets/segment-anything-downloads/` in your browser. If prompted, log in and personally read/accept the dataset license.
Select only sa_000014.tar, sa_000016.tar, and sa_000017.tar; provide the generated official download URLs or place the downloaded files in this audit directory.
Signed URLs are credentials: do not commit them or include them in public reports. Expired URLs need to be regenerated.
If your browser also cannot reach the official page, report that result; a login/access requirement must not be inferred from a network timeout.

## Recovery isolation

Existing normal HF/ODL supervisors remain running; neither their source plan nor archives were modified.
No images were newly installed by this audit, so live completeness and shard state remain owned by the existing recovery supervisor.
No smoke or training was started. recovery-operation-policy.json explicitly disables the smoke launcher, including a future automatic call by the existing supervisor.
Current full training_index_missing snapshot: 290323 (normal background recovery may change this).
