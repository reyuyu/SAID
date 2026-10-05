# Corrupt SA-1B shard JPEG rescue

Training consumes original JPEG bytes, not tar container validity.
Evidence originals are never modified. Extraction uses independent copies and retains only necessary image data.
No resize, recompression, rename or format conversion. Frozen training index SHA256: `0fed1fe12b625ba1f8e762b3545eb115ba084fb314743a74f9e90103e57a27c8`.

| Shard | Required | Valid original JPEGs recovered | Missing | Decode failures remaining | Status |
|---|---:|---:|---:|---:|---|
| sa_000014.tar | 11168 | 11168 | 0 | 0 | TRAINING_IMAGES_FULLY_RECOVERED |
| sa_000016.tar | 11164 | 11164 | 0 | 0 | TRAINING_IMAGES_FULLY_RECOVERED |
| sa_000017.tar | 11168 | 11168 | 0 | 0 | TRAINING_IMAGES_FULLY_RECOVERED |
| sa_000019.tar | 11164 | 11164 | 0 | 0 | TRAINING_IMAGES_FULLY_RECOVERED |

Exact missing IDs: MISSING_REQUIRED_SA_IMAGES.json. Per-image hashes/dimensions: evidence/sa1b-jpeg-rescue/<shard>/images.jsonl.
000014's historical local archive was lost; historical1MiB fragments were not a recoverable full archive.
The latest minimum-wall-clock instruction authorizes an independent full ODL14 download and conditional17 download; old evidence and all installed JPEGs are preserved.
Fresh checksum and focused extraction evidence: evidence/sa1b-wallclock/forensic/ and evidence/sa1b-wallclock/focused-rescue/.
The archives are gzip compressed despite .tar names: compressed size modulo512 and zero padding do not repair gzip corruption.
Tail repair is attempted only on an independent decompressed tar copy if member bytes are demonstrably complete.
Pending rescue proofs are not counted as completed rescue. Original-byte staged JPEGs may exist before full decode/atomic installation proof finishes.
Additional000019 is now checksum-matching but container-invalid. Its ID range212562..223749 is bounded by fully listed/verified18 and verified20; it gets independent JPEG rescue without a same-object redownload.
Candidate single-image access investigations: evidence/sa1b-jpeg-rescue/third-source-search.json and third-source-candidates.json.
ODL official file catalog exposes original raw tar objects, not individual JPEG objects. No individual-JPEG object API has been established.
Existing DCI/Urban archives have no exact SA image-ID overlap with the quarantined ranges.
Candidate kkkkkcm/SA-1B-400k has seekable JPEG members, but original-byte equivalence is UNPROVEN; only bounded header/member probes are permitted and it is not used for recovery.
k-m-irfan/sa1b is gated; current metadata probes time out. Authorization/network status remains unresolved, and no permission gate is bypassed.
