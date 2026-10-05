# ModelScope SA-1B image equivalence audit

Checked UTC: 2026-10-05T07:33:31.857680+00:00.
Repository: Tongyi-DataEngine/SA1B-Paired-Captions-Images.
Pinned revision: 4859a29a7aa78102d3364852755c7818e7583366.
Only the opensource_url full-image field is used; matched_local_urls cropped images are excluded.
Only source-indexed IDs already installed from checksum-/tar-verified normal SA-1B shards are references.
Local encoded bytes are independently rehashed against their original tar extraction inventories.
Only parquet footer/projected URL-column HTTP ranges were fetched, not complete parquet files or the full dataset.

## Exact comparisons

Requested distinct IDs: 200.
Compared: 200; distinct: 200.
Classification counts: {"BITWISE_EQUIVALENT": 200}.
Replacement gate passed: True.
Normal-shard coverage: {"sa_000039.tar": 9, "sa_000045.tar": 14, "sa_000044.tar": 28, "sa_000012.tar": 14, "sa_000015.tar": 10, "sa_000002.tar": 7, "sa_000011.tar": 13, "sa_000038.tar": 24, "sa_000009.tar": 15, "sa_000013.tar": 8, "sa_000004.tar": 13, "sa_000008.tar": 13, "sa_000006.tar": 3, "sa_000041.tar": 12, "sa_000020.tar": 11, "sa_000001.tar": 1, "sa_000005.tar": 5}.
Dimensions, ID/basename, encoded-JPEG SHA256, RGB-pixel SHA256/byte equality and native-preprocess tensor SHA256/torch.equal are recorded per image.

## Actual training preprocessing

Factory: train.said_cvssl_data.reference_view_a_transform.
Canonical source SHA256: 3cd78dfe7a82e1482166665ee452eedd15c8342b42268e4e89590ef1ea74339e.
Canonical git revision: 14653c92c6da9d552a2b624ab169eaaa275cdde8.
Preprocessing is imported from the existing canonical training module, never reimplemented or substituted.

## Safety decision

Replacement is authorized only if all200 distinct references are BITWISE_EQUIVALENT or TRAINING_INPUT_EQUIVALENT with no dimension/ID errors.
Any pixel/tensor/dimension/ID mismatch closes the gate; no required missing JPEG may be installed from this source after a failed audit.
No missing-ID batch is downloaded by this audit. Existing HF/ODL recovery is not stopped or modified.
Per-image results: evidence/modelscope-sa1b/equivalence-audit.json.
Source row-group and download provenance: evidence/modelscope-sa1b/audit-samples.json, audit-downloads.json, parquet-scans.json.
