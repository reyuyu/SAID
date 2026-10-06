# S=0.2 full local training cache

Status: `LOCAL_FULL_PREFLIGHT_READY`.
Local root: `/root/said_s02_stage500/ShareGPT4V`. Ephemeral Docker overlay; risk explicitly accepted. Pod rebuild may discard the cache.
NFS is the persistent source of truth: `/opt/data/private/lklk/SAID/local_assets/training/ShareGPT4V`. Never delete source images based on local presence.
After cache loss, restore the frozen manifest from NFS and stage missing files again. GitHub backs up code/config/reports.

{
  "status": "LOCAL_FULL_PREFLIGHT_READY",
  "manifest": {
    "path": "/opt/data/private/lklk/SAID/recovery/required_training_images.jsonl",
    "persistent_target": "/opt/data/private/lklk/SAID/recovery/evidence/s02-local-full-data-local/required_training_images.jsonl",
    "sha256": "60106915fb9ba11ba4686081d7bddb42536d9779b0d32ae3df83166d1e456c2f",
    "index_sha256": "0fed1fe12b625ba1f8e762b3545eb115ba084fb314743a74f9e90103e57a27c8",
    "bytes": 316768447,
    "reconstruction_byte_exact": true
  },
  "family_counts": {
    "coco": 118287,
    "llava": 558128,
    "sam": 569486
  },
  "total_images": 1245901,
  "total_bytes": 610063394179,
  "total_GB": 610.063394179,
  "total_GiB": 568.1658109454438,
  "local_existing_images": 116817,
  "local_existing_bytes": 11519230546,
  "remaining_images": 1129084,
  "remaining_bytes": 598544163633,
  "unledgered_existing_images": 0,
  "destination_free_bytes": 1259174772736,
  "queue_order": "family (COCO,LLaVA,SAM), source parent directory, filename",
  "storage": {
    "mount": "/      overlay overlay",
    "lifecycle": "ephemeral Docker overlay cache; pod rebuild may discard it",
    "risk_accepted_by_user": true,
    "persistent_source": "/opt/data/private/lklk/SAID/local_assets/training/ShareGPT4V",
    "source_deletion_forbidden": true
  },
  "prepared_utc": "2026-10-06T06:12:18.328728+00:00"
}

No training is started; no global drop_caches, VM changes, archives/masks/eval/cache copying.
Resource PSI is host-scoped on cgroup v1. Benchmarks use different real sorted remaining ranges; family mix is recorded.
Large hash ledgers, manifests, images, partials and raw logs remain local; publication includes only reviewed summaries.
