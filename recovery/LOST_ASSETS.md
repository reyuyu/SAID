# Lost local assets

## Reconstructed exactly

- Original annotation and skip1000 indexed record bytes, verified against historical SHA256.
- Official OpenAI CLIP B/16 and common step000000.pt, both with matching historical SHA256.
- Three extended evaluation manifests with matching historical SHA256; five evaluation protocols/data restored.
- Full all-branch Git history and reports, pinned environment, training configurations and original code.

## Rebuildable when source images are available

- Training image trees, path indexes, decoded/cache assets, native bare exports and future run artifacts.
- Public COCO/LLaVA image archives can be downloaded again; original SAM shard mirror/revision
  is now recorded in SA1B_SHARDS.json, with actual checksum/recovery status in its dedicated report.

## Not reconstructible from reports

- Lost trained historical step500/step4868 checkpoints, optimizer moments, rank RNG and loader states.
- Scalar metrics and provenance cannot recreate those exact tensors/files. A new run creates a new trajectory,
  not continuation of the lost checkpoint. Backups, if found later, can change this assessment.
- External baseline checkpoints are intentionally deferred until needed; no historical experiments are rerun.
