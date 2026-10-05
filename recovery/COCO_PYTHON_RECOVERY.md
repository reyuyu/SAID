# COCO original-image recovery

Completed physical frozen-index audit: 2026-10-04T18:54:13.618443+00:00.
Evidence: `evidence/coco-independent-index-audit.json` and `evidence/hf-training-coco-zipfile-test.json`.

- Reused the already-complete original `train2017.zip`; no duplicate archive download.
- Cached repository: `danasone/coco2017`, revision `2e5331942f3d99a6f3f09591cdf4694abf2642e5`.
- Size: 19336861798 bytes. Actual SHA256:
  `69a8bb58ea5f8f99d24875f21416de2e9ded3178e903f1f7603e283b9e06d929`.
  This exactly matches the original COCO ZIP digest specified for `pcuenq/coco-2017-mirror`.
- Python standard-library `ZipFile.testzip()` passed all 118287 original image members.
- Safe atomic original-byte extraction completed all 118287 images, preserving canonical filenames.
- Target: `local_assets/training/ShareGPT4V/coco/train2017/`.
- Physical directory entry set exactly equals original ZIP file names and all frozen COCO training-index paths.
- Physical image count: 118287. Recovered required images: 118287. COCO index missing: 0.

No `unzip` executable is required. No resizing, recompression or preprocessing changes occurred.
The old curl partial and complete ZIP remain preserved.

## NFS audit handoff

The extraction worker finished all original-byte writes, then spent more than ten minutes in
redundant per-entry NFS metadata enumeration. That read-only stage was stopped, preserving all assets.
An independent audit enumerated actual physical names, compared them exactly against both the
SHA/CRC-verified safe ZIP file set and frozen index, and passed. Regular image creation is proven by
the completed controlled atomic extraction writes; archive counts alone were not used as an existence audit.
The complete handoff/progress provenance is retained in `evidence/coco-index-audit-handoff.json`.

This is an image-path/archive audit, not a full JPEG-decode claim.
TRAIN_IMAGE_COMPLETENESS.json was refreshed independently without waiting for SAM or LLaVA.
Full image decode and any five-step smoke remain gated on all 1245901 required training images.
No formal training is started.
