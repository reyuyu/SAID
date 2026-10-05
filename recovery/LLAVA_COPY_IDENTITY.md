# Original LLaVA archive identity gate

Identity audit passed at 2026-10-04T18:34:16.942049+00:00.
Machine-readable evidence: `evidence/llava-copy-identity-audit.json`.

## Immutable sources

- Official repository: `liuhaotian/LLaVA-Pretrain`.
- Official revision: `70f9d1e5e1a697fe35830875cfc7de1dd590d727`.
- Explicitly approved copy: `332F/LLaVA-Pretrain`.
- Copy revision: `698a737896f11f8cd9548d2989d2c2875405e955`.
- Only archive: `images.zip`, exactly 27356108382 bytes.
- Identical official/copy LFS SHA256 and live HEAD digest:
  `05459d8cb059bd32322b1c466c1cbd4568b09b1ce1db748425b7977236912660`.

The direct official download received no payload during its 600-second low-rate window.
It was stopped with its SDK partial retained. The copy was not used until all checks below passed.
Requests use the official Hugging Face SDK through the reachable HF API relay with TLS verification enabled.
`hf_xet` is installed/enabled; the relay's HEAD exposes no Xet metadata, so transfer uses SDK HTTP fallback.

## Original image identity

Both immutable ZIP central directories were fetched with bounded HTTP range reads.
Both have exactly 558128 distinct original image paths, with equal raw path sets.
Their sorted, newline-delimited raw-path SHA256 is
`62a95db3d4a90d3106ba83ae94c5e5ab8f6d62863cc01a07936c03991ad44b4d`.
The installed path set equals all 558128 LLaVA paths in the frozen ShareGPT4V training index.

The raw archive stores `<5-digit>/<9-digit>.jpg`, not a top-level `images/` wrapper.
Those raw paths are extracted unchanged beneath
`local_assets/training/ShareGPT4V/llava/llava_pretrain/images/`.
This sets the extraction root; it does not rename, resize, recompress or convert any image.

Five encoded-image byte samples were read from the revision-pinned official SDK partial
and the approved copy. Each passed original ZIP CRC/size checks, and both sides had identical SHA256:

| Original raw path | SHA256 of original/copy encoded image bytes |
| --- | --- |
| `00001/000015879.jpg` | `b347764a7833bc56eb14142675c8bf03c4956f0d30878193249a2a93af7296cb` |
| `00148/001489150.jpg` | `4a553d8d42ca27a9ad140413c4a064c1c2f6b3f45c84add202f2f437d6d78afe` |
| `00377/003773629.jpg` | `c5be271b1297273aa33d54273aab45e1bbcb1b745551a76b4d5d88b3c52b2dbd` |
| `00177/001774045.jpg` | `bb58335723868edd6435446aa87e75dec483d3c077cd3011f475f93d9943db6d` |
| `00004/000047774.jpg` | `d1b89555b0975e0c593db29241ff87360e6f7b22b667966882e1f8c2479a308f` |

This identity gate is not full archive-download acceptance. Final SHA256, ZipFile.testzip,
safe original-byte extraction, physical count and frozen-index existence checks must still pass.

## Independent progress and safety

The copy has its own state (`HF_LLAVA_ASSET.json`), download lock and supervisor log,
so COCO extraction and the existing SA1B supervisor remain independent.
Payload throughput is sampled every 30 seconds. A 600-second average or continuous low-rate
window below 1MiB/s stops this download, preserving its SDK partial and reporting the blocker.
No unrelated or repacked dataset is substituted automatically.

Existing slow-source partials and completed archives remain retained.
Independent family completion updates TRAIN_IMAGE_COMPLETENESS.json immediately,
preserving other families' recorded snapshots. Full decode and smoke are not started
while any of the 1245901 required training paths is missing. No formal training is started.
