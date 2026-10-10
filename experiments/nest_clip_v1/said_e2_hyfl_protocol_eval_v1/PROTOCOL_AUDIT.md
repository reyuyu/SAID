# HyFL/SAID protocol audit

Fixed model: E2@4868, SHA256 `50634512e226e79d526e269ba1a6ca75d7248f47e75417537f1605ca71ac2a0e`. No training, optimizer restoration, model changes, mask/fusion/HNS inference, reranking, TTA or ensemble. Final actual inference uses native ViT-B/16, original preprocess, LongCLIP tokenizer248 with truncate=True, FP32 weights/inputs/outputs/normalized features, L2 normalization and cosine. Autocast is disabled. Original native backend settings are preserved: CUDA matmul TF32=False, cuDNN TF32=True (permitted TF32 image convolution with FP32 tensors).

Read arXiv v1 Table1/Table3/S1.1. S1.1 specifies full Flickr30k, COCO2017-Val5K, 248-token truncation, and normalized cosine for CLIP baselines. HyFL's Lorentzian similarity is NOT introduced into SAID. Published R@1 and our added R@5/10 are distinct.

| Dataset | Status | Images | Captions | Protocol |
| --- | --- | --- | --- | --- |
| DOCCI | VERIFIED_SOURCE_PROTOCOL | 5000 | 5000 | DOCCI-official-test-5000-cosine248 |
| DCI | DCI_PROTOCOL_UNVERIFIED | 7805 | 7805 | SAID-DCI-short-plus-extra-7805-NOT-HyFL-verified |
| Long-DCI | PROTOCOL_UNVERIFIED | 7602 | 7602 | TULIP-constructor-reconstruction-7602 |
| Urban-1k | VERIFIED_SOURCE_PROTOCOL | 1000 | 1000 | Urban-full1k-firstline-cosine248 |
| COCO | VERIFIED_SOURCE_PROTOCOL | 5000 | 25000 | COCO2017-Val5K-first5 |
| Flickr30k-Full | BLOCKED_PROTOCOL_DATA | 0 evaluated / 31783 annotated | 0 evaluated / 158915 annotated | Full images missing |

`VERIFIED_SOURCE_PROTOCOL` means the available official source, selection and metric agree with the published/official-loader definition. It does not imply a bytewise comparison against unavailable HyFL-private derived JSON or pixels.

DOCCI: official docci_descriptions.jsonlines test descriptions and image filenames are exact; 5000 test records, filename ordering, basename-test filtering match HyFL loader. Local images originate from the previously fingerprinted mirror; every image has been decoded and hashed, but HyFL-private image byte identities are unavailable.

DCI: original archive contains 7805 records/images. HyFL reads DCI_test.json and sorts filename, but neither its JSON nor its caption construction was available. The provisional evaluation explicitly uses the existing SAID short_caption + space + extra_caption convention and filename sort. It is NOT a verified HyFL DCI result, and is not ranked against the paper.

Long-DCI: official TULIP constructor sorts annotation filenames, uses raw extra_caption and skips exactly empty strings. 7805 raw records, 203 empty exclusions, 7602 nonempty pairs. We reproduced this constructor using the verified official DCI archive, without additional filtering. 3025 historical captions differ in leading/trailing whitespace. All 7602 image/order identities and all 248-token arrays match the historical stripped manifest (LONG_DCI_TOKEN_EQUIVALENCE.json). The official dci_long.csv is still unavailable; CSV filepath/title and missing-row equivalence remain UNVERIFIED. The reconstruction result is provisional, not a full official CSV replication.

COCO: official 5000 images and 25014 annotations. Explicit positive mappings retain the first five captions per image (25000); 14 extras are excluded by the original documented rule. Sorted image IDs and original annotation order reproduce torchvision CocoCaptions. TULIP eval/coco.py explicitly uses captions[0:5].

Urban: 1000 pairs, fixed stems, first-line captions. Raw newline versus legacy strip yields identical tokenizer output; full candidate pools preserved.

Flickr Full: official Illinois annotation archive retrieved; results_20130124.token has 31783 distinct images and 158915 unique caption IDs with slots #0–#4 per image. Caption-to-image positives are parsed explicitly, without i//5. Source token SHA256 `fad17aa6894489708a31f67d6055a6935fc5d0e3bc632ff45f5bbba6924bf9ee`. Only 1000 named image files are available, 30783 missing. The official site requests image access via a form; direct image URLs return404 and HF endpoints timed out. No form/message was submitted. No full Flickr inference ran. TULIP results.csv was not available for row comparison.

Flickr test1K remains a separate legacy-only 1000/5000 protocol. Its exact Karpathy origin has not been independently proved, so it is not relabeled as Karpathy or Full.

Ranking rule: descending FP32 similarity; exactly equal candidates ordered by ascending fixed manifest index. Near ties are not rounded. Query/gallery blocks retain global top11, allowing exact top10 and K/K+1 margin auditing. GPU GEMM block shape can alter last-bit near ties; per-query receipts and boundary counts are retained locally. Frozen legacy ranking is separately regressed, including canonical COCO CPU chunk512/1D argsort. Any new deterministic-tie differences are disclosed in LEGACY_COMPARISON.md.

COCO exception required by the existing canonical protocol: PRIMARY results retain CPU FP32/query512/full gallery/per-row1D argsort from the frozen original evaluator. Generic bounded GPU scores are retained only as diagnostics. On the same freshly inferred SHA-verified feature bank, GPU streaming yields I2T R1=61.84 versus canonical61.86, one query (image456303/query3934). Four highest candidates have exactly equal similarity, and the ascending-index rule chooses a different candidate than the frozen per-row argsort rule. CANONICAL_COCO_REPROOF.json records every query difference and independently reproduces all historical counts. No better-score selection is used; the previously established original COCO numeric rule controls primary results regardless of direction of the difference. Full Flickr remains bounded GPU streaming.

Precision compatibility: initial implementation incorrectly disabled the frozen evaluator's default cuDNN TF32 permission. Those incompatible attempts were stopped and retained. Independent old-path DOCCI inference reproduced all historical counts exactly; a single-variable TF32 flag test matched old/new cached image embeddings bit-for-bit. Query test_02516 moved rank6→5 in the incompatible strict-TF32-off variant, changing I2T R5 by one. Across all DOCCI images the two profiles differ by max0.000900492, exceeding5e-6. The final system restores the ORIGINAL FP32 backend profile, verified on all four GPUs at the actual batch64 before launch. We do not conflate FP32 tensor dtype with pure IEEE32 contraction everywhere.

Native batch3 versus batch16 also exceeds5e-6 (max0.000456773), with no small-sample query-hit differences. This numerical batch variation remains a FAILED optimization check; no tolerance is increased. Only the verified original batch64 is authorized by the final plan. Cache identity includes batch and backend flags; changing them refuses cache reuse. CrossGPU/same-batch64 tests pass. See NATIVE_PRECISION_RESTORE_AUDIT.json and EVAL_TESTS.json for failed and passing receipts.

Source hashes/commits: REFERENCE_VERSIONS.json. Full image/caption inventories and feature shards stay local. No silent missing-image deletion, invalid-caption filtering or dataset substitution.
