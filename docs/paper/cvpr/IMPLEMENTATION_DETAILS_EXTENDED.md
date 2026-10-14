# Sema-CLIP implementation details (audited)

This document records the settings recovered from the formal B/16, L/14, and ABCDE experiment commits. Paths and commit identifiers are retained so that every paper statement can be checked against the repository history.

## Model and architecture

- **Initialization.** Both models load OpenAI CLIP weights through `longclip.load_from_clip`. The B/16 formal method is identified by branch `experiment/nested-d3-hns-full-v1`, commit `45468e1`; the L/14 method by `experiment/hns-v1-l14-full5epoch6085-fromscratch-v1`, commit `8a0b6e8`.
- **Image encoder.** Input resolution is 224×224 with CLIP bicubic resize, center crop, RGB conversion, tensor conversion, and normalization by mean `(0.48145466, 0.4578275, 0.40821073)` and standard deviation `(0.26862954, 0.26130258, 0.27577711)`. ViT-B/16 uses 16×16 patches (196 patch tokens plus CLS, visual width 768, 12 visual blocks, 12 heads, projected width 512). ViT-L/14 uses 14×14 patches (256 patch tokens plus CLS, visual width 1024, 24 visual blocks, 16 heads, projected width 768).
- **Text encoder.** The repository OpenAI CLIP `SimpleTokenizer` BPE is used with a 248-token context. Start/end tokens count toward the context budget.
- **Training-only mask path.** A one-layer mask Transformer and attention pooling produce the text and visual conditions. `FusionBranch` copies the visual mask blocks, maps visual features with a Xavier-initialized adapter, and applies the production visual-patch detach (`hidden.detach().float()`).
- **BalancedStack.** For paired features, the gate is `sigmoid(W_t(t)+W_v(v))`, followed by `g*t+(1-g)*v`. The gate matrix is zero initialized; the adapter is Xavier initialized. The initialization is isolated with a forked RNG so constructing the auxiliary branch does not perturb the training stream.
- **Inference.** Strict bare-student export is used. Retrieval consumes native full-image and full-caption embeddings only; training masks, Fusion, reranking, ensembles, and TTA are not used.

## Training data and view construction

The ShareGPT4V annotation contains 1,246,901 records. The formal training stream deterministically skips the first 1,000 records and therefore contains 1,245,901 records. The audited annotation SHA is `5c5f0f4ee58d7b7467f9e49eb5b17f930890a8a0c18a4e2a5be6b15714ef8b3c`; the retained training-record SHA is `0fed1fe12b625ba1f8e762b3545eb115ba084fb314743a74f9e90103e57a27c`.

Caption preprocessing replaces newlines with spaces, strips surrounding whitespace, and splits literally on `. `. Empty segments are removed. The loader selects the longest valid prefix whose full, all-detail, and end-of-detail tokenizations fit the 248-token budget, counting SOT/EOT. If the first segment itself exceeds the budget, the caption is truncated and detail views are disabled. The views are:

1. `F`: the valid full-caption prefix;
2. `Dall`: all detail sentences after the first;
3. `D3`: an ordered strict nested subset of detail sentences.

`nested_detail_d3` uses a stateless SHA256-derived RNG keyed by sampling seed 0, epoch, sample ID, and namespace `nested_detail_d3_v1`. It selects up to three details (with fewer details for short captions); it is not the separate random-K mode. Shuffle seed and training seed are both zero.

## Optimization and resources

The formal HNS setup uses AdamW with betas `(0.9, 0.999)`, epsilon `1e-8`, and four parameter groups: backbone, shared/text mask, visual mask, and Fusion adapter. Weight decay is `1e-2` for the backbone and `0` for mask and adapter groups. Peak learning rates are:

| Group | Peak LR | Schedule |
|---|---:|---|
| Backbone | `1e-6` | linear warmup through update 200, then cosine decay |
| Text/shared mask | `1e-3` | cosine factor `0.5(1+cos(pi*s/H))` |
| Visual mask | `1e-3` | same cosine factor |
| Fusion adapter | `2e-4` | same cosine factor |

Training uses four NVIDIA A100 80GB GPUs, 256 examples per rank, global batch 1024, accumulation 1, and eight workers per rank. BF16 autocast is enabled; CLIP encoder checkpointing is enabled and pair-block checkpointing is disabled in the formal configurations. The random seed is 0. The B/16 full follow-up has 4 epochs and horizon 4868 (1217 updates per epoch); the formal paper node is step 3651. The L/14 model is independently initialized from common step 0, trained for 5 epochs / 6085 updates with horizon 6085, and is not a continuation of a B/16 or 4868-step run. Exact wall-clock durations are not uniformly available in the retained audits and are therefore not stated here.

## Objectives and masks

For each valid batch, symmetric image-to-text and text-to-image contrastive cross entropy is evaluated for `F`, `Dall`, and `D3`. The audited alignment expression is

```text
align = 10 / sum([1.35, 1.35, 0.30]) *
        (1.35 * CE_F + 1.35 * CE_Dall + 0.30 * CE_D3).
```

Sparsity uses weights `[1,2,2]`, implemented as `(s_F + 2*s_Dall + 2*s_D3)/3`. Hard Straight-Through masks are binary in the forward pass (sigmoid probability threshold 0.5) and use the sigmoid proxy in the backward pass. HNS is active with beta `[2,2]` on the detail chain `F -> Dall -> D3`; its hierarchy multiplier is `min(1, completed_updates/200)`, so the first 200 updates are linearly ramped. The HNS surcharge is based on `ReLU(child-parent)` hard-mask violations. `inclusion_max=0`, so ordinary inclusion loss is disabled; its violation telemetry is retained without gradient. HNS and ordinary inclusion must not be described as simultaneously active losses.

## Evaluation protocol

All native retrieval uses L2-normalized image and text embeddings, inner-product cosine similarity, and the complete candidate pool. No inference mask, Fusion module, reranking, ensemble, or TTA is used.

| Benchmark | Protocol recorded in the audits |
|---|---|
| COCO-Val5K | Standard 5k-image retrieval with native full-image/full-caption embeddings; both directions report R@1/5/10. |
| Flickr30k-Test1K | 1k test-image protocol; separate from Flickr30k-Full. |
| Flickr30k-Full | Official-token protocol with 31,783 images and 158,915 captions; reported as a separate supplementary evaluation. |
| DOCCI | Native full-image/full-caption retrieval with the repository evaluation entry point. |
| Long-DCI | Reconstructed fixed 7,602-pair protocol; it must not be called DCI-full. |
| Urban-1k | 1,000 image-caption pairs, diagonal positives, full 1,000-item candidate pool. |

The later evaluation audits document a scheduler that launches the first four jobs concurrently on GPUs 0–3 and starts Urban on the first released GPU. Historical reports do not prove that every earlier run used this exact scheduler, so paper text should say “four-GPU parallel evaluation” only for runs with that execution evidence.

## Audited headline nodes

- **B/16 HNS-v1, step 3651 (commit `45468e1`).** The checked-in `FULL_RESULTS.json` reports Score5 73.638084, J_long3 78.014807, J_long 86.915002, and Short4 67.073. The archive of component results contains a separately named `A.Full` node with Score5 73.774418; the naming/provenance difference is recorded in `OPEN_QUESTIONS.md`.
- **L/14 HNS-v1, step 6085 (commit `8a0b6e8`).** Score5 77.968310, J_long3 82.423184, J_long 90.775002, Short4 71.286, Urban I2T/T2I 95.3/95.7. The report also gives Flickr30k-Full R@1/5/10 of 64.701255/84.611270 (I2T) and 44.409275/67.100022 (T2I).
- **ABCDE archive, step 3651 (commit `4ef949c`).** A.Full/B/C/D/E are compared under the 4868-horizon schedule. The full table is preserved in the archive report; its headline rows are reproduced in `EXPERIMENT_CONFIG_TABLE.md` for configuration comparison, not as a replacement for source result files.

## Reproducibility limits

The formal checkpoints and runtime audits preserve rank synchronization, cursor continuity, and strict bare-student export checks. They do not provide a single uniform wall-clock measurement for both model families. Public retrieval sets have been observed across multiple experiments; reported comparisons should therefore be described as exploratory rather than as unbiased model selection evidence.
