# SAID HNS-v1 ViT-B/16 — Epoch3

This release identifies the third epoch endpoint (step 3651) of the four-epoch HNS-v1 ViT-B/16 schedule (native cosine horizon 4868). It uses 248-token LongCLIP text encoding and produces 512-dimensional native CLIP embeddings. MaskNet, FusionBranch and HNS are training-time mechanisms; default retrieval uses the exported student CLIP encoder only.

The ModelScope repository is `hongchenyang/SAID-HNS-v1-ViT-B16-Epoch3` and is currently private. Access requires a ModelScope token with permission to read the repository.

## Quick start

From a clone of the matching GitHub release branch:

```bash
python -m pip install -r requirements-inference.txt
ms-hub download hongchenyang/SAID-HNS-v1-ViT-B16-Epoch3 inference/student_step3651.pt --local-dir checkpoints/hns_v1_b16
sha256sum checkpoints/hns_v1_b16/inference/student_step3651.pt
python scripts/infer_hns_v1_b16.py --checkpoint checkpoints/hns_v1_b16/inference/student_step3651.pt --image examples/demo.jpg --texts "A person riding a bicycle." "A dog running."
```

The expected bare-student SHA256 is `a60150575f7e8f7b41effb730c0f9d866757954833a240813d4695d9a1cfcb44`. Pin a ModelScope revision with `--revision REVISION` when downloading a published repository. The repository uses SAID's loader and is not a Transformers `AutoModel` checkpoint.

## Historical results

These are single-seed, previously observed native retrieval results; they are not significance claims. Flickr30k-Test1K and Flickr30k-Full use different candidate pools.

| Dataset | I2T R@1 | T2I R@1 |
|---|---:|---:|
| Urban-1k | 94.100 | 92.600 |
| DOCCI | 80.300 | 81.160 |
| Long-DCI | 60.366 | 60.602 |
| COCO-Val5K | 61.900 | 43.036 |
| Flickr30k-Test1K | 89.700 | 73.980 |
| Flickr30k-Full | 56.968 | 36.777 |

Score5: **73.774418%**. Complete R@1/5/10 values and protocol details belong in the frozen historical evaluation artifacts; do not substitute the Test1K candidate pool for FlickrFull (31,783 images and 158,915 captions).

## Retrieval

```python
image_features = F.normalize(model.encode_image(images).float(), dim=-1)
text_features = F.normalize(model.encode_text(tokens).float(), dim=-1)
similarity = image_features @ text_features.T
```

Use `longclip.tokenize(texts, context_length=248, truncate=True)`, the repository's CLIP image transform, and strict loading. See [INFERENCE.md](INFERENCE.md) and [EVALUATION.md](EVALUATION.md).

## Training and ablations

The full step-3651 training state is a separate artifact. It contains model, adapter, optimizer, scheduler, RNG and sampler/cursor information and is required for exact continuation. The common step-0 state is required for controlled from-scratch ablations. A bare student file cannot restore MaskNet/Fusion or optimizer state. See [ABLATIONS.md](ABLATIONS.md), [REPRODUCE.md](REPRODUCE.md) and [PROVENANCE.md](PROVENANCE.md).

Do not package ShareGPT4V images or other datasets with this release. Users must obtain lawful access independently.
