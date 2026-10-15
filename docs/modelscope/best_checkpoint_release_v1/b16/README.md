# SAID HNS-v1 ViT-B/16 best checkpoint

This private ModelScope repository contains the exploratory best checkpoint selected from the verified SAID HNS-v1 checkpoint sweep.

- Backbone: `ViT-B/16`
- Training step: `3651`
- Scheduler horizon: `4868`
- Source branch: `experiment/nested-d3-hns-full-v1`
- Source commit: `45468e1`
- Training data: ShareGPT4V-1M, with the audited nested-detail D3 sampling protocol
- Text context: 248 tokens, native SAID tokenizer

The repository contains both the strict bare CLIP student for inference and the full training checkpoint for exact recovery. The bare student is the file to use for native retrieval inference.

## Inference

Inference uses native CLIP image and text embeddings, L2 normalization, and cosine similarity over the candidate pool. **Mask, Fusion, reranking, ensemble, and TTA are not required at inference time.**

```python
import torch
from model import longclip

model, preprocess = longclip.load_from_clip("ViT-B/16", device="cuda")
state = torch.load("bare_student.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
model.eval()
```

Use the corresponding SAID repository at commit `45468e1` for the tokenizer and preprocessing implementation.

## Audited retrieval results (percent)

| Metric | Value |
|---|---:|
| Score5 | 73.774418 |
| J_long3 | 78.188029 |
| J_long | 87.040003 |
| Short4 | 67.154000 |
| Urban I2T R@1 | 94.100004 |
| Urban T2I R@1 | 92.600006 |
| Flickr30k-Full I2T R@1/R@5/R@10 | 56.964415 / 79.023377 / 85.857219 |
| Flickr30k-Full T2I R@1/R@5/R@10 | 36.777523 / 59.283265 / 68.394425 |

The public benchmark selection is exploratory and is not an independent validation-set model choice.
