# SAID HNS-v1 ViT-L/14 best checkpoint

This private ModelScope repository contains the exploratory best checkpoint selected from the verified SAID HNS-v1 checkpoint sweep.

- Backbone: `ViT-L/14`
- Training step: `4651`
- Scheduler horizon: `6085`
- Source branch: `experiment/hns-v1-l14-full5epoch6085-fromscratch-v1`
- Source commit: `8a0b6e8`
- Training data: ShareGPT4V-1M, with the audited nested-detail D3 sampling protocol
- Text context: 248 tokens, native SAID tokenizer

The repository contains both the strict bare CLIP student for inference and the full training checkpoint for exact recovery. The bare student is the file to use for native retrieval inference.

## Inference

Inference uses native CLIP image and text embeddings, L2 normalization, and cosine similarity over the candidate pool. **Mask, Fusion, reranking, ensemble, and TTA are not required at inference time.**

```python
import torch
from model import longclip

model, preprocess = longclip.load_from_clip("ViT-L/14", device="cuda")
state = torch.load("bare_student.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
model.eval()
```

Use the corresponding SAID repository at commit `8a0b6e8` for the tokenizer and preprocessing implementation.

## Audited retrieval results (percent)

| Metric | Value |
|---|---:|
| Score5 | 78.003728 |
| J_long3 | 82.427547 |
| J_long | 90.650002 |
| Short4 | 71.368000 |
| Urban I2T R@1 | 95.300007 |
| Urban T2I R@1 | 95.600003 |
| Flickr30k-Full I2T R@1/R@5/R@10 | 64.735865 / 84.749709 / 90.375358 |
| Flickr30k-Full T2I R@1/R@5/R@10 | 44.472831 / 67.143441 / 75.373627 |

The public benchmark selection is exploratory and is not an independent validation-set model choice.
