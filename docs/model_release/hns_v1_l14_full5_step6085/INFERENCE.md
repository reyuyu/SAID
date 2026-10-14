# Native inference

Install the repository's runtime dependencies (or use an environment that already provides PyTorch, torchvision, Pillow and ftfy):

```bash
python -m pip install -r requirements.txt
```

Then run:

```bash
python scripts/infer_hns_v1_l14_full5_step6085.py \
  --checkpoint checkpoints/hns_v1_l14_full5_step6085/inference/student_step6085.pt \
  --image examples/demo.jpg \
  --texts "A person riding a bicycle." "A dog running."
```

The script loads the bare native ViT-L/14 state dict strictly, encodes images at 224px, tokenizes text with `context_length=248` and `truncate=True`, and L2-normalizes both modalities in FP32 before calculating inner products. The expected embedding dimension is 768.

This release is for native CLIP inference. MaskNet, FusionBranch, HNS, training-time losses, reranking, ensemble and TTA are intentionally absent from the inference path.
