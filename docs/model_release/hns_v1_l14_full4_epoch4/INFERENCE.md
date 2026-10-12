# Native inference

This release is the final epoch-4 endpoint of the HNS-v1 ViT-L/14 run. Inference uses only the exported student CLIP encoder. MaskNet, FusionBranch, HNS, reranking, TTA and ensembles are not used.

```bash
python -m pip install -r requirements-inference.txt
python scripts/infer_hns_v1_l14.py \
  --checkpoint checkpoints/hns_v1_l14/inference/student_step4868.pt \
  --image examples/demo.jpg \
  --texts "A person riding a bicycle." "A dog running."
```

The loader constructs the ViT-L/14@224 architecture from the student state itself and then performs a strict state-dict load. Text uses `longclip.tokenize(..., context_length=248, truncate=True)`. Both image and text embeddings are L2-normalized in FP32 before the inner product. The first model construction may download the upstream OpenAI initializer; it is only an architecture constructor and is not the trained model.
