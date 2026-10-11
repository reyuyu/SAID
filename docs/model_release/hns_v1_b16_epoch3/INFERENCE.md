# Inference

`scripts/infer_hns_v1_b16.py` loads the bare `student_step3651.pt`, selects CPU or CUDA, applies the SAID 224-pixel CLIP transform, tokenizes at context length 248, normalizes both modalities in FP32 and prints ranked text similarities. It does not construct or execute the training MaskNet/Fusion path.

The script requires the repository source and a local copy of the base ViT-B/16 initializer used by `model.longclip`. If that initializer is not cached, the upstream CLIP URL may be downloaded once; after the initializer and student file are cached, inference is offline. Use `--device cpu` for a CPU smoke test.

Embedding width must be 512. A strict state-dict load failure indicates a wrong model family or checkpoint; do not bypass it.
