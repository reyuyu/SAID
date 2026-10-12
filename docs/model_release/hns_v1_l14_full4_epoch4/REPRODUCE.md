# Provenance and reproduction

The training source is pinned to commit `c2b70f0805124e79e9e4dcb528b176189124843f` on `experiment/hns-v1-l14-full4epoch4868-v1`. The run used ViT-L/14@224, four GPUs, 256 samples/rank, global batch 1024, seed 0, 248-token text, native BF16 autocast with FP32 master parameters, and a four-epoch cosine horizon of 4868 updates. The selected endpoint is update 4868.

The private ModelScope repository contains the student, the full training checkpoint and the verified common Step0. It does not contain ShareGPT4V, Urban-1k, COCO, Flickr30k, DOCCI or Long-DCI files. Users must obtain those datasets lawfully and follow their terms.

The OpenAI CLIP ViT-L/14 initializer is identified by SHA256 `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`. The trained student is identified independently by the SHA256 in `MODEL_IDENTITY.json`.
