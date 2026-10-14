# Provenance and reproduction

The training source is pinned to commit `8a0b6e8894c166d40bced43625032fa7bbca61fe` on `experiment/hns-v1-l14-full5epoch6085-fromscratch-v1`. The run used ViT-L/14@224, four GPUs, 256 samples/rank, global batch 1024, seed 0, 248-token text, native BF16 autocast with FP32 master parameters, and a five-epoch cosine horizon of 6085 updates. The selected endpoint is update 6085.

The private ModelScope repository contains the student, the full training checkpoint and the verified common Step0. The checkpoint stores optimizer moments, per-rank RNG/sampler state, the completed-step counter and scheduler horizon; the cosine scheduler is reconstructed from that horizon and counter rather than stored as a separate object. It does not contain ShareGPT4V, Urban-1k, COCO, Flickr30k, DOCCI or Long-DCI files. Users must obtain those datasets lawfully and follow their terms.

The OpenAI CLIP ViT-L/14 initializer is identified by SHA256 `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`. The trained student is identified independently by the SHA256 in `MODEL_IDENTITY.json`.
