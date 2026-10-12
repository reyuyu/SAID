# SAID HNS-v1 ViT-L/14 — Full4 Step4868

This is the final epoch-4 model from a four-epoch cosine training schedule, not the independent Pure3 model. It is a private ModelScope release for the verified SAID HNS-v1 ViT-L/14 run.

ModelScope: `hongchenyang/SAID-HNS-v1-ViT-L14-Full4-Step4868`  
Training source: `experiment/hns-v1-l14-full4epoch4868-v1` at `c2b70f0805124e79e9e4dcb528b176189124843f`  
Backbone: OpenAI CLIP ViT-L/14@224, 768-dimensional native embeddings, 248-token text context.

## Download

Authenticate with a ModelScope token that can read the private repository, then run:

```bash
python scripts/download_hns_v1_l14.py --student-only --output-dir checkpoints/hns_v1_l14
```

To download the full training state and common Step0 as well:

```bash
python scripts/download_hns_v1_l14.py --output-dir checkpoints/hns_v1_l14
```

The script verifies every downloaded SHA256. Never put a token in a script, README, shell history, or commit message.

## Inference

```bash
python scripts/infer_hns_v1_l14.py \
  --checkpoint checkpoints/hns_v1_l14/inference/student_step4868.pt \
  --image examples/demo.jpg \
  --texts "A person riding a bicycle." "A dog running."
```

See [INFERENCE.md](INFERENCE.md) and [EVALUATION.md](EVALUATION.md) for CPU/CUDA use and native retrieval reproduction. The published student uses no inference-time mask or reranking.

## Results

Score5 **77.890862%**, J_long3 **83.191078%**, J_long **90.670000%**, Short4 **71.390%**, Urban Mean **95.25%**. Urban R@1 is **95.300% I2T / 95.200% T2I**. Complete R@1/5/10 tables and the Flickr30k-Full manifest are frozen in the GitHub training experiment.

## License and limitations

The repository source is Apache-2.0 and retains upstream SmartCLIP, LongCLIP and OpenAI CLIP attribution. Upstream model and dataset terms continue to apply. No dataset images or captions are included. This is a single-seed research release; the benchmark numbers are descriptive and are not a statistical significance claim.
