# SAID HNS-v1 ViT-L/14 — Full5 Step6085

This private ModelScope repository contains the verified SAID HNS-v1 ViT-L/14 model trained independently from the common Step0 through five epochs, ending at update 6085.

ModelScope: `hongchenyang/SAID-HNS-v1-ViT-L14-Full5-Step6085`  
Training source: `experiment/hns-v1-l14-full5epoch6085-fromscratch-v1` at `8a0b6e8894c166d40bced43625032fa7bbca61fe`  
Backbone: OpenAI CLIP ViT-L/14@224, 768-dimensional native embeddings, 248-token text context.

## Download

Authenticate with a ModelScope token that can read the private repository. The download script verifies every file against `CHECKSUMS.sha256`:

```bash
python scripts/download_hns_v1_l14_full5_step6085.py \
  --student-only --output-dir checkpoints/hns_v1_l14_full5_step6085
```

To download the complete training state and common Step0 as well:

```bash
python scripts/download_hns_v1_l14_full5_step6085.py \
  --output-dir checkpoints/hns_v1_l14_full5_step6085
```

Never put a token in a script, README, shell history, command line, or commit message.

## Native image-text inference

Inference uses only the exported student CLIP encoder. Training-time MaskNet, FusionBranch and HNS are not used.

```bash
python scripts/infer_hns_v1_l14_full5_step6085.py \
  --checkpoint checkpoints/hns_v1_l14_full5_step6085/inference/student_step6085.pt \
  --image examples/demo.jpg \
  --texts "A person riding a bicycle." "A dog running."
```

The loader performs a strict ViT-L/14 state-dict check, encodes text with context length 248, returns 768-dimensional L2-normalized image/text embeddings, and prints cosine similarities. The first model construction may download the upstream OpenAI ViT-L/14 initializer for architecture construction; it is not the trained checkpoint.

## Verified results

| Metric | Full5@6085 |
|---|---:|
| Score5 | 77.968310% |
| J_long3 | 82.423184% |
| J_long | 90.775002% |
| Short4 | 71.286% |
| Urban-1k I2T / T2I | 95.300% / 95.700% |
| Flickr30k-Full I2T / T2I R@1 | 64.7013% / 44.4093% |

The complete native five-dataset R@1/5/10 results are recorded in the GitHub training experiment. Flickr30k-Full is the verified 31,783-image / 158,915-caption token protocol and is distinct from Flickr30k-Test1K. No inference mask, reranking, ensemble, or TTA was used.

## Checksums and identity

- Bare student: `f360465fc02a5465d312c672d5179dcd8991d9c98ffd14325831b9150a8f7da1`
- Full training checkpoint: `8a036746e44a789e69d40987cf6852ee9603b6881e30ef30ab88c8463bcec538`
- Common Step0: `8fa8f7819f41f01635b95c0c7aed3c2a51f8dd58317666cacd74d3bc162fc912`

## License and limitations

The repository source is Apache-2.0 and retains upstream SmartCLIP, LongCLIP and OpenAI CLIP attribution. Upstream model and dataset terms continue to apply. No dataset images or captions are included. This is a single-seed research release; benchmark values are descriptive and are not a statistical significance claim.
