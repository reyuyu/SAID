# SAID HNS-v1 ViT-L/14 Full4 Step4868 release

This branch is the GitHub companion for the private ModelScope model `hongchenyang/SAID-HNS-v1-ViT-L14-Full4-Step4868`.

The published model is the final epoch-4 endpoint of the verified four-epoch cosine run, with 4868 optimizer updates, ViT-L/14@224, 248-token text context and 768-dimensional native embeddings. It is not the independent Pure3 model. The source training identity is commit `c2b70f0805124e79e9e4dcb528b176189124843f` on `experiment/hns-v1-l14-full4epoch4868-v1`.

## Quick start

After receiving read access to the private ModelScope repository:

```bash
python -m pip install -r requirements-inference.txt
python scripts/download_hns_v1_l14.py --student-only --output-dir checkpoints/hns_v1_l14
python scripts/infer_hns_v1_l14.py \
  --checkpoint checkpoints/hns_v1_l14/inference/student_step4868.pt \
  --image assets/cat.webp \
  --texts "A cat holding a sign." "A dog on a bicycle."
```

The expected bare student SHA256 is `4a1e5ed69b7aa728d7189518b9f285052e8293290963e079d886ddcab5fe819e`. The inference loader is strict and uses the native student only; it performs no inference mask, reranking, ensemble or TTA.

See [the release model card](docs/model_release/hns_v1_l14_full4_epoch4/README.md), [inference details](docs/model_release/hns_v1_l14_full4_epoch4/INFERENCE.md), and [evaluation instructions](docs/model_release/hns_v1_l14_full4_epoch4/EVALUATION.md).

## Frozen retrieval results

Score5 **77.890862%**, J_long3 **83.191078%**, J_long **90.670000%**, Short4 **71.390%**, Urban Mean **95.25%**. Urban R@1 is **95.300% I2T / 95.200% T2I**. Full R@1/5/10 tables remain in the source experiment artifacts; no data files are redistributed here.

The ModelScope repository is private. A valid ModelScope token with read permission is required; never put a token in source, documentation, logs or commits.
