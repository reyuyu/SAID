# Evaluation

Use `scripts/eval_hns_v1_b16.py --checkpoint ... --dataset urban1k --data-root /path/to/Urban1k --device cuda:0`. The script delegates to the repository's native evaluator and accepts explicit local roots. It uses 248-token text, the reference CLIP image preprocessing, FP32-normalized embeddings, a full candidate pool and the dataset's native positive mapping.

The same entry point exposes `coco` and `urban1k` today. DOCCI, Long-DCI and Flickr variants require their manifests and dataset-specific reference evaluators; no public script should silently replace Long-DCI or FlickrFull with another candidate pool. Historical numbers in the model card are frozen observations.
