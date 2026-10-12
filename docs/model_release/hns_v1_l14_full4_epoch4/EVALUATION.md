# Evaluation

The reported numbers use the native student encoders, full candidate pools, L2-normalized FP32 embeddings and inner-product retrieval. No inference mask, reranking, ensemble or TTA is used.

The release script runs Urban-1k directly:

```bash
python scripts/eval_hns_v1_l14.py \
  --checkpoint checkpoints/hns_v1_l14/inference/student_step4868.pt \
  --dataset urban1k --data-root /path/to/Urban1k --output urban.json
```

It also supports COCO-Val5K through the repository canonical annotation layout and arbitrary historical protocols through JSONL manifests. Each manifest row must contain `image_id`, `image_path`, `caption`, and `positive_image_id`; the image root is passed separately. Dataset files are intentionally not distributed with this model.

## Frozen results

| Dataset | I2T R@1 | T2I R@1 |
|---|---:|---:|
| COCO-Val5K | 64.760 | 48.620 |
| Flickr30k-Test1K | 93.200 | 78.980 |
| DOCCI | 85.440 | 86.180 |
| Long-DCI | 64.733 | 66.496 |
| Urban-1k | 95.300 | 95.200 |

Score5 is 77.890862%, J_long3 is 83.191078%, J_long is 90.670000%, Short4 is 71.390%, and Urban Mean is 95.25%. The verified Flickr30k-Full protocol uses 31,783 images and 158,915 captions; its R@1 values are 65.141742 (I2T) and 44.396061 (T2I). Source result files are `RESULTS_4868.json` and `FLICKR_FULL_RESULTS.json` in the training experiment.
