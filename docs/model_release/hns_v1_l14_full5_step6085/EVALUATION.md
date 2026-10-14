# Evaluation

The reported numbers use the native student encoders, full candidate pools, L2-normalized FP32 embeddings and inner-product retrieval. No inference mask, reranking, ensemble or TTA is used.

The repository's historical evaluation entry points can be used with the downloaded
`inference/student_step6085.pt` file and the corresponding private dataset manifests.
Dataset files are intentionally not distributed with this model.

## Frozen headline results

For Full5@6085, the verified headline values are Score5 77.968310%,
J_long3 82.423184%, J_long 90.775002%, Short4 71.286%, Urban-1k I2T/T2I
95.300%/95.700%, and Flickr30k-Full R@1 I2T/T2I 64.7013%/44.4093%.
Flickr30k-Full uses the verified 31,783-image / 158,915-caption protocol and
is distinct from Flickr30k-Test1K. The complete per-dataset R@1/5/10 tables
remain in the pinned GitHub training experiment; this release does not copy
dataset files or stale Full4 result tables.
