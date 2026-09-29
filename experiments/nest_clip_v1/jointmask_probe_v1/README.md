# NEST JointMask probe v1

Controlled A3-RandomK comparison of text-only (`T`), correctly paired image-conditioned
(`TI`), and globally shuffled image-conditioned (`TI-Shuffle`) pair masks. All runs use
the shared step-0 checkpoint, a 3651-step scheduler horizon, and independent 5-step smoke
and 500-step formal output directories.

Formal native evaluation is prepared for COCO, Urban-1k, Flickr30k test1K, DOCCI, and
Long-DCI. DCI Full is intentionally excluded by the user's latest instruction.
