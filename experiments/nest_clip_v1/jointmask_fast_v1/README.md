# NEST JointMask fast v1

Controlled A3-RandomK comparison of text-only (`T-fast`), correctly paired
image-conditioned (`TI-fast`), and globally shuffled image-conditioned
(`TI-Shuffle-fast`) pair masks.

All groups use the shared step-0 checkpoint, 4×256 global training batches, scheduler
horizon 3651, and stop at 500 synchronized updates. The diagnosed common execution setting
is 128×128 pair tiles, pair checkpoint disabled, and encoder checkpoint enabled. T retains
the pair implementation; no matrix special case is enabled.

The frozen native evaluation list is COCO, Urban-1k, Flickr30k test1k, DOCCI, and Long-DCI.
DCI Full is excluded.
