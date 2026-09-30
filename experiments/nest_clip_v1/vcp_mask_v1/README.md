# NEST VCP-Mask v1

Two fixed 500-update experiments from the shared NEST step-0 state:

- `VCP-Mask`: visual-conditioned original AttentionPool, with the A3 inclusion schedule.
- `TI-noInc`: the existing TI-fast architecture with `arm=A2`, retaining `condition_mode=joint_image` while fixing inclusion weight to zero.

The runner enforces a real four-GPU 5+30-step speed gate before smoke/formal execution. Native evaluation is limited to COCO, Urban-1k, Flickr30k test1K, and DOCCI.
