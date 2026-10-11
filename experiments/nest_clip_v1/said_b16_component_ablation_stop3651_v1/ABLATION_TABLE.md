# ViT-B/16 component ablation status

| Arm | Step3651 | Five-set native evaluation | Status |
|---|---:|---|---|
| A Full (historical) | yes | historical reference retained | complete reference |
| B Joint-Mask Only | no (stopped at 2261) | not run | engineering failure |
| C Nested Alignment Only | not started | not run | stopped after B |
| D Full + Text-only SmartMask | not started | not run | stopped after B |

B smoke and the 35-update resource probe passed.  The formal run then hit
three consecutive complete-cycle timings over the trainer's 3-second stop gate
at steps 2259–2261 and exited with a DataLoader worker abort during teardown.
The runtime checkpoint and logs are retained locally; no incomplete model is
used for comparison or export.
