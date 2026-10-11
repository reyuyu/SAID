# ViT-B/16 component ablation: A → B → C → D

This isolated branch compares the historical HNS-v1 Full arm (A) with three
predeclared objectives at the same 4-epoch cosine horizon (4868) and the same
step-3651 stop.  A is a read-only historical reference; B, C, and D each start
from the immutable common step-0 checkpoint and run independently.

| arm | objective | updates | scheduler horizon |
|---|---|---:|---:|
| A | historical Full HNS-v1 | reference 3651 | 4868 |
| B | Joint visual/text mask, F-only | 3651 | 4868 |
| C | Native F/Dall/D3 alignment, no mask | 3651 | 4868 |
| D | Full HNS with text-only mask logits | 3651 | 4868 |

The sequence is strictly serial.  A failed smoke, resource probe, training
audit, checkpoint restore, or native evaluation stops the remaining arms.
Public retrieval sets are exploratory comparisons and are not used for arm
selection.  No updates after 3651 are run.
