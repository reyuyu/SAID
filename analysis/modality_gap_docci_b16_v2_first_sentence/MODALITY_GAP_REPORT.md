# Phase 2 follow-up: first-sentence caption condition

The former 30% prefix condition was replaced with the first sentence of each caption. The previous 30% result is preserved in the v1 directory. This run uses the same 1,000 seed-0 DOCCI pairs and the same verified Long-CLIP and HNS-v1 B16 checkpoints.

| Model | Full gap | 50% gap | First-sentence gap | Full cosine | 50% cosine | First-sentence cosine |
|---|---:|---:|---:|---:|---:|---:|
| Long-CLIP | 1.048990 | 1.035737 | 0.964834 | 0.308560 | 0.309811 | 0.319803 |
| Ours masked | 0.686359 | 0.703691 | 0.772812 | 0.332476 | 0.312464 | 0.294320 |
| Ours native | 0.778371 | 0.763776 | 0.791741 | 0.371744 | 0.353470 | 0.327038 |

Average kept caption lengths are 135.73 BPE tokens (Full), 68.76 (50% prefix), and 22.87 (first sentence). Ours masked mean keep ratios are 0.6116, 0.5980, and 0.6083.

The first-sentence condition increases Ours masked centroid gap by 0.086453 from Full and lowers its paired cosine by 0.038156. Ours masked still has a smaller centroid gap than Long-CLIP at the first-sentence condition (0.772812 versus 0.964834), but the relative advantage is reduced. Long-CLIP's gap again decreases rather than increases under this probe. Ours native remains a useful control: its first-sentence gap is 0.791741, only 0.018929 above Ours masked, while its paired cosine remains higher.

These are geometric observations from one checkpoint, one seed, and 1,000 DOCCI pairs. They do not establish retrieval gains or causal claims about semantic masking.
