# Beta-CLIP versus SAID: comparison unavailable

Official checkpoint downloads failed, so Beta-CLIP native scores and deltas are unavailable. The existing SAID Balanced B16@4868 result is reused, not retrained or re-evaluated. TCI is excluded from the native comparison and Score5. The sole delta direction is BetaCLIP minus SAID.

| Model | Inference | COCO I2T/T2I % | Urban I2T/T2I % | Flickr I2T/T2I % | DOCCI I2T/T2I % | Long-DCI I2T/T2I % | Score5_R1 % |
|---|---|---|---|---|---|---|---:|
| Beta-CLIP CE official | Native CLS: not evaluated | N/A | N/A | N/A | N/A | N/A | N/A |
| Beta-CLIP BCE official | Native CLS: not evaluated | N/A | N/A | N/A | N/A | N/A | N/A |
| SAID Balanced B16 | Native dual tower | 61.700000/42.220000 | 92.100006/91.100007 | 88.900000/72.440000 | 78.760000/79.980000 | 59.668508/60.812944 | 72.768147 |

SAID J_long3=77.070244%; J_long=85.485003%.

## Reused SAID complete Recall

| Dataset | Direction | R@1 % | R@5 % | R@10 % |
|---|---|---:|---:|---:|
| COCO | I2T | 61.700000 | 84.100000 | 90.300000 |
| COCO | T2I | 42.220000 | 68.032000 | 77.720000 |
| Urban-1k | I2T | 92.100006 | 98.400003 | 99.300003 |
| Urban-1k | T2I | 91.100007 | 98.700005 | 99.200004 |
| Flickr30k-test1k | I2T | 88.900000 | 98.300000 | 99.400000 |
| Flickr30k-test1k | T2I | 72.440000 | 91.540000 | 95.500000 |
| DOCCI | I2T | 78.760000 | 95.720000 | 98.260000 |
| DOCCI | T2I | 79.980000 | 95.880000 | 98.200000 |
| Long-DCI | I2T | 59.668508 | 78.295185 | 83.793738 |
| Long-DCI | T2I | 60.812944 | 78.624046 | 84.109445 |

## Questions requested by the user

1. Can the official CE/BCE checkpoints reproduce README Urban scores? **Undetermined: no checkpoint obtained.**
2. What are their native CLS five-dataset scores under frozen SAID protocols? **Not evaluated.**
3. Which ten R@1 directions and aggregate scores beat SAID? **No comparison can be computed.**

No checkpoint SHA256, args, epoch, architecture or loss mode is inferred from script expectations. No training or substitution is performed. This run stops under the explicit unavailable-checkpoint rule.
