# DOCCI Modality Gap — Long-CLIP vs Sema-CLIP

This is a fixed-seed (0), 1,000-pair DOCCI exploratory analysis. Metrics are computed in the original 512-D L2-normalized feature space; plots use one shared PCA fit for all six main panels.

PCA explained variance: PC1=0.231368, PC2=0.096362.

## Caption lengths (BPE tokens including SOT/EOT)

| Setting | Mean | Min | Max |
|---|---:|---:|---:|
| Full | 135.73 | 35 | 246 |
| 50% prefix | 68.76 | 18 | 204 |
| First sentence | 22.86 | 3 | 64 |

## Metrics

| Model | Caption | Centroid gap | Paired cosine | Paired L2 | Mask keep |
|---|---|---:|---:|---:|---:|
| Long-CLIP | Full | 1.048990 | 0.308560 | 1.175900 | — |
| Long-CLIP | 50% prefix | 1.035737 | 0.309811 | 1.174780 | — |
| Long-CLIP | First sentence | 0.964834 | 0.319803 | 1.166118 | — |
| Ours masked | Full | 0.686359 | 0.332476 | 1.154985 | 0.6116 |
| Ours masked | 50% prefix | 0.703691 | 0.312464 | 1.172094 | 0.5980 |
| Ours masked | First sentence | 0.772812 | 0.294320 | 1.187467 | 0.6083 |
| Ours native | Full | 0.778371 | 0.371744 | 1.120183 | — |
| Ours native | 50% prefix | 0.763776 | 0.353470 | 1.136208 | — |
| Ours native | First sentence | 0.791741 | 0.327038 | 1.159182 | — |

## Trend summary

- Long-CLIP centroid gap Full→Full→50% prefix→First sentence: 1.048990 → 1.035737 → 0.964834
- Ours masked centroid gap Full→Full→50% prefix→First sentence: 0.686359 → 0.703691 → 0.772812
- Ours native centroid gap Full→Full→50% prefix→First sentence: 0.778371 → 0.763776 → 0.791741

The plot is descriptive and does not establish retrieval gains or causality. The Ours masked image feature follows the training-time caption-conditioned mask path; Ours native is an explicit no-mask control.
