# DOCCI Modality Gap — Long-CLIP vs Sema-CLIP

This is a fixed-seed (0), 1,000-pair DOCCI exploratory analysis. Metrics are computed in the original 512-D L2-normalized feature space; plots use one shared PCA fit for all six main panels.

PCA explained variance: PC1=0.233606, PC2=0.105252.

## Caption lengths (BPE tokens including SOT/EOT)

| Setting | Mean | Min | Max |
|---|---:|---:|---:|
| Full | 135.73 | 35 | 246 |
| 50% prefix | 68.76 | 18 | 204 |
| 30% prefix | 41.54 | 11 | 123 |

## Metrics

| Model | Caption | Centroid gap | Paired cosine | Paired L2 | Mask keep |
|---|---|---:|---:|---:|---:|
| Long-CLIP | Full | 1.048990 | 0.308560 | 1.175900 | — |
| Long-CLIP | 50% prefix | 1.035737 | 0.309811 | 1.174780 | — |
| Long-CLIP | 30% prefix | 1.008667 | 0.314210 | 1.170974 | — |
| Ours masked | Full | 0.686359 | 0.332476 | 1.154985 | 0.6116 |
| Ours masked | 50% prefix | 0.703691 | 0.312464 | 1.172094 | 0.5980 |
| Ours masked | 30% prefix | 0.734860 | 0.302094 | 1.180895 | 0.6013 |
| Ours native | Full | 0.778371 | 0.371744 | 1.120183 | — |
| Ours native | 50% prefix | 0.763776 | 0.353470 | 1.136208 | — |
| Ours native | 30% prefix | 0.770465 | 0.339486 | 1.148428 | — |

## Trend summary

- Long-CLIP centroid gap Full→50%→30%: 1.048990 → 1.035737 → 1.008667
- Ours masked centroid gap Full→50%→30%: 0.686359 → 0.703691 → 0.734860
- Ours native centroid gap Full→50%→30%: 0.778371 → 0.763776 → 0.770465

The plot is descriptive and does not establish retrieval gains or causality. The Ours masked image feature follows the training-time caption-conditioned mask path; Ours native is an explicit no-mask control.
