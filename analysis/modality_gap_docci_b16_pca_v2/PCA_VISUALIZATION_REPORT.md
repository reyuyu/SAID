# DOCCI V2 First Sentence — PCA Visualization

This is a feature-only, fixed-seed visualization using the existing DOCCI V2 First Sentence `features.npz`; no checkpoint was loaded and no features were re-extracted.

The main 2×3 layout has Long-CLIP in row 1 and Ours Masked in row 2, with Full, 50% Prefix, and First Sentence in columns. Image points are blue and text points orange (`s=3`, `alpha=0.12`); light KDE contours and both centroids are overlaid.

## PCA schemes

- Shared PCA: one fit over all 12,000 Long-CLIP and Ours Masked image/text points across the three caption settings.
- Per-model PCA: Long-CLIP and Ours Masked each have an independent fit over their own 6,000 points. Each row shares its own projection across columns; row coordinates must not be compared across models.
- PC2–PC3 versions are included as a fixed sensitivity view, with explained variance recorded in `PCA_PARAMETERS_AND_VARIANCE.json`.

## Explained variance

| Fit | PC1 | PC2 | PC3 |
|---|---:|---:|---:|
| Shared | 23.14% | 9.64% | 3.94% |
| Long-CLIP per-model | 49.08% | 1.99% | 1.95% |
| Ours Masked per-model | 16.12% | 3.41% | 3.15% |

## 512-D quantitative metrics

| Model | Caption | Centroid Gap | Paired Cosine | Paired L2 |
|---|---|---:|---:|---:|
| longclip | full | 1.048990 | 0.308560 | 1.175900 |
| longclip | prefix50 | 1.035737 | 0.309811 | 1.174780 |
| longclip | first_sentence | 0.964834 | 0.319803 | 1.166118 |
| ours_masked | full | 0.686359 | 0.332476 | 1.154985 |
| ours_masked | prefix50 | 0.703691 | 0.312464 | 1.172094 |
| ours_masked | first_sentence | 0.772812 | 0.294320 | 1.187467 |

## Visual comparison

The shared PCA preserves one global coordinate system and is the appropriate reference for comparing the two rows. Per-model PCA usually makes within-model local structure easier to inspect, but its two rows have independently rotated/scaled PCA axes and their cross-row distances are not scientifically comparable. The PC2–PC3 views are a sensitivity check rather than a replacement for the primary PC1–PC2 views.

The image/text double-cluster structure remains visible under both PCA fitting schemes. KDE contours make local density and shape easier to see, while the original 512-D Centroid Gap remains the quantitative modality-gap measure. Points were not filtered, shifted, rescaled by modality, or manually cleaned.

PCA is better suited for the paper main figure when the question is global cluster separation and cross-row comparison. Per-model PCA and PC2–PC3 are useful supplementary diagnostics for local structure and sensitivity.
