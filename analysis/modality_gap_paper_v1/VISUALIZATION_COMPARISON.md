# Modality Gap visualization comparison

This directory reuses the DOCCI First Sentence export from commit `124e89574d6ec586f5eaf4f5ff63f9679b3c89ff` on `analysis/modality-gap-docci-b16-v1`. The source contains 1,000 fixed samples, 512-dimensional unit-normalized image/text features, and the reported metrics in `input/modality_gap_metrics.csv`. No model, checkpoint, or sample-selection logic was changed. Every sample is retained in the scatter, histogram, KDE, and violin calculations; alpha is used only to make dense point clouds readable.

The complete PNG/PDF set is in `figures/`. The directly usable paper candidates are copied to `figures/main/`; the remaining PCA axis combinations and model-specific views are in `figures/appendix/`.

## Recommended paper figures

| Figure | Role | Why it is useful |
|---|---|---|
| `main/gap_axis_hist_kde_3x3` | Primary modality-gap figure | A single fixed Full-caption axis makes image/text separation and its caption-dependent change visible in one dimension. Centroid lines and density overlap show the effect without relying on a particular 2D rotation. |
| `main/caption_completeness_gap_trends_1x3` | Summary figure | Preserves the three original 512D quantities (centroid gap, paired cosine, paired L2) with bootstrap 95% bands. It is the clearest view of caption completeness effects. |
| `main/mask_displacement_gap_axis_violin` | Mechanism figure | Directly shows the distribution of Ours Masked minus Ours Native image displacement on the Ours Full gap axis, with zero and the textward fraction annotated. |
| `main/pca_shared_longclip_masked_pc1_pc2_2x3` | 2D context | A conventional shared PCA comparison that can accompany the 1D figure, while keeping the same points and centroid-gap segment. |

## Quantitative findings retained in the plots

The values below are recomputed from the 512D arrays and agree with the source CSV to below `8.1e-9`; see `tables/metrics_consistency_check.csv` and `tables/metrics_with_bootstrap_ci.csv`.

| Model | Full gap | 50% gap | First Sentence gap | Full cosine | 50% cosine | First Sentence cosine |
|---|---:|---:|---:|---:|---:|---:|
| Long-CLIP | 1.049 | 1.036 | 0.965 | 0.309 | 0.310 | 0.320 |
| Ours Native | 0.778 | 0.764 | 0.792 | 0.372 | 0.353 | 0.327 |
| Ours Masked | 0.686 | 0.704 | 0.773 | 0.332 | 0.312 | 0.294 |

Ours Masked has the smallest centroid gap for all three caption settings. Its advantage over Long-CLIP is `0.363`, `0.332`, and `0.192` in Full, 50% Prefix, and First Sentence, respectively. Masked image features move along the Ours Full gap axis by mean `-0.055`, `-0.027`, and `+0.001` for the three caption settings; negative is toward the text centroid. The fraction of samples moving textward is 63%, 49%, and 41%. These distributions are shown in `mask_displacement_gap_axis_violin`.

## Answers to the requested questions

1. **Best main-text figure:** the 3×3 `gap_axis_hist_kde_3x3` is the most direct modality-gap visualization. It explicitly shows both distributions, both centroids, and the projected separation under the same axis definition. Pair it with the trend summary when space permits.
2. **Most direct caption-completeness view:** `caption_completeness_gap_trends_1x3`, especially its centroid-gap panel. It shows that Long-CLIP's gap decreases as captions become shorter, while Ours Masked's gap increases and Ours Native stays nearly flat with a shallow U-shape. This is a model-dependent caption effect rather than a monotone universal rule.
3. **Where masked gap mitigation is clearest:** the gap-axis distributions and the centroid-gap trend. In every caption column, Ours Masked has the smallest image/text centroid separation. The mask-displacement violin adds the mechanism: the Full-caption distribution is shifted textward, and the shift weakens as the caption becomes less complete.
4. **Why Ours Native can look better:** Native has higher paired cosine and lower paired L2 in all three caption settings. Those are instance-level image/text pairing metrics, whereas centroid gap measures global modality alignment. Masking compresses the global gap and changes feature norms (`image centroid norm`: 0.616/0.661/0.715 for Masked versus 0.715 for Native) but can reduce pairwise semantic alignment; the figures should therefore be read together.
5. **Appendix-only figures:** PC2–PC3 and per-model PCA views are useful audits but are sensitive to variance choices and are harder to compare across models. The positive/shifted-negative similarity grid is also best in the appendix because it emphasizes pairwise alignment rather than the modality centroid gap. The model-specific 1×3 gap-axis plots are useful for detailed inspection but duplicate the compact 3×3 main view.

## Interpretation and safeguards

- The gap axis is the normalized Full-caption `(image centroid − text centroid)` vector computed separately for each model. Projected `Δμ_axis` is therefore a one-dimensional diagnostic; the exact Euclidean centroid gap is always reported in the trend plot and CSV table.
- Similarity negatives are a fixed, reproducible 137-sample cyclic shift of text features. They are a reference distribution, not a claim about a particular retrieval-negative sampler.
- Histogram density, KDE bandwidth, alpha, point size, PCA component pairs, and bootstrap settings are recorded in `PLOT_PARAMS.json`. The finite alternatives retained in the sweep are listed in `PARAMETER_SWEEP_SUMMARY.csv`.
- No UMAP/t-SNE result is used as a main claim: the requested source data are sufficient for PCA and the gap-axis view, and adding a stochastic nonlinear embedding would make cross-caption interpretation less direct.
