# Modality centroid-gap figure and table

This directory contains the reproducible artifacts for the DOCCI modality centroid-gap analysis used in the paper's one-figure/one-table presentation.

## Contents

- `input/`: the shared 1,000-sample DOCCI export, 512D features, caption-length metadata, metrics, and provenance.
- `fullcaption_main/`: the recommended Full-caption 2x3 figure (`modality_gap_main_2x3_fullcaption.{pdf,png}`), its metric CSV, parameter record, and notes.
- `formal_docci_table/`: the single-column DOCCI LaTeX table (`modality_gap_docci.tex`), PDF/PNG preview, source CSV, and verification record.
- `figures/main/`: main-text candidates for caption-completeness trends, gap-axis distributions, masking displacement, and shared PCA.
- `figures/appendix/`: PCA, similarity, gap-axis, and displacement alternatives retained for appendix comparison.
- `make_visualizations.py`: generator for the analysis figures.
- `formal_docci_table/make_docci_table.py`: generator and CSV verification for the formal DOCCI table.
- `PLOT_PARAMS.json`, `PARAMETER_SWEEP_SUMMARY.csv`, `FIGURE_MANIFEST.json`, and `VISUALIZATION_COMPARISON.md`: parameter provenance and comparison records.

## Data and metric definition

All features are 512-dimensional and use the same 1,000 DOCCI image-caption pairs in the shared sample manifest. The centroid gap is the Euclidean distance between the image and text centroids in the original 512D embedding space. The Full-caption main figure uses Long-CLIP, Sema-CLIP (Global; exported as `ours_native`), and Sema-CLIP (Masked; exported as `ours_masked`). The table reports DOCCI Full, Random 50%, and Random 10% conditions; the random conditions use the five-seed `random_distributed` mean from the formal CSV.

The feature export and historical metrics are reused from the existing SAID analysis artifacts. No training, checkpoint modification, sample re-selection, or feature extraction is performed by the plotting scripts.

## Reproduction

From this directory, install NumPy, pandas, Matplotlib, SciPy, scikit-learn, and LaTeX dependencies used by the scripts, then run:

```bash
python make_visualizations.py
python formal_docci_table/make_docci_table.py
```

The scripts read `input/features.npz`, `input/modality_gap_metrics.csv`, and the formal table CSV and write the figure/table artifacts into their respective directories. The private absolute paths recorded in the provenance JSON are historical source-environment paths; hashes and shared sample IDs are retained for auditability.
