# DOCCI modality-gap analysis (Long-CLIP vs Sema-CLIP)

This directory contains the fixed-seed (`seed=0`) 1,000-pair DOCCI analysis requested for Phase 2. `extract_features.py` performs the model-specific extraction and `analyze_and_plot.py` computes 512-D metrics and produces the figures.

The six main panels use one PCA fit over all Long-CLIP and Ours-masked image/text features. Ours-native is projected with the same transform. `modality_gap_metrics.csv` contains centroid gap, paired cosine, paired L2, and Ours mask keep ratio. `provenance.json` records the manifest, model checkpoint paths and hashes, extraction configuration, finite checks, and sampling list.

The output is descriptive: it uses 1,000 DOCCI pairs and one random seed, and it does not establish retrieval improvements or causal claims.
