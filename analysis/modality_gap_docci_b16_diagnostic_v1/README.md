# DOCCI Modality Gap double-cluster diagnosis

This independent read-only diagnosis uses the frozen 1,000-pair DOCCI sample, the first-sentence v2 feature set, and the verified Long-CLIP / HNS-v1 B16 checkpoints. It reruns the exact BalancedStack `pair_logits(..., paired=True) -> sigmoid -> hard_st` path to save per-sample mask, probability, logit, gate, and feature-norm statistics. Clustering is KMeans k=2 with `n_init=50`, `random_state=0` on the original 512-D normalized image features; stability and PCA-only comparisons are reported separately.

See `MODALITY_GAP_DIAGNOSTIC_REPORT.md` for the evidence-based interpretation. No optimizer, training step, checkpoint write, or production-model modification is performed.
