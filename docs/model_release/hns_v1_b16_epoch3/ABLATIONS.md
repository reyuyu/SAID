# Controlled ablations

There are three distinct workflows:

1. **Inference and feature analysis.** The bare student is sufficient for image/text embeddings, similarity, retrieval and feature statistics.
2. **Continuation.** Use the full training checkpoint to restore CLIP, MaskNet/Fusion adapter, AdamW moments, scheduler horizon 4868, RNG, sampler and cursor. Changing a loss or component makes a new experiment; it is no longer an exact continuation.
3. **From common step 0.** Use `common_step000000.pt`, the original HNS-v1 source commit, the frozen config and legally obtained ShareGPT4V data/index. Compare one component at a time: visual semantic mask, nested alignment, sparsity, or HNS inclusion/hierarchy. Do not report invented scores or mix results from other SAID arms.

Reference controls are ViT-B/16, four planned epochs, horizon 4868, nested_detail_d3, balanced_stack/patch, view weights `[1.35, 1.35, 0.30]`, sparsity weights `[1, 2, 2]`, sparsity scale 1, inclusion max 0, HNS beta `[2, 2]`, seed 0. Training requires the original data and substantial multi-GPU memory; inference memory is a separate, much smaller requirement. Dataset files are not included.
