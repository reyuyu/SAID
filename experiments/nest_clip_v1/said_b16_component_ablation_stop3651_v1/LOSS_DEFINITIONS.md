# Ablation objectives

* **A:** historical Full HNS-v1 result, retained unchanged.
* **B:** `10 * L_mask(F) + S_mask(F)`, where `L_mask` uses the complete global
  image/text candidate matrix and `S_mask` is the original F positive-mask
  sparsity term. Dall, D3, inclusion, and HNS are absent from the graph.
* **C:** `10/3 * (1.35*L_native(F) + 1.35*L_native(Dall) +
  0.30*L_native(D3))`, with normalized native CLIP embeddings and symmetric
  `100 * image @ text.T` CE. Mask/fusion parameters remain frozen.
* **D:** the historical Full objective, with mask logits generated from the
  text condition only.  The image embedding remains in masked similarity;
  replacing an image cannot change the mask logits.

The production HNS loss and retrieval evaluator are unchanged.  The only
production-source edit in this branch is the isolated text-only mask dispatch
and the explicit ablation trainer/module path.
