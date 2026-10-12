# Ablations and scope

HNS-v1 trains a native CLIP student with training-time semantic mask and fusion auxiliaries. The published inference artifact is the student encoder only. It does not contain a usable MaskNet/FusionBranch or AdamW continuation state.

The historical SAID ablations compare component and loss choices under separately tracked seeds and checkpoints. They are evidence about the training method, not additional inference modes of this file. A bare student is sufficient for retrieval; exact continuation requires the full Step4868 training checkpoint, the matching source commit, the common Step0, the fixed sampler and the private training data.

These single-seed, repeatedly observed public-benchmark results are descriptive. They do not establish statistical significance or an unbiased benchmark SOTA claim.
