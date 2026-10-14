# Full5 training plan and scope

- Model: Sema-CLIP ViT-L/14, `dual_branch` + `balanced_stack` + patch mask.
- Initialization: audited common Step0, not a historical trained checkpoint.
- Horizon: 6085 synchronized optimizer updates (five epochs); stop exactly at Step6085.
- Frozen variables: model structure, losses, sampling, data order, optimizer groups, BF16/checkpointing, seed and evaluation protocol.
- Evaluation nodes: Step4868 and Step6085, using strict bare-student exports.
- Stop rule: no Step6, no new seed, no additional training arm.
