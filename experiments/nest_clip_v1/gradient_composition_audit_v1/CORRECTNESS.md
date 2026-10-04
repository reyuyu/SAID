# Correctness contract

No training and no parameter updates: the audit constructs no optimizer, calls no step or scheduler, and traps exposed optimizer/scaler step methods. Each diagnostic batch hashes every model state tensor before and after on all4 ranks. First/last batches also repeat F I2T backward with bitwise deterministic gradient checks. Checkpoint file SHA256 is checked before/after each stage and globally.

Five targeted tests pass: read-only frame-local exposure reproduces production loss and every parameter gradient bitwise at both equal and Arm B weights; native Full CE has no auxiliary gradients; bounded per-parameter FP64 Gram matches direct dot products; full-FP32 component gradients reconstruct the actual weighted alignment gradient.

The successful real4-card first-batch pilot verifies1024×1024 native candidates, diagonal positives, direct same-embedding top1 ranking agreement, all16 objective backwards and20 actual gradient tensors per backward identical across ranks. Two failed preliminary checks are preserved separately: an initially too-strict bitwise matrix comparison rejected harmless FP32 GEMM tiling differences (maximum2.098e-5); a command-label edit then had a syntax error before model execution. Both were corrected before the full96-batch audit.

Default production sources are byte-for-byte unchanged from b85f73a. Live masked losses are captured from existing production function return-frame locals using an experiment-local Python profile callback; no detached CE log is used as a differentiable loss.

Native reference uses actual native image/text towers, scale100, full1024 candidates and no mask/fusion. Backbone autocast matches BF16 training. FP64 accumulators receive bounded tiles of a single parameter's gradients, never a concatenated150M-parameter vector. Native zero/absent auxiliary gradients are labeled N/A for cosine, rather than zero alignment.

BF16 backward quantization means separately backwarded gradients need not sum bitwise to a combined backward. The pilot's overall alignment reassembly relative L2 difference was0.002871; its vision group was0.011057 and text group0.001864. All FP32 mask/adapter/gate groups agreed within about5.2e-7. Report actual errors for every batch; one full-FP32 real-batch control per stage separately validates mathematical linearity without this quantization. Counterfactual weights are local linear gradient diagnostics, not training outcomes.

Main inclusion uses production inclusion_weight(A3,200)=1, shared by all checkpoints, so initialization stage does not zero inclusion. The appendix uses0.5 by exactly scaling the measured inclusion gradient. Every main objective is backwarded via a real DDP-wrapped module and receives standard all-reduce averaging. The profile wrapper reads production mathematics only.
