# Summary alignment dose correctness

Two preregistered arms: E1[1.4,0.2,1.4], E2[1.5,0,1.5]. The only production edit relaxes weight validation from positive to nonnegative, requiring a positive total. Every production mathematical function/class outside hparams validation remains AST-identical to bbe5daa; forward is unchanged. No branch is removed or skipped.

E2 is **zero Summary alignment weight**, not removal of Summary supervision. The Summary view and raw I2T/T2I CE remain computed. Its original2/3 sparsity term and the original inclusion target/ramp remain. Tests verify zero gradient through the Summary CE alignment route and nonzero gradient through Summary sparsity. Negative, nonfinite and all-zero weight configurations remain rejected.

26 targeted/regression tests passed, including explicit CE/gradient references for both registered weights and bitwise auxiliary/directional CE invariance. Preflight compares1000 real samples to the pinned Arm B sampler and all500×4 sampler-ID hashes to S0.4. Live training verifies each F/S/D raw/token/K/detail-index stream against S0.4.

Each arm independently runs fresh smoke5 and fresh formal500 from the common step0; resume=None, horizon4868,4×256 candidates,accumulation1. Frozen native inference/evaluators and strict export are unchanged. No adaptive weights or full training are launched.

Last50 raw and weighted CE describe actual training updates451..500. **Last50 weighted gradient norm shares** are measured separately by read-only replay of those50 input batches at the final step500 checkpoint. They are not per-update historical gradients, because only final checkpoint is retained. The native-gradient spot check uses the same first8 seed0 global batches as previous S0.4/B audits. Diagnostic backward passes never call optimizer/scaler step; model digests and checkpoint SHA are verified unchanged. Norm shares are diagnostic, not exact summed-gradient contribution percentages.
