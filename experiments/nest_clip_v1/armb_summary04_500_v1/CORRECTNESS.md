# Arm B S0.4 correctness

This experiment changes only normalized alignment weights from[1.2,0.6,1.2] to[1.3,0.4,1.3]. All production model, data, optimizer, scheduler, sampling/RNG, inclusion ramp and native evaluator sources remain byte-identical to b85f73a. Six directional CE weights remain equal. The weight sum is3.

Two tests pass: actual S0.4 loss and gradients match explicit global CE reference; directional CE, sparsity and inclusion remain bitwise identical to S0.6 on the same model/input. Preflight proves every F/S/D string, token tensor,K and detail-index field equals the pinned Arm B sampler on1000 seed0 real samples. All500×4 live sample/F/S/D/token/K/index digests are checked against original Arm B during training.

Smoke5 and formal500 each construct fresh models/empty AdamW state from common step0. Formal never resumes from smoke or a trained model; horizon4868,256/rank×4,accumulation1. Strict bare export and unchanged five-dataset native evaluation follow.

Step1/100/200/500 and last50 diagnostics retain raw directional CE, normalized weighted contributions, keep/inclusion/S-D IoU and gate statistics. Gate/F-D diagnostics in last50 have only one scheduled observation; CE/keep/inclusion/S-D IoU use all50.

A separate read-only spot check uses only8 fixed1024-candidate global batches and five objectives (F/S/D combined,actual alignment,native Full CE). No full32-batch audit is rerun. It uses standard synchronized DDP backwards, per-batch model digests, checkpoint SHA checks and hard optimizer/scaler step traps. Full gradients and checkpoints remain outside Git.
