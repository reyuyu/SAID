# View-Relation / Cross-View Sibling Discrimination500

Use the original verified approximately2-second B16 Balanced runtime, compact
old-R, best coefficients2e-4/1/[1,1,1]/1/1 and common step0. Reuse the completed
matched500/H4868 baseline (full SHA f44e3ab0566a12e189514a92fedac3416299f87ca129b649a2c76a63b222b3f8).
No baseline retraining or runtime optimization.

Only change the relation term: original inclusion plus equal-weight, unscaled
FP32 zero-margin sibling violation. P text compares its own Hard-ST mask to
detached R mask; R text compares its own mask to detached P mask. Image and text
retain gradients in all comparisons. Reuse existing positive masks and P/R
embeddings, with no extra encoder, new mask or CE negative/denominator changes.
Keep the original200-update ramp, global valid reduction and F-only fallback.

Correctness tests include isolated gradient directions and zero-sibling bitwise
loss, all-parameter-gradient and AdamW recovery. Four-rank explicit-global tests
check scaling across normal, sparse, global1/global0 fallback and tail cases.
The5-step full-batch smoke is independent from the fresh shared-step0 formal500.

Monitor resources under3s/65GiB and positive keep/P-R IoU. Freeze the diagnostic
stop guard before training:25-update running means,8 consecutive bad windows,
IoU below max(.15,.25*matched baseline window mean) or any F/P/R positive keep
outside(.02,.995). Stop and record; never retune coefficients or mask semantics.

The supervisor then exports/strictly checks the bare student, runs the five frozen
native protocols once, writes all R@1/5/10, ten R1 deltas, raw scores and relation
diagnostics at1/100/200/500/last50, and pushes compact evidence to
codex/nest-balanced-view-relation-500-v1. Full checkpoints/data/cache remain local.

Runtime: /root/lk_projects/SAID-nest-clip-v1/balanced_view_relation_500_v1.
Stop after500 and five evaluations. No coefficient, margin, transpose constraints,
2x2 CE, cross-mask gradients, sampling variants, seeds or full4868 promotions.
