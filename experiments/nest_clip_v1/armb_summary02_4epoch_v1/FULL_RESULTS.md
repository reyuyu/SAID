# S02 full recovery: prefix gate blocked

**BLOCKED_PREFIX_GATE** — full started from common step0, but stopped after
optimizer update5 and before update6. No4868 result or scientific classification
is claimed. Final export and five-set full evaluation have not run.

All5120 sample IDs/F/S/D strings/tokens and LR match the original smoke.
First three losses match exactly; update4/5 differ by0.005729675/0.011646271.
An unchanged original native five-update smoke independently reproduces
cross-run numerical differences. Optimizer counters/groups/LR/RNG match, but
model/adapter/moment tensor contents differ. The precise underlying mechanism
is not proven and the numerical gate has not been loosened.

See `PREFIX_REPRODUCIBILITY.md`, `PREFIX_REPRODUCIBILITY.json`,
`PREFIX_STATE_FORENSICS.json`, `FIRST_FIVE_GATE.json`, and
`TRAINING_DIAGNOSTICS.json`. All forensic checkpoints remain local and marked
do-not-resume. A future authorized retry must start fresh from common step0.
No S0.1 or other sampling/sparsity/loss/optimizer experiment was launched.
