# Authorized S02 full recovery run

Authorization: user instruction on2026-10-05, after DATA_AUDIT_PASS and S02
five-update smoke. Branch `codex/nest-balanced-armb-summary02-4epoch-recovery-v1`.

Run `python -m experiments.nest_clip_v1.armb_summary02_4epoch_v1.recovery_full run`
using the recovered `.venv`. Only this new recovery supervisor is authorized.
The preserved historical `run.py` resumes500 and must NOT be invoked.

The native trainer/model/data/loss/optimizer/scheduler remain unchanged.
F/S/D weights `[1.4,0.2,1.4]`; Summary+RandomDetail; Balanced-Stack-Patch B/16;
four A10080GB,256/rank, accumulation1, seed0, workers8; H4868; stop4868.
Only initialize from common step0, SHA
`54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`.

A read-only LR-call guard synchronizes all ranks after completed update5 and
before update6 forward/backward. It compares every sample ID, F/S/D string/token
stream digest, LR, loss and serialized model/adapter/AdamW state to smoke5.
Deviation fails closed before optimizer update6. Extra step5 checkpoint is I/O
only; epoch checkpoints1217/2434/3651/4868 retain the original state format.
Original tail180/rank/global720 once per epoch remains unchanged.

Progress: `FULL_PROGRESS.json`, `TRAINING_DIAGNOSTICS.json`; runtime native steps,
cycles and console logs in `runtime/SAID-nest-clip-v1/armb_summary02_4epoch_recovery_v1/`.
No data downloads or repeated full image audits.

After native4868 acceptance, strict bare export/encoder equality check, then
original COCO canonical, Urban1k, Flickr30k-test1K, DOCCI, reconstructed
Long-DCI7602 evaluators. Normalize native image/full-caption embeddings and take
their plain inner product only. No alternate inference/split or further trial.

Outputs: `FULL_RESULTS.md/json`, `TRAINING_DIAGNOSTICS.json`,
`STRICT_EXPORT_PROVENANCE.json`, `CHECKPOINT_SHA256.json`, `FIRST_FIVE_GATE.json`,
plus numeric per-update diagnostics and frozen native raw metrics.
Only code/reports/sanitized evidence go to GitHub. Checkpoints remain local.
