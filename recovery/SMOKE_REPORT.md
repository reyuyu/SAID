# Recovery smoke report

Four-GPU smoke executed: False.
Four-GPU smoke passed: False.
The prior training-data blocker is cleared: `training-audit.json` and the final
1245901-image exact-path/full PIL/RGB audit passed at2026-10-05T15:18:47UTC.
Smoke has not been executed; data acceptance is not smoke/DDP/export acceptance.
Only the previously authorized four-A100 five-update smoke may follow; no
500/4868-update training is authorized or automatically launched.
Original CPU/unit tests passed: True.
Passing original suites: research 93 core +11 legacy; canonical 69 core +11 legacy.
The legacy retrieval tests run separately because their old sys.path change shadows the train package in spawned workers.
Real CPU RandomK and Summary0.2 model/optimizer construction passed: True.
Construction uses original modules, H4868, context248, empty optimizer, zero updates, and original F/S/D weights.
Recovery's fail-closed guard tests: four passed; see `test_recovery.py`.

The only recovery trainer command is canonical RandomK, four A100s, batch256/rank,
global1024, accumulation1, seed0, workers8, H4868, 1217 updates/epoch, stop exactly5.
No synthetic-image or reduced-index smoke is used as a substitute for recovered training assets.
DDP parameter agreement and native trained-checkpoint export remain unverified unless the real
five-update acceptance/export JSONs pass. Missing assets never count as a passed smoke.

Historical steady-state reference: about2.1s/update and27.8GiB peak allocated/rank.
These are references, not recovery measurements. Five warmup updates cannot establish a
495-cycle steady-state average. Investigate significant memory/time deviations before continuation.

No 500/4868-step formal training, historical searches, or gradient audit have been run.
