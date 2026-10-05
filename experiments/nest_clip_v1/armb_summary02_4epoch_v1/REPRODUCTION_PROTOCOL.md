# S02 full trajectory with a500-step retrieval gate

Latest authorization supersedes the failed cross-run numerical-prefix gate, not any training hyperparameter.

- Fresh common step0, SHA256 `54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`.
- Frozen `recovery/configs/summary02.json`, F/S/D `[1.4,0.2,1.4]`, horizon4868 in both stages.
- First5: exact sample/text/token/LR/config/source/optimizer-group invariants, finite state, actual AdamW5 updates and within-run four-rank parameter agreement. No cross-run loss/parameter/moment comparison.
- Native trainer, model, losses, preprocessing, sampling, LR and distributed tail remain byte-identical to the passed smoke. New wrapper only checks invariants and adds checkpoint metadata.
- Stop native formal stage at500. Complete model/adapter/AdamW/four-rank Python/NumPy/CPU/CUDA/loader RNG and epoch/batch cursor are saved. Native manual LR scheduler is documented explicitly. Historical BF16 has no GradScaler; checkpoint records that fact rather than adding one.
- Export and five frozen full-caption native evaluations run in separate processes with no training process alive. Checkpoint SHA256 must be unchanged before/after evaluation.
- Pass requires Score5>=70.164367, J_long3>=73.403324, Urban I2T>=89.1 and T2I>=87.0, no obvious collapse. A conservative collapse screen flags invalid/nonfinite R1 or loss of more than50% of historical directional R1; all directional deltas are reported, no search is performed. Borderline failures remain failures.
- Only this trajectory's new step500 may resume, and only on `REPRODUCTION_PASS`. Original native resume validator and additional exact restoration checks enforce AdamW/model/RNG/scheduler/cursor continuity before update501.
- Stop at4868, strict export, five-set native evaluation, classification and stop. No other experiments.

New isolated local runtime: `runtime/SAID-nest-clip-v1/armb_summary02_500gate_recovery_v1/`.
Previous failed step5 and native replay remain preserved and are never resumed.

Launch: `.venv/bin/python -m experiments.nest_clip_v1.armb_summary02_4epoch_v1.reproduction_full`.
Status: `FULL_PROGRESS.json` (UTC timestamps), native logs under the isolated runtime.
Publication uses an explicit report whitelist and a secret scan; checkpoint/image/archive bytes are never staged.
