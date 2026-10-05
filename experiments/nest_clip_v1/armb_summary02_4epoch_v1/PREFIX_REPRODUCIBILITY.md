# S02 full prefix gate: stopped before update6

**BLOCKED_PREFIX_GATE**. Authorized full started2026-10-05T16:11:11UTC, from
the exact common step0, with horizon/stop4868 and no resume. Only optimizer
updates1–5 executed. All four training processes stopped; no full checkpoint4868,
final bare export, five-set full evaluation or full-result classification exists.

## Observed trajectories

| Update | Original smoke | Stopped full | Native smoke replay | Full minus original |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 60.999504089 | 60.999504089 | 60.999504089 | 0 |
| 2 | 23.328746796 | 23.328746796 | 23.328746796 | 0 |
| 3 | 20.395172119 | 20.395172119 | 20.395172119 | 0 |
| 4 | 18.684776306 | 18.690505981 | 18.690505981 | +0.005729675 |
| 5 | 21.589082718 | 21.600728989 | 21.600372314 | +0.011646271 |

Maximum observed full loss difference is0.053945% of the original loss.
The conservative implemented gate uses `rtol=1e-5, atol=1e-4`, and stopped at
the first out-of-tolerance comparison, update4, before any sixth update.
This numerical tolerance was not a user-provided hyperparameter; it has not
been silently loosened after seeing the failure.

## What matches exactly

- All5120 sample IDs across four ranks/five updates. Complete F/S/D string and
  token-ID stream SHA256 matches for every rank/update in all three trajectories.
- Original native code SHA256 manifest, model/fusion initialization digests,
  runtime model settings, dataset metadata, PyTorch/CUDA/NCCL versions, recorded
  NCCL environment and four GPU UUIDs match the original smoke.
- F/S/D weights `[1.4,0.2,1.4]`; H4868; original inclusion/sparsity/optimizer/LR;
  seed0;256/rank; accumulation1; all native preprocessing unchanged.
- Common step0 SHA
  `54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`.
  All four rank CPU/CUDA/loader-generator RNG states match the original step5.
- All329 initialized AdamW states have counter5, matching keys/shapes/dtypes;
  optimizer groups, ordering, LR, betas/epsilon match. All state tensors finite.

## What does not match

- Earliest *observed logged* difference: update3 backbone gradient norm differs
  by0.000579834, even though update3 forward CE/loss match. This does not prove
  that every individual earlier gradient tensor was equal; no earlier full
  tensor snapshots exist for the original smoke.
- At step5,271/317 native model tensors and14/14 adapter tensors differ.
  Maximum parameter absolute differences: model0.000523973, adapter0.000436173.
  All329 `exp_avg` and329 `exp_avg_sq` tensors differ; maximum absolute
  differences0.260688484 and0.036436081. Counters are exact and finite.
- The largest native-model parameter delta is a mask-pool bias. Do not dismiss
  the entire state comparison simply because total-loss differences are small.

## Bounded investigation, not additional research trials

1. Read-only full/smoke checkpoint and metadata comparison; no data re-audit or
   download. `PREFIX_STATE_FORENSICS.json` records actual tensor differences.
2. Five repeated identical-input forwards/backwards of the original text block
   and original visual block, loaded strictly from common step0. Native BF16
   autocast, FP32 masters, original dimensions. **Zero optimizer updates**.
   Text selects efficient attention; visual selects flash attention. Both
   isolated probes were bitwise reproducible in this test. They do not reproduce
   or explain the entire four-rank training discrepancy, and cannot be used as
   proof that the whole training computation is deterministic.
3. One additional original, unmodified native four-A100 smoke, exactly5 updates,
   H4868, common step0, same frozen config. It reproduces a different update4/5
   trajectory from the original smoke despite matching sample/text/token streams
   and construction metadata. Native acceptance passes; no wrapper/gate code is
   used. Thus the discrepancy is not unique to the full-run wrapper or its
   uncapped4868 sampler. It is an observed cross-run numerical reproducibility
   issue; the exact responsible backend/collective mechanism is **not proven**.

No deterministic-backend setting, sampler, loss formula, optimizer epsilon,
learning rate or other training hyperparameter has been changed to chase the
reference. No automatic numeric-gate bypass or full restart is performed.

## Operational state and retained evidence

The outer background coordinator no longer exists and failed to finalize its
initial `TRAINING/0` progress snapshot. State is reconciled to the observed
five-update rejection, not left claiming a live run. No child/GPU training
process remains. A small unrelated-process heartbeat test confirms `nohup` and
`setsid` can both survive a normal command exit here; it does not explain the
coordinator disappearance. No container OOM kill was recorded at inspection.
Future guard failures directly publish a blocked marker before rank teardown,
and coordinator collection refuses a rejected marker.
Future launches isolate the torchrun process group for reliable termination,
and the read-only pre-update guard refuses to continue if its supervisor PID
disappears. These safety changes were made after the failed attempt, not used
to alter its numerical trajectory. They do not authorize a new attempt.

Full step5 remains local for forensics, explicitly marked do-not-resume. The
native diagnostic smoke is also quarantined. Neither may initialize a future
full run; any separately authorized retry must begin at the common step0.

Proofs: `FULL_PREFLIGHT.json`, `FIRST_FIVE_GATE.json`,
`PREFIX_REPRODUCIBILITY.json`, `PREFIX_STATE_FORENSICS.json`,
`NATIVE_BACKWARD_PROBE.json`, `VISUAL_BACKWARD_PROBE.json`, and local raw logs.
No4868 performance or `FULL_POSITIVE`/tradeoff/scaling conclusion is claimed.
