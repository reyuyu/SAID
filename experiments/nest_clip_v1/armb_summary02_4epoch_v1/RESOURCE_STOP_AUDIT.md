# S02 stopped before500: resource evidence

**STOPPED_BEFORE_500_RESOURCE_GATE**. Exactly33 finite AdamW updates, from fresh common step0, horizon4868.

Minimal first5 input/text/token/LR/seed/source invariants passed; four-rank parameter difference0. No cross-run numerical gate.

| Update | Full-cycle seconds |
|---:|---:|
| 31 | 11.166840 |
| 32 | 3.752889 |
| 33 | 6.774076 |

Original native monitor stops after three consecutive cycles>3 seconds. It was not relaxed.
Early worker shutdown subsequently raised SIGABRT, so there is no final native acceptance/DDP proof beyond the passed first5.

Median full cycle steps7–30: 2.104327s. Peak allocated: 27.759881GiB/card.

Container limit500GiB, substantial file cache, historical peak at the limit, no recorded OOM kill. Counters are cumulative, not proof of this run's cause.
NFS/data waits and host-side synchronization/reclaim remain hypotheses, not confirmed causes.
Advised only37 exact completed archive paths to release unused cache; no image/byte/deletion/global-cache operation. No material memory reduction and no new training trial.

Full step33 CPU state audit passed: model/adapter/AdamW finite, counter33, manual scheduler4868, BF16/no scaler, four-rank RNG and epoch0/batch33 cursor.
Step33 is evidence only and must not be resumed under the current500 continuation protocol. No step500 exists; retrieval gate is NOT RUN, not REPRODUCTION_FAIL_AT_500.
