# S=0.2 final five-update smoke gate

**READY_TO_START_S02_FULL** — verified on 2026-10-05 UTC.

No data was downloaded or re-audited. No formal training was started or
authorized. The native trainer executed exactly updates 1–5; update 6 did not
execute. All four GPUs are idle after successful process-group teardown.

## Frozen protocol and initialization

- Source protocol: `52bb7ae2d77aae8c2b1f69877aed5f7cc5d85b18`, original
  Summary0.2 full-preflight branch. Native model, trainer, sampler and exporter
  source remain byte-identical; no historical branches were merged.
- Config: `recovery/configs/summary02.json`, ViT-B/16, context248,
  `dual_branch`, `balanced_stack`, `patch`, Summary+RandomDetail.
- F/S/D alignment weights: `[1.4, 0.2, 1.4]`. F is the historically packed Full
  view; S is its first literal sentence; D is the original seeded random subset
  of visible non-summary sentences. Native logs call S/D `O`/`E`.
- Four distinct A100 80GB PCIe GPUs; batch256/rank, global1024,
  accumulation1, seed0, workers8/rank, epochs4, updates/epoch1217.
- Scheduler horizon4868, never5. Fusion peak LR0.0002, mask/visual peak LR0.001,
  visual LR scale1, sparsity scale1, inclusion max1. Historical optimizer,
  LR warmup/cosine, inclusion ramp, preprocessing and inference are unchanged.
- Initialized only from `runtime/SAID-nest-clip-v1/shared/step000000.pt`, SHA256
  `54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`.
  Provenance: `OpenAI CLIP + original random MaskNetwork`; optimizer state empty;
  completed_steps0; no resume checkpoint or old500-step checkpoint.
- Reused the existing `DATA_AUDIT_PASS` evidence: SAM569486, COCO118287,
  LLaVA558128; total1245901, missing0, corrupt0, IO errors0.

## Five optimizer updates

CE entries below are `I2T / T2I`. Weighted alignment contributions use the
unchanged native expression `10/sum(weights) * weight * (I2T + T2I)`.

| Step | Total loss | F CE | S CE | D CE | Weighted F/S/D alignment |
| ---: | ---: | --- | --- | --- | --- |
| 1 | 60.999504 | 2.163231 / 1.935223 | 4.266203 / 3.869996 | 4.090154 / 3.553579 | 19.126116 / 5.424132 / 35.670754 |
| 2 | 23.328747 | 0.604773 / 0.581604 | 1.843994 / 1.980533 | 1.570009 / 1.457112 | 5.536428 / 2.549685 / 14.126562 |
| 3 | 20.395172 | 0.357487 / 0.460196 | 1.758470 / 2.112269 | 1.273225 / 1.436210 | 3.815853 / 2.580493 / 12.644031 |
| 4 | 18.684776 | 0.342492 / 0.380125 | 1.697274 / 1.772411 | 1.383880 / 1.088642 | 3.372214 / 2.313123 / 11.538440 |
| 5 | 21.589083 | 0.283931 / 0.381447 | 1.699852 / 1.900032 | 1.733724 / 1.385681 | 3.105098 / 2.399923 / 14.557222 |

| Step | Sparsity contribution | Raw inclusion | Inclusion ramp | F/S/D keep ratio | Backbone LR | Backbone grad norm |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | 0.778505 | 0.030308 | 0.000 | 0.466459 / 0.467762 / 0.466766 | 5e-9 | 3418.261475 |
| 2 | 1.115907 | 0.033210 | 0.005 | 0.667164 / 0.670959 / 0.669319 | 1e-8 | 716.629578 |
| 3 | 1.354511 | 0.028362 | 0.010 | 0.820330 / 0.808243 / 0.813358 | 1.5e-8 | 508.014435 |
| 4 | 1.460650 | 0.023208 | 0.015 | 0.883739 / 0.871264 / 0.877842 | 2e-8 | 570.623352 |
| 5 | 1.526450 | 0.019482 | 0.020 | 0.920935 / 0.912233 / 0.916975 | 2.5e-8 | 890.842163 |

Inclusion contribution equals raw inclusion times its ramp. Native sparsity is
`(F_sparse + 2*S_sparse + 2*D_sparse)/3`, not dose-weighted. Every recorded total
matches the full weighted alignment + sparsity + inclusion decomposition.
Exact unrounded LR for every optimizer group and finite gradient norms for
every rank/group/adapter parameter are in `evidence/s02-smoke-audit.json`.

Hard gate all-open/all-closed fractions are0 for F/S/D, positive and negative
pairs, at all five updates. Every step records positive/negative keep statistics.
The unchanged historical soft-gate diagnostic schedule records detailed gate
quantiles only at update1 during a five-step smoke: F/S/D gate mean0.5,
variance0, saturation0. Steps2–5 have hard-gate statistics, not invented soft
distribution measurements. Full native diagnostics are retained in local
`steps.jsonl` and the sanitized proof JSON.

## Timing and memory investigation

| Step | Slowest-rank update seconds | Full-cycle seconds |
| ---: | ---: | ---: |
| 1 | 8.209250 | 46.545925 |
| 2 | 1.997547 | 2.065025 |
| 3 | 1.962250 | 2.039596 |
| 4 | 1.962709 | 2.041760 |
| 5 | 1.955800 | 2.030112 |

The first update exceeds the4-second reference threshold and is explicitly
flagged as startup/warmup, not hidden in a throughput average. Investigation:
the first cycle contains38.336676seconds outside the measured update, including
initial spawn-worker startup/first-batch loading. The first measured update also
has cold-start overhead; these logs do not separately attribute kernel setup
versus other startup costs. Updates2–5 have only0.067–0.079seconds of extra
end-to-end cycle overhead, no image failure/timeout, and all full cycles are
below2.066seconds. There is no sustained NFS stall or repeated slow-update abort.

Median update time1.962709seconds; post-first full-cycle median2.040678seconds.
Peak allocated27.759881GiB on each rank, about the historical27.8GiB; peak
reserved28.447266GiB maximum. Step5 checkpoint save16.479959seconds is separate
from update timing. Total loop/teardown time about73.515seconds. Five warmup
updates are not proof of long-run throughput or numerical stability.

## Acceptance evidence

- Four-rank gather-gradient startup check passed; final NCCL all-reduce10.
  Every rank completed5 optimizer updates, with final parameter difference from
  rank0 exactly0. All losses, gradients and serialized optimizer/model tensors
  are finite. No duplicate global image IDs in any smoke batch.
- All5120 samples replayed from the exact seed0 DistributedSampler prefix,
  skip-first1000 mapping, original frozen captions and historical text sampler.
  Every rank/step sample ID and complete F/S/D text/token stream digest matches.
  This metadata replay does not read images or repeat the dataset audit.
- Checkpoint saved at step5 with H4868, stop_updates5, next_epoch0/next_batch5,
  four rank RNG records,329 initialized AdamW parameter states all at step5,
  and290 changed native model tensors. All317 model state keys and shapes match
  step0. Historical BF16 autocast uses FP32 master parameters without GradScaler;
  no scaler state is expected. The original manual scheduler's horizon and
  serialized LR are verified, not replaced with a five-step scheduler.
- Strict native bare student export passed. Reloaded bare and checkpoint native
  image/text encoder outputs are bit-exact: image/text maximum difference0.
  Context248; the2x2 normalized image-text similarity forward is finite:
  `[[0.2989732623,0.0673418641],[0.1515310407,0.3197045326]]`.
- Checkpoint SHA256:
  `362555b4774335cc2cbd77adbdaa86268e3f234388dbc808a5fffd85f743f5bf`.
  Bare SHA256:
  `c61b1ef5d19a2a524558cdc05e6b948542215ed402a08dbd0c876209fa37e4ca`.
- Recovery CPU/unit suites148passed, including18 new verifier tests; previous
  native CPU/model/data/evaluator evidence remains
  valid. No core dependencies or historical native implementation were changed.

## Isolation and next action

All smoke checkpoints, bare weights and raw logs remain local in
`runtime/SAID-nest-clip-v1/quarantine/recovery-s02-smoke-5-20261005T153511Z/`,
marked `SMOKE_ONLY_DO_NOT_RESUME.md`. The active smoke output path no longer
exists. Data, common step0 and prior recovery evidence were not deleted.

Any separately authorized S02 full training must begin from the common step0,
never from these smoke/old500 checkpoints. Formal4868 training has not started.
Wait for the user's next explicit instruction.

Proofs: `evidence/s02-smoke-command.json`, `evidence/s02-smoke-audit.json`,
`evidence/s02-export-audit.json`. Code/report/proof synchronization excludes
datasets, checkpoints, credentials and raw training logs.
