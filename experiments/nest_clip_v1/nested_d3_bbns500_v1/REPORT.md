# BBNS: frozen support audit and local500 preparation

Status: PREPARED; new BBNS training has not started at this preparation commit.

Read-only successful Nested D3 Balanced Anchor checkpoint:
`runtime/SAID-nest-clip-v1/nested-detail-d3-balanced500-20261007/step500/step000500.pt`.
SHA256: `24792ba9e6e3c34f733820f489cb3a8953278246ae58f2f688652894be871e4b`.
Identity matches its actual five-set result/validation; completed500/horizon4868. Model digest and checkpoint SHA unchanged after audit. No optimizer or parameter update.

Deterministic cohort: first16384 valid examples of seed0 epoch0 fixed training stream, ordered batch/rank/within-rank. Candidate loader:17 global1024 batches;16384 selected IDs saved in ANCHOR_SUPPORT_COHORT.json. All images local-only. A first observational pass reached the same thresholds but emitted early-loader cleanup warnings; authoritative audit exhausted its bounded loader normally, exited0 and reproduced all six thresholds. Original raw audit directories retained locally; no training was involved.

Bands frozen from Anchor support only, before any BBNS training or retrieval result:

| Edge | kappa=P20(C) | tau_low=P20(R) | tau_high=P80(R) |
|---|---:|---:|---:|
| Dall->F | 0.6466294646263122 | 0.6404582858085632 | 0.6976943612098694 |
| D3->Dall | 0.6413809061050415 | 0.6085701584815979 | 0.6629944801330566 |

Population std and all requested P10/P20/P25/median/P75/P80/P90 distributions, hard keep/IoU/violations, cohort and local raw array size/SHA/time windows are in ANCHOR_SUPPORT_AUDIT.json/.md.

Exact loss: alignment + Omega_F/3 + edge(Dall,F) + edge(D3,Dall).
Edge: relu(kappa-C)^2 + relu(tau_low-R)^2 + relu(R-tau_high)^2; eps1e-6. Coverage uses detached child and updates parent only; refinement uses detached parent and updates child only. No old child-global sparsity, independent inclusion, inclusion ramp, beta or online threshold adaptation. Legacy inclusion_max1 remains only as unused construction metadata.

256 CPU tests passed. Manual tensor gradient cases saved in GRADIENT_ROUTING_AUDIT.json. Frozen config: configs/nested_d3_bbns500.json, identical experiment config copy. Classification tolerances declared before training in SEARCH_PLAN.json.

Fresh common0 SHA256: `54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`.
Frozen K3/text/sampler, alignment[1.35,1.35,.30], Balanced-Stack-Patch Hard-ST, ViT-B/16,4A10080GB,256/rank,global1024,accum1,seed0,workers8,horizon4868,original optimizer/LR/preprocess. Exactly500 updates; never resume Anchor. First5 gate and all512000-record stream proof against Anchor.

Image root: `/root/said_s02_stage500/ShareGPT4V/`. Local missing/symlink/escape fails; no NFS fallback or data copy. /root is disposable Docker overlay; NFS source images remain intact.

Detached runner will train500 -> unchanged8-batch gradient protocol -> strict native bare export/verify -> five retrieval datasets -> review/report -> commit/push/fetch/remoteHEAD check. Only this arm; no full4868, band/beta/K/weight changes or additional experiment. All binaries and raw large logs remain local.
