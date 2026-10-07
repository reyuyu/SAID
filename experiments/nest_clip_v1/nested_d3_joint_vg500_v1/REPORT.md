# Joint-VG: prepared independent local-only500 experiment

Status: PREPARED; no retrieval results yet. Read-only Anchor V/G audit on16384 valid examples passed; model digest/checkpoint SHA unchanged. Joint loss/gradient and pipeline tests:281 passed.

Frozen alignment [1.35,1.35,0.30], fixed K3, optimizer/LR/seed0/global1024/workers8/horizon4868. Fresh common0 SHA256:54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6. Exactly500 updates, never resume the Anchor. Local-only /root/said_s02_stage500/ShareGPT4V; missing/symlink/path escape fails, no NFS fallback. /root is disposable overlay; NFS sources retained.

Joint parent/child probabilities in V and G, including both denominators. Formula:alignment + Omega_F/3 + edge(Dall,F) + edge(D3,Dall). No stop-gradient, old child global sparsity, independent inclusion, old ramp, beta or online quantiles.

| Edge | epsV=P80(V) | gamma_low=P20(G) | gamma_high=P80(G) |
|---|---:|---:|---:|
| Dall_F | 0.013686737418174746 | 0.011261347867548467 | 0.024891537427902226 |
| D3_Dall | 0.020522184297442438 | 0.04266522899270058 | 0.08650477081537247 |

Audit mean/std/all requested percentiles, hard keep/IoU/violations, checkpoint identity and local raw artifact paths/SHA/size/time are in ANCHOR_VG_AUDIT.json/MD. No retrieval performance used to choose regions. Operational classification/telemetry criteria frozen before launch in SEARCH_PLAN.json.

Positive G lower bands penalize exact equality. PyTorch ReLU has zero subgradient at exact equality, so this does not guarantee escape from an exact-collapse kink; the formula remains exactly as requested. GRADIENT_AUDIT.json records joint endpoint gradients and this caveat.

Detached runner completes training, identical8-batch backbone gradient spot-check, strict bare native five-set evaluation, report and commit/push/fetch/remote-HEAD verification. Stops500, with no full/new arms/quantile/beta/lambda/SG changes. Checkpoints,bare,images,cache and raw large logs remain local.
