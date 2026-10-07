# KR2M1 and WeakSparse: two independent local500 experiments

Queue: KR2M1 -> five native benchmarks -> complete report/review -> fresh WeakSparse -> same evaluation/report/review.
Both fresh common0, exactly500 optimizer updates, local-only; no full/combined/third experiment.
KR2M1 changes only K sampling. WeakSparse restores fixed K3 and changes only absolute sparsity coefficients to [.5,1.,1.5], mass3, without normalization.

| Arm | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I |
|---|---:|---:|---:|---:|---:|---:|
| Anchor | 71.063600 | 74.954666 | 84.025001 | 65.227000 | 90.900000 | 89.700000 |
| KR234 | 71.107901 | 75.046502 | 84.120002 | 65.200000 | 91.600000 | 89.200000 |
| KR2M1 | 71.091514 | 74.959857 | 84.115002 | 65.289000 | 91.400000 | 89.500000 |
| WeakSparse-051015 | 70.897922 | 74.811870 | 83.870002 | 65.027000 | 90.700000 | 89.100000 |

Every directional R@1/5/10, all anchor deltas, K histogram/median/coverage, CE/shares, gradients, keep/IoU/violations and immutable checkpoint/bare identities are in each arm REPORT.md/JSON.
Raw logs and binary artifacts stay local. Each RUNTIME_STATS.json records local paths,size,SHA256,time windows.
No automatic follow-up beyond these two experiments.

## Direct comparison with KR234

| Arm vs reference | ΔScore5 | ΔJ_long3 | ΔJ_long | ΔShort4 | ΔUrban I2T | ΔUrban T2I |
|---|---:|---:|---:|---:|---:|---:|
| KR2M1 vs Anchor | +0.027914 | +0.005190 | +0.090001 | +0.062000 | +0.500000 | -0.200000 |
| KR2M1 vs KR234 | -0.016387 | -0.086645 | -0.005000 | +0.089000 | -0.200000 | +0.300000 |
| WeakSparse-051015 vs Anchor | -0.165678 | -0.142796 | -0.154999 | -0.200000 | -0.200000 | -0.600000 |
| WeakSparse-051015 vs KR234 | -0.209979 | -0.234631 | -0.250000 | -0.173000 | -0.900000 | -0.100000 |

KR2M1 improves Short4 and Urban T2I over KR234, but its Score5/J_long3 are slightly lower. Neither random-K arm preserves Anchor Urban T2I. WeakSparse is below both Anchor and KR234 on all six summary metrics; lowering these absolute sparsity coefficients did not improve retrieval.

Retain Anchor; no automatic full, combination, or third experiment. RESULTS.json also records every dataset/direction R@1/5/10 delta against KR234 and the reference result SHA256.
