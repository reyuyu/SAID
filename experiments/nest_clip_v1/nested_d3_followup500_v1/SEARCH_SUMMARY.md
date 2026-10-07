# KR2M1 and WeakSparse: two independent local500 experiments

Queue: KR2M1 -> five native benchmarks -> complete report/review -> fresh WeakSparse -> same evaluation/report/review.
Both fresh common0, exactly500 optimizer updates, local-only; no full/combined/third experiment.
KR2M1 changes only K sampling. WeakSparse restores fixed K3 and changes only absolute sparsity coefficients to [.5,1.,1.5], mass3, without normalization.

| Arm | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I |
|---|---:|---:|---:|---:|---:|---:|
| Anchor | 71.063600 | 74.954666 | 84.025001 | 65.227000 | 90.900000 | 89.700000 |
| KR2M1 | 71.091514 | 74.959857 | 84.115002 | 65.289000 | 91.400000 | 89.500000 |
| WeakSparse-051015 | 70.897922 | 74.811870 | 83.870002 | 65.027000 | 90.700000 | 89.100000 |

Every directional R@1/5/10, all anchor deltas, K histogram/median/coverage, CE/shares, gradients, keep/IoU/violations and immutable checkpoint/bare identities are in each arm REPORT.md/JSON.
Raw logs and binary artifacts stay local. Each RUNTIME_STATS.json records local paths,size,SHA256,time windows.
No automatic follow-up beyond these two experiments.
