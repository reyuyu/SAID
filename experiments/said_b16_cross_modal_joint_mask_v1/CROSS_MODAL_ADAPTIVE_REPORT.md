# Cross-Modal Interaction Joint Mask: adaptive 1217-step decision

This report records the authorized X1/X2 continuation from the independently
trained step-500 checkpoints. X3 is retained as a 500-step diagnostic result
only: its paired implementation omitted the second bidirectional modulation
term, so it is not a compliant candidate for promotion.

The controller used a fixed rule before the 1217 evaluations: engineering
audits had to pass, then both `Score5 >= 72.8292674138102` and
`J_long3 >= 76.98411235635034`, the A-Matched@1217 reference values. No test
set result was used to change a configuration or add an arm. Because neither
X1 nor X2 met both inequalities, no candidate was continued to 3651.

| Arm | Score5 | Δ | J_long3 | Δ | J_long | Δ | Short4 | Δ | Urban I2T | Δ | Urban T2I | Δ | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A-Matched@1217 | 72.829267 | — | 76.984112 | — | 85.905003 | — | 66.597 | — | 93.100 | — | 91.600 | — | reference |
| X1@1217 | 72.738336 | -0.090932 | 76.904560 | -0.079553 | 85.835003 | -0.070000 | 66.489 | -0.108 | 92.600 | -0.500 | 91.600 | 0.000 | stop |
| X2@1217 | 72.826483 | -0.002784 | 76.944806 | -0.039307 | 85.800003 | -0.105001 | 66.649 | +0.052 | 93.300 | +0.200 | 91.100 | -0.500 | stop |

The five-dataset R@1/R@5/R@10 values are in the two JSON summaries next to
this report. All are native student-only retrieval evaluations with the frozen
protocols; no inference mask, reranking, ensemble, or TTA was used.

Both continuations passed four-rank synchronization, finite-gradient and
checkpoint acceptance. Each compared 1,217 updates and 1,245,904 sample
positions against the A-Matched stream with zero mismatches. X1 full/bare
checkpoint SHA256 are `04712f85e15e7011a11ac42172e34516fef99b4b13d5b915f372123264616aeb`
and `828a4287f02ac558c7e592f0afe5c4213c7c32c040aba34de5d5719832be0861`.
X2 full/bare SHA256 are `01fe7e2e0f387622074cd8acbd5292a34645a246aa0a951d4767cbbc88fd0329`
and `167ada2e68c4dfbc7f3701b61da5faa8d94c3eaf6cd5b4e831804749e33e526d`.

The observed X2 Urban I2T gain is accompanied by a 0.5-point T2I loss and
does not establish an overall improvement. The comparisons remain exploratory:
the public retrieval sets have been repeatedly observed and no independent
selection set was available. All training and evaluation processes have
exited; the four GPUs are idle.
