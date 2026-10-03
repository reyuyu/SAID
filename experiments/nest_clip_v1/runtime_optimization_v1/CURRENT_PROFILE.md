# Current Audited B16 Profile

Unprofiled exact original trainer: 5 warmup +30 updates, four-rank maximum mean 2.061873s; allocated 27.760GiB. Profiler timings are not speed claims.

Profile:5 warmup +10 active updates on all4 ranks; instrumented mean 2.248166s. Same256/rank,global1024,compact old-R,context248,seed0 and initial weights.

Top-level CPU wall scopes include waiting for their GPU work; nested forward scopes overlap and cannot be added to parent scope. CUDA kernel totals come only from Chrome trace cat=kernel, excluding GPU user annotations.

| Stage | Mean wall ms/update | Fraction of profiled wall % |
|---|---:|---:|
| data_wait | 0.128 | 0.01 |
| host_to_device | 0.396 | 0.02 |
| model_forward | 611.992 | 27.22 |
| backward | 1449.612 | 64.48 |
| gradient_diagnostics | 40.321 | 1.79 |
| optimizer | 14.834 | 0.66 |
| integrity_audit_logging | 125.259 | 5.57 |
| step_synchronization | 1.917 | 0.09 |

Nested forward detail; inclusive CUDA kernel times:

| Scope | Mean CUDA ms/update |
|---|---:|
| image_encoder_forward | 98.405 |
| text_encoder_F | 74.978 |
| text_encoder_P | 73.494 |
| text_encoder_R | 73.485 |
| visual_mask_branch | 26.762 |
| text_mask_branch_F | 29.121 |
| text_mask_branch_P | 29.133 |
| text_mask_branch_R | 29.142 |
| pair_score_F | 16.866 |
| pair_score_P | 16.885 |
| pair_score_R | 16.902 |
| collectives_gather | 35.003 |
| shared_attention_pool | 1.474 |

DDP communication 702.402ms overlaps compute by 695.318ms; do not add communication to backward wall time.
Fixed unused trainable parameter(s): ['clip.logit_scale']. Preserve their absent gradients/AdamW state; test static graph with this stable unused set.

CUDA active-kernel timeline fraction by rank: [84.66, 85.33, 85.19, 86.18]%.

Backward/recompute is the largest scope; normal data_wait is tiny and is not the regular bottleneck. Forward transformer work, repeated pair tiles, gradient diagnostics and CPU audit logging are measured secondary costs. Prioritize checkpoint strategy, audit decoupling and F/P/R/gate/scoring execution reuse; validate each numerically before recommendation.

Multistep profiled/nonprofiled losses began to diverge at step3 despite identical text/sample hashes; no bitwise multistep equivalence is claimed from this timing run. Fixed real-input one-update parameter/optimizer regression with deterministic controls and self-repeat calibration is required. Exact input-image hashing is added to later probes.

Traces and fixed input .pt files remain local under runtime_optimization_v1/profile-current/. All scopes/operators/kernels and DDP timing are in evidence/current-profile.json.
