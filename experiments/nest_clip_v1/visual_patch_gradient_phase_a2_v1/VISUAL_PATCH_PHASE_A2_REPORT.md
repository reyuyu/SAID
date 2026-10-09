# SAID Phase A.2: visual Patch gradient mechanism

This is a read-only audit of E2-Uniform at steps 500 and 1217. It compares the
production route A (`hidden.detach().float()`) with route B (`hidden.float()`).
No optimizer was created, no update was performed, no checkpoint was written,
and no evaluator was run. The raw receipt is `/root/said_visual_patch_gradient_phaseA2_v1/run4/A2_RAW_RECEIPT.json`.

## Acceptance

Both checkpoints matched their required SHA256. The global16 preflight and the
global1024 next-step probe completed at both nodes. All captured forward
tensors and loss components were exactly equal between A and B; input hashes
were equal, parameters remained unchanged, all values were finite, and the
four-rank collectives completed. A's Patch-boundary gradient was exactly zero;
B's was nonzero. The A/A and B/B repeats quantify numerical noise rather than
serving as a production training step.

## True total visual-backbone gradient

| checkpoint / batch | A norm | B norm | Δ norm | Δ/A | cos(A,B) | cos(Δ,A) |
|---|---:|---:|---:|---:|---:|---:|
| 500 / global1024 | 96.2100567 | 102.639882 | 17.9746378 | 0.186827 | 0.985734 | 0.276256 |
| 1217 / global1024 | 75.4998997 | 78.1525063 | 11.7083197 | 0.155077 | 0.98898 | 0.152999 |

The added route changes the native visual-backbone gradient by 18.68% and
15.51% of the A norm (the B norm itself is 6.68% and 3.51% higher). The total
A/B cosine remains high (0.9857 and 0.9890),
so the aggregate direction is not strongly opposed. The added delta has a
positive projection on A in both full batches (0.276 and 0.153), although some
late transformer-layer delta cosines are near zero or slightly negative.
Text backbone, fusion gate, and the mask/adapter groups are unchanged to the
receipt precision; the tiny formal-batch differences in mask/adapter are at
about 1e-9 relative scale.

Repeat noise is far below the A/B change: native visual B/B relative noise is
0.0842% at 500 (A/A is 0) and 0 at 1217. Thus the observed total-gradient
difference is resolvable above repeat noise.

## Patch-boundary attribution

At `hidden[:,1:]`, A is exactly zero. B at global1024 has:

| checkpoint | total | alignment | sparsity | hierarchy | boundary sum error |
|---|---:|---:|---:|---:|---:|
| 500 | 0.0289188009 | 0.0289906972 | 0.000646618878 | 3.1195541e-06 | 0.00235088 |
| 1217 | 0.0272295251 | 0.0272810955 | 0.000458056984 | 3.94808649e-06 | 0.00229861 |

Alignment supplies essentially all of the boundary signal in both formal
batches (relative norms 1.0025 and 1.0019); sparsity contributes 2.24% and
1.68% of the total norm, and hierarchy 0.011% and 0.015%. Component norms are
not additive as norms; the reported vector sum error is the native BF16/
checkpointing numerical residual (about 0.23%). At the small global16
preflight, the relative component mix is sample-dependent, so it is retained
in the JSON rather than generalized to a training-wide claim.

## Resource and reproducibility result

Both four-rank global1024 probes ran on NVIDIA A100 80GB PCIe GPUs. Peak
allocated memory was about 31.1 GB per rank (reserved about 33.2 GB), with no
OOM, NaN/Inf, or DDP failure. The exact per-run times, memory, rank identity,
and sampling summaries are in `GLOBAL1024_RESOURCE_VALIDATION.json`. The
next-step sampler assertion passed on all ranks against the frozen reference;
the probe did not infer model-dependent values as data-stream evidence.

## Answers and classification

1. The added Patch path is a substantial visual-backbone signal: 15.5--18.7%
   relative total-gradient change in the tested full batches.
2. Its aggregate direction is mostly aligned with the original gradient, not
   strongly conflicting, while individual late layers show mixed projections.
3. At the Patch boundary, alignment dominates; sparsity is secondary and
   hierarchy is negligible at these nodes.
4. The mechanism is present at both 500 and 1217, though its relative norm is
   smaller at 1217.
5. A real four-rank global1024 batch is feasible on the observed hardware.
6. Mechanistically, an isolated Phase B experiment is justified by this audit,
   subject to ordinary single-seed uncertainty and the fact that no retrieval
   improvement has been measured here.

**Classification: `PHASE_B_FEASIBLE`**

This classification covers mechanism and resource feasibility only. It does
not claim an Urban or retrieval gain. No training or evaluation was run.
