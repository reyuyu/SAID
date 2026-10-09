# Phase A.1: Visual Patch gradient additivity root cause

**Classification: `PROBE_IMPLEMENTATION_ISSUE`**

The original 1.539450022% failure is reproduced on E2-Uniform@500, global16 (4 samples/rank), then localized without training or evaluation. The checkpoint SHA is unchanged. No optimizer is constructed, no parameter is updated and no checkpoint is written.

## Direct results

| Precision | Variant | Separate component-gradient error | Threshold | Separate status | One-pass component-sum vs total |
|---|---|---:|---:|---|---:|
| BF16 native | A | 0.000000000% | 1.0000% | PASS | 0.000000000% |
| BF16 native | B | 1.539646588% | 1.0000% | FAIL | 0.000000000% |
| FP32 diagnostic | A | 0.000000000% | 0.0300% | PASS | 0.000000000% |
| FP32 diagnostic | B | 0.045099488% | 0.0300% | FAIL | 0.000000000% |

BF16 B separate error is **1.539646588%** (the previous 1.539450022% was the earlier probe implementation). FP32 B separate error is **0.045099488%**, still above the 0.03% diagnostic reference. A is exactly additive in both paths because its regularizer gradients are disconnected from the native visual backbone.

## Evidence separating graph correctness from probe error

1. The scalar graph identity `total_training - (weighted_align + weighted_sparse + weighted_hierarchy)` is exactly zero for A and B in both BF16 and FP32. The loss decomposition used by the probe therefore matches the actual HNS macro loss graph.

2. A one-pass `autograd.grad(weighted_align + weighted_sparse + weighted_hierarchy, params)` matches `autograd.grad(total_training, params)` with zero measured error for every reported parameter group and layer, in both precision modes and both variants. This is the relevant control for the production one-scalar backward.

3. The failure appears only when the probe calls `autograd.grad` separately for each component and adds those independently recomputed gradients. It is largest after the visual Patch route reaches `visual_conv1`: BF16 B layer error is 2.329102%, FP32 B layer error is 0.073110%. Other visual Transformer layers are much smaller.

4. The Patch input gradient is zero for every A component and nonzero for B alignment, sparsity, hierarchy and total, proving that the requested route is actually disconnected/connected as intended.

Because FP32 also fails the separate-backward threshold while the one-pass control passes exactly, the root cause is the **probe implementation/gradient decomposition method**, with BF16 recomputation amplifying the discrepancy. It is not justified to label the original failure a pure BF16 effect or a production loss-graph inconsistency.

## A/B and scope

- A: original `hidden.detach().float()`; native visual regularizer gradients are zero; separate additivity PASS.
- B: only `hidden.float()`; Patch route live; separate additivity FAIL in BF16 and FP32, but one-pass production-style control PASS.
- Forward tensors (global image/text embeddings, conditions, probabilities, Hard-ST masks, component scalars and total) are exactly equal A/B within each precision path.
- No global1024, no E2@1217, no Urban evaluation and no training were run.

The earlier failed runs due output-directory creation, an undefined local route variable, and Tensor JSON serialization are preserved outside the repository as launch evidence; they occurred before a valid scientific result. The final A.1 receipt is from the completed audit6 run.

## Limits and next action

This proves the previous acceptance failure was caused by separately differentiated component recomputation in the audit, not that the Patch-gradient mechanism improves retrieval. A future probe should report the one-pass total gradient as the production-equivalent acceptance and use separate component gradients only as diagnostic decomposition, with explicit recomputation error. No Phase B training is authorized by this task.
