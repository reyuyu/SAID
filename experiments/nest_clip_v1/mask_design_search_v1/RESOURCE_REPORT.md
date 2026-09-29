# NEST mask design search v1: resource-gate report

Date: 2026-09-29 UTC

This round implemented the exact input-level JointInput definitions for CLS/Patch visual tokens and All/Text readout. No matching prior implementation or formal result was found. All four new candidates failed the frozen compute gate before smoke training, so no new 500-step run or retrieval evaluation was started.

## Decision

All new candidates are `resource_infeasible`. The measured pair path alone projects to roughly 11.6-11.7 minutes per CLS step and 22.1-22.3 minutes per Patch step, before image/text encoding, feature gather, logging, checkpoint saving, or DataLoader work. The gate is 30 seconds for the complete step. Larger safe microbatches did not materially improve CLS throughput and only modestly improved Patch throughput.

The full pair matrix contains 1,572,864 joint sequences per rank per regular step: three views, two retrieval directions, 256 local queries, and 1024 global candidates. Direct forward work is 2.664 PFLOP/rank for CLS and 5.029 PFLOP/rank for Patch. Even exact reuse of modality-independent projections cannot remove the pair-dependent MLP over interacted text tokens, so this is a compute limitation rather than a tensor-materialization mistake.

## Resource matrix

| Candidate | Tokens | Readout | New params | Microbatch | Pair throughput/rank | Projected pair-only step median/P90 | Measured peak alloc/reserved | 500-step wall lower bound | Status |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| C1 CLS-All | 1 | all | 393,216 | 512 | 2259.6/s | 696.1/701.7 s | 7.62/8.33 GiB | 96.7 h | resource_infeasible |
| C2 CLS-Text | 1 | text | 393,216 | 512 | 2238.6/s | 702.6/708.6 s | 7.62/8.33 GiB | 97.6 h | resource_infeasible |
| P1 Patch-All | 196 | all | 393,216 | 512 | 1175.5/s | 1338.0/1340.9 s | 13.62/15.09 GiB | 185.8 h | resource_infeasible |
| P2 Patch-Text | 196 | text | 393,216 | 512 | 1184.6/s | 1327.7/1332.6 s | 13.62/15.10 GiB | 184.4 h | resource_infeasible |

The memory figures above are pair-microbenchmark peaks, not complete training-step peaks. They show that chunking controls resident tensors; they do not establish full-step memory. A complete step was not launched because the pair-only time lower bound already exceeds the gate by 23x for CLS and 44x for Patch.

## Correctness and implementation checks

- ViT-B/16 returns `X_I=[B,197,768]`, post-LN/pre-projection CLS `[B,768]`, and patches `[B,196,768]` from one visual forward. Native `z` and `h_cls @ visual.proj` match exactly.
- Joint lengths are exactly 249 and 444; Text readout pools transformed text tokens after the shared MaskNetwork block. No residual JointMaskAdapter is called.
- The shared projection A has 393,216 parameters and identical initialization SHA256 `63aad777951b024c898d602556bafb44bbe409d0cd77d2bc1ef574cdab6f2a42` for all candidates.
- Changing the image while holding text fixed changes mask probabilities in all four modes. At initialization, All pooling assigns visual-token mass 0.0031 for CLS and 0.2688 for Patch.
- Condition hidden states are detached before A; A and the original MaskNetwork receive finite gradients. Native CLIP state remains strictly loadable, and bare-student export omits A.
- 23 focused/regression tests pass. The two-rank NCCL reference covers all-valid, a rank with zero valid samples, V=0, V=1, tail batches, checkpoint on/off, all four structures, named gradient None states, and AdamW one-step comparison.
- Maximum two-rank gradient error is 0.0004883. Maximum AdamW update error is 0.0019873, confined to verified near-zero attention key-bias null directions.

## Existing 500-step references

| Model | Method | New params | Updates | J_long | Stable step median | Peak allocated |
|---|---|---:|---:|---:|---:|---:|
| B0 T-fast | text-only NEST | 0 | 500 | 81.82 | 1.974 s | 28.46 GiB |
| B1 Residual-CLS TI-fast | low-rank residual CLS condition | 114,688 | 500 | 82.28 | 2.001 s | 28.84 GiB |

`J_long` is the frozen mean of Urban I2T/T2I R@1 and DOCCI I2T/T2I R@1. B1 improves it by +0.46 pp over B0 in the existing single-seed 500-step run. New candidates have no Recall or J_long entries because training and evaluation were correctly skipped.

## Recommendation

Retain **B1 Residual-CLS TI-fast** as the only next-stage candidate from this comparison. It has completed quality evidence and stays near 2 seconds/step. No exact JointInput candidate should advance under the current 4xA100 budget.

If input-level interaction is revisited, define a new compressed experiment in advance: a small latent cross-attention set, explicit text-token compression, or patch-token selection. Those alter the current experiment and were not run here. No 3651-step training, extra seed, learning-rate scan, Long-DCI, or DCI Full evaluation was started.

## Reproduction and artifacts

- `run.sh tests` runs focused tests and the two-rank NCCL reference.
- `run.sh validate` checks the real ViT/token definitions and conditioning diagnostics.
- `run.sh resource C1|C2|P1|P2` repeats the four-rank resource benchmark.
- `RESULTS.json` is the machine-readable status and evidence index, including source and configuration SHA256 values.
- Raw small resource JSON and validation logs are under `evidence/`. No weights, data, or large profiler trace is committed.
