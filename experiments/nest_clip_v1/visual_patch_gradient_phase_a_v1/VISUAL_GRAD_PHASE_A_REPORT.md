# Visual Patch conditional gradient: Phase A stopped preflight

Conclusion: `INCONCLUSIVE`. The gradient acceptance failed before any formal global1024 run. No Phase B training is authorized or started.

## What ran

CPU tests and checkpoint identities passed. GPU0–3 ran a global16 (4/rank) preflight at fixed E2-Uniform@500, with native BF16 encoder autocast/checkpointing and original FP32 masks/fusion. Both variants used identical model parameter objects, frozen values, input tensors and RNG. Only the visual-adapter input detach was removed in the local B override; production code and text detach remained unchanged. No optimizer was constructed; no parameters were updated and no checkpoint was saved.

The full256/rank next501/502 inputs were prepared; all original JSONL sample IDs, F/Dall/D3 text/token summaries, K and selected-index fields matched before slicing to preflight4/rank. The initial comparison failed because the probe omitted the observer detail-index field and compared Python histogram integer keys to JSONL string keys; read-only CPU reconstruction established exact original serialized-field agreement. A regression test was added. The subsequent GPU preflight failed the independent gradient acceptance below. Original failure logs are preserved locally.

## Failed numerical acceptance

All four ranks reported native visual backbone component-gradient additivity relative L2 error **1.539450022%**, exceeding the predeclared **1%** BF16 tolerance. The probe uses true per-parameter component and total autograd gradients, native differentiable all-gather and mean four-rank reduction. FP32 downstream tolerance was predeclared0.03%. Tolerances were not widened after observing failure.

Separate BF16 backward passes can round shared encoder-path accumulations differently, so this discrepancy could be numerical. That explanation has not been independently established. It must not be labelled either a beneficial Patch signal or a harmful training mechanism. The failed traceback omitted the A/B variant, and numeric gradient vectors were not persisted before failure; this limitation is explicit in the JSON files.

The initial script checked all captured A/B forward tensors for bit-exact equality and checked complete state-digest equality and unchanged parameter structure after each backward pass, before reaching the failed gate. All four ranks reached that gate. Thus global16 forward equivalence and in-memory immutability checks passed, but detailed per-tensor receipts and literal state hashes were not persisted. No formal-batch conclusion is claimed. All gradient finite checks preceding the failed gate passed.

The probe has now been improved to persist numerical failure receipts before raising and cleanly release its process group. This reporting-only revision was CPU tested and was **not rerun on GPU**. It changes no production code, loss, precision or numerical threshold.

## Required research questions

| Question | Evidence and status |
|---|---|
| Q1: What does detach block? | Production graph and CPU tests show that adapter-input Patch gradients are blocked, while the native CLS alignment path stays live. Real E2 regularizer-zero/path-norm receipts are UNVERIFIED because the preflight stopped before publishing them. |
| Q2: How large, from which component? | UNVERIFIED; A/B components were differentiated, but norms were not persisted before failure. |
| Q3: Cooperative, orthogonal or conflicting? | UNVERIFIED; no numerical cosines are reported. |
| Q4: Similar across500/1217? | UNVERIFIED;1217 was not run after500 preflight failure. |
| Q5: Extra memory/global1024 feasibility? | UNVERIFIED; no OOM in global16, but this does not establish global1024 feasibility. Per-variant memory/time receipts were not persisted. |
| Q6: Proceed to500-step training? | No recommendation to train yet. First resolve component-gradient consistency in a separately authorized read-only probe without changing production precision/model/loss. |

A future user-approved Phase B, if numerical evidence supports it, would change only this single visual input detach on the frozen E2 protocol. This task does not start it. No retrieval evaluation or Urban feedback was used. Gradient direction alone cannot prove retrieval benefit.

## Immutable identities and stopping

- E2@500: `74271df5298525f924834b3c74fa228eceff623020291c775808f1994214cb9a`; before/after expected hashes match.
- E2@1217: `d276813e12b1b6da9a1f0247d9bbe13c6471276b45e852e06b149b9ad3cb5ead`; before/after expected hashes match.

Both checkpoint manifests match the untouched production sources in the analysis worktree. Existing E2 reports, evaluator code and weights were not written. All formal gradient, layerwise, stability and resource claims remain UNVERIFIED. Scientific status is INCONCLUSIVE, not MECHANISM_UNFAVORABLE.

Reproduction scripts: `recovery/visual_patch_gradient_phase_a.py` (GPU; further execution requires user direction after this failed acceptance) and `recovery/collect_visual_patch_gradient_phase_a.py --stopped-preflight` (CPU-only report from saved failure logs).
