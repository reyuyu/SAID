# Phase 2.4 Local Semantic Evidence protocol

Frozen CLIP evaluation only. Formal maps remain in identity raster coordinates.
Two backbones (initial OpenAI ViT-B/16 and Phase 2.2 final) use zero-based blocks
3, 6, 9, 11 and four stages: residual, after_attention, attention_delta, mlp_delta.
Input states are additionally evaluated to isolate each sampled block's attention update.
All intermediate and branch-only states reuse final ln_post + proj diagnostically;
this does not assume natural text alignment. No Said q/k is applied.

## Sample and metrics

The original Phase 2.3 small sample is fixed: official val first 128 images,
1,908 phrases, exactly the same 104 switching groups and 237 unordered pairs.
All-phrase means and 474 directed paired observations have separate denominators.
Semantic mass excess is the mean true-minus-shuffled mass on those paired
observations. Its mean equals the symmetric pair switch-margin mean algebraically;
these are not independent selection signals.

Primary temperature is 0.07. Robustness temperatures are 0.03, 0.05, 0.07, 0.10,
0.20; no temperature is tuned or promoted using validation performance.
Pre-softmax orientation margin uses true GT coverage versus rotate180 of the
same GT, over all phrases. It never substitutes another entity for opposite GT.
Spatial prior correlations use average ranks for ties. Random pointing is
computed from patch centers, separately from the mean GT area.

## Pareto gate and full confirmation

Pareto maximizes pointing, mass gain, semantic mass excess, localization margin,
switch margin, target>distractor rate and orientation semantic gap, with no
weighted total score. Candidate eligibility requires improvement over its own
backbone's final residual in mass gain, semantic excess, switch margin and
target>distractor, plus nonnegative orientation semantic and mass-gain gaps.

The corrected complete small audit has one global Pareto-front candidate:
initial_block11_attention_delta. Full confirmation retains the two candidates
already fixed during small-audit inspection: initial_block11_attention_delta
and phase22_block11_attention_delta (the matched trained-backbone counterpart).
Both satisfy the eligibility gate. Each is compared with its own block11
residual baseline. This is two candidates, plus the required backbone-specific
baselines. Full-val results will not change this list.

The full run uses 1,000 images / 14,334 phrases, the identical 791 switching
groups / 1,837 pairs from the previous full audit. The small sample is contained
inside full validation; this is an expanded confirmation, not independent test
set evidence. Locality perturbation is conditional on every candidate having
nonpositive mass gain and switch margin. Positive block11 attention-delta
results mean that condition is not triggered.

## Reproduction

From the repository root, in said-smartclip, set FLICKR30K_ROOT and
FLICKR30K_ENTITIES_ROOT to the original dataset directories.

```bash
python -m eval.salu.local_evidence_eval --image_root "$FLICKR30K_ROOT" --entities_root "$FLICKR30K_ENTITIES_ROOT" --output outputs/local_semantic_evidence/small --max_images 128
python -m eval.salu.local_evidence_eval --image_root "$FLICKR30K_ROOT" --entities_root "$FLICKR30K_ENTITIES_ROOT" --output outputs/local_semantic_evidence/full --max_images 0 --candidates configs/local_evidence_full_candidates.json
```

Use fresh output directories. Logits are persisted after each backbone, enabling
artifact-only postprocessing with --postprocess. A complete manifest is emitted
only after all summaries and maps are written. Final residual equals current
native/LongCLIP patch extraction exactly on eight real images per backbone;
actual observed max difference is zero. Model and training sources are unchanged.
