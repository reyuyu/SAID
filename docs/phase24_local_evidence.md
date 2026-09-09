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

## Completed results and interpretation

Both runs completed. See [the complete metric tables](phase24_local_evidence_results.md)
for every candidate, orientation/logit control, spatial correlation, block update
and selected-candidate temperature sweep. The small global Pareto front contains
only initial block11 attention_delta. Selection remained fixed for full validation.

On full validation, initial attention_delta increases pointing from 0.233710 to
0.647691, mass gain from -0.019051 to +0.025210, semantic excess/switch margin
from -0.014220 to +0.018114, and target>distractor from 0.485074 to 0.533093.
Its localization margin is +0.008029 and orientation semantic gap +0.023265.
The matched Phase 2.2 attention_delta also passes all gates: pointing 0.592159,
mass gain +0.020611, semantic excess +0.018543, target>distractor 0.535014,
localization margin +0.006839, orientation semantic gap +0.024100.
Initial has better pointing and mass gain; Phase 2.2 has slightly higher paired
semantic excess and target>distractor. Do not claim initial dominates on full val.

Within both backbones, block3 has the highest mass gain and semantic excess among
residual candidates and among after-attention candidates, but both are negative.
Block11 attention_delta is the clear useful branch candidate. "Best" here names
these explicit comparisons, not an invented aggregate score.

MLP updates reduce mass gain and semantic excess at every sampled block in both
backbones. Attention updates already slightly reduce mass gain at block3, with
joint semantic deterioration at block6; they improve mass and semantics at blocks9
and11. Thus it would be incorrect to blame every attention update or infer a
monotonic loss of locality. These are projected localization diagnostics, not
causal pixel-locality measurements. The difference between a branch-only feature
and a residual mixture also does not by itself establish a sole causal source.

Both orientation gaps are positive for initial attention_delta blocks3/11 and
Phase 2.2 attention_delta blocks3/9/11. Some other candidates have mixed signs;
all residual and after-attention candidates retain negative semantic orientation
gaps. No rotated map is a benchmark or corrected representation.

The selected attention_delta candidates retain positive mass gain and semantic
excess across all five temperatures on small and full evaluation; final residual
baselines retain negative values. Primary temperature remains 0.07.
Paired phrase controls argue against a purely fixed spatial-prior explanation;
prior correlation alone neither proves nor disproves grounding. No uncertainty
interval or independent held-out test claim is made.

Conclusion **A** is best supported: a suitable relative improvement in local
semantic representation exists inside the current frozen CLIP ViT, specifically
the final attention branch under diagnostic final projection. This is a candidate
for subsequent review, not a production-readiness claim: target>distractor is
only about 53%, and small-sample mean localization margins remain negative.
The evidence supports MLP-associated degradation but does not establish the
strong causal claim in conclusion C. Optional locality perturbation was not
triggered. No training, q/k modification, production integration, Prototype or
Unsaid work was performed.

## Dashboard and validation

```bash
streamlit run tools/said_dashboard/app.py --server.address 127.0.0.1 --server.port 8501
```

Select **Local Semantic Evidence** and artifact root
`outputs/local_semantic_evidence/small` (or `outputs/local_semantic_evidence/full`).
Panels: sample comparison and per-phrase metrics; identity/rotate180 diagnostic;
shared-scale 4x4 layer/stage matrix; layer metric evolution; Pareto ranking and
temperature robustness; spatial priors. The full run only contains its four
selected entries. All dashboard content is read from precomputed artifacts.

Final pytest: **80 passed, 1 skipped, 3 warnings**. The seven decomposition/metric
tests include exact reconstruction, patch order, 768-to-512 projection, native
and LongCLIP final equivalence, normalization, opposite-GT synthetic orientation,
and tied ranks. Two dashboard tests cover the complete page and actionable
missing-image handling. Browser smoke also renders the real small-run sample.
Artifact validation checked every saved map: 61,056 small and 57,336 full maps,
plus all 128/1,000 input crops. Shapes, finiteness, unit sums, phrase counts and
metric means pass. Saved maps match softmax of persisted logits with maximum
absolute errors 1.86e-9 (small) and 3.54e-9 (full). Results are saved outside git
in `outputs/local_semantic_evidence/validation_checks.json`.
