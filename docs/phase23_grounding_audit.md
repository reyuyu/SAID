# Phase 2.3: Semantic Grounding Audit

This is frozen-model evaluation, not training. No changes are made to
SaidRouter, projection initialization, losses, supervision or the backbone.
Caption-dependent routing is not assumed to be correct semantic grounding.

## Data preparation and provenance

Use the [official Flickr30K Entities annotation repository](https://github.com/BryanPlummer/flickr30k_entities)
and its `val.txt` or `test.txt`. The annotation revision used here is
`68b3d6f12d1d710f96233f6bd2b6de799d6f4e5b`. Its README describes phrase chains,
official categories, XML boxes and the original image request procedure.
Obtain the images through the linked Flickr30K project page if a local copy is
not available. Set both variables explicitly; the evaluator never guesses paths
or silently skips missing image files:

```bash
export FLICKR30K_ROOT=/path/to/flickr30k/images
export FLICKR30K_ENTITIES_ROOT=/path/to/flickr30k_entities
```

Required layout:

```text
$FLICKR30K_ROOT/<image_id>.jpg
$FLICKR30K_ENTITIES_ROOT/val.txt
$FLICKR30K_ENTITIES_ROOT/test.txt
$FLICKR30K_ENTITIES_ROOT/Annotations/<image_id>.xml
$FLICKR30K_ENTITIES_ROOT/Sentences/<image_id>.txt
$FLICKR30K_ENTITIES_ROOT/UNRELATED_CAPTIONS
```

Extract `annotations.zip` from the official repository at its root. Images for
this run came from the public `nlphuji/flickr30k` image archive, downloaded via
`hf-mirror.com`, with only the 1,000 official validation image IDs extracted.
The mirror's dataset split labels are not used. Image dimensions must match
the official XML. Image/annotation files, checkpoints and all audit outputs
are outside version control. Follow the source dataset's conditions of use.

## Protocol

* Primary denominator: every visible boxed phrase **mention**, over all five
  sentences. Repeated mentions of an entity remain distinct queries, matching
  the phrase-level task. Only the phrase string is tokenized. The full sentence
  is stored for human context, never used as a query; no silent truncation.
* Standard validation file order, first 128 images for the small audit, then
  the complete 1,000-image validation split. Selection never uses model scores.
* Known unrelated sentences listed by the official repository are excluded.
  Scene/nonvisual/no-box mentions and fully cropped-away targets are excluded
  with separate counts. Missing images or malformed boxes raise errors.
* PASCAL VOC one-based inclusive boxes become continuous zero-based half-open
  rectangles `[xmin-1, ymin-1, xmax, ymax]`. Apply the actual integer dimensions
  of short-side resize to 224 and torchvision's rounded 224 center crop. Clip
  the resulting boxes to the model input. Partially visible targets are scored
  on their visible union, with the retained area fraction saved per phrase.
* Every model sees the same crop, same phrase and same visible target boxes.
  Unit tests compare the crop pixels with torchvision on odd/asymmetric sizes.
  Scores describe the visible crop, **not** full-image grounding performance.
* Run inference in float32 for all four models, with TF32 disabled and fixed
  seeds. No optimizer, backward pass, gradients or checkpoint mutation.

## Four variants

| artifact name | image/text backbone | attention |
| --- | --- | --- |
| initial_direct | native OpenAI CLIP ViT-B/16, original 77-position text encoder | cosine of projected final patch and phrase, softmax / tau_eval |
| phase21_router | Phase 2.1 final positive-Said checkpoint | existing SaidRouter, checkpoint temperature |
| phase22_router | Phase 2.2 final identifiable-Said checkpoint | existing SaidRouter, checkpoint temperature |
| phase22_direct | same Phase 2.2 final image **and text** backbone as above | direct patch/text cosine; no q/k projections |

Native CLIP's final patches are extracted using the same final ln_post/proj
convention as the reviewed Phase 1 interface. SALU backbones use that interface
directly. Initial CLIP remains native; trained models retain their reviewed
LongCLIP text interface. `tau_eval=0.07` is configurable and applies only to
direct attention. Checkpoint state dict loading is strict; source checkpoint
SHA256, training mode and router temperature are recorded in the manifest.

## Metrics and denominators

1. Exact rectangle-union intersection with each 16x16 patch gives `W_GT` in
   `[0,1]`. Overlapping instances are not counted twice; spaces between boxes
   are not filled in.
2. Pointing: center of the argmax patch inside any visible target box. Argmax
   ties follow NumPy's first-index rule.
3. GT mass = `sum(A * W_GT)`. Report mean, median, p25 and p75.
4. Area fraction = `mean(W_GT)`; mass gain = GT mass minus area fraction.
   Uniform attention has exactly zero gain; mass lift is supplementary only.
5. Distractors: other visible annotated entity unions with exact union IoU
   `<0.1` against the target. Margin = target mass minus the **maximum** valid
   distractor mass. Rate uses strict `>`. Queries with no distractor have null
   margin and are excluded from that metric, with eligible count recorded.
   This raw-mass margin is size-sensitive, so read it alongside area/mass gain.
6. Switching: choose one short annotated mention per entity, then deterministically
   select 2-4 entities with pairwise union IoU `<0.1` and bounding-envelope
   center separation `>=0.2 * 224` pixels. Evaluate all pairs in these groups.
   `switch_margin = ((M_aa-M_ab)+(M_bb-M_ba))/2`; report distribution, positive
   fraction, both-target preference and both-pointing-correct rate. All models
   share the same pairs. At least the first 32 multi-entity images get saved
   overlays; the UI can render every stored group. JSD is supplementary.
7. Peak labels: target takes priority, then another visible annotated entity,
   then background. **Background means outside annotations; it may contain
   unannotated entities.** This is evidence of location, not proof that a token
   contains global context. No new categories are invented; official multiple
   phrase types contribute to each corresponding category breakdown.

## Running the small and full audits

```bash
python -m pytest tests/test_semantic_grounding_eval.py -q

python -m eval.salu.semantic_grounding_eval \
  --split val --max_images 128 --tau_eval 0.07 \
  --phase21_checkpoint runs_salu/phase21/salu_said_only_last.pt \
  --phase22_checkpoint runs_salu/phase22/salu_said_only_last.pt \
  --output_dir outputs/semantic_grounding_small

# Only after inspecting the small audit:
python -m eval.salu.semantic_grounding_eval \
  --split val --max_images 0 --tau_eval 0.07 \
  --phase21_checkpoint runs_salu/phase21/salu_said_only_last.pt \
  --phase22_checkpoint runs_salu/phase22/salu_said_only_last.pt \
  --output_dir outputs/semantic_grounding
```

Use a fresh output directory: the evaluator refuses to mix runs. A manifest
with `status=complete` is published only after all four variants and overlays
are finished. JSON rejects NaN/Inf. Best/worst 50 phrase lists for each model
are ranked by area-adjusted mass gain, with their IDs saved in `summary.json`.

```text
outputs/semantic_grounding/
  manifest.json       # protocol, split/hash, checkpoint hashes, images, switch groups
  per_phrase.json     # phrase, context, GT boxes, categories, per-model metrics
  summary.json        # overall/category distributions; best50 / worst50
  switching.json      # per-pair four masses, margin, correctness and JSD
  images/            # exact model-input RGB crops
  attention/<model>/ # per-phrase float32 14x14 arrays
  overlays/<model>/  # selected cases, target boxes and peak crosses
```

## Interpretation

Compare C against D to isolate the effect of adding q/k routing to the **same**
backbone. Compare A against D for pretrained versus fine-tuned direct geometry;
these are different text/image backbones, not a causal component ablation.
Positive mass gain and target switching matter alongside pointing, and a high
pointing score alone does not establish selectivity. Correlated mentions and
repeated entities mean phrase counts are not independent statistical samples.
This audit measures our specified pointing/mass protocol on the official split;
it is not the standard proposal-based Flickr30K recall@K benchmark.

Any failing audit is reported as such. Algorithm changes require another phase.

## Completed audit results

Both runs completed successfully on the official validation split. The small run was inspected before launching the full run. These are frozen evaluation results; model, router, initialization, loss and training code were unchanged.

| Run | Images | Phrase mentions | Spatially separated multi-entity images | Pairs per model |
|---|---:|---:|---:|---:|
| Small | 128 | 1,908 | 104 | 237 |
| Full | 1,000 | 14,334 | 791 | 1,837 |

Full-run exclusions: 2,082 `no_box`, 1,002 `missing_box_or_nonvisual`, 211 `scene`, and 99 fully outside the center crop. There were no missing images. Metrics retain all eligible phrase mentions across the five sentences, so entities can recur. Localization metrics have 12,495 eligible target/distractor comparisons.

Evaluation code commit: `cdc8125d3fe60eba173e9949cd7879bc2808b520`. The dashboard-only commit made while the full run was finishing did not alter evaluation. Annotation repository revision: `68b3d6f12d1d710f96233f6bd2b6de799d6f4e5b`. The artifact manifest contains split and annotation archive SHA-256 hashes.

| Checkpoint | SHA-256 | Router mode / temperature |
|---|---|---|
| phase21 | `402f94d0ef0f014db8242f6d3c8eb8297c8313211ea41a8db3e5774f73d88193` | positive / 0.07 |
| phase22 | `e748a73b7e567aac79b1a6fc0523c3ca64fbd5f509bee5b0e09e77c600015ab7` | identifiable / 0.07 |

A uses the initial official OpenAI CLIP ViT-B/16 weights with its native text encoder. B and C use their respective final checkpoints; D uses exactly C's text/image backbone without q/k projection. All direct evaluations use cosine softmax at 0.07.

### Overall grounding

Mass and margins below are means. The shared mean GT area fraction is **0.319635**. A uniform attention map has zero mass gain; every model has negative mean mass gain.

| Model | Pointing | GT mass | Mass gain | Target > distractor | Localization margin | Switch margin |
|---|---:|---:|---:|---:|---:|---:|
| A Initial direct | 23.37% | 0.300584 | -0.019051 | 48.51% | -0.039170 | -0.014220 |
| B Phase 2.1 Router | 25.67% | 0.294295 | -0.025340 | 49.49% | -0.030674 | -0.003229 |
| C Phase 2.2 Router | 29.65% | 0.291263 | -0.028372 | 47.07% | -0.056122 | -0.026515 |
| D Phase 2.2 direct | 23.24% | 0.283930 | -0.035705 | 47.27% | -0.051162 | -0.020026 |

| Model | GT mass median | GT mass p25 / p75 | Switch median | Switch p25 / p75 | Positive switch | Both queries prefer own target |
|---|---:|---:|---:|---:|---:|---:|
| A Initial direct | 0.208158 | 0.062049 / 0.471063 | -0.010373 | -0.021766 / -0.002599 | 16.00% | 0.05% |
| B Phase 2.1 Router | 0.204715 | 0.063764 / 0.450922 | -0.001990 | -0.010797 / 0.004577 | 39.68% | 0.82% |
| C Phase 2.2 Router | 0.201201 | 0.061735 / 0.447331 | -0.018412 | -0.041722 / -0.004046 | 17.04% | 0.38% |
| D Phase 2.2 direct | 0.188283 | 0.052441 / 0.442275 | -0.015323 | -0.031361 / -0.004689 | 13.45% | 0.05% |

A positive pair-averaged switch margin does not imply both individual queries prefer their own target. For C this stronger check passes only 7/1,837 pairs (0.38%). Differing box sizes affect individual target/distractor preference; symmetric switching cancels a query-independent attention map exactly.

### Peak location analysis

| Model | Target | Other annotated object | Outside all annotations |
|---|---:|---:|---:|
| A Initial direct | 3,350 (23.37%) | 6,732 (46.97%) | 4,252 (29.66%) |
| B Phase 2.1 Router | 3,679 (25.67%) | 6,665 (46.50%) | 3,990 (27.84%) |
| C Phase 2.2 Router | 4,250 (29.65%) | 7,262 (50.66%) | 2,822 (19.69%) |
| D Phase 2.2 direct | 3,331 (23.24%) | 6,919 (48.27%) | 4,084 (28.49%) |

For C, other-object peaks are more common than background peaks. This does not establish that background tokens encode global context: the diagnostic identifies peak coordinates only, and annotations are incomplete.

### Official category breakdown

Each model cell reports pointing accuracy / mean mass gain. Categories are used verbatim and can overlap. Full mass/localization distributions are in `summary.json` and the dashboard.

| Category | Mentions | A | B | C | D |
|---|---:|---:|---:|---:|---:|
| animals | 523 | 35.37% / -0.0101 | 21.61% / -0.0315 | 28.68% / -0.0562 | 24.28% / -0.0517 |
| bodyparts | 537 | 7.26% / -0.0122 | 9.31% / -0.0173 | 6.33% / -0.0252 | 8.57% / -0.0196 |
| clothing | 2317 | 10.70% / -0.0243 | 10.79% / -0.0337 | 12.60% / -0.0374 | 11.31% / -0.0380 |
| instruments | 155 | 21.29% / -0.0333 | 10.32% / -0.0345 | 20.65% / -0.0347 | 12.26% / -0.0497 |
| other | 3243 | 18.75% / -0.0203 | 20.88% / -0.0190 | 24.05% / -0.0199 | 18.87% / -0.0327 |
| people | 5790 | 24.94% / -0.0173 | 29.86% / -0.0277 | 36.37% / -0.0295 | 25.15% / -0.0368 |
| scene | 1521 | 46.75% / -0.0174 | 50.95% / -0.0145 | 50.82% / -0.0162 | 48.85% / -0.0288 |
| vehicles | 337 | 32.64% / -0.0260 | 30.27% / -0.0347 | 33.23% / -0.0427 | 29.08% / -0.0542 |

### Small-run results and qualitative review

| Model | Pointing | GT mass | Mass gain | Switch mean | Positive switch |
|---|---:|---:|---:|---:|---:|
| A Initial direct | 24.79% | 0.315033 | -0.018680 | -0.014978 | 16.46% |
| B Phase 2.1 Router | 27.36% | 0.305277 | -0.028437 | -0.004214 | 43.46% |
| C Phase 2.2 Router | 31.55% | 0.304099 | -0.029614 | -0.028197 | 17.72% |
| D Phase 2.2 direct | 24.16% | 0.298650 | -0.035063 | -0.019651 | 17.72% |

Both output roots contain 50 best and 50 worst phrase IDs **per model**, ranked by mass gain, plus exported overlays. These are selected extremes, not a representative random sample. Examples from the full C ranking:

| Selection | Phrase ID | Phrase | Pointing | GT mass | Mass gain |
|---|---|---|---|---:|---:|
| worst50 | `7652712058_s3_p0` | A nearly full stadium of people | False | 0.196722 | -0.339278 |
| worst50 | `3260191163_s1_p0` | Two brown dogs | False | 0.269540 | -0.303006 |
| best50 | `3299820401_s3_p0` | A man | True | 0.505440 | 0.276754 |
| best50 | `416825249_s4_p1` | a toddler swing | True | 0.671504 | 0.208575 |

The stadium query places its peak on the pitch; the two-dog query peaks on grass below the dogs. Conversely, the cyclist query can focus on the man, and the toddler-swing query concentrates on the swing. The lowest switching pair reverses grass versus blue-jacket attention (margin -0.2400). Even the highest switch-margin pair (+0.0736) fails the stronger two-query target-preference check.

### Router versus direct CLIP and conclusion

**Selected conclusion: B. Caption-dependent, but semantic grounding is only partial.**

C beats same-backbone D in pointing by 6.41 percentage points and GT mass by 0.007333. However, its mean localization margin is worse by 0.004960 and its mean switch margin is worse by 0.006489. Its positive-switch fraction is 17.04% versus D's 13.45%, so even switching has mixed comparisons. There is no across-metric winner. Higher pointing does not demonstrate reliable entity selectivity.

Compared with initial direct A, fine-tuned direct D has almost unchanged pointing (-0.13 percentage points), but GT mass declines by 0.016654 and switch margin by 0.005806. This supports degradation of direct final-patch localization under this protocol. It does not isolate which component of training caused it.

All four models have negative mean mass gain and switch margin. Weak direct baselines make final-layer representation inadequacy a plausible hypothesis, but no layer comparison or alternative readout was evaluated: conclusion D as an inherent representation claim is not established. The results also do not establish that routing is *mostly* a non-semantic identification code (conclusion C). The grounding success claim (conclusion A) is unsupported. No algorithm changes follow from this audit.

### Dashboard and validation

Start from the repository root in the said-smartclip environment:

```bash
python -m streamlit run tools/said_dashboard/app.py --server.address 127.0.0.1 --server.port 8501
```

Select **Semantic Grounding Audit** and artifact root `outputs/semantic_grounding` (or `outputs/semantic_grounding_small`). This is the existing app. Four-model heatmaps share a colour scale by default, with GT boxes, peak crosses, same-backbone difference, A/B switching, failure filters, metric sorting and official-category breakdown. UI actions only read precomputed artifacts.

- Full test suite: **63 passed, 1 skipped, 3 warnings**. The skipped test is the existing distributed torchrun-only test; the warnings come from existing dependency/checkpoint behavior.

- Independent artifact validation: all 1,000 crops readable at 224x224, all 57,336 attention arrays finite/normalized at 14x14, pointing and summary arithmetic checked, all switching pair arithmetic checked, selected overlays present. **PASS** (`validation.json`, untracked output).

- Small and full evaluator exits: **0**; complete manifests emitted for all four models.

- Dashboard AppTest fixtures cover phrase selection, filters, GT overlays, maps, metrics and controlled missing-artifact errors; real-browser checks loaded both small and complete full artifacts, four-model panels and switching. Streamlit health: **ok**.

- Git comparison against the merged Phase 2.2 baseline contains no changes under `model/` or `train/`. Datasets, images, outputs and checkpoints remain outside Git.

Phase 2.3 remains on its review branch. Stop here for Review.
