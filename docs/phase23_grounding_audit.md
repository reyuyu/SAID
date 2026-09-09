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
