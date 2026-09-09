# Phase 2.5 Local-Evidence Router Integration

## Controlled experiment

Only the router's visual feature source changes: final residual versus final
attention delta, projected by the existing visual ln_post and proj. The existing
SaidRouter, q/k initialization and identifiable objective are reused unchanged.
Global CLIP inference and the existing residual patch API retain their semantics.
Production code does not import the diagnostic extractor. Local features are
computed in one differentiable forward; local checkpointing explicitly raises
NotImplementedError while the existing standard checkpoint path is untouched.
No parameters or state-dict keys are added by the interface or source selector.
Missing checkpoint args.said_feature_source means residual. A resume request
with a different source raises an error instead of silently changing the experiment.

Seed 25, initial OpenAI ViT-B/16, 4 A800 GPUs, batch 256/GPU (global 1024),
one epoch / 659 steps, backbone LR 1e-6, router LR 1e-4, weight decay 1e-2,
200 warmup steps, lambda_global=lambda_said=1, tau_said=0.07,
identifiable loss, pair_chunk_size=64, bf16, eight loader workers per rank.
Both arms use the existing 676,415-entry no-SAM JSON, with the unchanged first
1,000 entries reserved by the dataset class, leaving 675,415 training examples.
This is a mechanism experiment, not full official SmartCLIP training or a final
paper result. No box/phrase annotation is used in training.

Python, NumPy, torch and CUDA RNGs are seeded. DistributedSampler uses the same
seed and epoch; each rank has an explicit loader generator and workers receive
Python/NumPy seeds from torch.initial_seed. Initialization, complete sampler order
and the actually consumed caption stream are SHA-256 audited independently on
every rank. This guarantees auditable input matching, not a claim of bitwise
deterministic CUDA optimization or exact resumed-run RNG restoration.

Run from the repository root in said-smartclip with SHARE4V_DATA_ROOT and
SHARE4V_JSON set to the existing dataset locations:

```bash
bash train/run_local_evidence_ab.sh
```

Independent output directories are under runs_salu/phase25. Each arm saves the
initial model, steps100/200/400, and final; existing checkpoint behavior also
saves the CLIP-only counterparts. No extra checkpoint intervals are introduced.
Training records retain legacy fields and add router_input_feature_norm and
completed_steps. Checkpoint-step diagnostics correspond to the forward pass
immediately before that update; they are minibatch diagnostics, not held-out
retrieval or grounding scores. Identifiable routing uses 256 local candidates,
so its training chance is 1/256; the global contrastive batch is 1024.

## Predefined evaluation

Fixed small audit: identical 128 images / 1,908 phrases / 237 unordered switching
pairs. Both arms at initial, step100, step200, step400, final produce direct cosine
and trained router maps. Primary tau stays 0.07 and raster orientation stays
identity. All localization metrics reuse the reviewed Phase 2.4 geometry and
paired controls. Mean semantic mass excess and symmetric switch margin are
algebraically equal on this paired subset, not independent evidence.

The full Flickr gate requires final attention-delta router mass gain, semantic
excess and switch margin all positive, and all three plus target>distractor
better than final residual router. Report effect sizes without inventing a
significance threshold. If passed, compare residual router, attention-delta direct
and attention-delta router on the full 1,000 images / 14,334 phrases. No other
checkpoint gets full validation. The small subset is contained in full val.

Independently of this benchmark gate, a geometry-only diagnostic uses the first
5,000 phrases in official val order (drawn from its first 400 images), for both
final arms. It measures average-tie Spearman patch ranking and top-k intersection
fraction at k=1/5/10/20. Constant rankings are reported as undefined, not zero.
Ranking correlation versus per-phrase router mass gain is descriptive; quartile
tables do not establish causality. This is not a full-val grounding benchmark.

```bash
python -m eval.salu.local_router_eval --mode small --image_root "$FLICKR30K_ROOT" --entities_root "$FLICKR30K_ENTITIES_ROOT"
python -m eval.salu.local_router_eval --mode geometry --image_root "$FLICKR30K_ROOT" --entities_root "$FLICKR30K_ENTITIES_ROOT"
# Only if the recorded small gate passed:
python -m eval.salu.local_router_eval --mode full --image_root "$FLICKR30K_ROOT" --entities_root "$FLICKR30K_ENTITIES_ROOT"
python -m eval.salu.local_router_coco --coco_root "$COCO_DATA_ROOT"
```

Both final arms get the same standard COCO val2017 5,000-image / 25,000-caption
retrieval evaluation, with fp32 standard encode_image and encode_text only. Local
inference is explicitly guarded against accidental use. Report both directions'
R@1/5/10 to assess relative standard utility under the same evaluation protocol.

All evaluation artifacts live under outputs/local_evidence_router and are excluded
from git. Evaluators refuse an existing manifest to prevent accidental mixed runs.
Phase 2.3/2.4 official results are untouched.

## Review boundary

No multi-head router, q/k redesign, parser, box supervision, locality loss,
prototype or unsaid component is introduced. Heatmaps are caption-conditioned
evidence maps or token weighting, not segmentation masks. Final interpretation
must distinguish source geometry from learned projection distortion and backbone
fine-tuning effects. Stop for review after reporting results.

## Completed result

See [the complete measured tables](phase25_local_evidence_router_results.md).
Both 659-step arms completed. Every rank has identical initial-state, sampler-order
and actual caption-stream hashes across arms; initial parameter tensors also
compare bitwise equal. The only argument differences are feature source and
output directory. All 38,160 small attention maps pass saved-logit reconstruction.

The best-supported classification is **B: good local evidence, but learned free
q/k projections substantially distort its geometry**. This is qualified rather
than a claim that the router has zero semantic information: its final paired
semantic excess is positive and much better than residual routing.

On the fixed small sample, attention-delta direct pointing improves from 64.15%
to 72.69%, mass gain from +0.024277 to +0.075119, and semantic excess from
+0.018517 to +0.036780. Thus the evidence contradicts case C for this run.
The attention-delta router reaches only 39.41% pointing and -0.001668 mass gain,
although semantic excess/switch remain +0.028151 (residual router: -0.028177).
Target>distractor is 52.09% versus 47.97% for residual router and 55.43% for
attention-delta direct. The final router's negative mass gain fails the predefined
full gate. No full Flickr grounding benchmark was run and no earlier checkpoint
was substituted after seeing the trajectory.

Across the fixed 5,000-phrase geometry diagnostic, attention-delta router/direct
Spearman is -0.179125 and top5 overlap is 7.516%. Higher agreement accompanies
higher router mass gain: Pearson +0.532418, Spearman +0.528367. The bottom/top
correlation quartiles have mean mass gains -0.031980 / +0.050243. This supports
the geometry-distortion interpretation without proving a unique training cause.

Both arms become caption-dependent. Final minibatch route accuracy is reported
in the measured tables with chance 1/256; it is not a held-out grounding metric.
COCO I2T R@1 changes 58.54% -> 58.22%, and T2I R@1 40.172% -> 39.516%.
These are modest observed declines, not a statistical equivalence claim.

## Numerical audit detail

Real ViT-B/16 production extraction versus Phase 2.4 diagnostic extraction on
the same weights and input has max absolute difference **0**. The new global
output, standard global/patch APIs and residual forward_train versus the actual
pre-merge source also have max difference **0**. Old Phase 2.2 strict loading
passes; the full SALU parameter count remains 153,513,474.

A separate comparison with Phase 2.4's *native OpenAI* initial logits initially
rejected an overstrict cross-framework 1e-6 equality assumption. Existing
LongCLIP load_from_clip converts text positional embeddings through fp16 before
expansion; its first 20 positions exactly match that fp16 round trip, with a
6.0171e-5 difference from native fp32 positions. Native and production initial
visual state tensors are exactly equal. Measured residual/attention-delta logit
differences are 3.3364e-5 / 2.3335e-5, and attention differences 1.9056e-6 /
1.0016e-6, with **zero changed pointing peaks** over 1,908 phrases. This inherited
text-loading behavior was preserved for the single-variable A/B experiment;
it does not violate same-model production/diagnostic extraction equivalence.

Final tests: **91 passed, 1 skipped, 3 warnings**. Real 4-GPU batch256 smoke,
gradient sanity, legacy source equivalence, state loading and browser rendering
all pass. The dashboard displays the requested four maps, GT, pair metrics and
ranking diagnostics from completed artifacts. All work stops here for review;
no q/k constraint, additional objective or downstream component was introduced.
