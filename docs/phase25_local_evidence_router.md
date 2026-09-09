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
R@1/5/10. Dataset overlap with training may limit interpretation; this comparison
assesses relative standard utility and does not claim an uncontaminated test.

All evaluation artifacts live under outputs/local_evidence_router and are excluded
from git. Evaluators refuse an existing manifest to prevent accidental mixed runs.
Phase 2.3/2.4 official results are untouched.

## Review boundary

No multi-head router, q/k redesign, parser, box supervision, locality loss,
prototype or unsaid component is introduced. Heatmaps are caption-conditioned
evidence maps or token weighting, not segmentation masks. Final interpretation
must distinguish source geometry from learned projection distortion and backbone
fine-tuning effects. Stop for review after reporting results.
