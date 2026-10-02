# Balanced L14 Four-Epoch Migration

Frozen B16 reference: 14653c92c6da9d552a2b624ab169eaaa275cdde8.
Branch: codex/nest-balanced-l14-4epoch-v1.
Runtime: /root/lk_projects/SAID-nest-clip-v1/balanced_l14_4epoch_v1/.

OpenAI CLIP ViT-L/14,224 input, context248, visual hidden1024, text/native/mask768,
256 patches, one12-head MaskNet block and its independent visual copy, shared
pool, Xavier1024-to768 visual adapter, zero1536-to768 gate. The explicit model
name is recorded in configs/checkpoints and inferred/validated for strict native
export and evaluation. No B16 initialization or fine-tuned L14 checkpoint is used.

The fixed best recipe is fusion_lr2e-4, visual_mask_lr_scale1, view_weights[1,1,1],
sparsity_scale1, inclusion_max1, seed0, epochs4, horizon4868. Losses and detached
conditioning paths remain unchanged. Regularization stays a channel mean.

Correctness gates include B16 regressions, real L14 architecture/native interfaces,
position extension, strict native export, and two-rank768-channel scores/loss,
named gradients and AdamW updates. The distributed fixture uses tiny native
encoders for ten edge cases. A separate full-real-L14 FP32 two-rank reference uses
the fixed first two valid actual training samples and matched per-sample kernel
shapes; every named gradient and AdamW update is checked at the original tolerance.
Unnormalized Gaussian-image full-L14 stress failures remain explicitly recorded.
Real L14 shape/interface tests and real4x256 BF16 resource probes are separate.

The user subsequently approved a5s L14 update limit and requested continuation.
The original3s rejection is preserved. Fresh resource acceptance uses5 warmup
plus30 complete updates<=5s and allocated
memory<=65GiB per rank. Try the direct4x256 route first; mathematically equivalent
pair chunk/checkpoint optimizations are available only if memory is limiting.
A failed gate stops the task at resources and is reported explicitly. No speed
limit is relaxed, no ordinary accumulation substitutes a smaller candidate pool,
and no unvalidated Gradient Cache path is enabled.

After both correctness and resource acceptance, run an independent5-step smoke,
then the sole formal run from L14 step0 to4868. Save0/500/1217/2434/3651/4868
full checkpoints; evaluate0/500/3651/4868 sequentially after training. Formal
nonfinite errors, communications/data errors or three consecutive slow updates
preserve a checkpoint and stop. Native five-dataset protocols are unchanged.

The first live run inherited an initial-checkpoint suppression from the generic
search helper. Its omitted complete step0 was reconstructed CPU-only from the
verified unchanged prepared weights/empty optimizer and pristine per-rank RNG
records. See evidence/initial-checkpoint-reconstruction.json for source hashes
and the explicit derived-checkpoint provenance. The live training was not
restarted, and actual formal configuration remains unchanged. New L14 configs
now retain the requested initial-checkpoint option.

See REPORT.md, RESULTS.json and evidence for actual progress, measurements,
uncompleted nodes, hashes and commands. Push only code/configs/small evidence;
large weights and full data/token logs stay server-local. End after this run.
