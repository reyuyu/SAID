# NEST JointMask probe v1 — pre-formal report

Date: 2026-09-29 UTC

The implementation is based on `25a5d1242231050ce544c4a1df76613cd6a69930` and uses the
shared initialization whose SHA256 is
`54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`.
The three groups are A3-RandomK with `full_native_mix=0`: T (`text_only`), TI
(`joint_image`), and TI-Shuffle (`joint_shuffled_image`).

All 37 focused unit/regression tests passed. The two-rank NCCL reference passed for
text-only, real image conditioning, shuffled conditioning, a rank with zero valid rows,
global V=0/1, and a non-full local batch. Its maximum one-step parameter error against the
single-process global reference was `2.98e-8`.

Each group completed a real four-A100 smoke from the common step 0. Every rank used batch
256, all F/O/E candidate counts were 1024, all four ranks completed five synchronized
updates, all losses and gradients were finite, and the final maximum parameter difference
from rank 0 was exactly zero. Peak allocated memory was 17.268 GiB for T and 17.272 GiB for
TI/TI-Shuffle; steady-state steps took about 4.3–5.0 seconds.

The three step-0 CLIP/MaskNetwork states are byte-identical by tensor digest. TI and
TI-Shuffle adapter states are also byte-identical, all optimizer states are empty, and the
saved per-rank RNG states match. All three five-step sample/text/token/RandomK streams match
each other and the existing A3-RandomK smoke.

At step 1, all three losses are exactly `59.76386642456055` and all pair deltas are zero.
For TI and TI-Shuffle, WI/WT gradients are zero at step 1 while WO is nonzero; from step 2
onward WI, WT, and WO all receive nonzero gradients. Every shuffled offset is nonzero and
within the 1024-element global batch.

TI step 5 was additionally exported and strictly verified. The adapter loaded from the full
checkpoint, the optimizer step was 5, and the bare student's native image and text embeddings
matched the training module exactly (`max_abs=0`).

Machine-readable details, including all rank PIDs/GPU UUIDs, checkpoint hashes, timings and
memory, are in [smoke-audit.json](evidence/smoke-audit.json). Environment and resource checks
are in [preflight.json](evidence/preflight.json), and the two-rank result is in
[ddp-test.console.txt](evidence/ddp-test.console.txt).

Formal execution is configured to run T, TI, and TI-Shuffle sequentially for 500 updates,
each independently from the shared step 0 with scheduler horizon 3651. Evaluation commands
cover COCO, Urban-1k, Flickr30k test1K, DOCCI, and Long-DCI. DCI Full is excluded as requested.
