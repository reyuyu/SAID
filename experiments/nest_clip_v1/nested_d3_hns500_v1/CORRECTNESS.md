# HNS correctness and source isolation

The only mother implementation is fetched remote INC0 commit
`e201975982809961cf4f078522e3137203c52726`. BASELINE_PROVENANCE.json pins its
actual checkpoint, source hashes, optimizer groups, parameter counts, candidate
protocol, initialization and precision. The actual production mask width is512;
the attachment's256 is illustrative and does not change the architecture.

`tests/test_nested_d3_hns500.py` checks every binary support case using actual
sigmoid-logit Hard-ST masks. ReLU violation gradients are zero at00,01,11;
at10 gradient descent contracts child and expands parent. The erroneous product
`child*(1-parent)` is specifically rejected at00. Neither endpoint is detached.

At lambda0 the actual fetched INC0 forward is bound to the same synthetic
production module. Total loss, every original component, every parameter
gradient and one AdamW update compare exactly (atol=rtol=0). Independent
all-pairs CE/positive-mask objective reconstruction checks loss and every
gradient at completed0,99,199,499. Actual ramp at updates1/100/200/500 is
0/.495/.995/1. Default INC0 arithmetic remains unchanged.

DDP_CORRECTNESS.json records a real four-process CPU/gloo production objective
comparison to a single global8-example reference, with uneven valid counts
0/2/1/1. Loss, component reductions and every gradient agree at explicit FP32
tolerances; every rank's parameters agree exactly after AdamW. The extra
cross-partition AdamW update comparison was intentionally excluded: tiny
near-zero reduction-roundoff gradients can be magnified by epsilon1e-8. This
does not alter the production optimizer. Identical-partition lambda0 AdamW
equivalence is separately exact. No double world scaling occurs.

MATCHED_PREFLIGHT.json records1000 real local-image samples compared with the
fetched dataset class and actual INC0 frozen sampler. IDs, all strings/tokens,
K, indices, resolved paths and image preprocessing tensors match exactly.
Global Python/NumPy/Torch RNG states are unchanged. Frozen data/model encoder,
fusion, preprocessing, optimizer/scheduler and evaluator sources are pinned.
The export change only forwards the HNS constructor flag; export math is frozen.

CPU_TESTS.json contains actual test count, exit status and local JUnit SHA.
SMOKE_EVIDENCE.json is generated only after an independent fresh-common0
smoke5 completes with finite loss/gradients/states, exact frozen stream/LR and
four-rank parameter agreement. Formal500 restarts common0 with resume=None.
HNS_FORMAL_ACCEPTANCE.json is generated only after exactly500 updates, the
first-five gate, complete512000-record stream proof and strict native export.
These runtime evidence files are pending until the relevant stage finishes.

F/Dall/D3 share real mask parameters. GRADIENT_AUDIT.json reports shared
parameter groups and per-view output endpoints without inventing independent
mask branches. Native encoder hidden detach is inherited unchanged, so native
backbone gradients from mask regularizers are zero; alignment updates them.
All mask-to-mask hierarchy gradients remain joint. Positive paired mask
outputs and the CE all-pairs mask computation share parameters but are separate
graphs; a disconnected CE gradient at a paired output is labelled explicitly.

Collapse/inflation/equality are measured scientific outcomes, never early-stop
criteria. Only technical health failures halt this single arm. Reports compare
raw recall fractions and exact evidence to INC0, Anchor,KR234 and Joint-VG.
No full, SG variant, coefficient search or extra arm is authorized by the runner.
