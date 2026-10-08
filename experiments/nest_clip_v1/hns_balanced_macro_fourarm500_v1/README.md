# HNS and D3 Balanced macro-loss search @500

Four sequential independent arms: HNS-A8, Balanced-A8, Balanced-S12,
Balanced-H125. Each runs smoke5 from common0, then a new process and fresh
common0 for exactly500 formal optimizer updates, followed by gradient audit,
strict bare export/verification, native five-set parallel evaluation, report,
commit/push/fetch. No other arm or long continuation is scheduled.

Only macro coefficients change. Alignment uses original1.35/1.35/.30;
sparsity uses original1/2/2. HNS uses the original Hard-ST ReLU support loss,
beta2/2 and no mask-to-mask SG, with soft inclusion disabled. Balanced uses
original soft probability detached-child detail-chain inclusion, with HNS
disabled. Both retain the original200-step completed-before-update ramp.
The macro interface refuses a second sparsity_scale or inclusion_max.

Preparation and launch use `python -m recovery.hns_balanced_macro_fourarm
--prepare` and `--launch`. Launch requires all five evidence gates, the
published setup HEAD, an untouched common0 SHA, idle GPUs and an unstarted
queue. Launch creates an independent session with stdin disconnected and
SIGHUP ignored. The controller writes QUEUE_STATE.json and runtime/runner.log;
failure preserves artifacts and stops without automatic retry.

Validation evidence includes fetched default-objective exact CPU loss,
all trainable parameter gradients and AdamW updates; four-rank gloo versus
single-global-batch normalization; real native BF16/NCCL gradients at
completed0/99/199/200/499. The fixed real equivalence batch is32 total
(8/rank) to reduce validation cost; formal training remains1024(256/rank).
Default checkpoint gradient audits use a matched immutable1024-sample
cohort atstep500 for both methods, with no optimizer or parameter updates.
Component and per-view weighted norms are measured through autograd;
scalar-scaled raw norms are labeled estimates because BF16 rounding can
make them differ from measured weighted gradients.

Every arm verifies512000 sample IDs, F/Dall/D3 strings, token IDs, selected
indices and actual group LR values against its archived method baseline.
All image reads resolve only under `/root/said_s02_stage500/ShareGPT4V`;
missing, symlink or escaping paths fail. NFS fallback and data copy are
disabled. `/root` is disposable Docker overlay cache; NFS originals persist.

Four-GPU evaluator mapping:0 COCO,1 DOCCI,2 Long-DCI,3 Flickr then Urban.
Evaluator mathematics, batch64 and native full-caption protocol are frozen.
Training and gradient audits exit before evaluation begins. Results include
all30 recalls, existing aggregate scores, own-baseline deltas and HNS-v1
deltas. Scores below0.05pp improvement on one seed are weak signals;
no500-step candidate becomes a production winner automatically.

GitHub publication uses an explicit allowlist and small text artifacts only.
Checkpoints, bare weights, dataset, mirror, cache and large raw logs remain
local. After all four reports are synced and remote HEAD verified, the
controller exits and leaves GPUs idle.
