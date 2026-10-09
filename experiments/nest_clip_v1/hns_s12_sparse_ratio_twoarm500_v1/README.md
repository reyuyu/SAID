# HNS-S12 sparse allocation @500

Two independent fresh-common0 arms, sequential E1 Alignment-Matched then E2
Uniform. Both use macro scales 10/1.2/1, fixed K3, unchanged HNS beta2/2,
Hard-ST and no-SG, soft inclusion off, and original 200-update ramp.

| Arm | Raw ratio | Internal weights (mass5) | Effective sparse coefficients (mass6) |
|---|---|---|---|
| HNS-S12 baseline | 1/2/2 | 1/2/2 | 1.2/2.4/2.4 |
| E1 Alignment-Matched | 1.35/1.35/.30 | 2.25/2.25/.50 | 2.7/2.7/.60 |
| E2 Uniform | 1/1/1 | 5/3,5/3,5/3 | 2/2/2 |

Each arm runs independent smoke5 then starts formal500 again from common0.
Formal training is 4-GPU DDP, batch256/rank, seed0, workers8, BF16,
horizon4868. All data uses `/root/said_s02_stage500/ShareGPT4V/`; missing
local inputs fail without fallback. Post500: immutable gradient audit,
strict bare export/verify, native four-GPU five-dataset evaluation,
report and sanitized GitHub synchronization. Stop after E2; no continuation.

Commands from the isolated worktree:

```bash
.venv/bin/python -m recovery.hns_s12_sparse_ratio_twoarm500 --prepare
.venv/bin/python -m recovery.hns_s12_sparse_ratio_twoarm500 --publish-setup
.venv/bin/python -m recovery.hns_s12_sparse_ratio_twoarm500 --launch
```

Preparation must pass current CPU tests, measured four-process DDP with
uneven valid populations, and original-S12@500 real-global1024 loss and
all330-trainable-parameter gradient checks. The real audit runs pinned
original, repeated original, current default, and actual E1/E2 forwards
without updating parameters. Deterministic kernels are enabled only in
this audit so implementation equality is not obscured by GPU atomic
accumulation noise. Formal training retains the original kernel settings.
Scalar coefficient arithmetic is supplemental, not equivalence evidence.

`--launch` creates a detached session with closed stdin and independent
stdout log. `QUEUE_STATE.json` tracks progress. The runtime `runner.log`
and checkpoints remain local. Any gate/child failure stops the queue and
preserves evidence; retries require explicit preparation of fresh paths.
Completion commits/pushes/fetches and records matching remote/local HEAD.

Evaluation mapping: GPU0 COCO, GPU1 DOCCI, GPU2 Long-DCI, GPU3 Flickr then
Urban. Training and gradient processes must exit before evaluation.
Every arm publishes all30 recalls, aggregate and per-recall deltas,
sampling/LR proof, mask coverage/order/equality, loss contributions,
gradient norms/cosines/additivity, export and runtime evidence.

An earlier setup incorrectly treated a coefficient-only receipt and
inherited tests as new equivalence evidence. That launch was stopped,
evidence retained locally, and these receipts replaced with measured
current-code gates before the fresh launch. No scores came from that
attempt. Small single-seed gains remain exploratory; Urban0.1pp is one
query. No mask density or IoU alone establishes retrieval improvement.
