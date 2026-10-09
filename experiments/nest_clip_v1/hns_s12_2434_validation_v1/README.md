# HNS-S12 frozen continuation to2434

Resume only the evaluated S12@500 checkpoint SHA `cc0b27c871f717c8eb821b1e7a425a8e70ba82f771ee82f0061ca2e6652a172d`. Stop at1217 for a read-only gradient audit, strict bare export/verify, four-GPU native five-set evaluation and report. Only after that succeeds, resume the complete1217 checkpoint to2434, audit/evaluate/report and stop. No other arm or long continuation is scheduled.

All production model, objective, trainer, sampler, optimizer and evaluator sources are byte-identical to the S12@500 run. Static macros are10/1.2/1; internal alignment1.35/1.35/.30, sparsity1/2/2, HNS beta2/2, no-SG, soft inclusion0, fixedK3, ramp200 and horizon4868 remain frozen.

Each resume is checked before its first update: exact model/adapter/AdamW state, CPU/CUDA/Python/NumPy RNG and loader generator, four-rank parameter agreement, correct epoch/batch cursor. Independent CPU references verify the next five batches' sample IDs, F/Dall/D3 strings, token IDs and indices, plus native LR. The first segment replays consumed epoch0 loader batches without updates; the second starts epoch1/batch0. Loader workers8, batch256/rank and native epoch-tail semantics remain unchanged.

All image reads are fail-fast local-only from `/root/said_s02_stage500/ShareGPT4V`; no copy or full decode audit. `/root` is disposable Docker overlay. NFS originals remain the persistent source of truth and must be retained. Full checkpoints, bare weights and raw logs stay under persistent project runtime and are excluded from GitHub.

The existing immutable gradient audit now accepts `--expect-updates 1217|2434`; its matched global1024 audit batch and differentiation protocol are unchanged. This affects only read-only auditing, never training. Evaluator math/batch64/protocol are unchanged. GPU0 COCO, GPU1 DOCCI, GPU2 Long-DCI, GPU3 Flickr then Urban; all training/audit processes must exit before evaluation.

Setup/launch after checkpoint/local-path/test gates pass:

```bash
.venv/bin/python -m recovery.hns_s12_2434_validation --publish-setup
.venv/bin/python -m recovery.hns_s12_2434_validation --launch
```

Detached new-session runner has no terminal/stdin dependency. Persistent runtime: `/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/hns-s12-2434-validation-v1`. Check `runner.log` and experiment `STATE.json`. Errors preserve evidence and block subsequent work; there is no implicit retry. Each node's results/diagnostics are committed, pushed, fetched and remote-HEAD checked. Completion leaves GPUs idle. No3651/4868, coefficient changes or additional experiments.
