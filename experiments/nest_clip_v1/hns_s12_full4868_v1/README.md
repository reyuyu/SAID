# HNS-S12 fixed weights through4epochs

Only this continuation is authorized: immutable S12@2434 ->3651 ->4868. Parent full checkpoint SHA: `4cde7d4b91af80755215f20687b80856266fa1faed4b2db864a88ae72b030975`. The2434 checkpoint was strictly exported/evaluated without changes. It contains model/adapter, complete AdamW state, native scheduler metadata, all four rank RNG states, loader generator, sampler and cursor. At2434 the next cursor is epoch2/batch0, so no prior images or updates are replayed.

Static macros remain10/1.2/1, alignment1.35/1.35/.30, sparsity1/2/2, beta2/2, Hard-ST/no-SG, old soft inclusion0, fixedK3, original200-update warmup and horizon4868. No E3/E4 coefficient switch or retrieval-based configuration change. Training, objective, sampler, optimizer and evaluator source bytes remain unchanged from the evaluated2434 run.

The reviewed two-boundary controller is reused. Its stop/session metadata now derives from its configured targets; old1217/2434 defaults remain intact. The read-only gradient auditor additionally accepts3651/4868, using the same immutable matched global1024 batch and protocol. No gradient-audit operation updates model parameters.

At each resume, exact model/adapter/optimizer/CPU-CUDA-Python-NumPy RNG and loader generator are checked before the first update, plus zero four-rank parameter difference. CPU metadata/token references check the next five batches' IDs, text, tokens and selected indices; native LR is checked too. Four-card DDP,256/rank,accum1,workers8,seed0,BF16 and native epoch-tail behavior stay frozen.

Sequence: train2434->3651; gradient audit; strict bare export/verify; original parallel five-set evaluation; report/sync; resume **full3651 training checkpoint** to4868; repeat audit/export/verify/evaluation/report/sync; stop. Evaluation starts only after all training/audit GPU processes exit. GPU0 COCO, GPU1 DOCCI, GPU2 Long-DCI, GPU3 Flickr then Urban. Evaluator math, splits, captions, preprocessing, tokenizer, batch64, normalization and recalls remain unchanged.

Original E3/E4 HNS-v1, D3 Balanced and HNS-Half JSONs are pinned with git/blob provenance and the user-quoted Score5 values checked. Compare only matched nodes. Final scientific analysis waits for4868; partial E3 reporting does not steer E4. Last50 loss/CE/share/keep/violations/IoU, real component gradient norms/cosines and speed/resources are retained. Lower keep alone is not proof of better evidence selection; structural telemetry alone cannot establish the causal contribution of HNS.

Images are fail-fast local-only `/root/said_s02_stage500/ShareGPT4V`; no NFS fallback/copy/full decode audit. `/root` remains disposable overlay, and persistent NFS originals must be retained. Checkpoints, bare weights, full stream logs and other large assets stay server-local in persistent runtime. Only allowlisted small source/config/tests/Markdown/JSON are committed.

After preparation and CPU gates pass:

```bash
.venv/bin/python -m recovery.hns_s12_full4868 --publish-setup
.venv/bin/python -m recovery.hns_s12_full4868 --launch
```

Detached new-session runner has no terminal/stdin dependency. Persistent runtime: `/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/hns-s12-full4868-v1`. Inspect `runner.log` and experiment `STATE.json`. Existing technical/nonfinite/local-read/DDP/OOM/>60s/supervisor stops preserve evidence; no automatic retry or tuning. Each node is committed/pushed/fetched and remote HEAD checked. After4868 evaluation the controller exits and GPUs are idle. No additional experiment is queued.
