# HNS static-strength two-arm @500

Only E1-HNS-S12 `(10,1.2,1)` and E2-HNS-H4 `(10,1,4)` are authorized, in that order. Each runs an independent smoke5, then a fresh-common0 formal500 with horizon4868. H4 remains fixed4 throughout. No continuation, combinations, third arm or automatic retry.

Training/kernel/sampling/optimizer/evaluator implementations remain unchanged from the verified parent commit. The controller calls the existing macro interface, strict export/verify and four-GPU native evaluation. Evaluation mapping: GPU0 COCO, GPU1 DOCCI, GPU2 Long-DCI, GPU3 Flickr then Urban.

All training reads are local-only from `/root/said_s02_stage500/ShareGPT4V`. This disposable Docker overlay is not the persistent source of truth; retain NFS originals. Checkpoints and raw logs stay in persistent project runtime and are excluded from GitHub.

Prelaunch gates: CPU tests; four-rank default-loss/gradient/AdamW equivalence and global-valid normalization; real BF16 default equivalence at completed0/99/199/200/499; immutable original-HNS@500 gradient audit with S12/H4 controls sharing exactly the same parameters, batch1024 and raw loss graph. Those coefficient controls are distinct from each arm's actual trained500 gradient audit.

Technical stops preserve evidence: existing finite/CUDA/OOM/local-read/DDP/supervisor/long-step guards, plus all three valid-population supports entirely empty or entirely full at the same endpoint for five consecutive updates. Ordinary support density/equality changes do not stop training or alter coefficients.

After all gate JSONs pass, publish setup and launch the detached queue:

```bash
.venv/bin/python -m recovery.hns_static_strength_twoarm --publish-setup
.venv/bin/python -m recovery.hns_static_strength_twoarm --launch
```

The new-session runner uses no terminal/stdin and survives the interactive shell. Inspect `QUEUE_STATE.json` and persistent runtime `runner.log`. Each completed arm is reported, committed, pushed, fetched and remote-HEAD checked. Only allowlisted small code/tests/Markdown/JSON are staged. The controller stops after exactly two arms.
