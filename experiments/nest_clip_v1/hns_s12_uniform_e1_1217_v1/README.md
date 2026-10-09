# E2-Uniform native resume500 to1217

Only the evaluated E2-Uniform checkpoint is accepted:
`74271df5298525f924834b3c74fa228eceff623020291c775808f1994214cb9a`.
Parent branch `experiment/hns-s12-sparse-ratio-twoarm500-v1`, commit
`9880a296fb6552c66cf45f9646c6f8a2f2e802e4`. Production model, training,
loss, sampling and evaluators remain byte-identical to its code manifest.

The isolated controller reuses the reviewed S12 continuation hooks and
calls `train.train_nested_semantic_mask.main()` with its native `--resume`
interface, `--run-type formal`, `--max-updates1217`, original init-state,
config and local data. It neither resets optimizer/RNG nor changes the
horizon4868, ramp200, macro10/1.2/1, uniform sparse allocation, Hard-ST,
no-SG, beta2/2, alignment1.35/1.35/.30 or fixedK3.

Before launch: complete identity/SHA/source/config/AdamW/RNG/sampler checks,
original next-five-batch replay, and5000 local paths sampled from the
unconsumed epoch0 suffix. Native loader replay decodes consumed batches
1..500 while performing no optimizer updates, preserving original data
cursor behavior and DataLoader-generator restoration. Model/fusion/
AdamW/RNG/loader state and four-rank parameter agreement are audited before
update501. Updates501..505 check sample strings/tokens/indices/LR, exact
optimizer counters, finite parameters and rank agreement.

Exactly717 updates produce checkpoint1217, next_epoch1/next_batch0, AdamW
steps1217, horizon4868. The native180/rank final batch is unchanged. All
733904 suffix sample records and LR are compared against the original
same-epoch S12 data stream; model parameters are intentionally not compared
to the1:2:2 baseline. The original500 checkpoint is never overwritten and
its SHA is rechecked. Ordinary mask ordering inversions are telemetry;
nonfinite values, resource failure and reviewed complete-mask collapse
rules stop with evidence and no automatic retry or parameter adjustment.

After training: immutable matched-global1024 gradient audit; strict bare
export and verify; original parallel native five-set evaluation; report,
sanitized commit/push/fetch and remote/local HEAD verification; stop.
GPU0 COCO, GPU1 DOCCI, GPU2 Long-DCI, GPU3 Flickr then Urban. No evaluator
changes, inference masks, rerank, ensemble or TTA. No E1, other arm or
2434/3651/4868 continuation.

From this isolated worktree:

```bash
.venv/bin/python -m recovery.hns_s12_uniform_e1_1217 --prepare
.venv/bin/python -m recovery.hns_s12_uniform_e1_1217 --publish-setup
.venv/bin/python -m recovery.hns_s12_uniform_e1_1217 --launch
```

Launch creates a detached session, closes stdin and writes a persistent
runtime `runner.log`. `STATE.json` and runtime `RESTORE_AUDIT.json`,
`batch/lr/update-501..505-rank0..3.json` expose startup evidence. Final
results include all30 recalls and all deltas against HNS-S12@1217 and own
500, CE/shares, mask hierarchy/equality, gradient norms/cosines, resource
and checkpoint audits. Checkpoints, bare weights, raw large logs, data and
image mirrors remain local. Small single-seed gains are exploratory;
Urban0.1pp is one query.
