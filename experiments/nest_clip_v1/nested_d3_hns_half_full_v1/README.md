# HNS-Half continuation: E3 then E4

Continue the evaluated, SHA-pinned Half@2434 checkpoint. This is a continuation
of the existing trajectory, not a new initialization or an experiment sweep.
All new updates keep the existing hierarchy schedule at0.5 and horizon4868.

Sequence: restore2434 → train3651 → strict export/verify → four-GPU native
five-dataset evaluation → report/push/fetch → restore3651 → train4868 → final
export/evaluation/report/sync → STOP. Scores never change this authorized plan.
Technical failure blocks continuation and preserves persistent checkpoints.

Use the isolated worktree `said-e2-worktrees/hns-half-full` on branch
`experiment/nested-d3-hns-half-full-v1`:

```bash
python -m recovery.hns_half_full --prepare
python -m unittest tests.test_e2_candidates tests.test_hns_half_full
python -m pytest tests/test_eval_five_parallel.py
python -m recovery.hns_half_full --publish-setup
python -m recovery.hns_half_full --launch
```

Preparation and launch fail on an existing run; no implicit retries. The detached
runner ignores SIGHUP, holds a trajectory lock and supervises separate process
groups. Each resume verifies model/adapter/AdamW/RNG/loader state and four-rank
parameter synchronization before the next update, then checks the actual first
five batches against the frozen full-HNS sample/text/token/index/LR trajectory.
The entire completed segment is checked before evaluation and continuation.

GPU0: COCO; GPU1: DOCCI; GPU2: Long-DCI; GPU3: Flickr followed by Urban.
Training and gradient work must exit before evaluation. Evaluator math, batch64,
splits, native full-caption inference and aggregate implementations are unchanged.

Images are exclusively `/root/said_s02_stage500/ShareGPT4V`; local misses fail
immediately. No NFS fallback, copying or full decode audit. `/root` is disposable;
NFS originals remain the source of truth. Checkpoints, bare weights and raw logs
stay in `/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/nested_d3_hns_half_full_v1`.
Only explicit small code/config/test/report/diagnostic paths are staged to Git.

Progress: `STATE.json`, `PROGRESS.json`, persistent `runner.log` and
`first-five-gate-2434.json`. Epoch results are `step3651_RESULTS.json` and
`step4868_RESULTS.json`; `RESULTS.json` includes the Half trajectory and complete
native recall deltas against Balanced/HNS at the same epoch.
