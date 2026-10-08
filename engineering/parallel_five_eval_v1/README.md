# Parallel native evaluation scheduling

Run after training and gradient audit processes have exited:

```bash
.venv/bin/python -m tools.eval_five_parallel \
  --checkpoint /absolute/path/student_step500.pt \
  --training-checkpoint /absolute/path/step000500.pt \
  --output-dir /absolute/path/new-evaluation-directory
```

`--checkpoint` is the existing strict bare export. The optional training checkpoint is hashed for provenance only. Export and verify-export remain separate unchanged steps. Existing outputs are never overwritten. The scheduler rejects occupied GPUs, active training/audit processes, ambiguous CUDA visibility and overlapping scheduler GPU locks.

GPU0 runs COCO, GPU1 DOCCI, GPU2 Long-DCI, GPU3 Flickr then Urban. `--gpus 0,1,2,3` changes physical indices in that order. All child commands use explicit `cuda:N`, inherit the environment, batch64 and the existing evaluator modules. No DDP, model reuse, metric implementation or retry exists in this scheduler.

Outputs retain their original names: `coco_native.json`, `urban_native.json`, `flickr_test1k/flickr_test1k.json`, `docci/docci.json`, `long_dci/long_dci.json`. Separate logs are in `logs/`. `EVAL_PARALLEL_RUN.json` records hashes, commit, exact commands, GPU mapping, UTC times, durations, return codes and failures. Any subprocess failure or missing/invalid JSON fails the overall run; completed results remain available. Flickr failure blocks Urban on its lane.

`--serial` is validation-only: the historical five-dataset order on GPU0. The real validation driver is `python -m recovery.validate_parallel_five_eval`; it uses the same pinned HNS-SG@500 bare export for both modes and reuses existing metric readers and aggregate functions. A numerical mismatch prohibits default integration.

The shared single-model experiment entrypoints may use this scheduler only after the recorded numerical validation succeeds. The completed HNS-Half specialized controller schedules three models concurrently with its own GPU allocation and is not changed into competing four-GPU schedulers.

CPU tests: `.venv/bin/python -m pytest -q tests/test_eval_five_parallel.py`. Weights, datasets and raw logs remain local; only code, curated receipts, results and timing/comparison reports are published.
