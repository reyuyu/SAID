# S0 / S0-TriMask-HS continuation to 1000 updates — implementation

Branch: `codex/s0-continue-1000` (S0 arm) and `codex/s0-trimask-hs-v02` (HS continuation support).
Both continue the *same* arms that already had 500-step checkpoints; no arm, objective, loss or
hyper-parameter was changed. This file describes how the continuation is executed and what makes
its numbers trustworthy. The measured results live in `report.md` / `results.json` next to it.

## What is being extended

| arm | objective | parent checkpoint (@500) |
| --- | --- | --- |
| `S0_smartclip` | `said_cls_cvssl` (SmartCLIP-derived SAID/UnSAID, text mask `none`) | `ddpfix_step500_S0_smartclip/cvssl_S0_smartclip_step000500.pt`, sha256 `758dcdd2…67c743` |
| `S0_TriMask_HS` | `said_trimask_hs` (hard straight-through text gate, `lambda_sparse_T = 0.2`) | `step500/trimask_S0_TriMask_HS_step000500.pt`, sha256 `46f6d9c7…c0c4ef` |

Both parents were trained by the *same* trainer lineage on the *same* data stream: their stored
`caption_stream_sha256` (`799f8efa…`), `sample_stream_sha256` (`39c9885c…`) and
`view_b_param_stream_sha256` (`cbfb7ffd…`) are identical. Whatever the two arms differ by at 1000
updates therefore comes from the objective, not from the data.

## The problem with "just train longer"

The trainers restart at batch 0 of epoch 0 on every launch. Loading a @500 checkpoint with
`--init_state` and setting `--max_steps 1000` would therefore have produced a run that *retrains*
batches 0–499 and then reports itself as a 1000-step checkpoint. That would silently invalidate the
whole comparison, so the continuation is built around an explicit cursor.

## Cursor + replay guard

`--resume <checkpoint>` (both trainers) does the following, in this order:

1. loads the payload, restores model, optimizer **and** mask optimizer, and asserts the
   experiment identity is unchanged (objective, arm, LR horizon 3651; for the HS arm also
   `text_gate_mode = hard_st` and `lambda_sparse_t = 0.2`);
2. derives the cursor from the payload: `next_batch_index` when present, otherwise
   `step_in_epoch + 1`, and refuses if the cursor disagrees with `completed_steps`
   (`completed_steps != epoch * steps_per_epoch + batch_index`);
3. iterates the dataloader for exactly that many batches, hashing the caption stream, the sample
   ids and the view-b parameters **without taking a single optimizer step**;
4. compares those digests against the parent's (`caption_stream_sha256`,
   `sample_stream_sha256`, `view_b_param_stream_sha256`) and aborts with
   `resume replay mismatch: …` on any difference;
5. prints `REPLAY_VERIFIED skipped_batches=<n> caption_sha=… sample_sha=… view_sha=…` and only then
   starts training.

The guard is real verification, not a formality: both parents do carry all three digest keys
(checked before the runs), `next_batch_index` is absent in the S0 parent and is recovered from
`step_in_epoch + 1`, and the verified values match the parents bit for bit:

```
REPLAY_VERIFIED skipped_batches=500
  caption_sha=799f8efae42541759ac92239643d2150dc8aa23d1da984676f029063c21c0fd4
  sample_sha=39c9885c41aaf11cd79a05b7828cf07b1cfa6af2d4ad507c978e870f3a422856
  view_sha=cbfb7ffdfee3151a214beb57474a796adc9f5545702a73df4f8f1cb5fbbe7c19
```

Cost of the guard: the 500 prefix batches are read once (images decoded, no forward, no backward,
no update). That is I/O, not compute, and it is the price of proving the stream position.

## Runner

`tools/s0_continue_runner.py` (S0 arm) and `tools/trimask_hs_runner.py --resume/--save-steps`
(HS arm) execute the same four phases with real exit codes and an atomically rewritten
`run_status.json` after every phase:

```
train (resume at batch 500, replay-verified) -> verify the exact @1000 checkpoint
      -> export the bare student -> COCO canonical -> Urban-1k
```

`--save_completed_steps 750,1000` writes an intermediate checkpoint at 750, so a later failure
costs at most 250 steps. The runners refuse to start if the target checkpoint already exists, if a
live lock is held, or if the four GPUs are not idle, so no two runs can overlap.

### Incident: the gradient probe OOM (attempt 2)

The first attempt reached step 740 and then died at step 750 with

```
torch.OutOfMemoryError: Tried to allocate 592.00 MiB. GPU 0 ... of which 169.81 MiB is free.
Process 235259 has 79.13 GiB memory in use.
```

on all four ranks, inside `grad_probe -> norms -> torch.autograd.grad(..., retain_graph=True)`.
The probe holds several retained backward graphs on top of the ~37 GiB training steady state.

The probe was **an extra diagnostic introduced by this runner**, not part of the frozen protocol:
the frozen S0@500 run's `config.json` has no `grad_probe` entry and the trainer's
`--grad_probe_steps` default is empty. It was therefore removed from the continuation invocation
(default `''`, opt-in via `--grad-probe-steps`), which also makes the trainer command identical to
the frozen one apart from `--resume` and the save steps. No reported metric depends on the probe,
and the failed attempt wrote no checkpoint, so the continuation restarts from the @500 parent.

## Evaluation

Unchanged, standard evaluators only:

* `tools/phase30a_fixed_cohort_eval.py --canonical --canonical_only --coco`
  (legacy_cls column, 5 captions per image) → `<arm>_step001000_canonical.json`;
* `/root/SAID-gap-completion/tools/eval_urban1k_cls.py` (native CLS, plain inner product over the
  full 1000-item pool) → `<arm>_step001000_urban1k.json`.

Both scripts write a new file name (`*_step001000_*`); the @500 results are never overwritten.
The two evaluators were checked to agree with each other on the same checkpoint: for S0@500 the
canonical table's `S0_step500` row and the Urban script's `coco_val2017` section both give
0.6058 / 0.41236.

The frozen promotion gate (COCO I2T R@1 ≥ 0.6058 **and** T2I R@1 ≥ 0.41236, at least one strictly
higher) is defined at 500 updates only. The report quotes it for the @500 rows as a reference and
explicitly does not apply it to the 1000-step budget.

## Paired per-query evidence (why point estimates are not enough)

A 0.5 pp difference on COCO val2017 (5,000 I2T / 25,000 T2I queries) is within one binomial
standard error (~0.7 pp at p ≈ 0.6), while Urban-1k (n = 1,000) has an SE of ~1.5 pp. Since the two
arms are scored on the *same* images and captions, the honest test is paired, not binomial:

* `tools/diag/coco_query_hits.py` re-runs one checkpoint over COCO val2017 and stores the
  per-query R@1/R@5/R@10 hit vectors for both directions. It imports `retrieval_metrics`,
  `_top_indices` and the 512-row chunk from `eval/retrieval/coco_retrieval.py` itself (recording
  the library's sha256) rather than reimplementing the protocol, and it **asserts** that the
  aggregates recomputed from its own feature pass equal the numbers the standard evaluator already
  wrote for the same checkpoint (`--expect`). Image and caption order are fingerprinted so a pair
  can only be formed from identical query sets.
* `tools/diag/coco_paired_report.py` consumes those files and reports, from the repository's own
  `eval/paired_statistics.py`, the paired bootstrap of the mean difference (percentile CI, SE) and
  the exact McNemar test. It refuses to pair files whose order digests differ.

These files add evidence; they never replace the standard numbers.

Two defects found while running them, both fixed in the tooling rather than worked around:

* `eval.paired_statistics.mcnemar_exact` divides by `float(2 ** discordant)` and raises
  `OverflowError` past ~1023 discordant queries — unreachable for the small cohorts it was written
  for, unavoidable with COCO's 25,000 T2I queries. `tools/diag/coco_paired_report.py` falls back to
  the identical two-sided binomial tail computed in log space, and `verify_mcnemar_fallback`
  records that the fallback reproduces the library bit for bit on four cases the library can still
  handle (it is stored in the paired JSON as `mcnemar_fallback_verification`).
* The HS runner's `_conclusion` applied the frozen 500-step gate unconditionally, so the @1000 run
  was labelled `PROMISING_AT_500`. It is now budget-aware: 500 steps keeps the old strings exactly,
  any other budget reports `GATE_NOT_APPLICABLE_AT_<steps>` plus `gate_would_say` as a reference.
  The completed run's `run_status.json` was left untouched and a `conclusion_budget_aware.json`
  sidecar records recorded-vs-recomputed.

## Reproducing

```bash
# S0 arm (branch codex/s0-continue-1000)
python tools/s0_continue_runner.py --run-id s0_continue_1000 \
    --run-dir runs_salu/said_cls_cvssl/s0_continue_1000 --steps 1000

# HS arm (branch codex/s0-trimask-hs-v02)
python tools/trimask_hs_runner.py --run-id s0_trimask_hs_v02 \
    --run-dir runs_salu/said_s0_trimask_hs_v02/cont1000 --resume \
    runs_salu/said_s0_trimask_hs_v02/step500/trimask_S0_TriMask_HS_step000500.pt \
    --steps 1000 --lambda-sparse-t 0.2

# comparison + paired evidence (read-only)
python tools/diag/report_s0_continuation.py --out docs/s0_continuation/results.json \
    --markdown docs/s0_continuation/report.md
```
