# Final E2 candidates

Prepared on an isolated worktree derived from this candidate's exact mother.
The canonical project worktree and its unrelated edits are preserved.

The durable sequential entrypoint is `python -m recovery.e2_candidates --queue`
in the HNS-Half worktree. It runs A to2434, evaluates/reports/publishes A,
then starts B from common0. B pauses at500 and1217 for export/evaluation,
restores all model/optimizer/RNG/loader state and continues to2434.

Progress: canonical `runtime/SAID-nest-clip-v1/e2-final-candidates-v1/QUEUE_STATE.json`;
each candidate has `STATE.json`, `PROGRESS.json`, per-stage results and a
separate persistent canonical runtime directory. Training never overlaps evaluation.

GPU0 COCO; GPU1 DOCCI; GPU2 Long-DCI; GPU3 Flickr then Urban. Batch64 and
all evaluator source files match the validated parallel evaluation.

No automatic retries, score-dependent changes,3651/full training or third experiment.
Actual first5/all-step stream gates pin samples/text/tokens/indices/LR to the
original trajectory. Checkpoints, bare weights and raw logs stay out of Git.

`/root` is a disposable Docker overlay image cache. NFS originals remain intact.
