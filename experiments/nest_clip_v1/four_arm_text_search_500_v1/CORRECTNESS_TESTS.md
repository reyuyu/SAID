# Correctness tests

Initial targeted and historical regression suite: **93 passed**. See evidence/correctness-tests.txt. Coverage: A bounds/complement/fallback; B/C exact1000 real-sample replay against3da12a3; C synthetic live CE formula and gradients, weight1 exact old loss/gradients/AdamW; D ordered contiguous visible-only spans; private RNG/image augmentation replay; existing modes and candidate/loss regressions.

Additional selection suite plus directional/sampling rerun:22 passed (5 new selection cases;98 unique tests total across the two suites). Additional CPU audit replayed1000 actual training images: all four modes have bitwise equal image tensors and Full tokens. See evidence/image-and-sampling-1000.json.

Final artifact verification passed:4 fresh500/H4868 runs,4 strict exports,120 raw native recalls,4 resource gates, all command exit codes, all500×4 stream hashes and selection consistency. A/D K or contiguous-index replay checked all512000 sample positions per arm. B/C local-view hashes match previous all500×4 batches, and weighted objectives reconstruct on all500 steps. See evidence/final-artifact-verification.json and each arm evidence/final-matching-and-objective.json.
