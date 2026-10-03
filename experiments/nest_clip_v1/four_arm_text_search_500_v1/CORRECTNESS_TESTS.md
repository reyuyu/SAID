# Correctness tests

Initial targeted and historical regression suite: **93 passed**. See evidence/correctness-tests.txt. Coverage: A bounds/complement/fallback; B/C exact1000 real-sample replay against3da12a3; C synthetic live CE formula and gradients, weight1 exact old loss/gradients/AdamW; D ordered contiguous visible-only spans; private RNG/image augmentation replay; existing modes and candidate/loss regressions.

Additional selection suite plus directional/sampling rerun:22 passed (5 new selection cases;98 unique tests total across the two suites). Additional CPU audit replayed1000 actual training images: all four modes have bitwise equal image tensors and Full tokens. See evidence/image-and-sampling-1000.json.
