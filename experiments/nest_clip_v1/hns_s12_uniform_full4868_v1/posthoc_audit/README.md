# Reproduce the read-only posthoc audits

From the repository worktree, with the existing local runtime artifacts and local data mounted:

```bash
.venv/bin/python -m pytest -q tests/test_e2_uniform_posthoc_audit.py
.venv/bin/python -m recovery.e2_uniform_posthoc_audit --output-dir /path/to/fresh/audit
```

The output directory must not exist. The script starts no training and makes no optimizer updates. It checks the pinned E2@4868 checkpoint, E2/S12 bare hashes and unchanged evaluator sources before using idle GPU3 for native Urban inference. Every original artifact it reads is rehashed after both audits. Evaluator math, precision and batch64 are unchanged; inference calls the existing evaluate_urban1k and _recall with a read-only observer.

The full stream proof compares all saved E2/S12 rows and independently reconstructs sampler IDs, K/indices and native4868 LR. It separately checks complete checkpoint states, original four-rank restore evidence, rank timing and acceptance records. There is no cross-model gradient/output/augmentation equality requirement.

The paired summary links two compact JSONs containing all 2000 directional queries, preserving correctness categories, Top1/GT IDs, GT ranks with tie bounds and similarity margins. Neither training text nor model weights are uploaded. McNemar is exact conditional binomial; the private-seed paired-query bootstrap covers test-query sampling uncertainty only.

The original reports/results/weights are unchanged. The expanded initial audit JSON stays server-local; publication separates query directions to meet the repository's 1MiB per-file cap without dropping queries.
