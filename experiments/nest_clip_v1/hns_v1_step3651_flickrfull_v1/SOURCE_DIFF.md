# Dedicated HNS entrypoint source difference

Base: `e305275c24bb6a80c2e87a674d50239d6a0f22f5`.

`tools/eval_hyfl_native.py`, `tools/retrieval_bounded.py`, the resource queue,
and all four encoder/tokenizer source files are byte-identical to the frozen
E2 implementation. Their Git diff is empty; exact SHA256 values are recorded
in `SOURCE_EQUIVALENCE.json`. The original E2 worker still binds its E2 SHA.

The new `tools/eval_hns_v1_flickrfull.py` imports that worker. Its only changes
to the worker's in-process configuration are:

```python
native.MODEL_SHA = MODEL_SHA  # exact HNS-v1@3651 SHA, not an arbitrary whitelist
native.SOURCES = native.SOURCES + (ENTRY,)  # dedicated entrypoint receipt
```

The wrapper verifies frozen source bytes before binding, restricts the exact
checkpoint path and Full Flickr job settings, and invokes `native.run(...)`.
`load_model`, `identity`, `encode`, `retrieval`, and `run` remain the identical
imported function objects. The regression test verifies this identity and
rejects an E2 checkpoint, source drift, changed batches/normalization/chunks,
and another dataset. Model loading still performs strict SHA and state checks.

The other new script only audits provenance, invokes the existing queue with
one dedicated-worker command, and reads completed receipts/cached features.
It defines no new encoder, normalization, similarity, Top-K, or Recall math.
Independent output/cache paths contain HNS-only features. No production file
or historical result has been edited.
