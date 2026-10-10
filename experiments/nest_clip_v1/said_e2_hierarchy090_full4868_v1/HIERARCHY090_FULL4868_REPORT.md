# E2-Hierarchy090 full4868: engineering stop

The authorized4868 endpoint was NOT reached. No new native Recall or aggregate result exists; final comparison with E2 is UNAVAILABLE.

Resume from the exact H0.9@500 checkpoint succeeded. All four ranks restored model, fusion adapter, AdamW, CPU/CUDA/Python/NumPy RNG and DataLoader generator exactly, with zero parameter difference. Horizon4868, cursorepoch0/batch500 and next-update501 LR passed. Original checkpoint SHA256 remains `1e71b433b4dd49977be701fc269f9852e023443560b23c2d88d80e111773fd90`.

The native trainer executed one new optimizer update501. During its post-update sampling telemetry, the new controller asserted equality between the in-memory Python sampling dictionary and a JSON-deserialized historical dictionary. All four ranks raised this assertion. No new recoverable training checkpoint was saved; update501 tensors were discarded on process exit. There is no checkpoint from which update502 can resume.

Root cause: sampling histograms use integer keys in Python, whereas JSON object keys are strings. CPU-only reconstruction for all four step501 batches reproduces the actual pre-update stream hashes and the E2 reference stream. Raw dictionary equality fails; JSON-normalized equality passes exactly for everyrank, including sample IDs, text/token digests, K and selected indices. This supports a representation bug in the audit controller rather than a detected data-stream drift.

Limitation: the failed live sampling dictionaries were not persisted before the assertion. The schema diagnosis therefore combines live pre-update receipts, frozen production code and deterministic text-only reconstruction; it is not a saved post-update sampling receipt. The original failure evidence and controller are preserved unchanged.

CPU regression43 passed before launch, but these tests compared deserialized log dictionaries and missed the in-memory versus JSON key-type boundary. The new read-only diagnostic regression covers this boundary. No OOM, nonfinite-loss/gradient, source drift or checkpoint corruption was observed before the telemetry failure. This does not constitute successful501–505 acceptance or a full4368-update audit.

No restart, coefficient change, second arm, export or public evaluation followed the failure. The original H0.9 and E2 checkpoint/bare artifacts remain hash-identical. All training workers and the supervisor exited; four GPUs have no compute processes. Large raw logs remain server-local.

Original E2@4868 reference: Score5=73.812186642, J_long3=78.193644403, J_long=87.055002253, Short4=67.240, Urban93.9/92.9, Mean93.4. H0.9@4868 values and deltas are UNAVAILABLE; no conclusion about long-text or Urban improvement can be drawn.

FAILURE_AUDIT.json and READONLY_SAMPLING_DIAGNOSIS.json contain the receipts. The failing controller has not been changed or re-launched.
