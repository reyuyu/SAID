# Official β-CLIP CE evaluation: complete

The manually transferred official CE checkpoint reproduces published Urban CLS 88.60/89.00%
(I2T/T2I), passes native consistency, and completes all five frozen SAID protocols.
BCE checkpoint is unavailable and did not block CE. No training or finetuning was performed.

Read [FINAL_COMPARISON.md](FINAL_COMPARISON.md) for every directional R1 difference,
aggregate scores, CLS versus TCI distinction and complete R@1/5/10.
[CE_SAID_PROTOCOL_RESULTS.json](CE_SAID_PROTOCOL_RESULTS.json) stores raw fractions/protocol hashes;
[CHECKPOINT_INVENTORY.json](CHECKPOINT_INVENTORY.json) stores actual args, SHA256 and strict loading.
Commands are structured argv arrays in `commands/official-urban-ce.json` and `commands/native-five-ce.json`.
Run from the SAID worktree using the independent beta-clip-official Python3.10 environment.
The pinned official checkout remains outside this repository and unmodified.

Implementation: `native_adapter.py`, `official_urban.py`, `evaluate_native.py`,
read-only `find_checkpoints.py`; `write_reports.py` publishes small runtime results only.
Old Google Drive timeout logs remain historical evidence, superseded by successful manual CE transfer.
No checkpoint, dataset, embedding cache, cloud access secret or Python environment is committed.
