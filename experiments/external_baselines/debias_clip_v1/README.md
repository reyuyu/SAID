# DeBias-CLIP official B/16 reproduction: complete

Read [DEBIAS_VS_SAID.md](DEBIAS_VS_SAID.md) for the separate published/reference,
official default-AMP reproduction, and frozen SAID/native-FP32 result tables.
The released three-epoch B/16 checkpoint is used throughout; no training or L/14 evaluation runs.
Both phases preserve the pinned official loader, including the recorded positional-residual overwrite.
Only machine paths/shell syntax and documented annotation schema are adapted; official Python source remains clean.

Code: common.py native adapter/strict audit; prepare_data.py/audit_data.py raw protocol checks;
official_repro.py observes the untouched main/evaluator; unified.py uses frozen SAID metrics;
precision_audit.py/said_urban.py/urban_errors.py support requested query diagnosis;
write_reports.py writes small sanitized evidence. Structured argv arrays are in commands/.
Official checkout, released weight, source datasets and embedding caches remain outside Git.
