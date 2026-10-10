# Historical protocol regression

| Dataset | Direction | K | Historical count | Frozen legacy count (new inference) | New deterministic count | Delta queries |
| --- | --- | --- | --- | --- | --- | --- |
| COCO | I2T | 1 | 3093 | 3093 | 3093 | 0 |
| COCO | I2T | 5 | 4183 | 4183 | 4183 | 0 |
| COCO | I2T | 10 | 4485 | 4485 | 4485 | 0 |
| COCO | T2I | 1 | 10750 | 10750 | 10750 | 0 |
| COCO | T2I | 5 | 17185 | 17185 | 17185 | 0 |
| COCO | T2I | 10 | 19540 | 19540 | 19540 | 0 |
| Long-DCI | I2T | 1 | 4582 | 4582 | 4582 | 0 |
| Long-DCI | I2T | 5 | 5944 | 5944 | 5944 | 0 |
| Long-DCI | I2T | 10 | 6380 | 6380 | 6380 | 0 |
| Long-DCI | T2I | 1 | 4612 | 4612 | 4612 | 0 |
| Long-DCI | T2I | 5 | 5983 | 5983 | 5983 | 0 |
| Long-DCI | T2I | 10 | 6363 | 6363 | 6363 | 0 |
| DOCCI | I2T | 1 | 4010 | 4010 | 4010 | 0 |
| DOCCI | I2T | 5 | 4821 | 4821 | 4821 | 0 |
| DOCCI | I2T | 10 | 4928 | 4928 | 4928 | 0 |
| DOCCI | T2I | 1 | 4061 | 4061 | 4061 | 0 |
| DOCCI | T2I | 5 | 4809 | 4809 | 4809 | 0 |
| DOCCI | T2I | 10 | 4921 | 4921 | 4921 | 0 |
| Flickr30k-Test1K | I2T | 1 | 901 | 901 | 901 | 0 |
| Flickr30k-Test1K | I2T | 5 | 985 | 985 | 985 | 0 |
| Flickr30k-Test1K | I2T | 10 | 993 | 993 | 993 | 0 |
| Flickr30k-Test1K | T2I | 1 | 3700 | 3700 | 3700 | 0 |
| Flickr30k-Test1K | T2I | 5 | 4604 | 4604 | 4604 | 0 |
| Flickr30k-Test1K | T2I | 10 | 4791 | 4791 | 4791 | 0 |
| Urban-1k | I2T | 1 | 939 | 939 | 939 | 0 |
| Urban-1k | I2T | 5 | 991 | 991 | 991 | 0 |
| Urban-1k | I2T | 10 | 996 | 996 | 996 | 0 |
| Urban-1k | T2I | 1 | 929 | 929 | 929 | 0 |
| Urban-1k | T2I | 5 | 991 | 991 | 991 | 0 |
| Urban-1k | T2I | 10 | 994 | 994 | 994 | 0 |

All frozen legacy counts come from newly inferred embeddings. Historical scores are used ONLY as expected regression targets. COCO retains original raw-CPU normalization and chunk512/1D argsort for independent compatibility verification. Extended legacy datasets use their frozen CPU full-matrix/topk reference (size-guarded; never used for Full Flickr), while the new main evaluator uses bounded GPU blocks. Urban retains its native GPU scoring.

New exact ties use manifest-index ordering, with no score quantization. Any count delta above is disclosed as an evaluator numerical difference, not concealed as training improvement. Earlier incompatible TF32-off attempts are retained separately; the final FP32 native backend is restored to its original defaults after an exact single-variable diagnosis. Independent old-default DOCCI inference reproduced ALL historical counts. Failed batch3 variation is retained and batch optimization is not approved; formal batch64 is frozen and validated across four GPUs. Evidence is in EVALUATION_RECOVERY_AUDIT.json and NATIVE_PRECISION_RESTORE_AUDIT.json. Any remaining count deviations are NOT marked exact regression PASS. Query-level top11/score/hit artifacts remain local.

COCO primary results retain the original canonical CPU512/1D-argsort numerical protocol on new feature banks. Generic GPU streaming gives I2T R1=61.84 instead of61.86; query3934/image456303 has four exactly tied highest candidates, where ascending-index versus original per-row argsort selects different items. That replacement is not promoted to primary. CANONICAL_COCO_REPROOF.json supplies full query evidence and all exact canonical counts.

Flickr-test1K (1000/5000) is not Full (31783/158915). Long-DCI reconstruction changed only raw whitespace; token/image/order equivalence is proven, CSV equivalence is not. DCI short-plus-extra is independent and not used as a replacement for verified Long-DCI. Old Score5/J_long3/J_long/Short4 definitions remain historical; no renamed six-set aggregate is used.
