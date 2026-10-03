# Official Urban reproduction

CE published reference: I2T 88.6%, T2I 89.0%. The untouched official evaluator reproduces this pair in CLS: **REPRODUCED**, delta 0 pp.

| Checkpoint | CLS I2T % | CLS T2I % | TCI I2T % | TCI T2I % |
| --- | --- | --- | --- | --- |
| CE (measured) | 88.60 | 89.00 | 85.40 | 95.50 |
| BCE (checkpoint absent) | N/A | N/A | N/A | N/A |

The measured TCI path uses the official evaluator settings (`use_model_text_settings=False`, `use_text_eos=True`, `return_first=True`, `return_avg_tci=False`). TCI is query-dependent and excluded from the fair main table and all aggregates. The supplied README pair matches CLS; it is not a separate TCI reference. BCE published I2T/T2I 92.3/91.8 remains untested.
