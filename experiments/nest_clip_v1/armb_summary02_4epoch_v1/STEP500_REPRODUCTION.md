# S02 step500 retrieval reproduction

**REPRODUCTION_PASS**

Fresh common step0; scheduler horizon4868 throughout. Only500 updates so far, not a500-horizon trial.

| Metric | Actual pp | Historical pp | Delta pp |
|---|---:|---:|---:|
| Score5_R1 | 70.379409 | 70.364367 | +0.015042 |
| J_long3 | 73.883681 | 73.726611 | +0.157070 |
| J_long | 82.800002 | 82.765002 | +0.035000 |
| Short4_R1 | 65.123000 | 65.321000 | -0.198000 |

Gate checks: `{"Score5_R1": true, "J_long3": true, "Urban_I2T": true, "Urban_T2I": true, "no_obvious_dataset_collapse": true}`.
Collapse screen: Invalid/nonfinite R1 or loss of more than50% of historical R1 in any direction; no weight search. All directional deltas and R@1/5/10 in STEP500_RESULTS.json.
Training checkpoint SHA256 unchanged before/after separate-process evaluation: `7213a8b2e79123b46783c02264fe2898a98bdefc8a890db7e28c819b27be7a65`.
Optimizer, manual scheduler, BF16/no-scaler metadata, four-rank RNG and exact sampler cursor saved.
No cross-run loss/model/moment equality gate. No hyperparameter or weight search.
