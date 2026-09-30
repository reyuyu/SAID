# Supplemental Long-DCI: Balanced / Cosine @500

This supplement was explicitly requested after the original four-dataset experiment. It keeps the same 7602 images/captions and full candidate pool as TI-fast@500. Native normalized student embeddings, batch64 on cuda:0; no mask, fusion or rerank.

Values are percentages; changes from TI-fast@500 are percentage points. J_long remains unchanged.

| Direction | Recall | TI-fast@500 | Balanced@500 | Cosine@500 | Balanced-TI pp | Cosine-TI pp |
|---|---|---:|---:|---:|---:|---:|
| I2T | R@1 | 55.59 | 55.21 | 54.81 | -0.38 | -0.78 |
| I2T | R@5 | 74.76 | 74.61 | 74.18 | -0.14 | -0.58 |
| I2T | R@10 | 81.35 | 81.04 | 80.54 | -0.30 | -0.80 |
| T2I | R@1 | 56.45 | 57.22 | 55.25 | +0.78 | -1.20 |
| T2I | R@5 | 76.03 | 76.40 | 74.72 | +0.37 | -1.32 |
| T2I | R@10 | 81.82 | 82.06 | 80.66 | +0.24 | -1.16 |

Original Stack-Patch/CrossScore-CLS have no Long-DCI measurements and are not assigned estimated scores. TI-fast@3651 is a different training budget and is not included in this ranking. These single-seed benchmark results do not establish statistical significance.

Manifest SHA256: `8890a2be15e2b64c142f9a1e39224d34b6ce92fc17ffb6099e14942cc8161c4b`.

Raw evaluator JSON, checkpoint hashes, commands, actual evaluator commit, timings and exit codes are preserved under evidence/long_dci/. Weights and datasets remain server-local. No DCI Full evaluation or additional training was started.
