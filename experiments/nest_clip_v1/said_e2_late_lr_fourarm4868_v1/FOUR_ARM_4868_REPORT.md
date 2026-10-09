# E2-Uniform late LR four-arm epoch4 experiment

All four multipliers were fixed before training. Every arm independently resumes the same E2@3651.
Results are exploratory: public benchmarks were repeatedly observed; no trustworthy independent, deduplicated validation protocol was available.

| Model | BB | Text mask | Visual mask | Adapter | Score5 | J_long3 | J_long | Short4 | Urban I2T/T2I | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| E2-Uniform | 1.0 | 1.0 | 1.0 | 1.0 | 73.812187 | 78.193644 | 87.055002 | 67.240000 | 93.900 / 92.900 | 93.400 |
| B1-BB085 | 0.85 | 1.0 | 1.0 | 1.0 | 73.791848 | 78.185747 | 87.030002 | 67.201000 | 93.900 / 92.900 | 93.400 |

## B1-BB085

Delta vs E2 (pp): `{"Score5": -0.020338226782428137, "J_long3": -0.007897044637388717, "J_long": -0.025000000000005684, "Short4": -0.03900000000000148, "Urban_I2T": 0.0, "Urban_T2I": 0.0}`
Checkpoint SHA256: `f925f3858e12f9f00f7db06f0ce66f06907f6988fb3a8d39feee699df1f94bbd`.
| Dataset | I2T R@1/5/10 (%) | T2I R@1/5/10 (%) | Δ I2T R@1/5/10 (pp) | Δ T2I R@1/5/10 (pp) |
|---|---|---|---|---|
| COCO | 61.880000 / 83.740000 / 89.760000 | 42.984000 / 68.724000 / 78.192000 | +0.020000 / +0.080000 / +0.060000 | -0.016000 / -0.016000 / +0.032000 |
| Urban-1k | 93.900007 / 99.200004 / 99.600005 | 92.900002 / 99.100006 / 99.400002 | +0.000000 / +0.099999 / +0.000000 | +0.000000 / +0.000000 / +0.000000 |
| Flickr30k-test1k | 89.900000 / 98.500000 / 99.300000 | 74.040000 / 92.100000 / 95.780000 | -0.200000 / +0.000000 / +0.000000 | +0.040000 / +0.020000 / -0.040000 |
| DOCCI | 80.140000 / 96.420000 / 98.580000 | 81.180000 / 96.160000 / 98.420000 | -0.060000 / +0.000000 / +0.020000 | -0.040000 / -0.020000 / +0.000000 |
| Long-DCI | 60.286767 / 78.150487 / 83.938437 | 60.707708 / 78.729282 / 83.714812 | +0.013154 / -0.039463 / +0.013154 | +0.039463 / +0.026309 / +0.013154 |

Urban 0.1pp is one query; single-seed differences are not established statistical gains.
Lower mask density/violation alone does not prove better evidence selection.
No fifth arm, new seed, new multiplier, checkpoint fusion or epoch5 is authorized.
