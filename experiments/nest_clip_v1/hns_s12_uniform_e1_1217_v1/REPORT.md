# E2-Uniform: native resume500 ->1217

Original evaluated500 checkpoint resumed without parameter/optimizer/RNG/scheduler reset;717 updates, horizon4868. Original500 SHA: `74271df5298525f924834b3c74fa228eceff623020291c775808f1994214cb9a`.

| Model | Step | Score5 | J_long3 | J_long | Short4 | Urban I2T/T2I | Urban Mean |
|---|---:|---:|---:|---:|---:|---|---:|
| E2 Uniform | 500 | 71.314371 | 75.120618 | 84.195003 | 65.605000 | 91.100 / 90.000 | 90.550 |
| HNS-S12 | 1217 | 72.906355 | 76.999258 | 85.760003 | 66.767000 | 93.200 / 91.100 | 92.150 |
| E2 Uniform | 1217 | 72.794967 | 76.922278 | 85.855003 | 66.604000 | 93.000 / 91.800 | 92.400 |

| Dataset | I2T R1/R5/R10 (%) | T2I R1/R5/R10 (%) |
|---|---|---|
| COCO | 61.940000 / 83.520000 / 89.620000 | 42.496000 / 68.208000 / 77.652000 |
| Urban-1k | 93.000007 / 98.900002 / 99.700004 | 91.800004 / 98.900002 / 99.400002 |
| Flickr30k-test1k | 89.000000 / 98.200000 / 99.200000 | 72.980000 / 91.840000 / 95.580000 |
| DOCCI | 79.020000 / 96.020000 / 98.320000 | 79.600000 / 95.740000 / 98.100000 |
| Long-DCI | 58.734544 / 77.111287 / 82.701921 | 59.379111 / 77.808471 / 83.122862 |

HNS_S12 deltas(pp): {"Score5": -0.1113884191649106, "J_long3": -0.0769806986081818, "J_long": 0.09499988079072352, "Short4": -0.1629999999999967, "Urban_I2T": -0.1999974250793457, "Urban_T2I": 0.6999969482421875, "Urban_Mean": 0.2499997615814209}.
| Dataset | Delta I2T R1/R5/R10(pp) | Delta T2I R1/R5/R10(pp) |
|---|---|---|
| COCO | +0.060000 / -0.060000 / +0.000000 | -0.052000 / -0.084000 / -0.184000 |
| Urban-1k | -0.199997 / -0.100005 / +0.099999 | +0.699997 / +0.199997 / -0.200003 |
| Flickr30k-test1k | -0.500000 / +0.000000 / +0.000000 | -0.160000 / -0.220000 / -0.040000 |
| DOCCI | -0.040000 / -0.020000 / -0.020000 | -0.080000 / -0.060000 / -0.080000 |
| Long-DCI | -0.302552 / -0.289398 / -0.499868 | -0.539332 / -0.210471 / -0.052618 |

E2_Uniform_500 deltas(pp): {"Score5": 1.480595532261134, "J_long3": 1.8016592204352406, "J_long": 1.6600000119209284, "Short4": 0.9989999999999952, "Urban_I2T": 1.8999993801116943, "Urban_T2I": 1.8000006675720215, "Urban_Mean": 1.850000023841858}.
| Dataset | Delta I2T R1/R5/R10(pp) | Delta T2I R1/R5/R10(pp) |
|---|---|---|
| COCO | +1.360000 / +0.580000 / +0.240000 | +0.936000 / +0.904000 / +0.668000 |
| Urban-1k | +1.899999 / +0.299996 / +0.000000 | +1.800001 / +0.299996 / -0.100005 |
| Flickr30k-test1k | +0.900000 / +0.600000 / +0.200000 | +0.800000 / +0.280000 / +0.340000 |
| DOCCI | +1.000000 / +0.700000 / +0.140000 | +1.940000 / +0.660000 / +0.360000 |
| Long-DCI | +1.578532 / +1.525914 / +1.026046 | +2.591423 / +1.775848 / +1.197053 |

Matched-node interpretation: {"Score5_improved": false, "Urban_I2T_improved": false, "Urban_T2I_improved": true, "Urban_Mean_improved": true, "weak_single_seed_Score5_signal": false}.
Mask coverage/order/violations/IoU and500/same-node baselines are in SCIENTIFIC_DIAGNOSTICS.json. Density inversions do not stop training and do not establish retrieval quality.
Visual-mask hierarchy/sparsity gradient cosine500=-0.8991869688034058,1217=-0.7198777794837952. Other groups/norms/cosines/additivity and CE/share are in node diagnostics.
Original HNS-S121217 sampler/text/token/K/index and LR suffix matches all733904 actual records, including180/rank tail. Different sparse allocation intentionally means model parameters are not required to match the1:2:2 baseline.
Model/fusion/AdamW, four-rank RNG, sampler/cursor and DataLoader generator restore audits plus501..505 parameter agreement/optimizer counters/finite checks passed. Production code remains byte-identical.
Five native datasets only; no mask inference/rerank/ensemble/TTA. Training and gradient processes exit before parallel evaluation. Parent and final checkpoint hashes remain unchanged through evaluation.
Only E2 Uniform; stop1217, no2434/3651/4868, E1 or other arm. Checkpoints/bare/large raw logs/data stay local. Urban0.1pp is one query; small single-seed differences are exploratory.
