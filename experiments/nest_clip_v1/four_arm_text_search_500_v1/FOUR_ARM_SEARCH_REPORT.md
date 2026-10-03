# Four-arm text-supervision search

**TRADEOFF**. Selected eligible champion: **A**; best observed Score5: **B**. No safe overall improvement over matched RandomK.

Four preregistered arms ran sequentially:5-step smoke followed by fresh500 from identical commonstep0, seed0,4×256 candidates,4868 scheduler horizon. Native inference is unchanged. No baseline retraining, adaptive tuning or4868-step continuation.

Long guard: J_long3≥73.403324%; Score5 near-ties strictly<0.05pp use J_long3,Urban T2I,Long T2I,Short4. Raw fractions determine selection.

| Model | Score5 | J_long3 | J_long | Short4 |
|---|---:|---:|---:|---:|
| RandomK baseline | 69.900394 | 73.603324 | 82.340003 | 64.346000 |
| Summary+RandomDetail previous | 69.556218 | 72.553029 | 81.590002 | 65.061000 |
| A | 69.710789 | 73.441982 | 81.950002 | 64.114000 |
| B | 69.911321 | 73.132202 | 82.245001 | 65.080000 |
| C | 69.777585 | 72.994642 | 81.940002 | 64.952000 |
| D | 69.312527 | 72.433544 | 81.345002 | 64.631000 |

| Arm | Dataset | I2T R1 | T2I R1 | Δ RandomK I/T pp | Δ previous I/T pp |
|---|---|---:|---:|---|---|
| A | COCO | 59.040000 | 40.936000 | -0.920000 / +0.012000 | -1.300000 / -0.768000 |
| A | Urban-1k | 89.300007 | 86.800003 | +0.300002 / -1.100004 | +0.800002 / -0.099999 |
| A | Flickr30k-test1k | 85.600000 | 70.880000 | -0.600000 / +0.580000 | -1.100000 / -0.620000 |
| A | DOCCI | 76.220000 | 75.480000 | -0.100000 / -0.660000 | +0.940000 / -0.200000 |
| A | Long-DCI | 55.814259 | 57.037622 | +0.420942 / +0.171008 | +2.328335 / +1.565378 |
| B | COCO | 60.360000 | 41.740000 | +0.400000 / +0.816000 | +0.020000 / +0.036000 |
| B | Urban-1k | 89.700001 | 87.400001 | +0.699997 / -0.500005 | +1.199996 / +0.500000 |
| B | Flickr30k-test1k | 86.800000 | 71.420000 | +0.600000 / +1.120000 | +0.100000 / -0.080000 |
| B | DOCCI | 75.660000 | 76.220000 | -0.660000 / +0.080000 | +0.380000 / +0.540000 |
| B | Long-DCI | 53.814786 | 55.998421 | -1.578532 / -0.868193 | +0.328861 / +0.526177 |
| C | COCO | 60.440000 | 41.628000 | +0.480000 / +0.704000 | +0.100000 / -0.076000 |
| C | Urban-1k | 89.100003 | 87.000006 | +0.099999 / -0.900000 | +0.599998 / +0.100005 |
| C | Flickr30k-test1k | 86.400000 | 71.340000 | +0.200000 / +1.040000 | -0.300000 / -0.160000 |
| C | DOCCI | 75.700000 | 75.960000 | -0.620000 / -0.180000 | +0.420000 / +0.280000 |
| C | Long-DCI | 54.340963 | 55.866877 | -1.052355 / -0.999737 | +0.855038 / +0.394633 |
| D | COCO | 59.660000 | 41.644000 | -0.300000 / +0.720000 | -0.680000 / -0.060000 |
| D | Urban-1k | 88.300002 | 86.500007 | -0.700003 / -1.400000 | -0.200003 / -0.399995 |
| D | Flickr30k-test1k | 86.400000 | 70.820000 | +0.200000 / +0.520000 | -0.300000 / -0.680000 |
| D | DOCCI | 74.980000 | 75.600000 | -1.340000 / -0.540000 | -0.300000 / -0.080000 |
| D | Long-DCI | 53.749013 | 55.472244 | -1.644304 / -1.394370 | +0.263089 / +0.000000 |

| Arm | Dataset | I2T R1/R5/R10 | T2I R1/R5/R10 |
|---|---|---|---|
| A | COCO | 59.040000 / 81.960000 / 88.560000 | 40.936000 / 66.500000 / 76.472000 |
| A | Urban-1k | 89.300007 / 97.800004 / 99.600005 | 86.800003 / 98.200005 / 99.000007 |
| A | Flickr30k-test1k | 85.600000 / 97.000000 / 99.000000 | 70.880000 / 90.880000 / 94.780000 |
| A | DOCCI | 76.220000 / 94.520000 / 97.420000 | 75.480000 / 94.240000 / 97.620000 |
| A | Long-DCI | 55.814259 / 75.138122 / 81.176006 | 57.037622 / 76.137858 / 82.057353 |
| B | COCO | 60.360000 / 82.720000 / 89.320000 | 41.740000 / 67.212000 / 77.032000 |
| B | Urban-1k | 89.700001 / 97.900003 / 99.500006 | 87.400001 / 98.100007 / 99.200004 |
| B | Flickr30k-test1k | 86.800000 / 97.600000 / 99.200000 | 71.420000 / 91.300000 / 95.280000 |
| B | DOCCI | 75.660000 / 94.460000 / 97.480000 | 76.220000 / 94.500000 / 97.540000 |
| B | Long-DCI | 53.814786 / 73.704288 / 79.926335 | 55.998421 / 75.703762 / 81.544330 |
| C | COCO | 60.440000 / 82.520000 / 89.540000 | 41.628000 / 67.220000 / 76.900000 |
| C | Urban-1k | 89.100003 / 97.900003 / 99.400002 | 87.000006 / 98.100007 / 99.000007 |
| C | Flickr30k-test1k | 86.400000 / 97.500000 / 99.000000 | 71.340000 / 91.260000 / 95.360000 |
| C | DOCCI | 75.700000 / 94.460000 / 97.480000 | 75.960000 / 94.540000 / 97.340000 |
| C | Long-DCI | 54.340963 / 74.072612 / 80.347277 | 55.866877 / 75.611681 / 81.504867 |
| D | COCO | 59.660000 / 82.340000 / 89.080000 | 41.644000 / 67.020000 / 76.888000 |
| D | Urban-1k | 88.300002 / 97.800004 / 99.000007 | 86.500007 / 98.000002 / 99.100006 |
| D | Flickr30k-test1k | 86.400000 / 97.500000 / 99.100000 | 70.820000 / 90.980000 / 95.200000 |
| D | DOCCI | 74.980000 / 94.260000 / 97.220000 | 75.600000 / 94.340000 / 97.460000 |
| D | Long-DCI | 53.749013 / 73.704288 / 80.281505 | 55.472244 / 75.309129 / 81.360168 |

| Arm | Mean normal full cycle seconds | Peak allocated GiB | Status |
|---|---:|---:|---|
| A | 2.1050 | 27.7599 | COMPLETE |
| B | 2.0843 | 27.7599 | COMPLETE |
| C | 2.0969 | 27.7599 | COMPLETE |
| D | 2.0987 | 27.7599 | COMPLETE |

Correctness:93 initial targeted/regression tests passed, including1000 real-sample exact3da12a3 sampling replay and default-weight bitwise loss/gradients/AdamW. Five additional selection tests pass (98 unique tests total). A separate1000-real-image replay also passes. All live500×4 ID/Full/token/reference hashes are checked against RandomK. Image equality follows from identical indexed image paths and unchanged deterministic Resize/CenterCrop/Normalize, and is independently verified on1000 actual images. Historical image tensor digests were not recorded; this is source/data/transform equivalence plus actual replay, rather than a retrospective hash comparison. B/C also match previous local strings/tokens/K/indices hashes on all500×4 rank batches; live objective reconstruction checks the declared weight formula on all500 steps.

Strict exports and frozen five-dataset metadata/checkpoint SHA checks are stored per arm. Long-DCI is7602/7602, manifestSHA8890a2be15e2b64c142f9a1e39224d34b6ce92fc17ffb6099e14942cc8161c4b.

Gate/F-D diagnostics are scheduled at1/100/200/500; last50 gate and F-D entries have one observation, while CE/keep/inclusion/S-D IoU use all50 updates. See TRAINING_DIAGNOSTICS.json per arm.

Most useful candidate for a future full-training confirmation: A. This recommendation does not authorize or launch full training.

See SHORT_LONG_ANALYSIS.md for hypothesis-specific evidence and the limits of a single-seed500-step search. Branch: codex/nest-balanced-four-arm-text-search-500-v1.

The eligible champion is only the best new arm under the guard. In this run A remains below RandomK Score5; the existing RandomK recipe remains the stronger formal reference. B is the highest observed Score5 and retains short-text gains, but is ineligible because of its long-text loss. C gives limited support for direction-specific ambiguity; D does not demonstrate a consistent continuity benefit. See the detailed short/long analysis.

Artifacts contain small JSON/Markdown/code and lossless compressed logs only; checkpoints, datasets and embeddings remain outside Git.
