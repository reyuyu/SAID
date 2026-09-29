# NEST JointMask fast v1: 500-step formal results

Date: 2026-09-29 UTC

Training commit: `f2a85284a1061d55fd24388bf82c7b2e59a89a3a` on `codex/nest-jointmask-fast-v1`. Performance base: `f614bda2eab5ad2b71a08ff9f62abdc68edb593a`.

All three groups started independently from the same step-0 checkpoint (`54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`) and ran 500 synchronized updates with 4 A100 80GB GPUs, batch 256 per rank, global batch 1024, seed 0, and scheduler horizon 3651. The common fast setting was 128x128 pair tiles, pair checkpoint off, and encoder checkpoint on.

## Result

Correct image conditioning improved native retrieval in this single-seed 500-step run. TI-fast beat T-fast on 26 of 30 reported recall metrics (2 ties, 2 losses), with mean change +0.543 pp and mean R@1 change +0.793 pp. TI-fast also beat the architecture-matched shuffled-image control on 26 of 30 metrics (1 tie, 3 losses), with mean change +0.453 pp and mean R@1 change +0.630 pp.

The TI-fast advantage over TI-Shuffle-fast was positive for all 24 non-Urban metrics. Urban-1k was mixed: TI-fast lost I2T R@1/R@10 and T2I R@1, tied T2I R@5, and won the remaining two recalls. This is one seed and one stopping point, so these differences are descriptive rather than statistically significant.

## Native retrieval

Values are percentages. Delta columns are percentage points.

| Dataset | Direction | Metric | T-fast | TI-fast | TI-Shuffle-fast | TI-T | TI-Shuffle |
|---|---|---:|---:|---:|---:|---:|---:|
| COCO | I2T | R@1 | 59.00 | 60.04 | 59.04 | +1.04 | +1.00 |
| COCO | I2T | R@5 | 81.48 | 82.12 | 81.36 | +0.64 | +0.76 |
| COCO | I2T | R@10 | 88.42 | 88.76 | 88.52 | +0.34 | +0.24 |
| COCO | T2I | R@1 | 40.44 | 41.15 | 40.56 | +0.70 | +0.59 |
| COCO | T2I | R@5 | 66.23 | 66.68 | 66.24 | +0.46 | +0.45 |
| COCO | T2I | R@10 | 75.81 | 76.57 | 75.90 | +0.76 | +0.67 |
| Urban-1k | I2T | R@1 | 89.30 | 89.20 | 89.60 | -0.10 | -0.40 |
| Urban-1k | I2T | R@5 | 97.80 | 98.10 | 97.80 | +0.30 | +0.30 |
| Urban-1k | I2T | R@10 | 99.20 | 99.20 | 99.30 | +0.00 | -0.10 |
| Urban-1k | T2I | R@1 | 86.50 | 87.10 | 87.30 | +0.60 | -0.20 |
| Urban-1k | T2I | R@5 | 97.70 | 97.90 | 97.90 | +0.20 | +0.00 |
| Urban-1k | T2I | R@10 | 98.80 | 99.00 | 98.70 | +0.20 | +0.30 |
| Flickr30k-test1k | I2T | R@1 | 84.90 | 86.80 | 85.20 | +1.90 | +1.60 |
| Flickr30k-test1k | I2T | R@5 | 97.20 | 97.20 | 97.10 | +0.00 | +0.10 |
| Flickr30k-test1k | I2T | R@10 | 98.70 | 98.90 | 98.80 | +0.20 | +0.10 |
| Flickr30k-test1k | T2I | R@1 | 70.10 | 70.70 | 69.88 | +0.60 | +0.82 |
| Flickr30k-test1k | T2I | R@5 | 90.18 | 90.70 | 90.40 | +0.52 | +0.30 |
| Flickr30k-test1k | T2I | R@10 | 94.38 | 94.82 | 94.34 | +0.44 | +0.48 |
| DOCCI | I2T | R@1 | 75.92 | 76.22 | 76.00 | +0.30 | +0.22 |
| DOCCI | I2T | R@5 | 94.50 | 94.98 | 94.62 | +0.48 | +0.36 |
| DOCCI | I2T | R@10 | 97.44 | 97.40 | 97.38 | -0.04 | +0.02 |
| DOCCI | T2I | R@1 | 75.56 | 76.60 | 75.52 | +1.04 | +1.08 |
| DOCCI | T2I | R@5 | 94.14 | 94.92 | 94.26 | +0.78 | +0.66 |
| DOCCI | T2I | R@10 | 97.16 | 97.48 | 97.24 | +0.32 | +0.24 |
| long-DCI | I2T | R@1 | 54.29 | 55.59 | 54.53 | +1.30 | +1.07 |
| long-DCI | I2T | R@5 | 73.82 | 74.76 | 73.99 | +0.93 | +0.76 |
| long-DCI | I2T | R@10 | 80.62 | 81.35 | 80.66 | +0.72 | +0.68 |
| long-DCI | T2I | R@1 | 55.91 | 56.45 | 55.92 | +0.54 | +0.53 |
| long-DCI | T2I | R@5 | 75.44 | 76.03 | 75.48 | +0.59 | +0.55 |
| long-DCI | T2I | R@10 | 81.29 | 81.82 | 81.40 | +0.53 | +0.42 |

## Training mechanism

The retrieval gain does not coincide with lower inclusion violation. Over the final 50 steps, TI-fast had higher hard inclusion violation than both controls. Its O/E IoU was lower, and its keep ratios were lower, while no view produced all-open or all-closed masks. The correct-image adapter correction was about twice the shuffled control by the end of training. This indicates that the paired visual condition was used and changed mask selection, but the native retrieval improvement cannot be explained as improved inclusion compliance.

| Group | common loss | hard violation | O/E IoU | F/O/E keep | F/O/E delta |
|---|---:|---:|---:|---:|---:|
| T-fast | 9.3020 | 0.0284 | 0.8389 | 0.9076/0.8751/0.8745 | 0.0000/0.0000/0.0000 |
| TI-fast | 8.8933 | 0.0345 | 0.8167 | 0.8882/0.8563/0.8530 | 0.0213/0.0218/0.0225 |
| TI-Shuffle-fast | 9.2910 | 0.0296 | 0.8327 | 0.9035/0.8676/0.8715 | 0.0115/0.0111/0.0114 |

At step 500, TI-fast's F/O/E mean absolute visual corrections were 0.0221/0.0226/0.0232, compared with 0.0119/0.0115/0.0117 for the shuffled control. T-fast has no joint adapter and therefore zero correction.

## Execution and integrity

| Group | Training min | Stable step median/P90 s | Peak allocated/reserved GiB | Parameter max diff |
|---|---:|---:|---:|---:|
| T-fast | 17.66 | 1.974/2.070 | 28.46/29.00 | 0.0 |
| TI-fast | 17.86 | 2.001/2.103 | 28.84/29.25 | 0.0 |
| TI-Shuffle-fast | 17.91 | 2.006/2.104 | 28.84/29.25 | 0.0 |

All groups have exactly 500 continuous step records, finite losses and gradients, zero fallback steps, zero duplicate-ID steps, successful final NCCL checks, and identical sample/text/token stream digests at every step and rank. Global valid O/E candidates ranged from 1022 to 1024; F always used 1024 candidates. All exports strictly loaded at optimizer step 500 and matched full-checkpoint native image and text embeddings with maximum absolute error 0.

T-fast has no joint adapter, so TI-fast versus T-fast also includes the adapter parameterization. TI-fast versus TI-Shuffle-fast is the stronger controlled test of whether the correct paired image matters: their adapter initial tensors are identical, and only the visual condition presented to the gate is shuffled.

## Evaluation protocol and artifacts

COCO uses 5000 images and 25000 captions with similarity chunk 512. Urban-1k uses 1000 image-caption pairs. Flickr30k test1k uses 1000 images and 5000 captions. DOCCI uses 5000 pairs, and long-DCI uses 7602 pairs. All use batch 64 and normalized native student image/text embeddings with plain inner product; no mask, fusion, or reranking is used. DCI Full was excluded as requested.

Machine-readable results: `FORMAL500_RESULTS.json`. Preserved small raw evaluator JSONs are under `evidence/native_results/`; stage commands, commits, wall times, console output, and exit codes are under `evidence/`. Training checkpoints, bare students, data, and large caches remain on the server and are not committed.
