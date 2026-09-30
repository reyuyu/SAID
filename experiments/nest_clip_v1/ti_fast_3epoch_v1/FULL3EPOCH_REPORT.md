# TI-fast full 3-epoch result

Date: 2026-09-30 UTC

TI-fast completed the original 3651-update scheduler horizon by continuing the audited step-500 run. The model was exported as a bare student and evaluated with native normalized image/text embeddings only. DCI Full and Long-DCI were not run.

## Retrieval results

All values are percentages. The delta is step3651 minus step500 in percentage points.

| Dataset | Direction | Metric | Step500 | Step3651 | Delta |
|---|---|---:|---:|---:|---:|
| COCO | I2T | R@1 | 60.04 | 61.66 | +1.62 pp |
| COCO | I2T | R@5 | 82.12 | 83.44 | +1.32 pp |
| COCO | I2T | R@10 | 88.76 | 90.14 | +1.38 pp |
| COCO | T2I | R@1 | 41.15 | 41.94 | +0.80 pp |
| COCO | T2I | R@5 | 66.68 | 67.61 | +0.92 pp |
| COCO | T2I | R@10 | 76.57 | 77.36 | +0.79 pp |
| Urban-1k | I2T | R@1 | 89.20 | 91.50 | +2.30 pp |
| Urban-1k | I2T | R@5 | 98.10 | 98.50 | +0.40 pp |
| Urban-1k | I2T | R@10 | 99.20 | 99.30 | +0.10 pp |
| Urban-1k | T2I | R@1 | 87.10 | 89.80 | +2.70 pp |
| Urban-1k | T2I | R@5 | 97.90 | 98.20 | +0.30 pp |
| Urban-1k | T2I | R@10 | 99.00 | 98.90 | -0.10 pp |
| Flickr30k-test1k | I2T | R@1 | 86.80 | 88.00 | +1.20 pp |
| Flickr30k-test1k | I2T | R@5 | 97.20 | 97.90 | +0.70 pp |
| Flickr30k-test1k | I2T | R@10 | 98.90 | 99.50 | +0.60 pp |
| Flickr30k-test1k | T2I | R@1 | 70.70 | 71.96 | +1.26 pp |
| Flickr30k-test1k | T2I | R@5 | 90.70 | 91.22 | +0.52 pp |
| Flickr30k-test1k | T2I | R@10 | 94.82 | 95.16 | +0.34 pp |
| DOCCI | I2T | R@1 | 76.22 | 78.48 | +2.26 pp |
| DOCCI | I2T | R@5 | 94.98 | 95.64 | +0.66 pp |
| DOCCI | I2T | R@10 | 97.40 | 98.16 | +0.76 pp |
| DOCCI | T2I | R@1 | 76.60 | 79.28 | +2.68 pp |
| DOCCI | T2I | R@5 | 94.92 | 95.56 | +0.64 pp |
| DOCCI | T2I | R@10 | 97.48 | 98.08 | +0.60 pp |

`J_long` increased from **82.28%** to **84.77%** (**+2.49 pp**). It is the frozen mean of Urban and DOCCI I2T/T2I R@1.

## Training validation

- Completed updates: 3651/3651; continuation log covers steps 501-3651 without gaps.
- All four ranks completed 3151 continuation updates and ended with maximum parameter difference 0.
- Losses and gradients remained finite. Final NCCL all-reduce passed on all ranks.
- Regular steps 502-3650: mean 2.028s, median 2.008s, P95 2.127s, max 3.875s.
- One regular step exceeded 3 seconds: step1218 at 3.875s. Step501 was the resumed warmup step at 5.001s.
- Peak memory per GPU: 28.84 GiB allocated and 29.92 GiB reserved.
- Continuation loop wall time: 1.92 hours.

## Mechanism summary

- Last-50 common loss fell from 8.8933 at step500 to 6.3650 at step3651.
- F/O/E keep ratios changed from 0.888/0.856/0.853 to 0.849/0.744/0.775.
- Hard inclusion violation increased from 0.0345 to 0.0460.
- O/E IoU decreased from 0.8167 to 0.6758; O/E selections became less similar.

## Integrity

- Training checkpoint SHA256: `c106802b5b7a5fcbc368b7e1ca02a09d83a7ce7caa147976160185f35db73036`
- Bare student SHA256: `e9290a6b0eafdf0bca78ee351791115493bca5d19461ad6760a8a304222e10df`
- Strict load passed; optimizer step is 3651; native image and text embedding maximum absolute differences are both 0.
- Run commit: `b2ec5ae93c9a24dd84875f121eaa8c4032ec2339`
- Checkpoints were saved at 500-step intervals and at the final step3651.
