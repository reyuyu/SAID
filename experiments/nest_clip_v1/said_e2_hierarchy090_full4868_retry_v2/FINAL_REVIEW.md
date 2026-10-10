# Hierarchy090 controlled retry: final scientific review

Recommendation: **retain E2-Uniform as the main model; do not replace it with Hierarchy090 on this evidence.** Engineering acceptance passed, but the predeclared4868 endpoint does not improve Score5, either long aggregate, Short4 or Urban Mean. This is a negative exploratory comparison, not evidence of statistically significant inferiority.

The original failed run is preserved. The only repaired comparison normalizes complete sampling dictionaries to their JSON representation, rejects colliding keys, and still requires exact equality of every field/value. Production model, loss, sampler, trainer and evaluator sources are unchanged. This one authorized attempt restored the original complete H0.9@500 state and re-executed501; it did not start at502 or common0.

All values below are percentages; differences are percentage points. Display rounding does not alter the raw JSON results.

| Metric | E2@4868 | Hierarchy090@4868 | Delta |
|---|---:|---:|---:|
| Score5 | 73.812187 | 73.736826 | -0.075361 |
| J_long3 | 78.193644 | 78.130043 | -0.063601 |
| J_long | 87.055002 | 86.930003 | -0.124999 |
| Short4 | 67.240000 | 67.147000 | -0.093000 |
| Urban I2T R1 | 93.900 | 93.900 | 0.000 |
| Urban T2I R1 | 92.900 | 92.600 | -0.300 |
| Urban Mean R1 | 93.400 | 93.250 | -0.150 |

| Dataset | H0.9 I2T R1/R5/R10 | Delta I2T R1/R5/R10 | H0.9 T2I R1/R5/R10 | Delta T2I R1/R5/R10 |
|---|---|---|---|---|
| COCO | 61.620 / 83.780 / 89.640 | -0.240 / +0.120 / -0.060 | 43.108 / 68.720 / 78.148 | +0.108 / -0.020 / -0.012 |
| Flickr30k-test1k | 90.000 / 98.300 / 99.100 | -0.100 / -0.200 / -0.200 | 73.860 / 91.940 / 95.840 | -0.140 / -0.140 / +0.020 |
| DOCCI | 80.180 / 96.480 / 98.600 | -0.020 / +0.060 / +0.040 | 81.040 / 96.100 / 98.380 | -0.180 / -0.080 / -0.040 |
| Long-DCI | 60.300 / 78.282 / 83.991 | +0.026 / +0.092 / +0.066 | 60.760 / 78.611 / 83.636 | +0.092 / -0.092 / -0.066 |
| Urban-1k | 93.900 / 98.900 / 99.600 | 0.000 / -0.200 / 0.000 | 92.600 / 99.100 / 99.400 | -0.300 / 0.000 / 0.000 |

Long-DCI R1 improves by2 image queries and7 text queries out of7602. Its T2I R5/R10 decrease, while DOCCI R1 decreases in both directions. These isolated gains do not improve J_long3 or J_long. COCO T2I R1 increases, but COCO I2T and both Flickr R1 directions decrease, so Short4 falls. Urban T2I loses a net3 correct queries out of1000; I2T is unchanged. No paired-query or significance analysis was run in this task, so net changes do not establish which individual queries improve or regress.

Last50-step mask statistics use the original valid-population telemetry; they are descriptive training measurements.

| Statistic | E2 | H0.9 | Delta(pp) |
|---|---:|---:|---:|
| F keep | 55.793215 | 55.998282 | +0.205067 |
| Dall keep | 58.364780 | 58.509826 | +0.145046 |
| D3 keep | 57.781077 | 57.987776 | +0.206699 |
| Dall to F hard violation | 6.941052 | 7.866857 | +0.925805 |
| D3 to Dall hard violation | 8.799074 | 9.474172 | +0.675098 |
| Dall/F IoU | 81.848948 | 79.115760 | -2.733188 |
| D3/Dall IoU | 73.231994 | 71.641618 | -1.590376 |

Both runs have density order Dall > D3 > F. This is a structural observation, not a numerical failure or complete mask collapse. The smaller hierarchy coefficient is accompanied by higher hard violation rates and lower IoU, but these measurements do not prove a causal explanation of the retrieval differences.

The final read-only gradient probe uses the same sample-ID digest as the E2 probe. Norms are from differentiated actual weighted losses, not estimates obtained by multiplying raw norms. Full per-view/per-module results and coefficient controls remain in `step4868/GRADIENT_AUDIT.json`.

| Module | Alignment norm | Sparsity norm | Hierarchy norm | Cos(H,S) | Cos(H,A) | Cos(A,S) |
|---|---:|---:|---:|---:|---:|---:|
| Text mask/shared pool | 0.696342 | 0.216359 | 0.010584 | +0.381961 | -0.026226 | -0.116856 |
| Visual mask | 0.413983 | 0.123022 | 0.000944 | +0.216754 | +0.017472 | -0.181104 |
| Fusion adapter/gate | 0.798518 | 0.211267 | 0.003224 | +0.177254 | +0.037282 | -0.164196 |

Hierarchy/sparsity cosine is positive in these three groups at this checkpoint; the E2 visual/fusion values were -0.354293/-0.072796. Alignment/sparsity conflict remains. Native text/visual total gradient norms are42.731773/67.330933; original condition detaches remain, so direct sparsity/hierarchy gradients to these native groups are zero. Maximum component-additivity relative error is0.0000359533, below the unchanged0.0003 tolerance. This single final-batch diagnostic does not establish a persistent trajectory-wide mechanism or generalization improvement.

Engineering acceptance:

- 61 required CPU regressions passed, including exact real501 metadata on all four ranks, genuine value/field mismatch rejection, key-collision rejection and source identity. The standard preparation also passed its51-test subset.
- Actual501–505 restore/update acceptance passed before continuing; all4368 updates in this run and4471616 sample positions match complete real E2 metadata/LR logs. No duplicate optimizer updates; every epoch tail remains180/rank.
- Full1217/2434/3651/4868 checkpoints retain model/adapter, AdamW moments/counters, RNG, DataLoader generator, sampler and cursor. Final counter4868, cursor epoch4/batch0, scheduler horizon4868. No optimizer reset or new warmup.
- All four ranks finish with exact parameter equality, finite gradients and successful final NCCL reduction. Peak allocated memory27.7584GiB/rank, cycle median2.1273s and p952.5799s; no OOM kill or resource failure.
- Strict bare export passes; native image/text embedding maximum absolute differences are both0. All five evaluations exit successfully and preserve checkpoint/bare hashes. No inference masks, reranking, ensemble or TTA.
- Protected baseline/resume artifacts and all34 inventoried failed-run evidence files remain unchanged. An additional check of the actual original failed worktree confirms its original commit and all13 tracked controller/report files byte-for-byte unchanged.

Final checkpoint SHA256: `cc3bbc112a492c41e6204572018f2fddc3923898febc24f7e91f4e4816e91a6d`.

Bare student SHA256: `0c77a0484954c07f15de33d4c94d2778b052dd9ee230471f60537b1d2dea49ed`.

This is one seed on repeatedly observed public benchmarks. Independent validation is NOT_ESTABLISHED. Small query-count changes do not support a stable-benefit or unbiased SOTA claim. The predeclared endpoint was retained; no intermediate public evaluation, checkpoint selection, second arm, new seed or fifth epoch occurred. Large weights, datasets and raw logs remain server-local. Training/evaluation/gradient processes have exited and all four GPUs are idle.
