# Balanced Hyperparameter Search: Fixed Seed0

## Winner

**Balanced hparam search best**: `6d44ae8d5c34438e672287b09f3e6930af01473a4aa4905cd90c02c259b2a4a8`.

Final checkpoint SHA256: `c48286d4a049c547bc16491ded093fee7cbbbd355bc0ade3593fe886655b826e`.

Chosen parameters: `{"fusion_lr": 0.0002, "inclusion_max": 1.0, "sparsity_scale": 1.0, "view_weights": [1.0, 1.0, 1.0], "visual_mask_lr_scale": 1.0}`.

This is the highest measured score in the requested coordinate search/promotion procedure, seed0. It is not a claim of global hyperparameter optimality or statistical significance.

## Search Rules

The Balanced network, data, RandomK, backbone/text/pool LR, precision, hard-ST and candidate pool are frozen. Coordinates: fusion LR5e-5/2e-4; visual mask LR scale0.5/2; normalized view weights [2,1,1]/[1,1,1.5]; sparsity scale0.75/1.25; inclusion maximum0.5/1.5. Each round evaluates both new candidates from the shared untrained state, retaining the current best as the third option. After all500-step evaluations, global Top2 continue directly from their own step500 to3651. The user replaced the original Top3-to1217/Top2-to3651 plan; no intermediate1217 screening is run.

Default B0@500/@3651 are reused after strict checkpoint/metric verification. Same-trial promotions restore optimizer moments, all RNG and loader cursor, without resetting the3651 cosine horizon. The B0 legacy optimizer is split by parameter name with exact moments/step/LR preservation. Default loss/gradient/next AdamW update regression is exact.

Ranking uses original Recall floats: Score5_R1 (ten equally weighted R1 directions), then J_long3 (Urban/DOCCI/Long-DCI), then J_long (Urban/DOCCI). No minimum gain is imposed.

## Coordinates

| Round | Coordinate | Previous best | Candidate IDs | New best |
|---|---|---|---|---|
| 1 | fusion_lr | a009f49f814a | d16d3af78e46,6d44ae8d5c34 | 6d44ae8d5c34 |
| 2 | visual_mask_lr_scale | 6d44ae8d5c34 | 61afe6117bae,ffb3402cbb33 | 6d44ae8d5c34 |
| 3 | view_weights | 6d44ae8d5c34 | 50cd96066da8,dcdf2119cb11 | 6d44ae8d5c34 |
| 4 | sparsity_scale | 6d44ae8d5c34 | e4724e65e6a4,a72ea0fb10cb | 6d44ae8d5c34 |
| 5 | inclusion_max | 6d44ae8d5c34 | d6542aae963d,2ea4a3ae907a | 2ea4a3ae907a |

## Leaderboard 500

| Trial | fusion_lr | visual scale | F/P/R weights | sparsity | inclusion | Score5_R1 % | J_long3 % | J_long % | Reused |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| 2ea4a3ae907a | 0.0002 | 1.0 | [1.0, 1.0, 1.0] | 1.0 | 1.5 | 70.265202 | 73.775337 | 82.575002 | False |
| 6d44ae8d5c34 | 0.0002 | 1.0 | [1.0, 1.0, 1.0] | 1.0 | 1.0 | 69.990027 | 73.734711 | 82.445002 | False |
| d6542aae963d | 0.0002 | 1.0 | [1.0, 1.0, 1.0] | 1.0 | 0.5 | 69.906756 | 73.610593 | 82.295001 | False |
| a72ea0fb10cb | 0.0002 | 1.0 | [1.0, 1.0, 1.0] | 1.25 | 1.0 | 69.888746 | 73.447909 | 82.245003 | False |
| 50cd96066da8 | 0.0002 | 1.0 | [2.0, 1.0, 1.0] | 1.0 | 1.0 | 69.887529 | 73.567215 | 82.470002 | False |
| a009f49f814a | 0.0001 | 1.0 | [1.0, 1.0, 1.0] | 1.0 | 1.0 | 69.858294 | 73.601824 | 82.295001 | True |
| 61afe6117bae | 0.0002 | 0.5 | [1.0, 1.0, 1.0] | 1.0 | 1.0 | 69.814630 | 73.395716 | 82.170002 | False |
| e4724e65e6a4 | 0.0002 | 1.0 | [1.0, 1.0, 1.0] | 0.75 | 1.0 | 69.744885 | 73.462809 | 82.195003 | False |
| d16d3af78e46 | 5e-05 | 1.0 | [1.0, 1.0, 1.0] | 1.0 | 1.0 | 69.698257 | 73.312428 | 82.295003 | False |
| dcdf2119cb11 | 0.0002 | 1.0 | [1.0, 1.0, 1.5] | 1.0 | 1.0 | 69.550168 | 73.256947 | 81.975002 | False |
| ffb3402cbb33 | 0.0002 | 2.0 | [1.0, 1.0, 1.0] | 1.0 | 1.0 | 69.332616 | 72.919026 | 81.820002 | False |

## Leaderboard 3651

| Trial | fusion_lr | visual scale | F/P/R weights | sparsity | inclusion | Score5_R1 % | J_long3 % | J_long % | Reused |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| 6d44ae8d5c34 | 0.0002 | 1.0 | [1.0, 1.0, 1.0] | 1.0 | 1.0 | 72.531265 | 76.723441 | 85.195001 | False |
| 2ea4a3ae907a | 0.0002 | 1.0 | [1.0, 1.0, 1.0] | 1.0 | 1.5 | 72.484610 | 76.689683 | 85.220003 | False |
| a009f49f814a | 0.0001 | 1.0 | [1.0, 1.0, 1.0] | 1.0 | 1.0 | 72.444202 | 76.657670 | 85.195002 | True |

## Final References

| Model | Score5_R1 % | J_long3 % | J_long % |
|---|---:|---:|---:|
| Search best@3651 | 72.531265 | 76.723441 | 85.195001 |
| TI-fast@3651 | 72.146116 | 76.316193 | 84.765002 |
| 3/0@3651 background | 71.946091 | 76.306819 | 84.820002 |

## Complete Native Recall

Each table includes all successful500 trials and final3651 results from direct Top2 promotion. Values are percentages. Native bare-student normalized image/text inner products only; no training mask or rerank.

### a009f49f814a @500

| Dataset | Direction | R1 | R5 | R10 |
|---|---|---:|---:|---:|
| COCO | I2T | 59.600 | 81.880 | 88.800 |
| COCO | T2I | 40.932 | 66.636 | 76.320 |
| Urban-1k | I2T | 89.700 | 98.400 | 99.500 |
| Urban-1k | T2I | 87.400 | 98.200 | 99.100 |
| Flickr30k-test1k | I2T | 86.100 | 97.500 | 98.800 |
| Flickr30k-test1k | T2I | 70.340 | 90.660 | 94.680 |
| DOCCI | I2T | 76.180 | 94.560 | 97.560 |
| DOCCI | T2I | 75.900 | 94.720 | 97.540 |
| Long-DCI | I2T | 55.209 | 74.612 | 81.044 |
| Long-DCI | T2I | 57.222 | 76.401 | 82.057 |
### d16d3af78e46 @500

| Dataset | Direction | R1 | R5 | R10 |
|---|---|---:|---:|---:|
| COCO | I2T | 59.820 | 81.980 | 88.780 |
| COCO | T2I | 40.768 | 66.416 | 76.176 |
| Urban-1k | I2T | 89.300 | 98.300 | 99.300 |
| Urban-1k | T2I | 87.100 | 97.900 | 98.900 |
| Flickr30k-test1k | I2T | 86.200 | 97.400 | 98.800 |
| Flickr30k-test1k | T2I | 70.320 | 90.580 | 94.740 |
| DOCCI | I2T | 76.700 | 94.800 | 97.520 |
| DOCCI | T2I | 76.080 | 94.620 | 97.480 |
| Long-DCI | I2T | 54.446 | 74.191 | 80.742 |
| Long-DCI | T2I | 56.248 | 75.704 | 81.663 |
### 6d44ae8d5c34 @500

| Dataset | Direction | R1 | R5 | R10 |
|---|---|---:|---:|---:|
| COCO | I2T | 59.780 | 82.140 | 89.100 |
| COCO | T2I | 41.112 | 66.728 | 76.368 |
| Urban-1k | I2T | 89.300 | 98.300 | 99.400 |
| Urban-1k | T2I | 88.200 | 98.400 | 99.200 |
| Flickr30k-test1k | I2T | 86.100 | 97.200 | 98.800 |
| Flickr30k-test1k | T2I | 70.500 | 90.700 | 94.660 |
| DOCCI | I2T | 76.220 | 94.680 | 97.620 |
| DOCCI | T2I | 76.060 | 94.680 | 97.600 |
| Long-DCI | I2T | 55.551 | 75.112 | 81.268 |
| Long-DCI | T2I | 57.077 | 76.335 | 81.965 |
### 61afe6117bae @500

| Dataset | Direction | R1 | R5 | R10 |
|---|---|---:|---:|---:|
| COCO | I2T | 59.780 | 82.000 | 88.960 |
| COCO | T2I | 40.912 | 66.532 | 76.244 |
| Urban-1k | I2T | 89.500 | 98.000 | 99.300 |
| Urban-1k | T2I | 87.200 | 98.500 | 99.100 |
| Flickr30k-test1k | I2T | 86.500 | 97.200 | 98.700 |
| Flickr30k-test1k | T2I | 70.580 | 90.680 | 94.720 |
| DOCCI | I2T | 76.040 | 94.640 | 97.600 |
| DOCCI | T2I | 75.940 | 94.360 | 97.440 |
| Long-DCI | I2T | 55.012 | 74.388 | 80.716 |
| Long-DCI | T2I | 56.682 | 76.098 | 81.926 |
### ffb3402cbb33 @500

| Dataset | Direction | R1 | R5 | R10 |
|---|---|---:|---:|---:|
| COCO | I2T | 59.140 | 81.380 | 88.220 |
| COCO | T2I | 40.692 | 66.204 | 76.052 |
| Urban-1k | I2T | 89.400 | 97.700 | 99.300 |
| Urban-1k | T2I | 86.800 | 98.000 | 98.900 |
| Flickr30k-test1k | I2T | 85.600 | 97.200 | 98.700 |
| Flickr30k-test1k | T2I | 70.380 | 90.460 | 94.460 |
| DOCCI | I2T | 75.620 | 94.320 | 97.400 |
| DOCCI | T2I | 75.460 | 94.300 | 97.360 |
| Long-DCI | I2T | 53.920 | 73.468 | 80.203 |
| Long-DCI | T2I | 56.314 | 75.585 | 81.571 |
### 50cd96066da8 @500

| Dataset | Direction | R1 | R5 | R10 |
|---|---|---:|---:|---:|
| COCO | I2T | 59.740 | 81.920 | 88.800 |
| COCO | T2I | 41.112 | 66.652 | 76.328 |
| Urban-1k | I2T | 90.000 | 98.100 | 99.400 |
| Urban-1k | T2I | 87.700 | 98.100 | 99.000 |
| Flickr30k-test1k | I2T | 86.100 | 97.500 | 98.800 |
| Flickr30k-test1k | T2I | 70.520 | 90.760 | 94.620 |
| DOCCI | I2T | 76.480 | 94.620 | 97.700 |
| DOCCI | T2I | 75.700 | 94.560 | 97.480 |
| Long-DCI | I2T | 54.815 | 74.467 | 80.755 |
| Long-DCI | T2I | 56.709 | 76.085 | 81.821 |
### dcdf2119cb11 @500

| Dataset | Direction | R1 | R5 | R10 |
|---|---|---:|---:|---:|
| COCO | I2T | 59.300 | 81.940 | 88.680 |
| COCO | T2I | 40.780 | 66.448 | 75.936 |
| Urban-1k | I2T | 89.100 | 97.900 | 99.400 |
| Urban-1k | T2I | 87.000 | 98.200 | 99.100 |
| Flickr30k-test1k | I2T | 85.700 | 97.100 | 98.700 |
| Flickr30k-test1k | T2I | 70.180 | 90.380 | 94.380 |
| DOCCI | I2T | 75.920 | 94.520 | 97.480 |
| DOCCI | T2I | 75.880 | 94.400 | 97.420 |
| Long-DCI | I2T | 54.854 | 74.388 | 80.900 |
| Long-DCI | T2I | 56.788 | 76.019 | 81.847 |
### e4724e65e6a4 @500

| Dataset | Direction | R1 | R5 | R10 |
|---|---|---:|---:|---:|
| COCO | I2T | 59.520 | 81.580 | 88.680 |
| COCO | T2I | 40.772 | 66.500 | 76.092 |
| Urban-1k | I2T | 89.400 | 98.100 | 99.300 |
| Urban-1k | T2I | 87.500 | 98.100 | 99.100 |
| Flickr30k-test1k | I2T | 86.000 | 97.200 | 98.800 |
| Flickr30k-test1k | T2I | 70.380 | 90.640 | 94.640 |
| DOCCI | I2T | 76.080 | 94.480 | 97.560 |
| DOCCI | T2I | 75.800 | 94.420 | 97.440 |
| Long-DCI | I2T | 55.091 | 74.730 | 80.834 |
| Long-DCI | T2I | 56.906 | 76.085 | 81.926 |
### a72ea0fb10cb @500

| Dataset | Direction | R1 | R5 | R10 |
|---|---|---:|---:|---:|
| COCO | I2T | 59.840 | 81.980 | 88.980 |
| COCO | T2I | 41.060 | 66.604 | 76.240 |
| Urban-1k | I2T | 89.300 | 98.100 | 99.500 |
| Urban-1k | T2I | 87.700 | 98.400 | 99.100 |
| Flickr30k-test1k | I2T | 86.700 | 97.200 | 98.900 |
| Flickr30k-test1k | T2I | 70.600 | 90.680 | 94.640 |
| DOCCI | I2T | 76.100 | 94.640 | 97.560 |
| DOCCI | T2I | 75.880 | 94.600 | 97.400 |
| Long-DCI | I2T | 54.775 | 74.375 | 80.847 |
| Long-DCI | T2I | 56.932 | 76.243 | 81.847 |
### d6542aae963d @500

| Dataset | Direction | R1 | R5 | R10 |
|---|---|---:|---:|---:|
| COCO | I2T | 59.880 | 81.860 | 88.900 |
| COCO | T2I | 41.024 | 66.624 | 76.256 |
| Urban-1k | I2T | 89.700 | 98.100 | 99.400 |
| Urban-1k | T2I | 87.400 | 98.200 | 99.100 |
| Flickr30k-test1k | I2T | 86.000 | 97.400 | 98.800 |
| Flickr30k-test1k | T2I | 70.500 | 90.740 | 94.640 |
| DOCCI | I2T | 76.320 | 94.680 | 97.600 |
| DOCCI | T2I | 75.760 | 94.520 | 97.480 |
| Long-DCI | I2T | 55.380 | 74.849 | 81.005 |
| Long-DCI | T2I | 57.103 | 76.480 | 81.860 |
### 2ea4a3ae907a @500

| Dataset | Direction | R1 | R5 | R10 |
|---|---|---:|---:|---:|
| COCO | I2T | 60.040 | 82.340 | 89.100 |
| COCO | T2I | 41.440 | 66.804 | 76.712 |
| Urban-1k | I2T | 89.200 | 98.400 | 99.200 |
| Urban-1k | T2I | 87.500 | 98.300 | 99.100 |
| Flickr30k-test1k | I2T | 87.300 | 97.200 | 99.000 |
| Flickr30k-test1k | T2I | 71.220 | 90.860 | 94.940 |
| DOCCI | I2T | 76.940 | 94.980 | 97.480 |
| DOCCI | T2I | 76.660 | 94.820 | 97.660 |
| Long-DCI | I2T | 55.604 | 75.257 | 81.689 |
| Long-DCI | T2I | 56.748 | 76.309 | 81.965 |
### a009f49f814a @3651

| Dataset | Direction | R1 | R5 | R10 |
|---|---|---:|---:|---:|
| COCO | I2T | 61.400 | 84.100 | 90.080 |
| COCO | T2I | 42.116 | 67.844 | 77.580 |
| Urban-1k | I2T | 92.200 | 98.300 | 99.400 |
| Urban-1k | T2I | 90.900 | 98.500 | 99.200 |
| Flickr30k-test1k | I2T | 89.200 | 98.200 | 99.400 |
| Flickr30k-test1k | T2I | 71.780 | 91.480 | 95.340 |
| DOCCI | I2T | 78.160 | 95.660 | 98.240 |
| DOCCI | T2I | 79.520 | 95.640 | 98.060 |
| Long-DCI | I2T | 58.958 | 77.374 | 83.439 |
| Long-DCI | T2I | 60.208 | 78.203 | 83.373 |
### 6d44ae8d5c34 @3651

| Dataset | Direction | R1 | R5 | R10 |
|---|---|---:|---:|---:|
| COCO | I2T | 61.840 | 83.940 | 90.180 |
| COCO | T2I | 42.092 | 67.788 | 77.628 |
| Urban-1k | I2T | 91.900 | 98.400 | 99.300 |
| Urban-1k | T2I | 91.000 | 98.600 | 99.100 |
| Flickr30k-test1k | I2T | 89.000 | 98.100 | 99.500 |
| Flickr30k-test1k | T2I | 72.040 | 91.400 | 95.420 |
| DOCCI | I2T | 78.300 | 95.740 | 98.220 |
| DOCCI | T2I | 79.580 | 95.820 | 98.120 |
| Long-DCI | I2T | 59.129 | 77.730 | 83.636 |
| Long-DCI | T2I | 60.431 | 78.400 | 83.623 |
### 2ea4a3ae907a @3651

| Dataset | Direction | R1 | R5 | R10 |
|---|---|---:|---:|---:|
| COCO | I2T | 61.660 | 83.780 | 90.120 |
| COCO | T2I | 42.168 | 68.020 | 77.500 |
| Urban-1k | I2T | 92.100 | 98.600 | 99.400 |
| Urban-1k | T2I | 90.900 | 98.600 | 99.200 |
| Flickr30k-test1k | I2T | 88.600 | 98.100 | 99.500 |
| Flickr30k-test1k | T2I | 72.280 | 91.500 | 95.340 |
| DOCCI | I2T | 78.260 | 95.640 | 98.180 |
| DOCCI | T2I | 79.620 | 95.740 | 98.260 |
| Long-DCI | I2T | 58.787 | 77.559 | 83.346 |
| Long-DCI | T2I | 60.471 | 78.506 | 83.860 |

## Diagnostics and Failures

Actual LR and gradients are logged for backbone, text_mask_and_shared_pool, visual_mask, fusion_adapter. Requested gate moments/quantiles/saturation, F/P/R differences, modality norms and replaced-image mask switching at1/100/200/500/1217/2000/3000/3651 are preserved per successful stage. Diagnostics do not enter selection. All full-step measurements include real DataLoader and ordinary logs; checkpoint writes remain separate. Full candidate/tail handling and finite gradients are validated.

Failures: none.

## Artifacts

Parameter JSON hashes are trial IDs. SEARCH_STATE.json records runtime commits, commands/exit codes, checkpoint/student hashes, metric JSON, stage costs, promotion lineage and all failure statuses. leaderboard_500.csv / leaderboard_3651.csv preserve original score floats. The existing empty1217 leaderboard records that the superseded stage was not run. Raw evaluator JSON and compact evidence are committed here; large training checkpoints, RNG/optimizer payloads, full token logs, data and caches remain server-local.

Only the five authorized native protocols are used: COCO5000/25000, Urban1000/1000, Flickr1000/5000, DOCCI5000/5000, reconstructed Long-DCI7602/7602. Image batch64; COCO chunk512. No DCI Full run. All five sets were used in selection, so these are tuning results on previously examined benchmarks.
