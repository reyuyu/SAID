# HNS-v1 ViT-L/14 Pure3 — v2 final report

The v1 resource-gate failure remains preserved in `../hns_v1_l14_pure3epoch3651_v1/RESOURCE_GATE_FAILURE.json`. v2 changed only the authorized resource acceptance rule and added audit/evaluation tooling; the HNS-v1 model, loss, sampler, optimizer, and training mathematics were unchanged.

## Training and identity

The official ViT-L/14 source hash is `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`. The common HNS-L14 Step0 hash is `8fa8f7819f41f01635b95c0c7aed3c2a51f8dd58317666cacd74d3bc162fc912`. The completed checkpoint is `b5024cedf0f3ddceb72d59c81d809026b54bc36ef344b3e9925842ac767c5417`; its strict bare student export is `7a55b8e190ceae1bcf54a850a3452e2299fdfa4dc30f8e5c4cdb504309a9b731`. Export image/text outputs matched the checkpoint with max absolute error 0.0.

The formal run completed exactly 3,651 optimizer updates with scheduler horizon 3,651. All four ranks finished with maximum parameter disagreement 0.0, NCCL all-reduce count 10, finite losses/gradients, and peak allocated memory about 47.36 GiB per GPU. Checkpoint audits at steps 0, 1217, 2434 and 3651 passed and remained immutable. There is deliberately no standalone step-500 formal checkpoint: the configured checkpoint interval was 1217, and a later checkpoint is not presented as step 500.

The revised gate passed 5 warmup plus 30 measured cycles: mean 4.3500 s, P95 4.4736 s, maximum 4.5176 s, with every GPU below 65 GiB. Formal wall-clock time was about 16,867 s. Formal phase means were data wait 0.1061 s, update compute 4.3082 s, post-update logging 0.1326 s, barrier 0.00037 s, and monitor reduction 0.00022 s. The compute bucket includes transfer, GPU forward/backward, optimizer, DDP and stream serialization. Because v1 had no phase timers, the exact cause of its step-9 5.0773 s cycle remains INCONCLUSIVE; v2 timings do not retrospectively attribute that individual v1 cycle.

## Native five-dataset results

All values are percentage recall from normalized FP32 full-image/full-caption embeddings, with no inference mask, reranking, ensemble or TTA. DCI was not evaluated; Long-DCI is the fixed reconstructed 7,602-pair protocol.

| Dataset | I2T R@1 | R@5 | R@10 | T2I R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|---:|---:|
| COCO | 64.880 | 86.260 | 91.400 | 48.496 | 73.148 | 81.840 |
| Flickr30k-Test1K | 91.600 | 99.200 | 99.700 | 78.580 | 94.660 | 96.980 |
| DOCCI | 85.040 | 97.520 | 98.940 | 84.900 | 97.360 | 98.940 |
| Long-DCI | 63.746 | 80.860 | 86.346 | 64.930 | 81.137 | 85.688 |
| Urban-1k | 94.800 | 99.600 | 99.700 | 94.700 | 99.000 | 99.600 |

| Aggregate | Value |
|---|---:|
| Score5 | 77.167266 |
| J_long3 | 81.352777 |
| J_long | 89.860000 |
| Short4 | 70.889000 |
| Urban Mean | 94.750000 |

Relative to the historical HNS-v1 B16 Pure3 full result (Score5 73.638084, J_long3 78.014807, J_long 86.915002, Short4 67.073, Urban I2T/T2I 93.800/92.700), this L14 run changes the aggregates by +3.529182, +3.337970, +2.944998, +3.816000, +1.000/+2.000 percentage points respectively. This is an architecture comparison, not a controlled backbone ablation. No verified Balanced-L14 historical result was used or substituted.

## Flickr30k-Full

The separately audited official manifest contains 31,783 images and 158,915 captions. Results are I2T 64.506183/84.516880/90.403675 and T2I 43.617657/66.421043/74.783375 for R@1/R@5/R@10. Correct-query counts are I2T 20,502/26,862/28,733 and T2I 69,315/105,553/118,842. Manifest SHA256 is `bfed72a299e4d8cbfef817c13c836cd5aa189e18dd2fd29b632f5808e8dd12bc`; the complete receipt is `final_eval/flickr_full.json`.

## Mask and HNS diagnostics

Last-50-update means were F/Dall/D3 keep ratios 0.6436/0.6059/0.6081; F→Dall hard violation was 0.05975 with IoU 0.7762, and D3→Dall hard violation was 0.08350 with IoU 0.7666. These are descriptive training diagnostics, not evidence of retrieval causality. No non-finite values or rank divergence occurred.

## Evidence and stopping

`CHECKPOINT_AUDIT.json`, `DATA_STREAM_AND_LR_AUDIT.json`, and `RESOURCE_AND_TRAINING_AUDIT.json` contain the saved-state, every-update stream/LR, rank, finite-value and resource evidence. The stream proof covers all 3,651 updates and 3,737,712 rank sample positions, matching every shared field of the frozen reference schema; one optional telemetry digest present only in the reference was intentionally excluded from the intersection comparison and recorded as a schema difference. The raw failed attempt remains in `*.attempt1.json`.

No training or evaluation process remains active after this report. This was a single seed and the public retrieval sets have been repeatedly observed; the results are exploratory and do not establish statistical significance or an unbiased new SOTA.
