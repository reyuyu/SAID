# NEST-HybridF-E11: step500 original retrieval

**Only the saved step500 checkpoint is evaluated.** The user stopped the previously authorized 3651-step run after it had already logged step1123; the torchrun process received SIGINT and exited 1 by design. No updates after 500 contribute to this report. This is not a completed 3-epoch run, and the final cross-rank parameter-agreement check did not run at step500.

Training code commit `0deaa86e2749b20f8ab84e8f1c5095a6306e326d`; seed0, 4×A100 80GB, batch256/rank, no accumulation, shared init SHA256 `54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`, eta0.25, arm=A3, random K unchanged, epochs3 and cosine horizon3651. From the common step0, all four ranks logged steps1–500 with finite loss and gradients; all 2000 rank-step sample/F/P/R/K streams match the A3-RandomK reference. Maximum full-loss formula reconstruction error 2.45e-06. The step500 checkpoint SHA256 is `d74896147acafc1fa3eb9718315d58f5ec0a99f5189c2bf9c94c5847d56a6bce`; strictly loaded bare student SHA256 `a71ef81847fb774c86a234c0d0360c174c75a560e040b8d69a66ce07ea55fb2f`. Optimizer step500 and exact image/text embedding equality passed; all four native evaluation commands exited 0. The [interrupted-run audit](evidence/formal500-audit.json) records the intentional nonzero exit and later unused checkpoints.

The first 5-step smoke was rejected because rank3 reported an NCCL error after its updates. A new four-rank 5-step smoke with `NCCL_DEBUG=INFO` for diagnostics passed the strict audit; both original failure and retry remain recorded. The first error's cause was not established.

**Primary same-budget comparison: E11@500 vs A3-RandomK@500.** The following 24 values are percentages. The difference is `100×(E11 recall−A3 recall)` in percentage points, not relative percent. 9 of 24 values are higher for E11.

| Dataset | Direction | Metric | E11 (%) | A3-RandomK@500 (%) | Δ vs A3-RandomK (pp) |
|---|---|---|---:|---:|---:|
| coco | I2T | R@1 | 59.160 | 58.860 | +0.300 |
| coco | I2T | R@5 | 81.640 | 81.700 | -0.060 |
| coco | I2T | R@10 | 88.480 | 88.640 | -0.160 |
| coco | T2I | R@1 | 40.588 | 40.532 | +0.056 |
| coco | T2I | R@5 | 66.252 | 66.196 | +0.056 |
| coco | T2I | R@10 | 75.928 | 75.900 | +0.028 |
| urban | I2T | R@1 | 89.400 | 89.700 | -0.300 |
| urban | I2T | R@5 | 97.800 | 97.800 | +0.000 |
| urban | I2T | R@10 | 99.400 | 99.100 | +0.300 |
| urban | T2I | R@1 | 86.600 | 86.700 | -0.100 |
| urban | T2I | R@5 | 97.800 | 97.900 | -0.100 |
| urban | T2I | R@10 | 98.800 | 98.800 | +0.000 |
| flickr_test1k | I2T | R@1 | 85.400 | 85.500 | -0.100 |
| flickr_test1k | I2T | R@5 | 97.300 | 97.300 | +0.000 |
| flickr_test1k | I2T | R@10 | 98.700 | 98.900 | -0.200 |
| flickr_test1k | T2I | R@1 | 69.900 | 69.920 | -0.020 |
| flickr_test1k | T2I | R@5 | 90.180 | 90.200 | -0.020 |
| flickr_test1k | T2I | R@10 | 94.380 | 94.340 | +0.040 |
| docci | I2T | R@1 | 75.780 | 76.000 | -0.220 |
| docci | I2T | R@5 | 94.580 | 94.580 | +0.000 |
| docci | I2T | R@10 | 97.400 | 97.380 | +0.020 |
| docci | T2I | R@1 | 75.600 | 75.860 | -0.260 |
| docci | T2I | R@5 | 94.480 | 94.400 | +0.080 |
| docci | T2I | R@10 | 97.300 | 97.260 | +0.040 |

On COCO and Urban, additional existing 500-step references are available. Fixed-A3/Clean/Full results were verified in their original reports. S0 numbers come from a historical step500 report; its original checkpoint is not present on this server, so S0 was not re-evaluated here. These other methods are context, not a one-variable controlled comparison.

| Dataset | Direction | Metric | E11 (%) | A3-RandomK@500 (%) | Fixed-A3@500 (%) | S0@500 (%) | Clean@500 (%) | Full@500 (%) | Δ vs A3-RandomK (pp) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| coco | I2T | R@1 | 59.160 | 58.860 | 59.060 | 60.580 | 60.540 | 58.780 | +0.300 |
| coco | I2T | R@5 | 81.640 | 81.700 | 82.080 | 82.200 | 82.480 | 81.540 | -0.060 |
| coco | I2T | R@10 | 88.480 | 88.640 | 88.460 | 89.060 | 89.180 | 88.500 | -0.160 |
| coco | T2I | R@1 | 40.588 | 40.532 | 40.908 | 41.236 | 41.464 | 40.216 | +0.056 |
| coco | T2I | R@5 | 66.252 | 66.196 | 66.444 | 67.092 | 66.672 | 65.764 | +0.056 |
| coco | T2I | R@10 | 75.928 | 75.900 | 76.636 | 76.620 | 76.552 | 75.620 | +0.028 |
| urban | I2T | R@1 | 89.400 | 89.700 | 88.700 | 87.000 | 88.200 | 88.400 | -0.300 |
| urban | I2T | R@5 | 97.800 | 97.800 | 97.700 | 97.100 | 97.500 | 97.900 | +0.000 |
| urban | I2T | R@10 | 99.400 | 99.100 | 99.100 | 98.800 | 99.200 | 99.300 | +0.300 |
| urban | T2I | R@1 | 86.600 | 86.700 | 85.400 | 84.200 | 85.800 | 87.800 | -0.100 |
| urban | T2I | R@5 | 97.800 | 97.900 | 97.100 | 96.700 | 97.500 | 98.000 | -0.100 |
| urban | T2I | R@10 | 98.800 | 98.800 | 98.800 | 98.100 | 99.100 | 99.100 | +0.000 |

Per the user's latest instruction, DCI and Long-DCI were not evaluated. Prespecified `J_long` includes DCI, so it cannot be computed from this round's evaluations. No historical DCI score is substituted.

Existing training-log diagnostics, arithmetic mean of updates451–500 (O=P and E=R). A3's F uses masked-only CE; E11 combines masked and native CE. Different total losses do not measure retrieval quality; `common_loss=loss−inc_weight×inc`.

| Last 50 updates | E11@500 | A3-RandomK@500 |
|---|---:|---:|
| loss | 9.274462 | 9.367558 |
| common_loss | 9.242774 | 9.337752 |
| F_mask_i2t | 0.055876 | 0.055246 |
| F_mask_t2i | 0.057311 | 0.057091 |
| F_native_i2t | 0.049500 | n/a |
| F_native_t2i | 0.058600 | n/a |
| F_hybrid | 0.111916 | n/a |
| O_i2t | 0.198219 | 0.198190 |
| O_t2i | 0.240939 | 0.242640 |
| E_i2t | 0.905252 | 0.919622 |
| E_t2i | 0.880228 | 0.887423 |
| F_sparse | 0.900763 | n/a |
| O_sparse | 0.864378 | n/a |
| E_sparse | 0.866636 | n/a |
| inc | 0.031687 | 0.029806 |
| inc_weight | 1.000000 | 1.000000 |
| hard_inclusion_violation | 0.031070 | 0.028587 |
| F_keep_ratio | 0.900763 | 0.909265 |
| O_keep_ratio | 0.864378 | 0.876621 |
| E_keep_ratio | 0.866636 | 0.874311 |
| F_all_open | 0.000000 | 0.000000 |
| O_all_open | 0.000000 | 0.000000 |
| E_all_open | 0.000000 | 0.000000 |
| F_all_closed | 0.000000 | 0.000000 |
| O_all_closed | 0.000000 | 0.000000 |
| E_all_closed | 0.000000 | 0.000000 |
| oe_iou | 0.824680 | 0.836251 |
| valid_global | 1023.940000 | 1023.940000 |

The observed first500 step times sum to 929.0s (excludes checkpoint writes and startup); the interrupted full command ran for 2213.8s before the user-requested SIGINT. Highest per-rank GPU allocated memory through step500: 17.311GiB; reserved: 18.479GiB. The source training log and all checkpoints, including unused step600–1100 files, remain on the server at `/root/lk_projects/SAID-nest-clip-v1/hybridf_v1/E11/formal`. Only the step500 bare student under `step500/` was scored. The [first500 step logs](step500_steps.jsonl.gz), [machine-readable scores and exit codes](step500_results.json), [snapshot audit](evidence/formal500-audit.json) and [evaluation JSONs](step500_evaluation) retain the evidence. Only one seed was run; no significance, broad superiority, or independent effect of the inclusion term is claimed. The native-F mix also reduces the masked-F weight, so this experiment does not isolate those two changes.
