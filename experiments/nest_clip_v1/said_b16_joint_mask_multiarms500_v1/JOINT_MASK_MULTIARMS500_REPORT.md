# Joint Mask multi-arm experiment status

This report records the completed read-only evaluations and the controlled pause requested after the two-arm 1217 update comparison. The experiment uses the fixed ViT-B/16, four-GPU, global-batch-1024 protocol and the verified common Step0. J3 and J4 used the content-identical `/root/said_s02_stage500` data mirror; its records and offsets hashes match the NFS index, and sampled image hashes were checked before training.

The only model change is the predeclared mask-logit rule. A-Matched is the original BalancedStack rule; J1–J4 are the fixed formulas in the experiment source. The evaluator is the frozen native evaluator at commit `88d5db6`.

## 500-update comparison

| Arm | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I |
|---|---:|---:|---:|---:|---:|---:|
| A-Matched | 71.173740 | 75.009567 | 84.035003 | 65.420 | 91.300 | 89.400 |
| J1 | 71.105938 | 74.969231 | 84.060003 | 65.311 | 91.200 | 89.500 |
| J2 | 70.869028 | 74.548380 | 83.580003 | 65.350 | 90.400 | 89.300 |
| J3 | 70.889795 | 74.563659 | 83.695001 | 65.379 | 90.500 | 89.200 |
| J4 | 71.021642 | 74.732071 | 83.895001 | 65.456 | 90.400 | 89.700 |

At 500 updates, A-Matched ranked first by Score5 and J1 second. J4 was third. J1 traded a small Score5 decrease for a +0.1 percentage-point Urban T2I change versus A; this is an exploratory observation, not evidence of a generalization gain.

All five arms completed 500 updates with four-rank parameter difference 0 and NCCL aggregation 10.0. The training checkpoint and bare-student hashes are recorded in `JOINT_MASK_MULTIARMS_STATUS.json`; the per-dataset R@1/R@5/R@10 receipts are in each arm's `results/<arm>/` directory.

## 1217-update comparison for the two preselected arms

| Arm | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I |
|---|---:|---:|---:|---:|---:|---:|
| A-Matched | 72.829267 | 76.984112 | 85.905003 | 66.597 | 93.100 | 91.600 |
| J1 | 72.824649 | 76.993082 | 86.050002 | 66.572 | 93.700 | 91.800 |

Using the frozen selection order with Score5 as the primary key, A-Matched is the 1217 candidate by a margin of 0.004618 percentage points. J1 remains a candidate worth retaining for its slightly higher J_long3, J_long, and Urban values, but no 3651 claim is made for it.

## Controlled stop

The A-Matched 1217→3651 continuation was started from the complete A@1217 checkpoint, then stopped at the user's request before GitHub synchronization. It reached approximately step 1325 but did not persist a new checkpoint and is excluded from all results. The complete A@1217 checkpoint remains the valid parent. No J1 or other arm was started beyond 1217. All GPUs are idle after termination.

The next authorized action, if desired, is a fresh controlled continuation from A@1217 or J1@1217. This commit intentionally does not claim a 3651 result.
