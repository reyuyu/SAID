# Arm B Summary weight0.4 matched500 experiment

**STRONG POSITIVE**. All500 updates, strict bare export, frozen native5 evaluations and read-only spot8 are complete.

Only normalized alignment weights changed: F/S/D[1.2,0.6,1.2]→[1.3,0.4,1.3]. Sampling/RNG, model, candidates, directional weights, sparsity, inclusion/ramp, optimizer/LR,horizon4868 and native inference are unchanged. Smoke5 and formal500 independently begin at common step0; no resume or full continuation.

| Model | Score5 | J_long3 | J_long | Short4 |
|---|---:|---:|---:|---:|
| RandomK baseline | 69.900394 | 73.603324 | 82.340003 | 64.346000 |
| Arm B S=0.6 | 69.911321 | 73.132202 | 82.245001 | 65.080000 |
| Arm B S=0.4 | 70.160995 | 73.503658 | 82.595003 | 65.147000 |

| Reference | ΔScore5 pp | ΔJ_long3 pp | ΔJ_long pp | ΔShort4 pp |
|---|---:|---:|---:|---:|
| ArmB_S06 | +0.249674 | +0.371456 | +0.350002 | +0.067000 |
| RandomK | +0.260601 | -0.099666 | +0.255000 | +0.801000 |

| Dataset | I2T R1 | T2I R1 | Δvs S0.6 I/T pp | Δvs RandomK I/T pp |
|---|---:|---:|---|---|
| COCO | 60.320000 | 41.888000 | -0.040000 / +0.148000 | +0.360000 / +0.964000 |
| Urban-1k | 90.200007 | 87.600005 | +0.500005 / +0.200003 | +1.200002 / -0.300002 |
| Flickr30k-test1k | 86.800000 | 71.580000 | +0.000000 / +0.160000 | +0.600000 / +1.280000 |
| DOCCI | 76.060000 | 76.520000 | +0.400000 / +0.300000 | -0.260000 / +0.380000 |
| Long-DCI | 54.314654 | 56.327282 | +0.499868 / +0.328861 | -1.078664 / -0.539332 |

| Dataset | I2T R1/R5/R10 | T2I R1/R5/R10 |
|---|---|---|
| COCO | 60.320000 / 82.620000 / 89.520000 | 41.888000 / 67.268000 / 77.120000 |
| Urban-1k | 90.200007 / 98.000002 / 99.500006 | 87.600005 / 98.300004 / 99.200004 |
| Flickr30k-test1k | 86.800000 / 97.700000 / 99.100000 | 71.580000 / 91.560000 / 95.520000 |
| DOCCI | 76.060000 / 94.720000 / 97.540000 | 76.520000 / 94.840000 / 97.500000 |
| Long-DCI | 54.314654 / 74.427782 / 80.610366 | 56.327282 / 76.045777 / 81.899500 |

## Summary loss and gradient dominance

Raw CE and weighted CE are separately recorded at1/100/200/500 and last50. Last50 combined CE and relative weighted contributions (common10/3 omitted here):

| View | Raw combined CE | Weighted contribution | Weighted CE share | Old S0.6 weighted CE share |
|---|---:|---:|---:|---:|
| F | 0.115003 | 0.149503 | 7.310% | 6.382% |
| S | 1.630428 | 0.652171 | 31.886% | 41.173% |
| D | 0.956650 | 1.243645 | 60.804% | 52.445% |

Summary is not the largest weighted CE route. Largest: D. Raw loss magnitude alone is not a gradient conclusion.

| Spot8 native backbone view | Raw gradient norm mean | Weighted norm share | Old S0.6 weighted norm share |
|---|---:|---:|---:|
| F | 11.217986 | 24.340% | 20.165% |
| S | 29.731231 | 19.907% | 27.662% |
| D | 25.632570 | 55.753% | 52.173% |

Summary is not the largest weighted gradient-norm route. Largest: E_combined. Norm shares are diagnostics, not percentages of the summed gradient or AdamW parameter displacement.

| Spot8 group | Actual alignment/native cosine S0.6 | Actual alignment/native cosine S0.4 | Delta |
|---|---:|---:|---:|
| G1_vision_backbone | 0.528846 | 0.574895 | +0.046049 |
| G2_text_backbone | 0.364349 | 0.415255 | +0.050907 |
| native_backbone_total | 0.467383 | 0.517283 | +0.049900 |

Spot8 uses exactly the same first8 seed0/epoch0 global batches as the prior audit. Actual image/F/S/D digests match. Five objectives per batch are synchronized DDP backwards with20 parameter-tensor agreement checks; model/checkpoint remain unchanged. Different learned checkpoints limit attribution of gradient changes to immediate weighting alone. No full32-batch audit was rerun.

## Resources and correctness

Normal full-cycle mean 2.088410s; peak allocated 27.759893GiB. All500×4 F/S/D/token/K/indices stream hashes match Arm B. Common initialization, trainable parameter counts and production source hashes match. Fixed1000 sampling replay and explicit CE/gradient tests passed. All500 actual loss values reconstruct from the declared weighting and unchanged auxiliary terms.

Gate/F-D fields are scheduled diagnostics; last50 gate/F-D values have one observation. CE/keep/inclusion/S-D IoU use all50. Full details are in TRAINING_DIAGNOSTICS.json and compressed complete logs.

Long-DCI remains7602/7602 with manifestSHA8890a2be15e2b64c142f9a1e39224d34b6ce92fc17ffb6099e14942cc8161c4b. Native inference uses normalized image and full-caption text embeddings only.

## Decision

Full4868 verification recommended: **True** under the registered STRONG POSITIVE rule. Full training launched: **False**. This experiment stops at500.

POSITIVE requires both Score5>69.911321% and J_long3>73.132202%; STRONG POSITIVE additionally requires Score5≥70.000394% and J_long3≥73.403324%. Raw fractions, not rounded display values, determine decisions. Overlapping negative/tradeoff conditions are retained in DECISION.json.
