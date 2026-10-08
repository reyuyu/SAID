# HNS-Half1217

This is a branch-from-step500 intervention experiment, not an independently trained 0–1217 run.

Status: `PROMISING_FOR_FULL`. Exactly717 new updates501–1217; no additional training.
Only hierarchy surcharge is halved. Alignment/sparsity/beta2/2/no-SG/optimizer/RNG/sampler/horizon4868 frozen.
Schedule uses completed BEFORE update:completed499/update500=1;completed500/update501=.5.
All three checkpoints re-exported and evaluated with identical evaluator sources, splits, batch64 and native normalized image/full-caption embeddings. No masks/gates/rerank/ensemble/TTA.

| Model @1217 | Score5 | J_long3 | J_long | Short4 |
|---|---:|---:|---:|---:|
| HNS-Half | 72.869675 | 77.012792 | 85.925003 | 66.655000 |
| HNS-v1 | 72.836442 | 76.971404 | 85.955001 | 66.634000 |
| D3 Balanced | 72.812657 | 77.001762 | 86.040002 | 66.529000 |

## HNS-Half

| Dataset | I2T R1/R5/R10 | T2I R1/R5/R10 |
|---|---|---|
| COCO | 61.960000 / 83.560000 / 89.500000 | 42.480000 / 68.200000 / 77.828000 |
| Urban-1k | 93.000007 / 99.000007 / 99.700004 | 91.800004 / 98.900002 / 99.500006 |
| Flickr30k-test1k | 88.900000 / 98.100000 / 99.300000 | 73.280000 / 92.000000 / 95.680000 |
| DOCCI | 79.180000 / 96.100000 / 98.400000 | 79.720000 / 95.660000 / 98.220000 |
| Long-DCI | 58.760852 / 77.242831 / 83.214943 | 59.615891 / 77.861089 / 83.188635 |

## HNS-v1

| Dataset | I2T R1/R5/R10 | T2I R1/R5/R10 |
|---|---|---|
| COCO | 61.800000 / 83.860000 / 89.580000 | 42.476000 / 68.136000 / 77.704000 |
| Urban-1k | 93.400002 / 98.900002 / 99.700004 | 91.500002 / 98.600006 / 99.500006 |
| Flickr30k-test1k | 89.100000 / 98.100000 / 99.400000 | 73.160000 / 92.000000 / 95.660000 |
| DOCCI | 79.320000 / 96.160000 / 98.340000 | 79.600000 / 95.720000 / 98.180000 |
| Long-DCI | 58.458300 / 77.295449 / 83.083399 | 59.550118 / 77.690082 / 83.122862 |

## D3 Balanced

| Dataset | I2T R1/R5/R10 | T2I R1/R5/R10 |
|---|---|---|
| COCO | 62.020000 / 83.640000 / 89.660000 | 42.496000 / 68.180000 / 77.824000 |
| Urban-1k | 93.600005 / 98.900002 / 99.700004 | 91.900003 / 98.900002 / 99.500006 |
| Flickr30k-test1k | 88.400000 / 98.200000 / 99.200000 | 73.200000 / 92.000000 / 95.560000 |
| DOCCI | 79.160000 / 96.020000 / 98.280000 | 79.500000 / 95.560000 / 98.040000 |
| Long-DCI | 58.339911 / 77.203368 / 82.794002 | 59.510655 / 77.703236 / 83.136017 |

All aggregate and directional R1/R5/R10 deltas:comparison1217.json.
Deltas vs HNS-v1: `{"Score5": 0.0332331373155057, "J_long3": 0.04138856219252318, "J_long": -0.02999818801880849, "Short4": 0.021000000000000796}`.
Deltas vs D3 Balanced: `{"Score5": 0.057018037351539874, "J_long3": 0.011030062252558537, "J_long": -0.1149992370605446, "Short4": 0.12600000000000477}`.
Matched actual sampling payloads (IDs/F/Dall/D3 texts/tokens/indices) and LR for all717 updates are exact against existing HNS full logs. First501–505 full-state gates:RESUME_GATE.json.
Same fixed production graph tests prove forward violations unchanged, hierarchy gradient halved, alignment/sparsity gradients unchanged. Default production sources remain unchanged.
Full recommendation: True. Retrieval is primary; no violation veto or density-alone health claim. Single seed; material-long-decline threshold0.05pp declared before training.
Mask points/last50 and500→1217 changes:TRAINING_DIAGNOSTICS.json, MASK_HIERARCHY_AUDIT.json.
Parent/final/baseline checkpoint and bare SHA256 records:RESUME_PROVENANCE.json, BASELINE_PROVENANCE.json, RESULTS.json. Binary assets never uploaded.
Raw log local paths/sizes/SHA256/time ranges:RUNTIME_STATS.json. `/root` ephemeral cache; NFS originals retained. No copy/full image audit.
No training beyond1217 or additional experiment was started.
