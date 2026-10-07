# Nested D3 Balanced: seven isolated local500 arms

Status:`NO_IMPROVEMENT`. Winner:`None`. Exactly seven fresh-common0 runs; stopped at500, no full/combinations.
Winner/positive rules and tie-break were frozen before launch in SEARCH_PLAN.json. Primary Pareto objectives:Score5,J_long3,Urban T2I,Short4; six-metric secondary frontier also retained.

| Arm | Alignment | K | D3 sparsity r | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Anchor | 1.35/1.35/0.3 | 3 (bounded strict subset) | 2.0 | 71.063600 | 74.954666 | 84.025001 | 65.227000 | 90.900000 | 89.700000 |
| W20 | 1.4/1.4/0.2 | 3 (bounded strict subset) | 2.0 | 71.088347 | 74.957911 | 84.020003 | 65.284000 | 91.300000 | 89.400000 |
| W25 | 1.375/1.375/0.25 | 3 (bounded strict subset) | 2.0 | 71.009852 | 74.851087 | 83.965002 | 65.248000 | 91.400000 | 89.600000 |
| W35 | 1.325/1.325/0.35 | 3 (bounded strict subset) | 2.0 | 71.041286 | 74.842810 | 83.775002 | 65.339000 | 90.900000 | 89.100000 |
| W40 | 1.3/1.3/0.4 | 3 (bounded strict subset) | 2.0 | 71.069015 | 74.835025 | 83.885003 | 65.420000 | 91.200000 | 89.000000 |
| S25 | 1.35/1.35/0.3 | 3 (bounded strict subset) | 2.5 | 71.043678 | 74.930797 | 83.930002 | 65.213000 | 91.400000 | 89.100000 |
| S30 | 1.35/1.35/0.3 | 3 (bounded strict subset) | 3.0 | 71.043208 | 74.940014 | 83.980002 | 65.198000 | 90.900000 | 89.500000 |
| KR234 | 1.35/1.35/0.3 | Random valid2/3/4 | 2.0 | 71.107901 | 75.046502 | 84.120002 | 65.200000 | 91.600000 | 89.200000 |

| Arm | ΔScore5 | ΔJ_long3 | ΔJ_long | ΔShort4 | ΔUrban I2T | ΔUrban T2I | Classification |
|---|---:|---:|---:|---:|---:|---:|---|
| W20 | +0.024747 | +0.003245 | -0.004999 | +0.057000 | +0.400000 | -0.300000 | NO_IMPROVEMENT |
| W25 | -0.053748 | -0.103580 | -0.060000 | +0.021000 | +0.500000 | -0.100000 | NO_IMPROVEMENT |
| W35 | -0.022314 | -0.111856 | -0.250000 | +0.112000 | +0.000000 | -0.600000 | NO_IMPROVEMENT |
| W40 | +0.005415 | -0.119641 | -0.139999 | +0.193000 | +0.300000 | -0.700000 | NO_IMPROVEMENT |
| S25 | -0.019922 | -0.023870 | -0.095000 | -0.014000 | +0.500000 | -0.600000 | NO_IMPROVEMENT |
| S30 | -0.020392 | -0.014653 | -0.044999 | -0.029000 | +0.000000 | -0.200000 | NO_IMPROVEMENT |
| KR234 | +0.044301 | +0.091835 | +0.095001 | -0.027000 | +0.700000 | -0.500000 | NO_IMPROVEMENT |

Primary Pareto frontier:['Anchor', 'W20', 'W25', 'W35', 'W40', 'KR234']. Six-metric frontier:['Anchor', 'W20', 'W25', 'W35', 'W40', 'S25', 'KR234'].
All individual R@1/5/10 and complete direction/metric deltas:each arm REPORT.md/RESULTS.json and aggregate RESULTS.json.

| Arm | Lowest alignment share(%) | Weighted lowest/Dall gradient ratio | Lowest keep | Dall-lowest IoU | Lowest outside Dall(%) |
|---|---:|---:|---:|---:|---:|
| W20 | 40.800 | 0.325982 | 0.730469 | 0.836417 | 2.056 |
| W25 | 45.487 | 0.407136 | 0.767460 | 0.874819 | 1.802 |
| W35 | 53.609 | 0.598392 | 0.781702 | 0.878313 | 2.179 |
| W40 | 56.479 | 0.710553 | 0.816992 | 0.917275 | 1.739 |
| S25 | 50.165 | 0.488335 | 0.765533 | 0.862780 | 2.138 |
| S30 | 50.227 | 0.514499 | 0.754613 | 0.849416 | 1.690 |
| KR234 | 53.310 | 0.524341 | 0.782360 | 0.878394 | 2.274 |

Weight axis:compare W20/W25/Anchor(.30)/W35/W40 directly; each has identical512000 sample/text/token/indices trajectory. Sparsity axis:S25/S30 redistribute fixed coefficient mass5, no inclusion change. KR234 changes only K selection; original order/fallback and global RNG stay frozen.
Related CPU tests passed (exact count in CPU_TESTS.json), first5 hard gate,5000 prelaunch local paths,complete runtime stream/indices/LR review,8 frozen gradient batches,full optimizer/RNG/cursor checkpoint review and strict native export.
Checkpoint/bare/data/cache/raw logs retained locally, never uploaded. Per-arm RUNTIME_STATS.json records local raw evidence paths,size,SHA256 and UTC windows.
No automated full training,second round,or combination of weight/sparsity/K changes. Await human selection.

## Independent axis trends

| D3 alignment weight | Arm | Score5 | J_long3 | Urban T2I | Short4 |
|---:|---|---:|---:|---:|---:|
| 0.200 | W20 | 71.088347 | 74.957911 | 89.400000 | 65.284000 |
| 0.250 | W25 | 71.009852 | 74.851087 | 89.600000 | 65.248000 |
| 0.300 | Anchor | 71.063600 | 74.954666 | 89.700000 | 65.227000 |
| 0.350 | W35 | 71.041286 | 74.842810 | 89.100000 | 65.339000 |
| 0.400 | W40 | 71.069015 | 74.835025 | 89.000000 | 65.420000 |

Highest Score5 on the weight axis:W20. All four weight arms preserve fixed K3 and anchor sparsity; no extrapolation or combined arm.

| Sparsity r | Arm | Lowest keep | Dall-lowest IoU | Lowest outside Dall(%) | Score5 | J_long3 |
|---:|---|---:|---:|---:|---:|---:|
| 2.0 | Anchor | 0.792063 | 0.900053 | 1.681 | 71.063600 | 74.954666 |
| 2.5 | S25 | 0.765533 | 0.862780 | 2.138 | 71.043678 | 74.930797 |
| 3.0 | S30 | 0.754613 | 0.849416 | 1.690 | 71.043208 | 74.940014 |

KR234 valid K histogram:`{'2': 179406, '3': 175403, '4': 156622, '1': 505}`; mean K:2.953522; mean Dk effective tokens:65.513189; mean per-sample Dk/Dall content-token coverage:0.446772.
Conditioned K counts by pool size m,CE/shares/gradients and mask changes are in KR234/TRAINING_DIAGNOSTICS.json,GRADIENT_SPOTCHECK.json and MASK_HIERARCHY_AUDIT.json.

## Interpretation

No arm satisfies the predeclared replacement gate. Retain Anchor; `NO_IMPROVEMENT` here means no qualifying replacement, not that every individual metric decreased. The primary Pareto frontier contains Anchor, W20, W25, W35, W40 and KR234; no forced winner is selected.

Increasing the lowest-view alignment weight from .20 to .40 raises its weighted gradient ratio from .325982 to .710553 and last50 alignment share from 40.800% to 56.479%, without a monotonic Score5 gain. Anchor retains the best Urban T2I (89.7); W35/W40 drop to 89.1/89.0. W20 has a small Score5 gain (+.024747pp) with Urban T2I -.3pp.

Redistributing fixed sparsity mass toward D3 lowers its keep ratio from .792063 to .765533/.754613, but also lowers Dall-D3 IoU to .862780/.849416. Neither S25 nor S30 improves Score5 or J_long3. Hard violations do not decrease consistently; inclusion remains the original soft detached-child penalty.

KR234 achieves the highest Score5 (71.107901) and J_long3 (75.046502): +.044301/+.091835pp versus Anchor. Its Urban I2T rises .7pp, while Urban T2I falls .5pp and Short4 falls .027pp. It remains a Pareto tradeoff rather than `NEW_500_BEST`, and does not meet the frozen small-tradeoff rule for `PARETO_POSITIVE`. Mean valid K is 2.953522, mean lowest-view effective tokens 65.513189, and mean per-sample content-token coverage .446772. Its weighted lowest/Dall gradient ratio is .524341; last50 alignment share is 53.310%.

All seven runs stopped at exactly500 updates. No full4868, additional sampling/weight arm, or cross-axis combination was launched. Full five-set directional R@1/5/10, all deltas, masks, gradients, checkpoint identities and local raw-evidence inventories remain in the individual reports and JSON files.
