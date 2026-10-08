# W20 + KR234: one authorized combination at500

Classification: `MIXED`; full candidacy: `NO_FULL_CANDIDACY`.

Fresh common0, exactly500 updates, horizon4868, local-only. Only W20[1.4,1.4,.2] and historical KR234 sampling are combined. Original soft detached-child inclusion ramp200/max1 and sparsity[1,2,2] remain active. HNS is disabled.

| Method | Alignment | K | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Anchor | 1.35/1.35/.30 | fixed3 | 71.063600 | 74.954666 | 84.025001 | 65.227000 | 90.900000 | 89.700000 |
| W20 | 1.4/1.4/.2 | fixed3 | 71.088347 | 74.957911 | 84.020003 | 65.284000 | 91.300000 | 89.400000 |
| KR234 | 1.35/1.35/.30 | KR234 | 71.107901 | 75.046502 | 84.120002 | 65.200000 | 91.600000 | 89.200000 |
| W20-KR234 | 1.4/1.4/.2 | KR234 | 71.040797 | 74.800662 | 83.965002 | 65.401000 | 91.000000 | 89.300000 |

| Reference | ΔScore5 | ΔJ_long3 | ΔJ_long | ΔShort4 | ΔUrban I2T | ΔUrban T2I |
|---|---:|---:|---:|---:|---:|---:|
| Anchor | -0.022803 | -0.154004 | -0.059999 | +0.174000 | +0.099999 | -0.399995 |
| W20 | -0.047549 | -0.157249 | -0.055000 | +0.117000 | -0.300002 | -0.099999 |
| KR234 | -0.067104 | -0.245840 | -0.155000 | +0.201000 | -0.600004 | +0.100005 |

| Dataset | I2T R@1/5/10 | T2I R@1/5/10 |
|---|---|---|
| COCO | 60.580000 / 82.720000 / 89.320000 | 41.464000 / 67.256000 / 76.732000 |
| Urban-1k | 91.000003 / 98.500007 / 99.700004 | 89.300007 / 98.800004 / 99.400002 |
| Flickr30k-test1k | 87.800000 / 97.500000 / 99.000000 | 71.760000 / 91.360000 / 95.200000 |
| DOCCI | 77.680000 / 95.400000 / 98.180000 | 77.880000 / 95.120000 / 97.720000 |
| Long-DCI | 56.432518 / 75.217048 / 81.504867 | 56.511444 / 75.927388 / 81.846882 |

Lowest-view alignment share: 43.869127%; weighted lowest/Dall gradient norm ratio: 0.315456.
Identical eight-global-batch, fixed500-weight gradient spotcheck; raw directional CE on exact backbone group, all-reduce/4. Diagnostic norms are not signed net-gradient contributions.
Baseline gradient/CE/share comparisons, all five-set directional deltas, K statistics, mask hierarchy, gate statistics and runtime are recorded in JSON.

Q1: interaction = combination - W20 - KR234 + Anchor, in RESULTS.json. A positive value is a descriptive single-seed result, not proof of causal/statistical synergy.
Q2: raw Score5 exceeds KR234: False. Q3: exceeds known500 points: {'INC0': False, 'HNS-v1': False}.
Q4: Urban T2I, Short4, Flickr T2I and Long-DCI deltas vsKR234 identify whether W20 eases tradeoffs. Q5: J_long3/J_long/Long-DCI deltas vsW20 identify any rich-text benefit.
Selection uses raw Score5. STRONG_POSITIVE applies when aboveKR234 and J_long3 decline is at most0.20pp, a predeclared interpretation of “not markedly worse”; positiveScore5 cannot be labeledNEGATIVE by this guard. Full candidacy requires strictly greater rawScore5, with no rounding tolerance.
Github fetched configs/results and pinned source migrations are in BASELINE_PROVENANCE.json. Current non-HNS production loss/every gradient/AdamW were compared exactly to fetched historicalKR234. Native evaluator sources are unchanged, optimizer/LR/scheduler/checkpoint function ASTs match.
Strict bare/native full-caption inference only; no mask/gate/local-caption/rerank/ensemble/TTA.

Checkpoint SHA256: `e540a7d23be3c1db04d4a434be938300a2d007157c5b612c64a0cea07bdc6c82`. Bare SHA256: `2618cd73fdb9eb310bc5612312d1e989cabd11137795fd8fc9b3932e7fd7dc69`.
Large checkpoints/weights/1000-sample token evidence/rawlogs remainserver-local. RUNTIME_STATS.json inventories path/size/SHA/time windows. `/root` cache is disposable; NFS originals retained.
No full4868 or additional experiment launched, regardless of candidacy. Stop and wait for human decision.
