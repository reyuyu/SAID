# HNS static-strength two-arm @500

| Model | lambdaA | lambdaS | lambdaH | Score5 | J_long3 | J_long | Short4 | Urban I2T/T2I |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| HNS-v1 | 10 | 1 | 1 | 71.202056 | 75.152094 | 84.160001 | 65.277000 | 91.500 / 89.700 |
| E1-HNS-S12 | 10.0 | 1.2 | 1.0 | 71.307463 | 75.160439 | 84.235002 | 65.528000 | 91.400 / 89.500 |
| E2-HNS-H4 | 10.0 | 1.0 | 4.0 | 71.049193 | 74.885988 | 83.945003 | 65.294000 | 91.100 / 89.300 |

| Arm | Delta Score5 | Delta J_long3 | Delta J_long | Delta Short4 | Delta Urban I2T/T2I |
|---|---:|---:|---:|---:|---|
| E1-HNS-S12 | +0.105407 | +0.008345 | +0.075001 | +0.251000 | -0.100 / -0.200 |
| E2-HNS-H4 | -0.152864 | -0.266106 | -0.214997 | +0.017000 | -0.400 / -0.400 |

Selection: `{"BEST_OVERALL_500": "E1-HNS-S12", "RECOMMENDED_NEXT_VALIDATION": "E1-HNS-S12", "gain_pp": 0.10540696345462663, "weak_single_seed_signal": false, "automatic_continuation": false}`.

Q1: S12 Score5 delta 0.10540696345462663pp; keep changes {"F": -0.011252365112304763, "Dall": -0.019178892374038692, "D3": -0.02376084446907034}.
Q2: H4 hard violation deltas {"Dall_F_hard_violation": -0.0034335388615727436, "D3_Dall_hard_violation": -0.004320010840892792}.
Q3: H4 retrieval deltas {"Score5": -0.1528637950777636, "J_long3": -0.26610632512961274, "J_long": -0.21499742507934627, "Short4": 0.016999999999995907, "Urban_I2T": -0.3999948501586914, "Urban_T2I": -0.3999948501586914}. Lower violations alone do not imply retrieval gains.
Q4: Short4/long deltas {"E1-HNS-S12": {"Short4": 0.25099999999999056, "J_long3": 0.008344939091031733, "J_long": 0.07500096559525105}, "E2-HNS-H4": {"Short4": 0.016999999999995907, "J_long3": -0.26610632512961274, "J_long": -0.21499742507934627}}.
Q5: Gradient norms/cosines for every parameter group are in SCIENTIFIC_DIAGNOSTICS.json and GRADIENT_AUDIT.json. Pure coefficient controls share the original HNS@500 loss graph; learned-state gradients are separate.
Q6: E1-HNS-S12. Single-seed improvement below0.05pp is a weak signal, not full/E3 evidence.
All30 per-arm recalls and deltas are in RESULTS.json. Both arms fresh-common0, exactly500 updates, local-only. GPU idle.
No third arm, H2/H8, coefficient combinations, epoch-switching, new seed, new loss or1217/2434/3651/4868 continuation.
