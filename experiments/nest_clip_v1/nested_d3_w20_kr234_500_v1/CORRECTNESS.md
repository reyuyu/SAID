# W20 + KR234 correctness before training

This one authorized combination starts fresh from common step0 and stops at exactly500 updates with horizon4868. No production model, data, optimizer, scheduler or evaluator source was edited for this experiment.

GitHub branches were fetched before preflight. Original config/result blobs match the local Anchor, W20 and KR234 JSONs byte-for-byte. W20 uses literal[1.4,1.4,.2]; KR234 uses `nested_detail_kr234`. Both descend from D3 Balanced; all other method config values are identical. The combined config changes exactly these two fields.

`BASELINE_PROVENANCE.json` pins fetched commits/configs/results, common0 SHA256, current and predecessor source hashes. Since the original search, production source gained other sampling modes, inactive HNS routing and telemetry. These are explicitly audited rather than incorrectly claiming every source file is unchanged. Current non-HNS loss, every parameter gradient and one AdamW update match fetched historical KR234 exactly in CPU tests at completed0/99/199/499. Independent objective reconstruction checks W20 alignment plus sparsity[1,2,2] plus the original soft detached-child chain inclusion, ramp200/max1. HNS stays off.

Optimizer, learning-rate functions, seeding, horizon, consumed-batch logic and full checkpoint function ASTs match the fetched historical source. Native evaluator and local resolver sources are unchanged. Export has added inactive-HNS constructor routing; actual strict bare load and exact image/text embedding comparison are mandatory after500.

`MATCHED_PREFLIGHT.json` records1000 actual frozen seed0 samples,250/rank. Current output matches both fetched historical KR234 code and the original1000-sample audit on IDs, visible Dall pools, K, selected indices, F/Dall/Dk strings and token IDs, including fallbacks. Python/NumPy/Torch global RNG states are unchanged.5000 frozen sample paths resolve to the local mirror; missing, symlink or escape fails, with no NFS fallback. Full tokens/path proofs are local-only, indexed by path/size/SHA256.

92 CPU tests pass. Formal first5 checks happen before update6: common0 initialization, AdamW groups/order/counters, frozen data, exact KR234 text/token/index stream, W20shared F/Dall/LR, finite states and within-run rank synchronization. No cross-run floating-point equality is imposed. The same stream and literal loss checks cover all512000 actual training records before evaluation.

Read-only gradient diagnostics follow the original8-global-batch protocol at fixed step500 parameters, with no optimizer updates. Strict native bare evaluation covers five frozen datasets; full-caption inference only. No additional arm, weight change, HNS regularizer or full training starts automatically. Full candidacy is only a report field controlled by raw Score5 exceeding KR234.
