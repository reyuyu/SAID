# E2-Uniform posthoc audit (no training)

Engineering: all three complete saved-stream comparisons pass; original artifacts unchanged.
Retrieval: all 12 Urban Recall values exactly reproduce both original evaluations.

| Direction | Both correct | E2 only | S12 only | Both wrong | Delta pp | Exact McNemar p | Paired bootstrap 95% CI pp |
|---|---:|---:|---:|---:|---:|---:|---|
| I2T | 937 | 2 | 2 | 59 | +0.000 | 1.000000 | [-0.4, 0.4] |
| T2I | 923 | 6 | 3 | 68 | +0.300 | 0.507812 | [-0.3, 0.9] |

T2I net +3 is the difference between gains and regressions, not three gains with no regressions.
Paired bootstrap treats test queries as iid; it does not estimate training-seed variability. McNemar tests both discordant classes.
Both query directions and all improvement/regression IDs, target ranks, predictions, scores and margins are in the JSON.

| Updates | Epoch | Updates verified | Sample positions | Tail per rank |
|---|---:|---:|---:|---:|
| 1218–2434 | 1 | 1217 | 1245904 | 180 |
| 2435–3651 | 2 | 1217 | 1245904 | 180 |
| 3652–4868 | 3 | 1217 | 1245904 | 180 |

Actual E2 logs match independent actual S12 logs for every step/rank, including text/token/K/index summaries and native LR.
Independent frozen sampler, K/subset and LR reconstruction also passes. No gradients, outputs, parameters or augmented image pixels are compared.
Restoration, optimizer endpoint counts, scheduler horizon4868, next-epoch cursors and four-rank synchronization are checked from complete checkpoints and original restoration/acceptance evidence.
Raw per-query training text/token arrays were not saved: their identity is supported by matching cryptographic summaries, not a new full raw-text reconstruction.

Mechanism: word-length summaries of both improved and regressed queries are descriptive only. No causal specialization or statistically reliable training improvement is established.
No new training, model updates, parameter changes, checkpoint selection or evaluator mathematical changes occurred. Original reports were not overwritten.

## Exact native Recall reproduction

| Model | I2T R@1/R@5/R@10 (%) | T2I R@1/R@5/R@10 (%) |
|---|---|---|
| E2 | 93.900 / 99.100 / 99.600 | 92.900 / 99.100 / 99.400 |
| S12 | 93.900 / 99.200 / 99.600 | 92.600 / 99.200 / 99.400 |

All raw Recall floats match the originals exactly. Both directions fail the unadjusted 0.05 significance threshold and the two-direction Bonferroni 0.025 threshold. T2I exact p=0.5078125; paired query bootstrap 95% interval is [-0.3,+0.9]pp. This is not evidence of a statistically significant improvement.

## Symmetric descriptive cases

| Direction | Category | Query IDs | Mean caption words |
|---|---|---|---:|
| I2T | E2 only correct | 29, 292 | 100.500 |
| I2T | S12 only correct | 165, 233 | 120.000 |
| T2I | E2 only correct | 137, 182, 500, 583, 686, 798 | 104.333 |
| T2I | S12 only correct | 587, 725, 727 | 107.667 |

All 1000 queries per direction are saved in URBAN_4868_I2T_QUERIES.json and URBAN_4868_T2I_QUERIES.json; summary JSON contains their filenames and SHA256. Both improvement and regression groups are reported without selected success examples. Caption token counts count nonzero token positions including special tokens; word counts are whitespace counts. No entity/relation counts or semantic causal claims are inferred.

Engineering and requested log-summary audits: PASS; no failed or UNVERIFIED required item. Each stage has four complete rank timing files. All 3651 optimizer updates and 3,737,712 sample positions were verified. E2 stage hashes equal S12 hashes; frozen sampler, K/indices and native scheduler provide additional independent reconstruction. Complete checkpoint identities, restoration lineage and optimizer counters are recorded in FULL_STREAM_1218_4868_PROOF.json. Protected original artifacts are enumerated with unchanged SHA256 in IMMUTABILITY_AND_IDENTITY_AUDIT.json.

Scientific limitations: single training seed; paired-query uncertainty cannot establish seed robustness; raw per-example training token arrays are not independently reconstructed. Mechanistic specialization remains unproven. Fourteen CPU tests passed, including drift rejection, ties, exact paired statistics, output immutability and complete query artifact preservation.
