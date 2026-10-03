# Official default-AMP reproduction

Urban T2I target93.0%: **YES**, measured93.000000%, 930/1000.

| Official dataset | Direction | Published % | Released checkpoint + official evaluator % | Δ reproduced−published pp | Classification |
| --- | --- | --- | --- | --- | --- |
| COCO | I2T | 61.3 | 61.300000 | +0.000000 | REPRODUCED |
| COCO | T2I | 43.0 | 42.947949 | -0.052051 | NEAR-REPRODUCED |
| Flickr-full | I2T | 56.6 | 56.616367 | +0.016367 | REPRODUCED |
| Flickr-full | T2I | 36.6 | 36.603469 | +0.003469 | REPRODUCED |
| Urban1k | I2T | 93.1 | 93.300000 | +0.200000 | NEAR-REPRODUCED |
| Urban1k | T2I | 93.0 | 93.000000 | +0.000000 | REPRODUCED |
| DOCCI | I2T | 79.7 | 79.620000 | -0.080000 | NEAR-REPRODUCED |
| DOCCI | T2I | 80.0 | 79.960000 | -0.040000 | REPRODUCED |
| DCI-full | I2T | 68.5 | 68.468930 | -0.031070 | REPRODUCED |
| DCI-full | T2I | 67.6 | 67.469571 | -0.130429 | NEAR-REPRODUCED |

Source commit, strict loader anomaly, observed pools and precision diagnosis are retained in OFFICIAL_REPRO_RESULTS.json and DEBIAS_VS_SAID.md. All measured directions and recalls are retained, including unfavorable results.
