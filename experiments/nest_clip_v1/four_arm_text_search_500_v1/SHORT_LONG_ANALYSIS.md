# Short / long analysis

Status: **TRADEOFF**. Eligible: ['A']. Guard J_long3 ≥73.403324%.

| Model | Score5 | J_long3 | J_long | Short4 |
|---|---:|---:|---:|---:|
| RandomK baseline | 69.900394 | 73.603324 | 82.340003 | 64.346000 |
| Summary+RandomDetail previous | 69.556218 | 72.553029 | 81.590002 | 65.061000 |
| A | 69.710789 | 73.441982 | 81.950002 | 64.114000 |
| B | 69.911321 | 73.132202 | 82.245001 | 65.080000 |
| C | 69.777585 | 72.994642 | 81.940002 | 64.952000 |
| D | 69.312527 | 72.433544 | 81.345002 | 64.631000 |

A: vs RandomK, ΔScore5 -0.189605pp, ΔJ_long3 -0.161342pp, ΔShort4 -0.232000pp.

Removing extreme splits did not improve overall retrieval in this matched500-step run.

B: vs previous Summary+RandomDetail, ΔScore5 +0.355103pp, ΔJ_long3 +0.579172pp, ΔShort4 +0.019000pp.

Summary cosine nearest negative 0.739001, mean off-diagonal 0.321314; last50 CE Summary I/T 0.684915/0.826422, Detail I/T 0.454189/0.508366.

C: vs previous Summary+RandomDetail, ΔScore5 +0.221367pp, ΔJ_long3 +0.441612pp, ΔShort4 -0.109000pp.

Summary T2I downweight recovered T2I on ['Urban-1k', 'DOCCI', 'Long-DCI']; directional deltas versus previous are {'Urban-1k': 0.10000467300415039, 'DOCCI': 0.28000000000000247, 'Long-DCI': 0.39463299131807794}pp. This provides limited support for the T2I-ambiguity hypothesis. Short4 and overall/long-guard deltas above must be considered together: small positive directional changes do not establish that the long-text deficit was repaired. A single500-step seed does not establish causality. The12/11 normalization also strengthens the other five directions, as preregistered.

Summary cosine nearest negative 0.745524, mean off-diagonal 0.336501; last50 CE Summary I/T 0.640733/0.804365, Detail I/T 0.462126/0.518617.

D: vs previous Summary+RandomDetail, ΔScore5 -0.243691pp, ΔJ_long3 -0.119485pp, ΔShort4 -0.430000pp.

Contiguous Detail changes Long-DCI by +0.263089/+0.000000pp versus discrete RandomDetail. The directional results do not support a uniform continuity benefit. No statistical significance claim is made without replicated seeds.

Summary cosine nearest negative 0.742551, mean off-diagonal 0.330720; last50 CE Summary I/T 0.640434/0.779769, Detail I/T 0.539014/0.588234.

B is the strongest observed supervision change in this search. Against RandomK it gains Score5 +0.010927pp and Short4 +0.734000pp, but loses J_long3 -0.471122pp; it is0.271122pp below the safety threshold. Against previous Summary+RandomDetail it restores part of the long-text performance while preserving Short4. The very small overall Score5 increase is not a safe improvement.

B outperforms C by Score5 +0.133736pp, J_long3 +0.137560pp and Short4 +0.128000pp. These matched results favor reducing total Summary supervision over reducing only its T2I direction at these registered coefficients. C has modest positive long-T2I changes versus previous, but its Summary nearest-negative cosine rises; CE changes and cosine geometry do not independently prove a causal ambiguity mechanism.

D does not recover the previous Urban or DOCCI results; Long-DCI I2T gains0.263089pp while T2I is unchanged. This run offers no convincing overall benefit from enforcing contiguous Detail. Discrete sampling is not established as the main source of the long-text deficit.

Rule-selected eligible champion A is a relative selection among new arms, not an improvement over the existing RandomK. A is the only safe candidate for a future full-training confirmation under the requested rule, but its500-step evidence does not justify replacing the formal RandomK recipe. B is the more informative supervision direction for future research if the long-text deficit can be addressed. No full training is launched.