# Actual native retrieval results

| Model | DOCCI I2T/T2I | DCI I2T/T2I | Long-DCI I2T/T2I | Urban I2T/T2I |
| --- | --- | --- | --- | --- |
| E2@4868 (actual) | 80.200 / 81.220 | 69.058 / 69.289 (UNVERIFIED) | 60.274 / 60.668 (UNVERIFIED) | 93.900 / 92.900 |

| Model | COCO I2T/T2I | Flickr Full I2T/T2I |
| --- | --- | --- |
| E2@4868 (actual) | 61.860 / 43.000 | BLOCKED |

| Dataset | Direction | Queries | Candidates | R@1% (correct) | R@5% (correct) | R@10% (correct) |
| --- | --- | --- | --- | --- | --- | --- |
| DOCCI | I2T | 5000 | 5000 | 80.200000 (4010) | 96.420000 (4821) | 98.560000 (4928) |
| DOCCI | T2I | 5000 | 5000 | 81.220000 (4061) | 96.180000 (4809) | 98.420000 (4921) |
| DCI | I2T | 7805 | 7805 | 69.058296 (5390) | 86.367713 (6741) | 91.108264 (7111) |
| DCI | T2I | 7805 | 7805 | 69.288917 (5408) | 86.329276 (6738) | 90.787956 (7086) |
| Long-DCI | I2T | 7602 | 7602 | 60.273612 (4582) | 78.189950 (5944) | 83.925283 (6380) |
| Long-DCI | T2I | 7602 | 7602 | 60.668245 (4612) | 78.702973 (5983) | 83.701657 (6363) |
| Urban-1k | I2T | 1000 | 1000 | 93.900000 (939) | 99.100000 (991) | 99.600000 (996) |
| Urban-1k | T2I | 1000 | 1000 | 92.900000 (929) | 99.100000 (991) | 99.400000 (994) |
| COCO | I2T | 5000 | 25000 | 61.860000 (3093) | 83.660000 (4183) | 89.700000 (4485) |
| COCO | T2I | 25000 | 5000 | 43.000000 (10750) | 68.740000 (17185) | 78.160000 (19540) |
| Flickr30k-Full | I2T | NOT_RUN | NOT_RUN | N/A | N/A | N/A |
| Flickr30k-Full | T2I | NOT_RUN | NOT_RUN | N/A | N/A | N/A |
| Flickr30k-Test1K | I2T | 1000 | 5000 | 90.100000 (901) | 98.500000 (985) | 99.300000 (993) |
| Flickr30k-Test1K | T2I | 5000 | 1000 | 74.000000 (3700) | 92.080000 (4604) | 95.820000 (4791) |

DCI/Long-DCI are provisional protocols. Full Flickr has no result. Flickr1K is legacy-only. Missing scores are not zero; no six-set aggregate is reported. All public benchmark observations remain exploratory.
