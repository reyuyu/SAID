# Protocol and result provenance

Long-caption R@1 (I2T / T2I, %):

| Model | DOCCI | DCI (unverified) | Long-DCI (author CSV verified) | Urban-1k |
|---|---|---|---|---|
| E2-Uniform@4868 | 80.200 / 81.220 | 69.058 / 69.289 | 60.274 / 60.668 | 93.900 / 92.900 |

Short-caption R@1 (I2T / T2I, %):

| Model | COCO-Val5K | Flickr30k-Full |
|---|---|---|
| E2-Uniform@4868 | 61.860 / 43.000 | 56.713 / 36.725 |

| Dataset | I2T R@1/5/10 (%) | T2I R@1/5/10 (%) | Status / origin |
|---|---|---|---|
| DOCCI | 80.200000 / 96.420000 / 98.560000 | 81.220000 / 96.180000 / 98.420000 | VERIFIED_SOURCE_PROTOCOL; Historical real inference; reused unchanged |
| DCI | 69.058296 / 86.367713 / 91.108264 | 69.288917 / 86.329276 / 90.787956 | PROTOCOL_UNVERIFIED; Historical real inference; reused unchanged |
| Long-DCI | 60.273612 / 78.189950 / 83.925283 | 60.668245 / 78.702973 / 83.701657 | VERIFIED_EXACT_AUTHOR_CSV; Historical real inference; reused unchanged |
| Urban-1k | 93.900000 / 99.100000 / 99.600000 | 92.900000 / 99.100000 / 99.400000 | VERIFIED_SOURCE_PROTOCOL; Historical real inference; reused unchanged |
| COCO | 61.860000 / 83.660000 / 89.700000 | 43.000000 / 68.740000 / 78.160000 | VERIFIED_SOURCE_PROTOCOL; Historical real inference; reused unchanged |
| Flickr30k-Test1K | 90.100000 / 98.500000 / 99.300000 | 74.000000 / 92.080000 / 95.820000 | LEGACY_ONLY; Historical real inference; reused unchanged |
| Flickr30k-Full | 56.712708 / 78.969890 / 85.866658 | 36.725293 / 59.293333 / 68.391908 | VERIFIED_SOURCE_PROTOCOL; This task: actual Full inference |

Only verified source-protocol columns may be compared at the documented dataset level. HyFL-private annotation bytes and tie implementation are not available for bitwise reproduction. DCI is explicitly excluded from strict HyFL comparisons. Flickr-test1K is a separate legacy result, never Full. Long-DCI author CSV is exact against the raw manifest and token-equivalent against the historical stripped manifest (3,025 whitespace differences).
