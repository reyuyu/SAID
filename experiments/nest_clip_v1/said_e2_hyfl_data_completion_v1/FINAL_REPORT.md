# E2 HyFL data completion

P0 completed by actual full-library inference. P1 completed by direct author CSV comparison. P2 remains explicitly unverified. No training, model updates, alternate model selection or benchmark-driven protocol changes occurred.

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

ZIP SHA256 `2ce2420c0d17f0531deaa89ac657b4d5067ec519da16ff1ea12acbf5619c7391`; 31783 decoded images; missing/extra IDs: 0/0; 1000 historical images checked, byte-identical 1000, different bytes but equal pixels/preprocess 0.

Author TSV revision `95daae6e4b0b078f9af5fb7fcd8e95ab2201dc0a`, 7602 records / 5409968 bytes, SHA256 `019cfee5d9d1bf2ae10689628eb6a08f148efef65a15402420a1b0e195c940df`. Raw text, order, IDs, tokens and positive/candidate mappings exact against raw manifest. Historical stripped manifest has 3,025 text differences, zero token differences. Historical scores can therefore be reused.

DCI: absent author annotation/generator; GOAL subset and captions mismatch. The author-request draft was not sent. No strict HyFL-DCI comparison is claimed.

Full worker GPU0: image encoding 76.05s, text encoding 246.45s, scoring 9.51s, overall 346.31s. Peak CUDA allocated/reserved 1.477/2.432 GiB; CPU peak RSS 3.016 GiB. No OOM, NaN/Inf or image-read failures.

21 CPU tests and actual Full-image same-batch64 GPU correctness passed. The latter has exact per-query hit agreement; maximum feature error 0.0. Parameter digest and checkpoint SHA unchanged before/after. DCI identity is an unresolved protocol audit, not labeled PASS.

Necessary worker change only records separate timing and before/after model digest; retrieval functions and all production training/model files unchanged. An initial ZIP audit classified AppleDouble resource forks as JPEGs by suffix. Its failure evidence is preserved; confirmed magic/version permits classifying them as metadata. No image/caption subset was used. An initial small probe also failed before model loading because memory peak reset preceded CUDA initialization; its receipt is preserved and API call order corrected. New tools reject source SHA/count/ID errors, duplicate/escaping archive files, bad image versions, malformed TSV and token/order mismatch; original files are immutable.

Large ZIP/images, author TSV, manifests, per-image/token receipts, features and query outputs stay on server. Runtime sources: `/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/said-e2-hyfl-data-completion-v1`; image/features/evidence: `/root/said_hyfl_data_completion_v1`.

Remaining limitation: unavailable authoritative HyFL DCI annotation. Public datasets have been repeatedly observed; these scores are exploratory, not independently sealed generalization evidence. No fourGPU internal-shard equivalence is claimed. See DELIVERY_VERIFICATION.json for GPU/process cleanup and artifact hashes; final remote/local HEAD proof is saved on server at `/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/said-e2-hyfl-data-completion-v1/GITHUB_SYNC_VERIFIED.json` after push/fetch.
