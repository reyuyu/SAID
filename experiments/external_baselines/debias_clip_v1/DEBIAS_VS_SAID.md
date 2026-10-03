# DeBias-CLIP official B/16 reproduction and SAID frozen re-evaluation

Completed using the released `debias_vitb_3e.pt`, pinned official source, untouched model/evaluator,
and two separately audited dataset protocols. No training, checkpoint selection or inference augmentation ran.

## Q1: Official Urban T2I=93.0 reproduction

**YES.** Published93.0%; official default-AMP reproduction 93.000000%,
930/1000 correct; difference +0.000000pp,
+0 queries. Official I2T 93.300000%
versus paper93.1%, retained alongside T2I.

## Table A: Published reference only

| Dataset/protocol | Published I2T R1 % | Published T2I R1 % |
| --- | --- | --- |
| COCO | 61.3 | 43.0 |
| Flickr-full | 56.6 | 36.6 |
| Urban1k | 93.1 | 93.0 |
| DOCCI | 79.7 | 80.0 |
| DCI-full | 68.5 | 67.6 |
| Long-DCI (paper) | 57.8 | 57.4 |

Sources: [pinned official README](https://github.com/TRAILab/DeBias-CLIP/blob/18a06c98bcb50018e22c3febca2aefc8c4a12e0d/README.md)
and [paper2602.22419v2 Tables3/4](https://arxiv.org/html/2602.22419v2).
Table5's ablation DOCCI T2I79.7 differs from the primary80.0; both source entries are recorded.
Published full Flickr and DCI/Long-DCI protocols are not mixed into a SAID Score5.

## Table B: Official released-checkpoint reproduction

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

Actual full candidates: COCO5000/25014; Flickr31014/155070; Urban1000/1000;
DOCCI5000/5000; DCI7805/7805. The original base_full main entry point and its dataset readers
and ranking functions were called. Shell continuation syntax and machine paths were normalized;
evaluation flags were preserved. All official R@1/5/10 are retained in raw/official and the JSON report.
Long-DCI is a published reference but is not a separate dataset returned by released base_full.

## Q2 and Table C: Unified frozen SAID protocol

| Frozen dataset | Direction | DeBias B/16 R1 % | SAID B/16 R1 % | Δ SAID−DeBias pp |
| --- | --- | --- | --- | --- |
| COCO | I2T | 61.340000 | 61.700000 | +0.360000 |
| COCO | T2I | 42.948000 | 42.220000 | -0.728000 |
| Urban-1k | I2T | 93.300003 | 92.100006 | -1.199996 |
| Urban-1k | T2I | 93.100005 | 91.100007 | -1.999998 |
| Flickr30k-test1k | I2T | 89.800000 | 88.900000 | -0.900000 |
| Flickr30k-test1k | T2I | 73.600000 | 72.440000 | -1.160000 |
| DOCCI | I2T | 79.660000 | 78.760000 | -0.900000 |
| DOCCI | T2I | 79.920000 | 79.980000 | +0.060000 |
| Long-DCI | I2T | 59.431728 | 59.668508 | +0.236780 |
| Long-DCI | T2I | 59.168640 | 60.812944 | +1.644304 |

Unified Urban complete Recall:

| Direction | R@1 % | R@5 % | R@10 % |
| --- | --- | --- | --- |
| I2T | 93.300003 | 99.100006 | 99.700004 |
| T2I | 93.100005 | 98.700005 | 99.500006 |

| Metric | DeBias B/16 % | SAID B/16 % | Δ SAID−DeBias pp |
| --- | --- | --- | --- |
| Score5_R1 | 73.226838 | 72.768147 | -0.458691 |
| J_long3 | 77.430063 | 77.070244 | -0.359818 |
| J_long | 86.495002 | 85.485003 | -1.009999 |

Score5 averages all ten raw fractional directional R1 values; J_long3 averages Urban/DOCCI/Long-DCI's
six directions, J_long averages Urban/DOCCI's four. Differences are **SAID minus DeBias**.
All primary fair results use one checkpoint, native independent official encoders and plain normalized dot products.
No PCA, drop-first, sentence shuffle, shifted-padding augmentation, ensemble, reranking or conditional image feature runs.

## Q3: Located differences rather than a generic protocol label

Urban:1000 sorted raw image/caption pairs are identical (different_count=0);
sorted pair SHA256 `a69c44cdb450e72764374db4cc9886c6eca7856f4381fdbb77fac497ab89cfc6`. Official JSON uses filesystem enumeration,
SAID uses frozen lexical order; positive IDs are remapped by basename. Tokenizer remains the official
open_clip SimpleTokenizer at248 and preprocessing remains the official checkpoint factory transform.
Official default AMP and separate official FP32 differ in 1
T2I top1 queries, with correct counts 930 versus
931. Exact changed IDs are in URBAN_PRECISION_AUDIT.json.
This numerical precision difference is isolated using identical official raw rows/order/checkpoint.
Native FP32 embedding errors, frozen metric outputs and official FP32 consistency outcomes are in
raw_logs/native-urban-consistency.json; no default-AMP score is replaced by the diagnostic.
The first consistency gate found text max_abs0.0001925528 when text batch composition differed.
Matching the official batches (including final8 captions), then restoring the unchanged frozen candidate
order, reduces image/text max_abs to5.96046448e-8. Every R@1/5/10 value remains unchanged; no tolerance
is relaxed. Top1/ranking agreement is measured in raw_logs/native-urban-ranking-consistency.json.

COCO: released reader keeps25014 captions, frozen SAID keeps25000; it also changes image/caption ordering
and canonical chunk512/row-wise sorting. Flickr:31014 full-image candidates versus1000 test candidates,
not interchangeable scores. DOCCI: raw test membership/text are equal; precision and official argsort
versus frozen topk implementation are distinct. DCI: all7805 split entries with short+extra are different
from reconstructed7602 long captions. PROTOCOL_AUDIT.md gives the dataset-by-dataset evidence.
Dataset differences are established from raw data; numerical changes not isolated to a single factor
are not claimed as causally proved.

## Checkpoint and official loader audit

File size 1797383582 bytes; SHA256 `5cdd75077c35ab4d074a8e430c2f7f182ff48d01858d6c345776bf5f2713aa84`; saved epoch3.
303 state tensors /149835265 tensor elements. Actual B/16 shapes: vision conv[768,3,16,16],
vision positions[197,768], joint projection[768,512], text positions/base and residual[248,512].
The file saves epoch/name/state_dict/optimizer/scaler, but no args/config. Script flags are provenance,
not independently stored checkpoint hyperparameters. Manual transfer is from the author-provided official
Drive checkpoint through user cloud disk; access passwords/signed URLs are not published.

Official strict loading has no missing/unexpected keys and matches every effective post-conversion tensor.
There is one raw-state mismatch: the pinned factory replaces saved `text.positional_embedding_res`
with the base `text.positional_embedding` for this already-stretched custom checkpoint.
Maximum difference0.000978708267211914. Both phases preserve this same official behavior; no fix is
applied to chase published scores. Thus strict key compatibility does **not** mean every raw tensor
survives the official loader unchanged. This anomaly is explicit in CHECKPOINT_INVENTORY.json.

The official Urban same-input adapter audit has zero image/text differences and zero direct scaled-score
differences; own Recall matches the evaluator. Primary AMP normalized outputs are FP32,[1000,512],
mean norms1.0, full[1000,1000] similarity. Scaling/unscaled top1 equivalence is measured in the raw JSON.
Phase B's PCA-call guard confirms zero training PCA calls.

## Q4: Fair comparison scope and error analysis

The reference is fixed SAID Balanced-Stack-Patch B/16, four epochs/4868 updates, fusion_lr2e-4,
inclusion_max1, view_weights[1,1,1], sparsity_scale1, native inference. Reference scores are reused,
and its Urban predictions were independently recomputed to match92.10/91.10 exactly within float32 mean.
Bare checkpoint SHA256 `f36438934947ed7f1523fe87e877a0e62dd807dc8a7c89c751eac537e0ab64bb`; full `42901d24ae3b0e0147213fd45f5d3f663d9193d17b690536957625901e75500c`.
This is a same-test-protocol inference comparison, not matched training compute: DeBias has three
released training epochs/global batch256; SAID has four epochs/global full candidates1024.
Neither model was trained or selected again in this task. No L/14 model is evaluated.

Urban correctness categories, every query ID and top10 candidates are saved in
URBAN_ERROR_ANALYSIS.json and URBAN_RANKING_AGREEMENT.json. No manual hard-query selection is used.

## Complete measured Recall@1/5/10

| Protocol | Dataset | Direction | R@1 % | R@5 % | R@10 % |
| --- | --- | --- | --- | --- | --- |
| Official default AMP | COCO | I2T | 61.300000 | 83.140000 | 89.440000 |
| Official default AMP | COCO | T2I | 42.947949 | 68.541617 | 78.004318 |
| Official default AMP | Flickr-full | I2T | 56.616367 | 79.015928 | 85.790288 |
| Official default AMP | Flickr-full | T2I | 36.603469 | 59.377056 | 68.443929 |
| Official default AMP | Urban1k | I2T | 93.300000 | 99.100000 | 99.700000 |
| Official default AMP | Urban1k | T2I | 93.000000 | 98.800000 | 99.500000 |
| Official default AMP | DOCCI | I2T | 79.620000 | 96.240000 | 98.540000 |
| Official default AMP | DOCCI | T2I | 79.960000 | 95.840000 | 98.280000 |
| Official default AMP | DCI-full | I2T | 68.468930 | 85.701473 | 90.506086 |
| Official default AMP | DCI-full | T2I | 67.469571 | 84.612428 | 89.493914 |
| SAID frozen / native FP32 | Urban-1k | I2T | 93.300003 | 99.100006 | 99.700004 |
| SAID frozen / native FP32 | Urban-1k | T2I | 93.100005 | 98.700005 | 99.500006 |
| SAID frozen / native FP32 | COCO | I2T | 61.340000 | 83.060000 | 89.480000 |
| SAID frozen / native FP32 | COCO | T2I | 42.948000 | 68.520000 | 78.004000 |
| SAID frozen / native FP32 | Flickr30k-test1k | I2T | 89.800000 | 97.900000 | 99.200000 |
| SAID frozen / native FP32 | Flickr30k-test1k | T2I | 73.600000 | 91.980000 | 95.360000 |
| SAID frozen / native FP32 | DOCCI | I2T | 79.660000 | 96.260000 | 98.540000 |
| SAID frozen / native FP32 | DOCCI | T2I | 79.920000 | 95.840000 | 98.300000 |
| SAID frozen / native FP32 | Long-DCI | I2T | 59.431728 | 77.532228 | 83.412260 |
| SAID frozen / native FP32 | Long-DCI | T2I | 59.168640 | 76.966588 | 82.530913 |

Evidence: CHECKPOINT_INVENTORY.json, PUBLISHED_REFERENCE.json, OFFICIAL_REPRO_RESULTS.json,
SAID_UNIFIED_RESULTS.json, PROTOCOL_AUDIT.md, NATIVE_INFERENCE_AUDIT.md, URBAN_DATA_AUDIT.json,
URBAN_PRECISION_AUDIT.json, environment.json, structured commands/, raw/ and small raw_logs/.
Source remains clean at `18a06c98bcb50018e22c3febca2aefc8c4a12e0d`. Large weights, datasets, embeddings and environments remain on server.
