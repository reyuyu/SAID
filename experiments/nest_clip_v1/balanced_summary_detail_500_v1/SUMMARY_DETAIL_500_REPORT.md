# Full–Summary–Detail: strict single-variable500-update result

**NEGATIVE.** Fixed Summary–Detail decomposition does not improve the matched RandomK baseline at500.
Only500 optimizer updates were trained, from the common step0 after an independent5-step smoke;
scheduler horizon4868 throughout. No full4868 training, mixture, extra loss or inference trick follows.

## Main matched comparison

| Model | Score5_R1 % | J_long3 % | J_long % |
| --- | --- | --- | --- |
| Matched RandomK old-R@500 | 69.900394 | 73.603324 | 82.340003 |
| Summary–Detail@500 | 69.405034 | 72.570390 | 81.570002 |
| Delta pp | -0.495360 | -1.032934 | -0.770000 |

| Dataset | Direction | RandomK baseline % | Summary–Detail % | Delta pp |
| --- | --- | --- | --- | --- |
| COCO | I2T | 59.960000 | 59.920000 | -0.040000 |
| COCO | T2I | 40.924000 | 41.288000 | +0.364000 |
| Urban-1k | I2T | 89.000005 | 89.000005 | +0.000000 |
| Urban-1k | T2I | 87.900007 | 86.200005 | -1.700002 |
| Flickr30k-test1k | I2T | 86.200000 | 86.800000 | +0.600000 |
| Flickr30k-test1k | T2I | 70.300000 | 70.620000 | +0.320000 |
| DOCCI | I2T | 76.320000 | 75.880000 | -0.440000 |
| DOCCI | T2I | 76.140000 | 75.200000 | -0.940000 |
| Long-DCI | I2T | 55.393318 | 55.169692 | -0.223625 |
| Long-DCI | T2I | 56.866614 | 53.972639 | -2.893975 |

All aggregates use unrounded fractions. Delta is **Summary–Detail minus matched RandomK old-R**.
The reporting rule treats a dataset mean-R1 drop beyond0.2pp as a clear tradeoff; it is not a
training/parameter-selection gate. This is one seed and an early500-update experiment.

## Sampling definition and full-corpus audit

Full raw string/tokens remain bitwise-equivalent to the baseline's longest visible complete-sentence prefix.
Summary is the first raw cleaned segment; Detail is every remaining raw segment in original order,
compactly retokenized with the existing248-token tokenizer. No random K/subset/drop/shuffle/prefix PAD runs.
For raw n<2, local captions are absent, valid=false and only Full contributes. The unchanged batch
interface uses collation PAD sentinels for disabled local views, not a copied summary as Detail.

Complete train corpus:1245901 rows, original skip1000 preserved. Average raw segments
8.158627, median8.0.
Fixed seed0/epoch0/sample-ID100k audit: old R covers
57.041459% of raw non-summary BPE content;
new Detail covers100% before truncation and
99.645323% after truncation.
Detail truncation fraction:2.923879% of valid complete-corpus samples.

The Full-preservation gate is1000/1000 raw strings and1000/1000 token tensors. The sentence parser
is unchanged. Raw Detail extends beyond the already-packed Full in
5.794058% of eligible audit cases;
raw-n validity differs in11 of100k samples from old visible-n validity.
This follows the requested all-raw-detail construction and is explicit: Detail⊆raw caption does
not guarantee Detail⊆post-budget Full tokens. Inclusion mathematics remains unchanged.
DATA_SAMPLING_AUDIT.md/SAMPLING_AUDIT.json contain all requested quantiles and coverage definitions.

## Correctness, matching and unchanged mathematics

61 sampling/RandomK/model/hyperparameter/scheduler regression tests passed before training.
No model source changed relative to14653c92c6da9d552a2b624ab169eaaa275cdde8.
Optimizer/scheduler/RNG helper ASTs are unchanged; no View-Relation/RMask/SentenceDrop code is added.
Balanced architecture, independent visual blocks, shared pool, Hard-ST pair masks, F/S/D symmetric CE,
10/3 alignment, (ΩF+2ΩS+2ΩD)/3 sparsity and200-update inclusion ramp are unchanged.

All500 updates ×4 ranks match baseline sample-ID/Full/reference-stream SHA256 values, checked during
the run. A nonuniform-image RNG test verifies identical image augmentation after the same RNG state.
Shared step0 SHA256:`54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6`; optimizer initially empty.
The baseline full/bare hashes were verified and its metrics reused without retraining.
Formal training starts at0 with resume=None, not from smoke or a trained checkpoint.

## Resources and diagnostics

4×A10080GB,256/rank,full1024 image candidates, accumulation1,workers8, encoder checkpoint ON,
pair checkpoint OFF,image_chunk128,text_chunk128; BF16 encoders and unchanged FP32 masks/scores/losses.
Slowest-rank normal-cycle mean2.096900s, median2.070134s,
P952.217584s,max2.345307s; all500 mean including startup
2.123989s.
Max peak allocated27.759892GiB, reserved28.447266GiB.
The ≤3s mean/≤65GiB resource gate passed. Warmup5 and checkpoint writes are recorded separately.
Loss/gradients are finite and final cross-rank parameter difference0. Step1/100/200/500 and last50
record actual F/S/D lengths, validity/truncation, mask keep/IoU and original loss/gate diagnostics.
Runtime O/E field names are translated to Summary/Detail only in reports; their mathematical roles do not change.

## Strict native export and complete Recall

Full checkpoint:`/root/lk_projects/SAID-nest-clip-v1/balanced_summary_detail_500_v1/step500/step000500.pt`; SHA256:`d5071fde3899d77462af097efa1ad505b2d34ecd7e0416b835274160e1760cf1`.
Bare student:`/root/lk_projects/SAID-nest-clip-v1/balanced_summary_detail_500_v1/step500/student_step500.pt`; SHA256:`a6844c72b95bee9633cbc62dd9100415f2e047fd9a9234bea02a2ede8e028513`.
Strict reconstruction/export matches training native image/text embeddings with max_abs0;
all optimizer state steps are500. Native inference uses full-caption text and unmasked native image
embeddings only. No Summary/Detail, training masks/gates, reranking or ensemble enter evaluation.

Frozen COCO5000/25000,Urban1000/1000,Flickr test1K1000/5000,DOCCI5000/5000,
Long-DCI reconstructed7602/7602 (SHA8890a2be15e2b64c142f9a1e39224d34b6ce92fc17ffb6099e14942cc8161c4b).
No DCI Full or alternative split runs.

| Model | Dataset | Direction | R@1 % | R@5 % | R@10 % |
| --- | --- | --- | --- | --- | --- |
| Matched RandomK | COCO | I2T | 59.960000 | 81.820000 | 88.680000 |
| Matched RandomK | COCO | T2I | 40.924000 | 66.660000 | 76.268000 |
| Matched RandomK | Urban-1k | I2T | 89.000005 | 98.200005 | 99.400002 |
| Matched RandomK | Urban-1k | T2I | 87.900007 | 98.200005 | 99.100006 |
| Matched RandomK | Flickr30k-test1k | I2T | 86.200000 | 97.200000 | 98.800000 |
| Matched RandomK | Flickr30k-test1k | T2I | 70.300000 | 90.520000 | 94.660000 |
| Matched RandomK | DOCCI | I2T | 76.320000 | 94.540000 | 97.560000 |
| Matched RandomK | DOCCI | T2I | 76.140000 | 94.480000 | 97.520000 |
| Matched RandomK | Long-DCI | I2T | 55.393318 | 74.875033 | 81.057616 |
| Matched RandomK | Long-DCI | T2I | 56.866614 | 76.229939 | 82.044199 |
| Summary–Detail | COCO | I2T | 59.920000 | 82.240000 | 89.120000 |
| Summary–Detail | COCO | T2I | 41.288000 | 66.824000 | 76.876000 |
| Summary–Detail | Urban-1k | I2T | 89.000005 | 97.800004 | 99.400002 |
| Summary–Detail | Urban-1k | T2I | 86.200005 | 97.900003 | 98.900002 |
| Summary–Detail | Flickr30k-test1k | I2T | 86.800000 | 97.600000 | 99.200000 |
| Summary–Detail | Flickr30k-test1k | T2I | 70.620000 | 90.940000 | 94.960000 |
| Summary–Detail | DOCCI | I2T | 75.880000 | 94.860000 | 97.880000 |
| Summary–Detail | DOCCI | T2I | 75.200000 | 94.200000 | 97.140000 |
| Summary–Detail | Long-DCI | I2T | 55.169692 | 74.177848 | 79.992107 |
| Summary–Detail | Long-DCI | T2I | 53.972639 | 74.112076 | 80.110497 |

Config and exact commands are in config.json/commands/. Evidence includes CPU correctness tests,
preflight, resource/stream matching, sampling and strict export; raw/ retains every native Recall.
Large checkpoints, raw data, caches and indexes remain on the evaluation server.
Full4868 confirmation is a future explicit decision, not started by this experiment.
