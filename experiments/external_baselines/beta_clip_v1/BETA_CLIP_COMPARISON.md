# β-CLIP CE official checkpoint versus SAID Balanced B/16

**CE complete; BCE checkpoint unavailable.** The CE file transferred from the user cloud share
strictly loads the fine-tuned B/16 model and reproduces published Urban CLS 88.60/89.00% (I2T/T2I).
Native adapter embeddings and rankings match the official CLS path exactly on identical inputs.
All five SAID frozen protocols have now been evaluated with query-independent official CLS/EOS.
SAID wins 10/10 directional R1 comparisons; CE wins 0/10.
Score5 difference (SAID−β) is +11.017828 percentage points.

## Unified SAID-protocol re-evaluation (fair main result)

| Dataset | β-CLIP CE CLS I2T % | SAID I2T % | Δ SAID−β pp | β-CLIP CE CLS T2I % | SAID T2I % | Δ SAID−β pp |
| --- | --- | --- | --- | --- | --- | --- |
| COCO | 45.580000 | 61.700000 | +16.120000 | 33.196000 | 42.220000 | +9.024000 |
| Urban-1k | 88.600004 | 92.100006 | +3.500003 | 89.000005 | 91.100007 | +2.100003 |
| Flickr30k-test1k | 74.500000 | 88.900000 | +14.400000 | 64.000000 | 72.440000 | +8.440000 |
| DOCCI | 67.980000 | 78.760000 | +10.780000 | 67.920000 | 79.980000 | +12.060000 |
| Long-DCI | 41.278611 | 59.668508 | +18.389897 | 45.448566 | 60.812944 | +15.364378 |

| Metric | β-CLIP CE CLS % | SAID % | Δ SAID−β pp |
| --- | --- | --- | --- |
| Score5_R1 | 61.750319 | 72.768147 | +11.017828 |
| J_long3 | 66.704531 | 77.070244 | +10.365713 |
| J_long | 78.375002 | 85.485003 | +7.110001 |

Aggregates use unrounded fractions: Score5_R1 averages ten directional R1 values;
J_long3 averages Urban/DOCCI/Long-DCI's six directions; J_long averages Urban/DOCCI's four.
All differences use **SAID minus β-CLIP**. TCI is excluded.

## Published reference and official checkpoint reproduction

Published CE I2T/T2I 88.6/89.0%; published BCE 92.3/91.8%, from the
[pinned official README](https://github.com/fzohra/B-CLIP/blob/7be4476f84654b0febe7224d5788868d61ecae8b/README.md).
These are reference values, not five-dataset SAID-protocol results. Measured CE:

| Checkpoint | CLS I2T % | CLS T2I % | TCI I2T % | TCI T2I % |
| --- | --- | --- | --- | --- |
| CE (measured) | 88.60 | 89.00 | 85.40 | 95.50 |
| BCE (checkpoint absent) | N/A | N/A | N/A | N/A |

The published CE pair matches measured CLS, **REPRODUCED** (0 pp delta).
Measured TCI 85.40/95.50% uses the untouched official evaluator's EOS conditioning settings;
no separate TCI published reference is asserted. BCE is unavailable, not assigned its published
numbers as measured results. CE was completed without waiting for BCE.

## Provenance, loading and actual saved configuration

Source: official author-provided Google Drive checkpoint, ID `1nNbYLlU_1VcbEioLL7yX_rzARAzxD6q_`;
user manually downloaded it, transferred through USTC cloud disk to the evaluation server.
Filename `ce_checkpoint_10.pt`; size 1835285659 bytes; epoch 10;
SHA256 `5d8c007f8d9512ecbf2c4a90d44b339d0241b666f8d6c8aa7615ab0ed904f224`. Official source pinned to `7be4476f84654b0febe7224d5788868d61ecae8b`, clean checkout.

Actual args: `CLIP_VITB16_OPENAI`, context 248, β=0.5, max_concepts=30,
`fg_loss_fn=cls+tcil`, `text_conditioning_mode=attn_pooling_mlp`, EOS/concepts/conditioned patches enabled.
Saved last-block and intermediate-block flags are both false; the official mixed-block CLS path is used.
Checkpoint loss mode is legacy `1_k_positives` with `use_softmax_for_multi_positives=True`,
where the current CE training script uses `k_positives_ce`. This difference is recorded without
rewriting saved args or running a training loss. CE identity combines saved softmax configuration,
model/conditioner structure, declared official transfer provenance and exact Urban CLS reproduction.

Loading uses official state conversion and positional setup plus only removal of leading `module.`.
362 state tensors strictly load, missing_keys=[], unexpected_keys=[], every loaded tensor equals
the converted checkpoint. Fine-tuned vision/text towers and all 11 conditioner tensors are loaded.
Model construction skips a redundant timm pretrained download; the strict full-state equality check
proves no temporary initializer weight survives. No substitute model or partial load is used.

## Protocol and limitations

COCO: 5000 images/25000 captions, sorted image IDs, first five source-order captions,
five positives/image, canonical similarity_chunk=512 and frozen row-wise argsort semantics.
Urban: 1000/1000 diagonal, raw captions equal to official captions (0 differences).
Flickr: frozen test1K manifest, 1000/5000. DOCCI: frozen 5000/5000.
Long-DCI: exact reconstructed 7602/7602 manifest, SHA256
`8890a2be15e2b64c142f9a1e39224d34b6ce92fc17ffb6099e14942cc8161c4b`;
no DCI Full or paper DCI number is used. Frozen metric functions and positive mappings are reused;
dataset records, caption fingerprints and metric-source SHA256 values are in the raw JSON.

Official `SimpleTokenizer(context_length=248)` is used on unchanged SAID caption strings.
Its original truncation behavior (including no forced EOT replacement for overlong captions)
is retained. Image transform: RGB, 224 bicubic Resize/CenterCrop, OpenAI CLIP normalization.
Precision: FP32/no autocast; matmul TF32 off, cuDNN convolution TF32 on to match official defaults.
The initial Urban gate detected a one-query difference when convolution TF32 was off; this was
diagnosed and resolved before the remaining datasets, without changing model/data/evaluator math.
Native evaluation has zero conditioner forward calls; no concepts, reranking, ensemble or augmentation.

SAID reference is reused: Balanced-Stack-Patch B/16, fusion_lr=2e-4, inclusion_max=1,
view_weights=[1,1,1], sparsity_scale=1, four epochs/4868 updates, native inference.
Full checkpoint SHA256 `42901d24ae3b0e0147213fd45f5d3f663d9193d17b690536957625901e75500c`;
bare SHA256 `f36438934947ed7f1523fe87e877a0e62dd807dc8a7c89c751eac537e0ab64bb`. SAID was not retrained or modified.
CE is the primary closer loss-family comparison: SAID F/P/R use softmax CE alignment, while
β-CLIP CE uses global CE and multi-positive fine-grained CE; their losses are not identical.
This compares released checkpoints with the same raw test protocols and native inference,
not equal training compute: β-CLIP saved training budget is 10 epochs versus SAID's four.

Official L/14 architecture support: yes. Separate released fine-tuned L/14 checkpoint found:
no within pinned README/download-script scope. This transferred checkpoint is B/16; BCE and L/14
fine-tuned weights are not claimed verified.

## Complete Recall@1/5/10

| Model | Dataset | Direction | R@1 % | R@5 % | R@10 % |
| --- | --- | --- | --- | --- | --- |
| β-CLIP CE CLS | COCO | I2T | 45.580000 | 74.060000 | 84.280000 |
| β-CLIP CE CLS | COCO | T2I | 33.196000 | 59.392000 | 70.796000 |
| β-CLIP CE CLS | Urban-1k | I2T | 88.600004 | 98.300004 | 99.500006 |
| β-CLIP CE CLS | Urban-1k | T2I | 89.000005 | 98.300004 | 99.100006 |
| β-CLIP CE CLS | Flickr30k-test1k | I2T | 74.500000 | 95.700000 | 98.600000 |
| β-CLIP CE CLS | Flickr30k-test1k | T2I | 64.000000 | 87.920000 | 93.120000 |
| β-CLIP CE CLS | DOCCI | I2T | 67.980000 | 88.820000 | 92.620000 |
| β-CLIP CE CLS | DOCCI | T2I | 67.920000 | 88.020000 | 91.960000 |
| β-CLIP CE CLS | Long-DCI | I2T | 41.278611 | 62.996580 | 69.586951 |
| β-CLIP CE CLS | Long-DCI | T2I | 45.448566 | 62.365167 | 68.021573 |
| SAID Balanced B/16 | COCO | I2T | 61.700000 | 84.100000 | 90.300000 |
| SAID Balanced B/16 | COCO | T2I | 42.220000 | 68.032000 | 77.720000 |
| SAID Balanced B/16 | Urban-1k | I2T | 92.100006 | 98.400003 | 99.300003 |
| SAID Balanced B/16 | Urban-1k | T2I | 91.100007 | 98.700005 | 99.200004 |
| SAID Balanced B/16 | Flickr30k-test1k | I2T | 88.900000 | 98.300000 | 99.400000 |
| SAID Balanced B/16 | Flickr30k-test1k | T2I | 72.440000 | 91.540000 | 95.500000 |
| SAID Balanced B/16 | DOCCI | I2T | 78.760000 | 95.720000 | 98.260000 |
| SAID Balanced B/16 | DOCCI | T2I | 79.980000 | 95.880000 | 98.200000 |
| SAID Balanced B/16 | Long-DCI | I2T | 59.668508 | 78.295185 | 83.793738 |
| SAID Balanced B/16 | Long-DCI | T2I | 60.812944 | 78.624046 | 84.109445 |

## Evidence and commands

[CHECKPOINT_INVENTORY.json](CHECKPOINT_INVENTORY.json),
[OFFICIAL_URBAN_REPRO_CE.json](OFFICIAL_URBAN_REPRO_CE.json),
[NATIVE_CLS_CONSISTENCY.md](NATIVE_CLS_CONSISTENCY.md),
[CE_SAID_PROTOCOL_RESULTS.json](CE_SAID_PROTOCOL_RESULTS.json),
[environment.json](environment.json), `raw/ce/`, `raw_logs/`, and `commands/`.
Cloud passwords, signed URLs, weights, datasets, cached embeddings and environments are excluded.
Earlier download failures remain historical evidence and do not describe current CE availability.
