# Native inference audit

The pinned local_clip.model.CustomCLIP.forward forces simple_forward whenever model.training
is false. In the simple path it calls each independent vision/text tower and L2 normalizes
its global features. Official retrieval calls model with simple_forward=True, gathers full
image/text feature banks, then constructs a full normalized dot-product matrix multiplied
by the positive scalar exp(logit_scale). No query-dependent image encoder or reranking runs.

Training may encode two captions/image and use the batch-wise image PCA branch with dimension32
for the sampled-caption alignment. Those PCA and short-caption branches do not execute in eval.
The augmentation in training.data.CsvDatasetMultiCap removes the first sentence when remaining
sentences exist, samples a uniformly chosen nonempty number without replacement (with shuffled
order), joins the sampled sentences, tokenizes sampled plus full text, and shifts PAD positions
after SOT. None of these operations are applied to official or unified evaluation captions.

DeBiasCLIPNativeAdapter calls official encode_image/encode_text with normalize=True,
output_tokens=False and the official tokenizer. It does not convert weights to SAID/LongCLIP,
call PCA, average captions, shift padding, sample sentences or condition image features on text.
Both Phase A and Phase B use the same pinned official checkpoint-loading behavior.

The current factory.load_checkpoint may replace a saved text.positional_embedding_res with
text.positional_embedding for already-stretched248-position custom-format states. This branch
is recorded and checked against the actual loaded tensors. It is not patched to improve scores.
Strict state-key compatibility must be distinguished from equality to every raw checkpoint tensor.
The final inventory/evaluation logs will report both tests separately.


Completed measured audit: primary AMP encoders return normalized FP32 image/text features; both norm means1.0. Same-input adapter differences and direct-score differences are0. Official loader raw positional overwrite is confirmed. Unified PCA calls0. AMP versus native FP32 changes one Urban T2I top1 query,930→931 correct; no image/caption/checkpoint change. Encoder batch composition was aligned for the consistency check and frozen candidate order restored, reducing feature max_abs to5.96e-8 while all six recalls remained identical. Same-input and frozen-ranking evidence is retained in raw_logs/.
