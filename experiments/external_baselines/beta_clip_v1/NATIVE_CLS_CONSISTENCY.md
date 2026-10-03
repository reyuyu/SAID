# Native CLS consistency: PASS

Official Urban CLS and adapter on identical inputs: image/text raw max absolute difference 0,
scaled scores max absolute difference 0, top-1 mismatches 0/1000 in both directions,
and scaled full-ranking query mismatches 0/1000 in both directions across four ranks.
R1: I2T 88.60%, T2I 89.00%. Raw evidence: [OFFICIAL_URBAN_REPRO_CE.json](OFFICIAL_URBAN_REPRO_CE.json).

The adapter extracts CLS from official `encode_image_by_block` with the saved last/intermediate-block flags,
then uses official `encode_text` for EOS. Both towers are independent. The standard `encode_image`
implementation differs by at most 3.576279e-6 on tested raw CLS values; the adapter follows the exact
official CLS path. No conditioned representation or conditioner forward runs in native evaluation
(a raising forward hook verifies zero calls).

Frozen SAID Urban raw captions: 1000; different_count=0; both caption-list SHA256 values:
`473427e3f0fe30f38e0dbec4fd7f6701d0bbe27ad29ada57bd47883dc1772abb`.
Final SAID frozen metric R1: I2T 88.600004%, T2I 89.000005% (float32 mean).

During the unified-harness gate, disabling cuDNN TF32 convolution shifted one T2I query (89.0→88.9%).
The official runtime defaults to convolution TF32 enabled. Matching that default resolves the difference.
Matmul TF32 remains disabled, weights/activations FP32, no autocast. No data, tokenizer, loss or model
math was changed. Details: [raw_logs/urban-gate-diagnosis.json](raw_logs/urban-gate-diagnosis.json).
This diagnostic failure occurred before the remaining four datasets ran.

Checkpoint: `5d8c007f8d9512ecbf2c4a90d44b339d0241b666f8d6c8aa7615ab0ed904f224`. Strict load: 362/362 tensors, empty missing/unexpected keys,
all loaded tensors exactly equal to the converted fine-tuned checkpoint. Avoiding timm's redundant
pretrained network bootstrap retains no temporary initializer tensors. Official sources are unmodified.
