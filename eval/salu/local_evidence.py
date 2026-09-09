"""Evaluation-only decomposition of the existing CLIP ViT residual blocks.

The model implementation is untouched. ``extract_stages`` mirrors the current
VisionTransformer and ResidualAttentionBlock equations exactly and returns
768D states before the existing final ``ln_post``/``proj`` diagnostic map.
"""
import torch

STAGES = ('residual', 'after_attention', 'attention_delta', 'mlp_delta')


def decompose_block(block, x):
    """Return x_in, attention delta, x_attn, MLP delta and x_out."""
    x_in = x
    attn_delta = block.attention(block.ln_1(x_in))
    x_attn = x_in + attn_delta
    mlp_delta = block.mlp(block.ln_2(x_attn))
    x_out = x_attn + mlp_delta
    return {'x_in': x_in, 'attention_delta': attn_delta,
            'after_attention': x_attn, 'mlp_delta': mlp_delta,
            'residual': x_out}


def visual_input_sequence(visual, image):
    x = visual.conv1(image.type(visual.conv1.weight.dtype))
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype)[None, None].expand(len(x), 1, -1)
    x = torch.cat([cls, x], dim=1)
    return visual.ln_pre(x + visual.positional_embedding.to(x.dtype))


def extract_stages(visual, image, layers=(3, 6, 9, 11), include_input=False):
    """Return ``{layer: {stage: [B,196,768]}}`` from a normal image tensor."""
    x = visual_input_sequence(visual, image).permute(1, 0, 2)
    requested = set(layers)
    result = {}
    for index, block in enumerate(visual.transformer.resblocks):
        states = decompose_block(block, x)
        x = states['residual']
        if index in requested:
            result[index] = {stage: value.permute(1, 0, 2)[:, 1:].contiguous()
                             for stage, value in states.items() if stage in STAGES or include_input}
    missing = requested-set(result)
    if missing:
        raise ValueError('requested transformer blocks unavailable: %s' % sorted(missing))
    return result


def project_candidate(visual, feature):
    """Reuse final projection diagnostically for any 768D candidate."""
    if visual.proj is None:
        raise ValueError('visual projection is required for this diagnostic')
    return visual.ln_post(feature) @ visual.proj
