"""Execution-only checkpoint and pair kernels; preserve FP32 Hard-ST derivatives."""
import types

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from model.nested_semantic_mask import _checkpoint_blocks


def _plain_blocks(self, x):
    return self.resblocks(x)


def _partial_blocks(self, x):
    segment_size = self.runtime_segment_size
    blocks = tuple(self.resblocks)
    for index, start in enumerate(range(0, len(blocks), segment_size)):
        selected = blocks[start:start+segment_size]
        def segment(value, selected=selected):
            for block in selected:
                value = block(value)
            return value
        # Alternate checkpointed segments and ordinary segments. No blocks omitted.
        x = checkpoint(segment, x, use_reentrant=False) if self.training and index % 2 == 0 else segment(x)
    return x


def configure_checkpointing(clip, strategy):
    if strategy not in ('full', 'none', 'partial2', 'partial3'):
        raise ValueError(strategy)
    for transformer in (clip.visual.transformer, clip.transformer):
        if strategy.startswith('partial'):
            transformer.runtime_segment_size = int(strategy[-1])
        implementation = (_checkpoint_blocks if strategy == 'full' else
                          _plain_blocks if strategy == 'none' else _partial_blocks)
        transformer.forward = types.MethodType(implementation, transformer)


def reduced_masked_score(images, normalized_text, mask):
    masked = images[None].float() * mask
    numerator = (masked * normalized_text[:, None]).sum(-1)
    # Clamp BEFORE sqrt to retain finite gradients when all channels are closed.
    # Keep the square: replacing m*m by m would corrupt the Hard-ST derivative.
    denominator = masked.square().sum(-1).clamp_min(1e-12).sqrt()
    return 100 * numerator / denominator


def canonical_projection(values, weight, chunk=128):
    # The original128-row GEMMs can differ by ULPs from one tall GEMM. Preserve
    # those shapes to avoid changing a thresholded Hard-ST mask near zero.
    return torch.cat([F.linear(part, weight) for part in values.split(chunk)])


class CachedLinearOriginalBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cached, values, weight):
        ctx.save_for_backward(values,weight)
        return cached

    @staticmethod
    def backward(ctx, gradient):
        values,weight=ctx.saved_tensors
        with torch.enable_grad():
            local_values=values.detach().requires_grad_(True)
            local_weight=weight.detach().requires_grad_(True)
            output=F.linear(local_values,local_weight)
        value_grad,weight_grad=torch.autograd.grad(output,(local_values,local_weight),gradient)
        return None,value_grad,weight_grad


def cached_linear_original_backward(cached, values, weight):
    return CachedLinearOriginalBackward.apply(cached,values,weight)


class CachedNormalizationOriginalBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cached, values):
        ctx.save_for_backward(values)
        return cached

    @staticmethod
    def backward(ctx, gradient):
        (values,)=ctx.saved_tensors
        with torch.enable_grad():
            local=values.detach().requires_grad_(True)
            normalized=F.normalize(local.float(),dim=-1,eps=1e-6)
        (value_grad,)=torch.autograd.grad(normalized,local,gradient)
        return None,value_grad


def cached_normalization_original_backward(cached, values):
    return CachedNormalizationOriginalBackward.apply(cached,values)


class FusedTextViewBackward(torch.autograd.Function):
    """One batched encoder forward, original per-view BF16 weight-gradient rounding.

    This is explicit text rematerialization. It must not be called a no-checkpoint
    execution path, even when the image encoder uses no block checkpointing.
    """
    @staticmethod
    def forward(ctx, clip, tokens, view_batch, *parameters):
        ctx.clip,ctx.view_batch,ctx.parameters = clip,view_batch,parameters
        ctx.save_for_backward(tokens)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=tokens.is_cuda):
            text,hidden=clip.encode_text(tokens,return_full=True)
        ctx.mark_non_differentiable(hidden)
        return text.float(),hidden

    @staticmethod
    def backward(ctx, grad_text, grad_hidden):
        (tokens,)=ctx.saved_tensors
        token_views=tokens.split(ctx.view_batch);gradient_views=grad_text.split(ctx.view_batch)
        gradients=[None]*len(ctx.parameters)
        transformer=getattr(ctx.clip,'transformer',None)
        original=transformer.forward if transformer is not None else None
        if transformer is not None:transformer.forward=types.MethodType(_plain_blocks,transformer)
        try:
            for token_view,gradient_view in reversed(list(zip(token_views,gradient_views))):
                with torch.enable_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=tokens.is_cuda):
                    text,_=ctx.clip.encode_text(token_view,return_full=True)
                    text=text.float()
                partial=torch.autograd.grad(text,ctx.parameters,gradient_view,allow_unused=True)
                for index,value in enumerate(partial):
                    if value is not None:
                        gradients[index]=value if gradients[index] is None else gradients[index]+value
        finally:
            if transformer is not None:transformer.forward=original
        return (None,None,None,*gradients)


def encode_fused_with_original_weight_gradients(clip, tokens, view_batch):
    parameters=tuple(p for name,p in clip.named_parameters() if p.requires_grad
        and not name.startswith(('visual.','mask_net.')) and name!='logit_scale')
    return FusedTextViewBackward.apply(clip,tokens,view_batch,*parameters)


def balanced_tile(images, normalized_text, visual_summary, text_summary, visual_gate, text_gate,
                  reduced=True):
    gate = (text_gate[:, None] + visual_gate[None]).sigmoid()
    logits = gate * text_summary[:, None] + (1-gate) * visual_summary[None]
    probability = logits.sigmoid()
    mask = (probability >= .5).float() - probability.detach() + probability
    score = (reduced_masked_score(images, normalized_text, mask) if reduced else
             100*(F.normalize(images[None].float()*mask,dim=-1,eps=1e-6)*normalized_text[:,None]).sum(-1))
    return score, logits, probability
