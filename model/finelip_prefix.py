"""FP0: independently implemented FineLIP-method prefix baseline.

Method reference: tiiuae/FineLIP, 2118312c9d640c71904379e90129649a46e6f2dd.
No upstream source import, mask, routing, reconstruction, or global contrastive loss.
"""
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather


def live_gather(x):
    return torch.cat(all_gather(x.contiguous()), 0) if dist.is_initialized() else x


def metadata_gather(x):
    if not dist.is_initialized():
        return x
    parts = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, x.contiguous())
    return torch.cat(parts, 0)


def eos_positions(ids, eos_id=49407):
    is_eos = ids == eos_id
    if not bool((is_eos.sum(-1) == 1).all()):
        raise ValueError('each input must contain exactly one EOS')
    positions = is_eos.long().argmax(-1)
    if not bool((positions > 0).all()):
        raise ValueError('EOS must follow SOT')
    return positions


class PrefixAggregation(nn.Module):
    def __init__(self, dim=512, slots=39):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.network = nn.Sequential(nn.Linear(dim, int(dim * .2)), nn.GELU(),
                                     nn.Linear(int(dim * .2), slots))
        self.scale = nn.Parameter(torch.ones(1, 1, 1))

    def forward(self, x, valid=None):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.to(self.scale.dtype)
            logits = self.network(self.norm(x)).transpose(1, 2) * self.scale
            if valid is not None:
                logits = logits.masked_fill(~valid[:, None], -torch.inf)
            weights = logits.softmax(-1)
            return weights @ x, weights


def scores(visual, text, matched=False):
    """Inputs are normalized output tokens. One cosine grid, both directions."""
    c = (torch.bmm(visual, text.transpose(1, 2)) if matched else
         torch.einsum('ipd,jqd->ijpq', visual, text))
    c = F.leaky_relu(c, negative_slope=.1)
    return c.max(-1).values.mean(-1) + c.max(-2).values.mean(-1)


def legal_pairs(local_ids, global_ids, local_tokens, global_tokens):
    # Token ID 0 is valid content. Callers canonicalize only positions after EOS.
    identical = (local_tokens[:, None] == global_tokens[None]).all(-1)
    return (local_ids[:, None] != global_ids[None]) & ~identical


def hinge_sums(matrix, row_positive, column_positive, legal):
    a = F.relu(.2 + matrix - row_positive[:, None]) * legal
    b = F.relu(.2 + matrix - column_positive[None]) * legal
    return a.sum(), b.sum(), (a > 0).sum(), (b > 0).sum()


def distributed_objective(visual, text, ids, tokens, chunk_text=32):
    world = dist.get_world_size() if dist.is_initialized() else 1
    candidates = live_gather(text)
    positive = scores(visual, text, matched=True)
    global_positive = live_gather(positive)
    legal = legal_pairs(ids, metadata_gather(ids), tokens, metadata_gather(tokens))
    # Include collective outputs even if there are no valid negative pairs.
    zero = (visual.sum() + candidates.sum() + global_positive.sum()) * 0
    left, right = zero, zero
    neg, active_i, active_t = zero.detach(), zero.detach(), zero.detach()
    from torch.utils.checkpoint import checkpoint
    for j in range(0, len(candidates), chunk_text):
        sl = slice(j, j + chunk_text)
        value = checkpoint(scores, visual, candidates[sl], use_reentrant=False) if torch.is_grad_enabled() else scores(visual, candidates[sl])
        a, b, ai, at = hinge_sums(value, positive, global_positive[sl], legal[:, sl])
        left, right = left + a, right + b
        neg = neg + (value.detach() * legal[:, sl]).sum()
        active_i, active_t = active_i + ai, active_t + at
    stats = torch.stack([left.detach(), right.detach(), legal.sum(), active_i, active_t,
                         positive.detach().sum(), positive.new_tensor(len(positive)), neg])
    if dist.is_initialized():
        dist.all_reduce(stats)
    return world * (left + right), stats


class FineLIPPrefix(nn.Module):
    def __init__(self, clip, seed=0, amp=True):
        super().__init__()
        self.clip, self.amp = clip, amp
        for name, parameter in clip.named_parameters():
            if name == 'logit_scale' or name.startswith('mask_net.'):
                parameter.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.image_aggregator = PrefixAggregation()
            self.text_aggregator = PrefixAggregation()

    def encode(self, images, token_ids):
        eos = eos_positions(token_ids)
        with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16, enabled=self.amp):
            image_cls, patches = self.clip.encode_image_with_patches(images, use_checkpoint=False)
            text_eos, hidden = self.clip.encode_text(token_ids, return_full=True)
            projected_text = hidden @ self.clip.text_projection
        with torch.autocast(device_type=images.device.type, enabled=False):
            valid = torch.arange(token_ids.shape[1], device=token_ids.device)[None] < eos[:, None]
            image_slots, iw = self.image_aggregator(patches)
            text_slots, tw = self.text_aggregator(projected_text, valid)
            v = F.normalize(torch.cat([image_cls.float()[:, None], image_slots], 1), dim=-1)
            t = F.normalize(torch.cat([text_slots, text_eos.float()[:, None]], 1), dim=-1)
        return v, t, iw, tw, image_cls, text_eos

    def forward(self, images, tokens, image_ids, diagnostics=False):
        v, t, iw, tw, native_i, native_t = self.encode(images, tokens)
        with torch.autocast(device_type=images.device.type, enabled=False):
            eos = eos_positions(tokens)
            canonical = tokens.masked_fill(torch.arange(tokens.shape[1], device=tokens.device)[None] > eos[:, None], -1)
            loss, stats = distributed_objective(v, t, image_ids, canonical)
            extra = {}
            if diagnostics:
                for name, z, weights in [('image', v[:, 1:], iw), ('text', t[:, :-1], tw)]:
                    n = z.shape[1]
                    extra[name + '_slot_cos'] = ((z.sum(1).square().sum(-1) - z.square().sum((1, 2))) / (n * (n - 1))).mean().detach()
                    extra[name + '_aggregation_entropy'] = -(weights * weights.clamp_min(1e-30).log()).sum(-1).mean().detach()
                extra['native_cls_positive_cos'] = F.cosine_similarity(native_i.float(), native_t.float()).mean().detach()
        return loss, stats, extra
