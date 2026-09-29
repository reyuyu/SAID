"""NEST-CLIP v1: one backbone/mask, three independently encoded text views."""
import types

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .said_cls_cvssl import said_mask_from_hidden


def world_rank():
    return (dist.get_world_size(), dist.get_rank()) if dist.is_initialized() else (1, 0)


def gather(x, differentiable=True):
    if world_rank()[0] == 1:
        return x
    if differentiable:
        # Flatten ensures Gloo's backward receives contiguous gradients too.
        return torch.cat(dist_nn.all_gather(x.contiguous().flatten())).reshape(-1, *x.shape[1:])
    parts = [torch.empty_like(x) for _ in range(world_rank()[0])]
    dist.all_gather(parts, x.contiguous())
    return torch.cat(parts)


def global_sum(x):
    value = x.detach().clone()
    if world_rank()[0] > 1:
        dist.all_reduce(value)
    return value


def masked_scores(images, texts, masks, chunk=64):
    """Exact pairwise SIDM scores, bounded tiles; no B x G x D allocation.

    ||z_i*m_j||^2 = z_i^2 @ m_j^2. Clamp before sqrt has a finite
    derivative for all-closed masks and implements normalize(eps=1e-6).
    """
    texts = F.normalize(texts.float(), dim=-1, eps=1e-6)
    images, masks = images.float(), masks.float()
    return torch.cat([
        100 * (z @ (masks * texts).T) /
        (z.square() @ masks.square().T).clamp_min(1e-12).sqrt()
        for z in images.split(chunk)
    ])


def view_terms(z, t, m, valid, valid_global, z_global=None, score_chunk=64):
    """Local query CE sums, original rank-order columns, W/N scaling once."""
    world, rank = world_rank()
    n = int(valid_global.sum())
    if n < 1:
        raise ValueError('An enabled view needs at least one candidate')
    zg = gather(z) if z_global is None else z_global
    tg, mg = gather(t), gather(m)
    labels = rank * z.shape[0] + torch.arange(len(z), device=z.device)
    qi = masked_scores(z, tg, mg, score_chunk)
    qt = masked_scores(zg, t, m, score_chunk).T
    # Every rank traverses ALL three differentiable gather paths, including
    # zero-query ranks. Never form zero from logits masked with -inf.
    zero = (zg.sum() + tg.sum() + mg.sum() + qi.sum() + qt.sum()) * 0
    if bool(valid.any()):
        ci = F.cross_entropy(qi[valid].masked_fill(~valid_global[None], -torch.inf),
                             labels[valid], reduction='sum')
        ct = F.cross_entropy(qt[valid].masked_fill(~valid_global[None], -torch.inf),
                             labels[valid], reduction='sum')
    else:
        ci = ct = zero
    sparse = m[valid].abs().mean(-1).sum()
    keep = m.detach()[valid]
    sums = torch.stack([ci.detach(), ct.detach(), sparse.detach(),
                        (keep == 1).all(-1).float().sum(),
                        (keep == 0).all(-1).float().sum()])
    logs = global_sum(sums) / n
    return (world / n) * (ci + ct + zero), (world / n) * sparse, {
        'i2t': logs[0], 't2i': logs[1], 'sparse': logs[2],
        'keep_ratio': logs[2], 'all_open': logs[3], 'all_closed': logs[4],
        'candidates': n}


def native_scores(images, texts):
    """Unmasked FP32 cosine with the same fixed scale and epsilon."""
    with torch.autocast(images.device.type, enabled=False):
        return 100 * (F.normalize(images.float(), dim=-1, eps=1e-6) @
                      F.normalize(texts.float(), dim=-1, eps=1e-6).T)


def native_full_terms(z, tf, z_global=None):
    """Full-view local query sums and global differentiable candidates.

    Reuses the encoded image/text features and the existing image bank. No
    encoder or mask is called here. Gather backward and DDP averaging account
    for exactly one W/N scaling, identical to the masked task.
    """
    world, rank = world_rank()
    zg = gather(z) if z_global is None else z_global
    tg = gather(tf)
    n = len(zg)
    labels = rank * len(z) + torch.arange(len(z), device=z.device)
    with torch.autocast(z.device.type, enabled=False):
        ci = F.cross_entropy(native_scores(z, tg), labels, reduction='sum')
        ct = F.cross_entropy(native_scores(zg, tf).T, labels, reduction='sum')
    mean = global_sum(torch.stack((ci.detach(), ct.detach()))) / n
    return (world / n) * (ci + ct), {'i2t': mean[0], 't2i': mean[1], 'candidates': n}


def inclusion(pf, po, pe):
    return .5 * (F.relu(po.detach() - pf).mean(-1) +
                 F.relu(pe.detach() - pf).mean(-1))


def inclusion_weight(arm, completed):
    return min(1., completed / 200.) if arm == 'A3' else 0.


def _checkpoint_blocks(self, x):
    for block in self.resblocks:
        x = checkpoint(block, x, use_reentrant=False) if self.training else block(x)
    return x


def enable_encoder_checkpointing(clip):
    # No wrappers/modules or changed state_dict names; original native interfaces.
    for transformer in (clip.visual.transformer, clip.transformer):
        transformer.forward = types.MethodType(_checkpoint_blocks, transformer)


class NestedSemanticMask(nn.Module):
    def __init__(self, clip, arm='A2', checkpoint_encoders=True, score_chunk=64,
                 full_native_mix=0.):
        super().__init__()
        if arm not in ('A2', 'A3'):
            raise ValueError(arm)
        self.clip = clip  # sole owner of the original mask network
        self.arm, self.score_chunk = arm, score_chunk
        if not 0. <= full_native_mix <= 1.:
            raise ValueError('full_native_mix must be in [0,1]')
        self.full_native_mix = float(full_native_mix)
        if checkpoint_encoders:
            enable_encoder_checkpointing(clip)

    def encode_view(self, tokens):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=tokens.is_cuda):
            t, h = self.clip.encode_text(tokens, return_full=True)
        with torch.autocast(tokens.device.type, enabled=False):
            m, p, logits = said_mask_from_hidden(self.clip.mask_net, h.float())
        return t.float(), m, p, logits

    def forward(self, images, tokens_f, tokens_o, tokens_e, valid, completed=0):
        world, _ = world_rank()
        vg = gather(valid, False)
        v = int(vg.sum())  # identical branch on every rank
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=images.is_cuda):
            z = self.clip.encode_image(images).float()
        tf, mf, pf, lf = self.encode_view(tokens_f)
        zg = gather(z)
        all_valid = torch.ones_like(valid)
        af, sf, logs_f = view_terms(z, tf, mf, all_valid, torch.ones_like(vg), zg,
                                    self.score_chunk)
        logs = {'F_' + k: value for k, value in logs_f.items()}
        if self.full_native_mix:
            native, native_logs = native_full_terms(z, tf, zg)
            eta = self.full_native_mix
            # Mix separately computed CE losses, never logits. Sparse/inc
            # coefficients below are intentionally independent of eta.
            af = (1 - eta) * af + eta * native
            logs.update(F_mask_i2t=logs_f['i2t'], F_mask_t2i=logs_f['t2i'],
                        F_native_i2t=native_logs['i2t'], F_native_t2i=native_logs['t2i'],
                        F_native_candidates=native_logs['candidates'], full_native_mix=eta,
                        F_hybrid=(1-eta)*(logs_f['i2t']+logs_f['t2i']) +
                                 eta*(native_logs['i2t']+native_logs['t2i']))
        weight = inclusion_weight(self.arm, completed) if v >= 2 else 0.
        nonfinite = (~torch.isfinite(lf)).sum() + (~torch.isfinite(z)).sum() + (~torch.isfinite(tf)).sum()
        if v >= 2:
            to, mo, po, lo = self.encode_view(tokens_o)
            te, me, pe, le = self.encode_view(tokens_e)
            ao, so, logs_o = view_terms(z, to, mo, valid, vg, zg, self.score_chunk)
            ae, se, logs_e = view_terms(z, te, me, valid, vg, zg, self.score_chunk)
            inc_sum = inclusion(pf, po, pe)[valid].sum()
            inc = world / v * inc_sum
            loss = (10 / 3) * (af + ao + ae) + (sf + 2 * so + 2 * se) / 3 + weight * inc
            hard_violation = .5 * (((mo.detach() > mf.detach()).float().mean(-1)) +
                                    ((me.detach() > mf.detach()).float().mean(-1)))
            overlap = (mo.detach() * me.detach()).sum(-1) / ((mo.detach() + me.detach()) > 0).sum(-1).clamp_min(1)
            extra = global_sum(torch.stack([inc_sum.detach(), hard_violation[valid].sum(),
                                            overlap[valid].sum()])) / v
            logs.update({'O_' + k: x for k, x in logs_o.items()})
            logs.update({'E_' + k: x for k, x in logs_e.items()})
            logs.update(inc=extra[0], hard_inclusion_violation=extra[1], oe_iou=extra[2])
            nonfinite = nonfinite + sum((~torch.isfinite(x)).sum() for x in (lo, le, to, te))
        else:
            loss = 10 * af + sf
            logs.update(inc=0., hard_inclusion_violation=0., oe_iou=0., O_candidates=0, E_candidates=0)
        logs.update(valid_global=v, inc_weight=weight, nonfinite=global_sum(nonfinite),
                    loss=global_sum(loss) / world)
        if self.full_native_mix:
            if v >= 2:
                alignment = (10/3) * (logs['F_hybrid'] + logs['O_i2t'] + logs['O_t2i'] +
                                      logs['E_i2t'] + logs['E_t2i'])
                sparse = (logs['F_sparse'] + 2*logs['O_sparse'] + 2*logs['E_sparse'])/3
            else:
                alignment, sparse = 10*logs['F_hybrid'], logs['F_sparse']
            regularizer = weight*logs['inc']
            rebuilt = alignment + sparse + regularizer
            logs.update(alignment_weighted=alignment, sparse_weighted=sparse,
                        inclusion_weighted=regularizer, loss_reconstructed=rebuilt,
                        loss_reconstruction_abs_error=(rebuilt-logs['loss']).abs())
        # Only the total loss carries an active graph out of the DDP forward.
        return loss, {k: x.detach() if torch.is_tensor(x) else x for k, x in logs.items()}
