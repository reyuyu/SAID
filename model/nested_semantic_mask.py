"""NEST pair-mask probe: text-only and low-rank image-conditioned gates."""
import hashlib
import types

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


CONDITION_MODES = ('text_only', 'joint_image', 'joint_shuffled_image')


def world_rank():
    return (dist.get_world_size(), dist.get_rank()) if dist.is_initialized() else (1, 0)


def gather(x, differentiable=True):
    if world_rank()[0] == 1:
        return x
    if differentiable:
        return torch.cat(dist_nn.all_gather(x.contiguous().flatten())).reshape(-1, *x.shape[1:])
    parts = [torch.empty_like(x) for _ in range(world_rank()[0])]
    dist.all_gather(parts, x.contiguous())
    return torch.cat(parts)


def global_sum(x):
    value = x.detach().clone()
    if world_rank()[0] > 1:
        dist.all_reduce(value)
    return value


def hard_st(probabilities):
    return (probabilities >= .5).float() - probabilities.detach() + probabilities


def masked_scores(images, texts, masks, chunk=64):
    texts = F.normalize(texts.float(), dim=-1, eps=1e-6)
    images, masks = images.float(), masks.float()
    return torch.cat([
        100 * (z @ (masks * texts).T) /
        (z.square() @ masks.square().T).clamp_min(1e-12).sqrt()
        for z in images.split(chunk)
    ])


class JointMaskAdapter(nn.Module):
    def __init__(self, image_dim=768, text_dim=512, rank=64, output_dim=512):
        super().__init__()
        self.image = nn.Linear(image_dim, rank, bias=False)
        self.text = nn.Linear(text_dim, rank, bias=False)
        self.output = nn.Linear(rank, output_dim, bias=False)
        nn.init.xavier_uniform_(self.image.weight)
        nn.init.xavier_uniform_(self.text.weight)
        nn.init.zeros_(self.output.weight)

    def image_condition(self, hidden):
        return torch.tanh(self.image(hidden.detach().float()))

    def text_condition(self, hidden):
        return torch.tanh(self.text(hidden.detach().float()))

    def delta(self, image_condition, text_condition):
        return self.output(image_condition[:, None] * text_condition[None])


def deterministic_shuffle_indices(size, seed, completed, device=None):
    if size < 2:
        return torch.arange(size, device=device)
    material = f'{int(seed)}:{int(completed)}:image_condition_shuffle'.encode()
    shift = 1 + int.from_bytes(hashlib.sha256(material).digest(), 'big') % (size - 1)
    return (torch.arange(size, device=device) + shift) % size


def pair_mask(base_logits, image_condition, text_condition, adapter=None):
    if adapter is None:
        delta = base_logits.new_zeros((len(image_condition), len(base_logits), base_logits.shape[-1]))
    else:
        delta = adapter.delta(image_condition, text_condition)
    probabilities = torch.sigmoid(base_logits[None] + delta)
    return hard_st(probabilities), probabilities, delta


def _pair_score_block(z, t, image_condition, base_logits, text_condition, adapter,
                      active_pairs):
    masks, probabilities, delta = pair_mask(base_logits, image_condition, text_condition, adapter)
    scores = 100 * (F.normalize(z[:, None].float() * masks, dim=-1, eps=1e-6) *
                    F.normalize(t.float(), dim=-1, eps=1e-6)[None]).sum(-1)
    selected = (probabilities.detach() >= .5)[active_pairs]
    selected_delta = delta.detach()[active_pairs]
    if selected.numel():
        summary = torch.stack((selected.float().sum(),
                               selected.new_tensor(selected.numel(), dtype=torch.float32),
                               selected.all(-1).float().sum(),
                               (~selected).all(-1).float().sum(),
                               selected_delta.abs().sum(),
                               selected_delta.new_tensor(selected_delta.numel(), dtype=torch.float32)))
    else:
        summary = scores.detach().new_zeros(6)
    return scores, summary.detach()


def pair_scores(images, texts, image_condition, base_logits, text_condition, adapter=None,
                image_chunk=32, text_chunk=64, row_valid=None, column_valid=None,
                checkpoint_blocks=False):
    """Pair-conditioned scores in two-dimensional tiles; collectives stay outside."""
    row_valid = (torch.ones(len(images), dtype=torch.bool, device=images.device)
                 if row_valid is None else row_valid)
    column_valid = (torch.ones(len(texts), dtype=torch.bool, device=texts.device)
                    if column_valid is None else column_valid)
    rows, summaries = [], []
    for image_start in range(0, len(images), image_chunk):
        image_stop = min(image_start + image_chunk, len(images))
        block_row = []
        for text_start in range(0, len(texts), text_chunk):
            text_stop = min(text_start + text_chunk, len(texts))
            active = row_valid[image_start:image_stop, None] & column_valid[None, text_start:text_stop]
            args = (images[image_start:image_stop], texts[text_start:text_stop],
                    image_condition[image_start:image_stop], base_logits[text_start:text_stop],
                    text_condition[text_start:text_stop])
            function = lambda *values, active=active: _pair_score_block(
                *values, adapter=adapter, active_pairs=active)
            if checkpoint_blocks and torch.is_grad_enabled():
                scores, summary = checkpoint(function, *args, use_reentrant=False)
            else:
                scores, summary = function(*args)
            block_row.append(scores)
            summaries.append(summary)
        rows.append(torch.cat(block_row, dim=1))
    return torch.cat(rows), torch.stack(summaries).sum(0)


def positive_masks(base_logits, image_condition, text_condition, adapter=None):
    if adapter is None:
        delta = torch.zeros_like(base_logits)
    else:
        delta = adapter.output(image_condition * text_condition)
    probabilities = torch.sigmoid(base_logits + delta)
    return hard_st(probabilities), probabilities, delta


def pair_view_terms(z, t, base_logits, text_condition, image_condition, valid,
                    valid_global, adapter=None, z_global=None, image_condition_global=None,
                    image_chunk=32, text_chunk=64, checkpoint_blocks=False):
    """Local CE queries over the complete global pair matrix and positive-only regularizers."""
    world, rank = world_rank()
    n = int(valid_global.sum())
    if n < 1:
        raise ValueError('An enabled view needs at least one candidate')
    zg = gather(z) if z_global is None else z_global
    ag = gather(image_condition) if image_condition_global is None else image_condition_global
    tg, lg, bg = gather(t), gather(base_logits), gather(text_condition)
    labels = rank * z.shape[0] + torch.arange(len(z), device=z.device)
    qi, pair_stats = pair_scores(z, tg, image_condition, lg, bg, adapter,
                                 image_chunk, text_chunk, valid, valid_global,
                                 checkpoint_blocks)
    qt, _ = pair_scores(zg, t, ag, base_logits, text_condition, adapter,
                        image_chunk, text_chunk, valid_global, valid,
                        checkpoint_blocks)
    qt = qt.T
    zero = (zg.sum() + ag.sum() + tg.sum() + lg.sum() + bg.sum() + qi.sum() + qt.sum()) * 0
    if bool(valid.any()):
        ci = F.cross_entropy(qi[valid].masked_fill(~valid_global[None], -torch.inf),
                             labels[valid], reduction='sum')
        ct = F.cross_entropy(qt[valid].masked_fill(~valid_global[None], -torch.inf),
                             labels[valid], reduction='sum')
    else:
        ci = ct = zero
    positive, probabilities, positive_delta = positive_masks(
        base_logits, image_condition, text_condition, adapter)
    sparse = positive[valid].abs().mean(-1).sum()
    selected = positive.detach()[valid] >= .5
    positive_summary = torch.stack((selected.float().sum(),
                                    selected.new_tensor(selected.numel(), dtype=torch.float32),
                                    selected.all(-1).float().sum(),
                                    (~selected).all(-1).float().sum(),
                                    positive_delta.detach()[valid].abs().sum(),
                                    positive_delta.new_tensor(positive_delta[valid].numel(),
                                                              dtype=torch.float32)))
    totals = global_sum(pair_stats)
    positives = global_sum(positive_summary)
    negatives = totals - positives
    ce_logs = global_sum(torch.stack((ci.detach(), ct.detach(), sparse.detach()))) / n
    ratio = lambda values, numerator, denominator: values[numerator] / values[denominator].clamp_min(1)
    pair_count = lambda values: (values[1] / positive.shape[-1]).clamp_min(1)
    logs = dict(i2t=ce_logs[0], t2i=ce_logs[1], sparse=ce_logs[2],
                keep_ratio=ce_logs[2], all_open=positives[2] / n,
                all_closed=positives[3] / n, candidates=n,
                positive_keep_ratio=ratio(positives, 0, 1),
                negative_keep_ratio=ratio(negatives, 0, 1),
                positive_all_open=positives[2] / n,
                positive_all_closed=positives[3] / n,
                negative_all_open=negatives[2] / pair_count(negatives),
                negative_all_closed=negatives[3] / pair_count(negatives),
                delta_abs_mean=ratio(totals, 4, 5),
                positive_delta_abs_mean=ratio(positives, 4, 5),
                negative_delta_abs_mean=ratio(negatives, 4, 5))
    return ((world / n) * (ci + ct + zero), (world / n) * sparse,
            positive, probabilities, logs)


def view_terms(z, t, m, valid, valid_global, z_global=None, score_chunk=64):
    """Original text-only matrix implementation retained as a regression oracle."""
    world, rank = world_rank()
    n = int(valid_global.sum())
    if n < 1:
        raise ValueError('An enabled view needs at least one candidate')
    zg = gather(z) if z_global is None else z_global
    tg, mg = gather(t), gather(m)
    labels = rank * z.shape[0] + torch.arange(len(z), device=z.device)
    qi = masked_scores(z, tg, mg, score_chunk)
    qt = masked_scores(zg, t, m, score_chunk).T
    zero = (zg.sum() + tg.sum() + mg.sum() + qi.sum() + qt.sum()) * 0
    if bool(valid.any()):
        ci = F.cross_entropy(qi[valid].masked_fill(~valid_global[None], -torch.inf),
                             labels[valid], reduction='sum')
        ct = F.cross_entropy(qt[valid].masked_fill(~valid_global[None], -torch.inf),
                             labels[valid], reduction='sum')
    else:
        ci = ct = zero
    sparse = m[valid].abs().mean(-1).sum()
    logs = global_sum(torch.stack((ci.detach(), ct.detach(), sparse.detach()))) / n
    return ((world / n) * (ci + ct + zero), (world / n) * sparse,
            {'i2t': logs[0], 't2i': logs[1], 'sparse': logs[2]})


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
    for transformer in (clip.visual.transformer, clip.transformer):
        transformer.forward = types.MethodType(_checkpoint_blocks, transformer)


class NestedSemanticMask(nn.Module):
    def __init__(self, clip, arm='A3', checkpoint_encoders=True, image_chunk=32,
                 text_chunk=64, condition_mode='text_only', shuffle_seed=0,
                 checkpoint_pair_blocks=True):
        super().__init__()
        if arm not in ('A2', 'A3') or condition_mode not in CONDITION_MODES:
            raise ValueError((arm, condition_mode))
        self.clip = clip
        self.arm = arm
        self.image_chunk, self.text_chunk = int(image_chunk), int(text_chunk)
        self.condition_mode, self.shuffle_seed = condition_mode, int(shuffle_seed)
        self.checkpoint_pair_blocks = bool(checkpoint_pair_blocks)
        if condition_mode == 'text_only':
            self.joint_adapter = None
        else:
            image_dim = int(clip.visual.proj.shape[0])
            text_dim, output_dim = map(int, clip.text_projection.shape)
            self.joint_adapter = JointMaskAdapter(image_dim, text_dim, 64, output_dim)
        if checkpoint_encoders:
            enable_encoder_checkpointing(clip)

    def encode_view(self, tokens):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=tokens.is_cuda):
            text, hidden = self.clip.encode_text(tokens, return_full=True)
        eot = tokens.argmax(dim=-1)
        text_hidden = hidden[torch.arange(len(tokens), device=tokens.device), eot].float()
        with torch.autocast(tokens.device.type, enabled=False):
            base_logits = self.clip.mask_net(hidden.float().detach())
            condition = (self.joint_adapter.text_condition(text_hidden)
                         if self.joint_adapter is not None else
                         text_hidden.new_zeros((len(text_hidden), 64)))
        return text.float(), base_logits, condition

    def forward(self, images, tokens_f, tokens_o, tokens_e, valid, completed=0):
        world, rank = world_rank()
        valid_global = gather(valid, False)
        valid_count = int(valid_global.sum())
        if self.joint_adapter is None:
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=images.is_cuda):
                z = self.clip.encode_image(images)
            z = z.float()
            image_hidden = z.new_zeros((len(z), 1))
            image_condition = z.new_zeros((len(z), 64))
        else:
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=images.is_cuda):
                z, image_hidden = self.clip.encode_image(images, return_hidden=True)
            z, image_hidden = z.float(), image_hidden.float()
            with torch.autocast(images.device.type, enabled=False):
                image_condition = self.joint_adapter.image_condition(image_hidden)
        z_global, condition_global = gather(z), gather(image_condition)
        permutation = torch.arange(len(z_global), device=z.device)
        if self.condition_mode == 'joint_shuffled_image':
            permutation = deterministic_shuffle_indices(
                len(z_global), self.shuffle_seed, completed, z.device)
            condition_global = condition_global[permutation]
        local_start = rank * len(z)
        condition_local = condition_global[local_start:local_start + len(z)]

        def terms(tokens, enabled, enabled_global):
            text, base, text_condition = self.encode_view(tokens)
            return pair_view_terms(z, text, base, text_condition, condition_local,
                                   enabled, enabled_global, self.joint_adapter,
                                   z_global, condition_global, self.image_chunk,
                                   self.text_chunk, self.checkpoint_pair_blocks)

        all_valid = torch.ones_like(valid)
        af, sf, mf, pf, logs_f = terms(tokens_f, all_valid,
                                       torch.ones_like(valid_global))
        logs = {'F_' + key: value for key, value in logs_f.items()}
        weight = inclusion_weight(self.arm, completed) if valid_count >= 2 else 0.
        nonfinite = ((~torch.isfinite(z)).sum() + (~torch.isfinite(image_hidden)).sum() +
                     (~torch.isfinite(pf)).sum())
        if valid_count >= 2:
            ao, so, mo, po, logs_o = terms(tokens_o, valid, valid_global)
            ae, se, me, pe, logs_e = terms(tokens_e, valid, valid_global)
            inc_sum = inclusion(pf, po, pe)[valid].sum()
            inc = world / valid_count * inc_sum
            loss = ((10 / 3) * (af + ao + ae) +
                    (sf + 2 * so + 2 * se) / 3 + weight * inc)
            hard_violation = .5 * ((mo.detach() > mf.detach()).float().mean(-1) +
                                   (me.detach() > mf.detach()).float().mean(-1))
            overlap = ((mo.detach() * me.detach()).sum(-1) /
                       ((mo.detach() + me.detach()) > 0).sum(-1).clamp_min(1))
            extra = global_sum(torch.stack((inc_sum.detach(), hard_violation[valid].sum(),
                                            overlap[valid].sum()))) / valid_count
            logs.update({'O_' + key: value for key, value in logs_o.items()})
            logs.update({'E_' + key: value for key, value in logs_e.items()})
            logs.update(inc=extra[0], hard_inclusion_violation=extra[1], oe_iou=extra[2])
            nonfinite += sum((~torch.isfinite(x)).sum() for x in (po, pe))
        else:
            loss = 10 * af + sf
            logs.update(inc=0., hard_inclusion_violation=0., oe_iou=0.,
                        O_candidates=0, E_candidates=0)
        logs.update(valid_global=valid_count, inc_weight=weight,
                    condition_mode=self.condition_mode,
                    shuffle_shift=(int(permutation[0]) if len(permutation) else 0),
                    nonfinite=global_sum(nonfinite), loss=global_sum(loss) / world)
        return loss, {key: value.detach() if torch.is_tensor(value) else value
                      for key, value in logs.items()}
