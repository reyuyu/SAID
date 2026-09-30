"""Visual-conditioned pooling masks for the NEST A3-RandomK objective.

The text token transformer is evaluated once per text/view.  A detached visual
CLS hidden state changes only the attention-pooling query used to turn those
transformed tokens into a pair-specific 512-dimensional mask.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from model.nested_semantic_mask import (
    _checkpoint_blocks,
    enable_encoder_checkpointing,
    gather,
    global_sum,
    hard_st,
    inclusion,
    inclusion_weight,
    world_rank,
)


class VCPQueryAdapter(nn.Module):
    """q_i = pool_weight + U(tanh(V(layer_norm(h_i.detach()))))."""

    def __init__(self, image_dim=768, rank=64, output_dim=512):
        super().__init__()
        self.visual = nn.Linear(image_dim, rank, bias=False)
        self.output = nn.Linear(rank, output_dim, bias=False)
        nn.init.xavier_uniform_(self.visual.weight)
        nn.init.zeros_(self.output.weight)

    def forward(self, hidden, pool_weight):
        hidden = F.layer_norm(hidden.detach().float(), (hidden.shape[-1],), eps=1e-5)
        return pool_weight.float() + self.output(torch.tanh(self.visual(hidden)))


def vcp_pair_mask(query, transformed_tokens, pool_bias):
    """Return [text,image,dim] masks without materializing token outputs."""
    energy = torch.einsum('id,tld->til', query.float(), transformed_tokens.float())
    energy = energy + pool_bias.float().reshape(1, 1, 1)
    weights = energy.softmax(dim=-1)
    logits = torch.einsum('til,tld->tid', weights, transformed_tokens.float())
    probabilities = torch.sigmoid(logits)
    return hard_st(probabilities), probabilities


def _vcp_score_block(images, texts, image_queries, transformed_tokens, pool_bias,
                     active_pairs=None):
    masks, probabilities = vcp_pair_mask(image_queries, transformed_tokens, pool_bias)
    # Pair tensors are [text,image,*].
    scores = 100 * (
        F.normalize(images.float()[None] * masks, dim=-1, eps=1e-6) *
        F.normalize(texts.float(), dim=-1, eps=1e-6)[:, None]
    ).sum(-1)
    if active_pairs is None:
        return scores
    selected = probabilities.detach() >= .5
    active = active_pairs.unsqueeze(-1)
    active_count = active_pairs.sum(dtype=torch.float32)
    summary = torch.stack((
        selected.masked_fill(~active, False).float().sum(),
        active_count * selected.shape[-1],
        (selected.all(-1) & active_pairs).float().sum(),
        ((~selected).all(-1) & active_pairs).float().sum(),
    ))
    return scores, summary.detach()


def vcp_pair_scores(images, texts, image_queries, transformed_tokens, pool_bias,
                    image_chunk=128, text_chunk=128, image_valid=None, text_valid=None,
                    checkpoint_blocks=False, collect_stats=True):
    """Local text rows against all image candidates, tiled in both dimensions."""
    image_valid = (torch.ones(len(images), dtype=torch.bool, device=images.device)
                   if image_valid is None else image_valid)
    text_valid = (torch.ones(len(texts), dtype=torch.bool, device=texts.device)
                  if text_valid is None else text_valid)
    rows, summaries = [], []
    for text_start in range(0, len(texts), text_chunk):
        text_stop = min(text_start + text_chunk, len(texts))
        block_row = []
        for image_start in range(0, len(images), image_chunk):
            image_stop = min(image_start + image_chunk, len(images))
            args = (
                images[image_start:image_stop],
                texts[text_start:text_stop],
                image_queries[image_start:image_stop],
                transformed_tokens[text_start:text_stop],
                pool_bias,
            )
            active = (text_valid[text_start:text_stop, None] &
                      image_valid[None, image_start:image_stop]) if collect_stats else None
            function = lambda *values, active=active: _vcp_score_block(
                *values, active_pairs=active)
            if checkpoint_blocks and torch.is_grad_enabled():
                output = checkpoint(function, *args, use_reentrant=False)
            else:
                output = function(*args)
            if collect_stats:
                scores, summary = output
                summaries.append(summary)
            else:
                scores = output
            block_row.append(scores)
        rows.append(torch.cat(block_row, dim=1))
    summary = torch.stack(summaries).sum(0) if collect_stats else None
    return torch.cat(rows, dim=0), summary


def vcp_positive_masks(image_queries, transformed_tokens, pool_bias, base_query=None):
    """Pair each local text with its image; optionally return text-only diagnostics."""
    energy = torch.einsum('bd,bld->bl', image_queries.float(), transformed_tokens.float())
    weights = (energy + pool_bias.float()).softmax(dim=-1)
    logits = torch.einsum('bl,bld->bd', weights, transformed_tokens.float())
    probabilities = torch.sigmoid(logits)
    masks = hard_st(probabilities)
    if base_query is None:
        return masks, probabilities
    with torch.no_grad():
        baseline_energy = torch.einsum(
            'd,bld->bl', base_query.float(), transformed_tokens.detach().float())
        baseline_weights = (baseline_energy + pool_bias.float()).softmax(dim=-1)
        baseline_logits = torch.einsum(
            'bl,bld->bd', baseline_weights, transformed_tokens.detach().float())
        baseline_masks = torch.sigmoid(baseline_logits) >= .5
        detached_weights = weights.detach()
        entropy = -(detached_weights.clamp_min(1e-12).log() * detached_weights).sum(-1)
        diagnostics = dict(
            pooling_entropy=entropy,
            pooling_l1_from_text_only=(detached_weights - baseline_weights).abs().mean(-1),
            hard_mask_switch_fraction=((masks.detach() >= .5) != baseline_masks).float().mean(-1))
    return masks, probabilities, diagnostics


def vcp_view_terms(z, text, transformed_tokens, image_queries, pool_weight, pool_bias, valid,
                   valid_global, z_global=None, query_global=None, image_chunk=128,
                   text_chunk=128, checkpoint_blocks=False):
    """Global two-way retrieval loss while each rank owns only local text-token rows."""
    world, rank = world_rank()
    n = int(valid_global.sum())
    if n < 1:
        raise ValueError('An enabled view needs at least one candidate')
    zg = gather(z) if z_global is None else z_global
    qg = gather(image_queries) if query_global is None else query_global
    local_scores, pair_stats = vcp_pair_scores(
        zg, text, qg, transformed_tokens, pool_bias,
        image_chunk=image_chunk, text_chunk=text_chunk,
        image_valid=valid_global, text_valid=valid,
        checkpoint_blocks=checkpoint_blocks, collect_stats=True)
    global_scores = gather(local_scores)
    labels = rank * z.shape[0] + torch.arange(len(z), device=z.device)
    local_start = rank * len(z)
    image_rows = global_scores.T[local_start:local_start + len(z)]
    zero = (zg.sum() + qg.sum() + transformed_tokens.sum() + text.sum() +
            local_scores.sum() + global_scores.sum()) * 0
    if bool(valid.any()):
        # Image queries retrieve global text columns; text queries retrieve global images.
        ci = F.cross_entropy(
            image_rows[valid].masked_fill(~valid_global[None], -torch.inf),
            labels[valid], reduction='sum')
        ct = F.cross_entropy(
            local_scores[valid].masked_fill(~valid_global[None], -torch.inf),
            labels[valid], reduction='sum')
    else:
        ci = ct = zero

    positive, probabilities, diagnostics = vcp_positive_masks(
        image_queries, transformed_tokens, pool_bias, pool_weight)
    sparse = positive[valid].abs().mean(-1).sum()
    selected = positive.detach()[valid] >= .5
    positive_summary = torch.stack((
        selected.float().sum(),
        selected.new_tensor(selected.numel(), dtype=torch.float32),
        selected.all(-1).float().sum(),
        (~selected).all(-1).float().sum(),
    ))
    totals = global_sum(pair_stats)
    positives = global_sum(positive_summary)
    negatives = totals - positives
    ce_logs = global_sum(torch.stack((ci.detach(), ct.detach(), sparse.detach()))) / n
    ratio = lambda values, numerator, denominator: values[numerator] / values[denominator].clamp_min(1)
    pair_count = lambda values: (values[1] / positive.shape[-1]).clamp_min(1)
    logs = dict(
        i2t=ce_logs[0], t2i=ce_logs[1], sparse=ce_logs[2], keep_ratio=ce_logs[2],
        all_open=positives[2] / n, all_closed=positives[3] / n, candidates=n,
        positive_keep_ratio=ratio(positives, 0, 1),
        negative_keep_ratio=ratio(negatives, 0, 1),
        positive_all_open=positives[2] / n,
        positive_all_closed=positives[3] / n,
        negative_all_open=negatives[2] / pair_count(negatives),
        negative_all_closed=negatives[3] / pair_count(negatives),
        delta_abs_mean=0., positive_delta_abs_mean=0., negative_delta_abs_mean=0.,
        pooling_entropy=global_sum(diagnostics['pooling_entropy'][valid].sum()) / n,
        pooling_l1_from_text_only=global_sum(
            diagnostics['pooling_l1_from_text_only'][valid].sum()) / n,
        hard_mask_switch_fraction=global_sum(
            diagnostics['hard_mask_switch_fraction'][valid].sum()) / n,
    )
    return ((world / n) * (ci + ct + zero), (world / n) * sparse,
            positive, probabilities, logs)


class NestedVCPMask(nn.Module):
    def __init__(self, clip, arm='A3', checkpoint_encoders=True, image_chunk=128,
                 text_chunk=128, condition_mode='vcp_mask', shuffle_seed=0,
                 checkpoint_pair_blocks=False):
        super().__init__()
        if arm not in ('A2', 'A3') or condition_mode != 'vcp_mask':
            raise ValueError((arm, condition_mode))
        self.clip = clip
        self.arm = arm
        self.checkpoint_encoders = bool(checkpoint_encoders)
        self.image_chunk, self.text_chunk = int(image_chunk), int(text_chunk)
        self.condition_mode, self.shuffle_seed = condition_mode, int(shuffle_seed)
        self.checkpoint_pair_blocks = bool(checkpoint_pair_blocks)
        image_dim = int(clip.visual.proj.shape[0])
        output_dim = int(clip.text_projection.shape[1])
        self.vcp_query = VCPQueryAdapter(image_dim, 64, output_dim)
        self.joint_adapter = None
        if checkpoint_encoders:
            enable_encoder_checkpointing(clip)

    def encode_view(self, tokens):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=tokens.is_cuda):
            text, hidden = self.clip.encode_text(tokens, return_full=True)
        # B_phi is exactly the original MaskNetwork token transformer, once per view.
        with torch.autocast(tokens.device.type, enabled=False):
            transformed = self.clip.mask_net.resblocks(
                hidden.detach().float().permute(1, 0, 2)).permute(1, 0, 2)
        return text.float(), transformed

    def forward(self, images, tokens_f, tokens_o, tokens_e, valid, completed=0):
        valid_global = gather(valid, False)
        valid_count = int(valid_global.sum())
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=images.is_cuda):
            z, image_hidden = self.clip.encode_image(images, return_hidden=True)
        z, image_hidden = z.float(), image_hidden.float()
        pool = self.clip.mask_net.attn_pool.attention
        with torch.autocast(images.device.type, enabled=False):
            image_queries = self.vcp_query(image_hidden, pool.weight.squeeze(0))
        z_global, query_global = gather(z), gather(image_queries)

        def terms(tokens, enabled, enabled_global):
            text, transformed = self.encode_view(tokens)
            return vcp_view_terms(
                z, text, transformed, image_queries, pool.weight.squeeze(0),
                pool.bias.squeeze(0), enabled, enabled_global, z_global=z_global,
                query_global=query_global,
                image_chunk=self.image_chunk, text_chunk=self.text_chunk,
                checkpoint_blocks=self.checkpoint_pair_blocks)

        all_valid = torch.ones_like(valid)
        af, sf, mf, pf, logs_f = terms(tokens_f, all_valid, torch.ones_like(valid_global))
        logs = {'F_' + key: value for key, value in logs_f.items()}
        weight = inclusion_weight(self.arm, completed) if valid_count >= 2 else 0.
        nonfinite = ((~torch.isfinite(z)).sum() + (~torch.isfinite(image_hidden)).sum() +
                     (~torch.isfinite(image_queries)).sum() + (~torch.isfinite(pf)).sum())
        if valid_count >= 2:
            ao, so, mo, po, logs_o = terms(tokens_o, valid, valid_global)
            ae, se, me, pe, logs_e = terms(tokens_e, valid, valid_global)
            inc_sum = inclusion(pf, po, pe)[valid].sum()
            inc = world_rank()[0] / valid_count * inc_sum
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
        query_delta = image_queries - pool.weight.squeeze(0).float()
        logs.update(valid_global=valid_count, inc_weight=weight,
                    q_minus_w_norm=global_sum(query_delta.norm(dim=-1).sum()) / len(query_global),
                    w_norm=pool.weight.squeeze(0).float().norm(),
                    condition_mode=self.condition_mode, shuffle_shift=0,
                    nonfinite=global_sum(nonfinite), loss=global_sum(loss) / world_rank()[0])
        return loss, {key: value.detach() if torch.is_tensor(value) else value
                      for key, value in logs.items()}
