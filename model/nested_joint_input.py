"""Input-level joint visual/text masks for the controlled NEST design search."""
from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from model.nested_semantic_mask import gather, global_sum, hard_st, inclusion, inclusion_weight, world_rank

VISUAL_MODES = ("cls", "patch")
READOUT_MODES = ("all", "text")


class VisualInputProjection(nn.Module):
    """Shared post-LN/pre-projection visual hidden adapter, 768 -> 512."""

    def __init__(self, image_dim=768, width=512):
        super().__init__()
        self.projection = nn.Linear(image_dim, width, bias=False)
        nn.init.xavier_uniform_(self.projection.weight)

    def forward(self, hidden):
        return self.projection(hidden.detach().float())


def mask_transform(mask_net, sequence, checkpoint_block=False):
    """Run B_phi once on [pair, tokens, width] without a causal mask."""
    x = sequence.permute(1, 0, 2)
    for block in mask_net.resblocks:
        if block.attn_mask is not None:
            raise AssertionError("MaskNetwork joint attention must remain non-causal")
        x = (checkpoint(block, x, use_reentrant=False)
             if checkpoint_block and torch.is_grad_enabled() else block(x))
    return x.permute(1, 0, 2)


def joint_logits(mask_net, visual, text, readout_mode, checkpoint_block=False,
                 return_pool_weights=False):
    """Direct reference for one pair batch; visual/text already share width 512."""
    if readout_mode not in READOUT_MODES:
        raise ValueError(readout_mode)
    if visual.ndim != 3 or text.ndim != 3 or visual.shape[0] != text.shape[0]:
        raise ValueError((visual.shape, text.shape))
    visual_count = visual.shape[1]
    transformed = mask_transform(mask_net, torch.cat((visual, text), dim=1), checkpoint_block)
    pooled_tokens = transformed if readout_mode == "all" else transformed[:, visual_count:, :]
    attention_logits = mask_net.attn_pool.attention(pooled_tokens)
    weights = F.softmax(attention_logits, dim=1)
    logits = (weights * pooled_tokens).sum(dim=1)
    if return_pool_weights:
        return logits, weights
    return logits


def _joint_score_chunk(z, text, visual, text_hidden, mask_net, readout_mode,
                       checkpoint_block, active_pairs=None):
    logits = joint_logits(mask_net, visual, text_hidden, readout_mode, checkpoint_block)
    probabilities = torch.sigmoid(logits)
    masks = hard_st(probabilities)
    scores = 100 * (F.normalize(z.float() * masks, dim=-1, eps=1e-6) *
                    F.normalize(text.float(), dim=-1, eps=1e-6)).sum(-1)
    if active_pairs is None:
        return scores
    selected = probabilities.detach() >= .5
    active = active_pairs.to(torch.float32)
    active_count = active.sum()
    summary = torch.stack((
        (selected.float() * active[:, None]).sum(),
        active_count * selected.shape[-1],
        (selected.all(-1).float() * active).sum(),
        ((~selected).all(-1).float() * active).sum(),
    ))
    return scores, summary.detach()


def joint_pair_scores(images, texts, visual_tokens, text_hidden, mask_net, readout_mode,
                      pair_microbatch=8, row_valid=None, column_valid=None,
                      checkpoint_block=True, collect_stats=True):
    """Cartesian pair scores without materializing [image,text,token,width]."""
    rows, columns = len(images), len(texts)
    row_valid = (torch.ones(rows, dtype=torch.bool, device=images.device)
                 if row_valid is None else row_valid)
    column_valid = (torch.ones(columns, dtype=torch.bool, device=images.device)
                    if column_valid is None else column_valid)
    flat_scores, summaries = [], []
    for start in range(0, rows * columns, pair_microbatch):
        stop = min(start + pair_microbatch, rows * columns)
        flat = torch.arange(start, stop, device=images.device)
        ii, jj = torch.div(flat, columns, rounding_mode="floor"), flat % columns
        active = row_valid[ii] & column_valid[jj] if collect_stats else None
        output = _joint_score_chunk(images[ii], texts[jj], visual_tokens[ii], text_hidden[jj],
                                    mask_net, readout_mode, checkpoint_block, active)
        if collect_stats:
            scores, summary = output
            summaries.append(summary)
        else:
            scores = output
        flat_scores.append(scores)
    summary = torch.stack(summaries).sum(0) if collect_stats else None
    return torch.cat(flat_scores).reshape(rows, columns), summary


def positive_joint_masks(visual, text_hidden, mask_net, readout_mode,
                         checkpoint_block=True, return_pool_weights=False):
    output = joint_logits(mask_net, visual, text_hidden, readout_mode, checkpoint_block,
                          return_pool_weights)
    if return_pool_weights:
        logits, weights = output
    else:
        logits, weights = output, None
    probabilities = torch.sigmoid(logits)
    result = (hard_st(probabilities), probabilities)
    return (*result, weights) if return_pool_weights else result


def joint_view_terms(z, text, visual, text_hidden, valid, valid_global, mask_net,
                     readout_mode, z_global=None, visual_global=None,
                     pair_microbatch=8, checkpoint_block=True):
    """Local CE query rows against complete global candidates with pair-specific masks."""
    world, rank = world_rank()
    n = int(valid_global.sum())
    if n < 1:
        raise ValueError("An enabled view needs at least one candidate")
    zg = gather(z) if z_global is None else z_global
    vg = gather(visual) if visual_global is None else visual_global
    tg = gather(text)
    hg = gather(text_hidden, differentiable=False)
    labels = rank * z.shape[0] + torch.arange(len(z), device=z.device)
    qi, pair_stats = joint_pair_scores(
        z, tg, visual, hg, mask_net, readout_mode, pair_microbatch,
        valid, valid_global, checkpoint_block, True)
    qt, _ = joint_pair_scores(
        zg, text, vg, text_hidden, mask_net, readout_mode, pair_microbatch,
        valid_global, valid, checkpoint_block, False)
    qt = qt.T
    zero = (zg.sum() + vg.sum() + tg.sum() + hg.sum() + qi.sum() + qt.sum()) * 0
    if bool(valid.any()):
        ci = F.cross_entropy(qi[valid].masked_fill(~valid_global[None], -torch.inf),
                             labels[valid], reduction="sum")
        ct = F.cross_entropy(qt[valid].masked_fill(~valid_global[None], -torch.inf),
                             labels[valid], reduction="sum")
    else:
        ci = ct = zero
    positive, probabilities = positive_joint_masks(
        visual, text_hidden, mask_net, readout_mode, checkpoint_block)
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
    ratio = lambda values, a, b: values[a] / values[b].clamp_min(1)
    pair_count = lambda values: (values[1] / positive.shape[-1]).clamp_min(1)
    logs = dict(
        i2t=ce_logs[0], t2i=ce_logs[1], sparse=ce_logs[2], keep_ratio=ce_logs[2],
        all_open=positives[2] / n, all_closed=positives[3] / n, candidates=n,
        positive_keep_ratio=ratio(positives, 0, 1),
        negative_keep_ratio=ratio(negatives, 0, 1),
        positive_all_open=positives[2] / n, positive_all_closed=positives[3] / n,
        negative_all_open=negatives[2] / pair_count(negatives),
        negative_all_closed=negatives[3] / pair_count(negatives))
    return ((world / n) * (ci + ct + zero), (world / n) * sparse,
            positive, probabilities, logs)


class NestedJointInputMask(nn.Module):
    """A3-RandomK with input-level joint visual/text mask generation."""

    def __init__(self, clip, visual_mode="cls", readout_mode="all", arm="A3",
                 pair_microbatch=8, checkpoint_pair_block=True,
                 checkpoint_encoders=True):
        super().__init__()
        if visual_mode not in VISUAL_MODES or readout_mode not in READOUT_MODES:
            raise ValueError((visual_mode, readout_mode))
        if arm not in ("A2", "A3"):
            raise ValueError(arm)
        self.clip = clip
        self.visual_mode = visual_mode
        self.readout_mode = readout_mode
        self.arm = arm
        self.pair_microbatch = int(pair_microbatch)
        self.checkpoint_pair_block = bool(checkpoint_pair_block)
        self.checkpoint_encoders = bool(checkpoint_encoders)
        self.visual_input_projection = VisualInputProjection(
            int(clip.visual.proj.shape[0]), int(clip.text_projection.shape[0]))
        if checkpoint_encoders:
            from model.nested_semantic_mask import enable_encoder_checkpointing
            enable_encoder_checkpointing(clip)

    @property
    def visual_token_count(self):
        return 1 if self.visual_mode == "cls" else 196

    def encode_visual(self, images):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=images.is_cuda):
            z, h_cls, h_patch = self.clip.encode_image_with_joint_tokens(images)
        raw = h_cls[:, None, :] if self.visual_mode == "cls" else h_patch
        return z.float(), raw.detach().float()

    def encode_text(self, tokens):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=tokens.is_cuda):
            text, hidden = self.clip.encode_text(tokens, return_full=True)
        return text.float(), hidden.detach().float()

    def forward(self, images, tokens_f, tokens_o, tokens_e, valid, completed=0):
        world, _ = world_rank()
        valid_global = gather(valid, differentiable=False)
        valid_count = int(valid_global.sum())
        z, visual_raw = self.encode_visual(images)
        visual_global_raw = gather(visual_raw, differentiable=False)
        with torch.autocast(images.device.type, enabled=False):
            visual = self.visual_input_projection(visual_raw)
            visual_global = self.visual_input_projection(visual_global_raw)
        z_global = gather(z)

        def terms(tokens, enabled, enabled_global):
            text, hidden = self.encode_text(tokens)
            return joint_view_terms(
                z, text, visual, hidden, enabled, enabled_global, self.clip.mask_net,
                self.readout_mode, z_global, visual_global, self.pair_microbatch,
                self.checkpoint_pair_block)

        all_valid = torch.ones_like(valid)
        af, sf, mf, pf, logs_f = terms(tokens_f, all_valid, torch.ones_like(valid_global))
        logs = {"F_" + key: value for key, value in logs_f.items()}
        weight = inclusion_weight(self.arm, completed) if valid_count >= 2 else 0.
        nonfinite = ((~torch.isfinite(z)).sum() + (~torch.isfinite(visual)).sum() +
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
            logs.update({"O_" + key: value for key, value in logs_o.items()})
            logs.update({"E_" + key: value for key, value in logs_e.items()})
            logs.update(inc=extra[0], hard_inclusion_violation=extra[1], oe_iou=extra[2])
            nonfinite += sum((~torch.isfinite(x)).sum() for x in (po, pe))
        else:
            loss = 10 * af + sf
            logs.update(inc=0., hard_inclusion_violation=0., oe_iou=0.,
                        O_candidates=0, E_candidates=0)
        logs.update(valid_global=valid_count, inc_weight=weight,
                    visual_mode=self.visual_mode, readout_mode=self.readout_mode,
                    visual_token_count=self.visual_token_count,
                    nonfinite=global_sum(nonfinite), loss=global_sum(loss) / world)
        return loss, {key: value.detach() if torch.is_tensor(value) else value
                      for key, value in logs.items()}
