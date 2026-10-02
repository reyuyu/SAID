"""Independent token branches with exact Stack-Pool or dense CrossScore readout."""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from model.nested_semantic_mask import (enable_encoder_checkpointing, gather, global_sum,
                                       hard_st, inclusion, inclusion_weight, world_rank)


def pool_summary(tokens, weight, bias):
    energy = F.linear(tokens.float(), weight.float(), bias.float()).squeeze(-1)
    return energy.logsumexp(-1), (energy.softmax(-1).unsqueeze(-1) * tokens).sum(-2)


def stack_logits(visual_summary, text_summary, paired=False):
    av, uv = visual_summary
    at, ut = text_summary
    if paired:
        fraction = torch.sigmoid(at - av)
        return fraction[:, None] * ut + (1 - fraction[:, None]) * uv
    fraction = torch.sigmoid(at[:, None] - av[None])
    return fraction[..., None] * ut[:, None] + (1 - fraction[..., None]) * uv[None]


class FusionBranch(nn.Module):
    def __init__(self, source_blocks, visual_width, width, fusion, visual_tokens, text_tokens=248, rank=64):
        super().__init__()
        self.visual_blocks = copy.deepcopy(source_blocks)
        self.fusion, self.visual_tokens, self.text_tokens, self.rank = fusion, visual_tokens, text_tokens, rank
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1763)
            self.visual_adapter = nn.Linear(visual_width, width, bias=False)
            nn.init.xavier_uniform_(self.visual_adapter.weight)
            if fusion == 'balanced_stack':
                self.gate = nn.Linear(width * 2, width, bias=False)
                nn.init.zeros_(self.gate.weight)
            if fusion in ('crossscore_flat', 'cosine_crossscore'):
                torch.manual_seed(1787)
                self.query = nn.Linear(width, rank, bias=False)
                self.key = nn.Linear(width, rank, bias=False)
                nn.init.xavier_uniform_(self.query.weight)
                nn.init.xavier_uniform_(self.key.weight)
                torch.manual_seed(1789)
                self.readout = nn.Linear(visual_tokens * text_tokens, width, bias=True)
                if fusion == 'crossscore_flat':
                    nn.init.uniform_(self.readout.weight, -1 / math.sqrt(visual_tokens * text_tokens),
                                     1 / math.sqrt(visual_tokens * text_tokens))
                    nn.init.zeros_(self.readout.bias)

    def projected_queries(self, tokens):
        queries = self.query(tokens)
        return F.normalize(queries, dim=-1, eps=1e-6) if self.fusion == 'cosine_crossscore' else queries

    def projected_keys(self, tokens):
        keys = self.key(tokens)
        return F.normalize(keys, dim=-1, eps=1e-6) if self.fusion == 'cosine_crossscore' else keys

    def balanced_gate(self, visual, text, paired=False):
        width = self.gate.out_features
        text_part = F.linear(text, self.gate.weight[:, :width])
        visual_part = F.linear(visual, self.gate.weight[:, width:])
        return torch.sigmoid(text_part + visual_part if paired else text_part[:, None] + visual_part[None])

    def encode_visual(self, hidden):
        tokens = self.visual_adapter(hidden.detach().float())
        return self.visual_blocks(tokens.permute(1, 0, 2)).permute(1, 0, 2)

    def text_condition(self, tokens):
        keys = self.projected_keys(tokens)
        return self.contract_keys(keys)

    def contract_keys(self, keys):
        if self.visual_tokens == 1:
            scale = 1. if self.fusion == 'cosine_crossscore' else math.sqrt(self.rank)
            return torch.einsum('dt,btk->bdk', self.readout.weight, keys) / scale
        weight = self.readout.weight.reshape(self.readout.out_features, self.visual_tokens, self.text_tokens)
        return torch.einsum('dat,btk->bdak', weight, keys) / math.sqrt(self.rank)

    def cross_logits(self, queries, condition, paired=False, explicit=False):
        if not explicit:
            if self.visual_tokens > 1:
                if paired:
                    return torch.einsum('bak,bdak->bd', queries, condition) + self.readout.bias
                return torch.einsum('iak,tdak->tid', queries, condition) + self.readout.bias
            if paired:
                return torch.einsum('bk,bdk->bd', queries[:, 0], condition) + self.readout.bias
            return torch.einsum('iak,tdk->tid', queries, condition) + self.readout.bias
        if paired:
            relations = torch.einsum('bak,btk->bat', queries, condition)
        else:
            relations = torch.einsum('iak,btk->biat', queries, condition)
        if self.fusion != 'cosine_crossscore':
            relations = relations / math.sqrt(self.rank)
        return F.linear(relations.flatten(-2), self.readout.weight, self.readout.bias)


def pair_logits(module, visual, text, paired=False):
    if module.fusion == 'stack_pool':
        return stack_logits(visual, text, paired)
    if module.fusion == 'balanced_stack':
        gate = module.fusion_branch.balanced_gate(visual[1], text[1], paired)
        return (gate * text[1] + (1 - gate) * visual[1] if paired else
                gate * text[1][:, None] + (1 - gate) * visual[1][None])
    return module.fusion_branch.cross_logits(visual[0], text[0], paired)


def score_block(module, z, text, visual, condition, active, collect_diagnostics):
    logits = pair_logits(module, visual, condition)
    probability = logits.sigmoid()
    mask = hard_st(probability)
    scores = 100 * (F.normalize(z[None].float() * mask, dim=-1, eps=1e-6) *
                    F.normalize(text.float(), dim=-1, eps=1e-6)[:, None]).sum(-1)
    with torch.no_grad():
        selected = probability >= .5
        count = active.sum(dtype=torch.float32)
        summary = torch.stack((selected.masked_fill(~active[..., None], False).float().sum(),
                               count * selected.shape[-1],
                               (selected.all(-1) & active).float().sum(),
                               ((~selected).all(-1) & active).float().sum()))
        extra = logits.new_zeros(9)
        if collect_diagnostics:
            enabled = active[..., None]
            extra[:5] = torch.stack((logits.masked_fill(~enabled, 0).sum(),
                                    logits.square().masked_fill(~enabled, 0).sum(),
                                    (((probability < .01) | (probability > .99)) & enabled).float().sum(),
                                    count * logits.shape[-1], count))
            if module.fusion == 'stack_pool':
                text_weight = torch.sigmoid(condition[0][:, None] - visual[0][None])
                extra[5] = text_weight.masked_fill(~active, 0).sum()
            elif module.fusion in ('crossscore_flat', 'cosine_crossscore'):
                # Raw K is retained as a view-local tensor for sparse diagnostics.
                keys = condition[1]
                relation = torch.einsum('iak,btk->biat', visual[0], keys)
                if module.fusion != 'cosine_crossscore':
                    relation = relation / math.sqrt(module.fusion_branch.rank)
                enabled_r = active[..., None, None]
                extra[5] = relation.masked_fill(~enabled_r, 0).sum()
                extra[6] = relation.square().masked_fill(~enabled_r, 0).sum()
                extra[7] = count * relation.shape[-1] * relation.shape[-2]
                extra[8] = relation.abs().masked_fill(~enabled_r, 0).max()
    return scores, summary, extra


def fusion_scores(module, images, texts, visual, condition, image_valid, text_valid, diagnostics=False):
    rows, summaries, extras = [], [], []
    for start_t in range(0, len(texts), module.text_chunk):
        stop_t = min(start_t + module.text_chunk, len(texts))
        row = []
        for start_i in range(0, len(images), module.image_chunk):
            stop_i = min(start_i + module.image_chunk, len(images))
            vv = tuple(x[start_i:stop_i] for x in visual)
            tt = tuple(x[start_t:stop_t] for x in condition)
            active = text_valid[start_t:stop_t, None] & image_valid[None, start_i:stop_i]
            def block(z, t, *args, active=active, nv=len(vv)):
                return score_block(module, z, t, args[:nv], args[nv:], active, diagnostics)
            arguments = (images[start_i:stop_i], texts[start_t:stop_t], *vv, *tt)
            if module.checkpoint_pair_blocks and torch.is_grad_enabled():
                scores, summary, extra = checkpoint(block, *arguments, use_reentrant=False)
            else:
                scores, summary, extra = block(*arguments)
            row.append(scores)
            summaries.append(summary)
            extras.append(extra)
        rows.append(torch.cat(row, 1))
    combined = torch.stack(extras).sum(0)
    combined[8] = torch.stack(extras)[:, 8].max()
    return torch.cat(rows), torch.stack(summaries).sum(0), combined


def fusion_view_terms(module, z, text, visual, condition, valid, valid_global,
                      global_z, global_visual, diagnostics=False):
    world, rank = world_rank()
    n = int(valid_global.sum())
    scores, summary, diagnostic = fusion_scores(module, global_z, text, global_visual, condition,
                                                valid_global, valid, diagnostics)
    global_scores = gather(scores)
    labels = rank * len(z) + torch.arange(len(z), device=z.device)
    image_rows = global_scores.T[rank * len(z):(rank + 1) * len(z)]
    zero = (global_z.sum() + global_scores.sum() + scores.sum() + text.sum() +
            sum(x.sum() for x in global_visual) + sum(x.sum() for x in condition)) * 0
    if bool(valid.any()):
        ci = F.cross_entropy(image_rows[valid].masked_fill(~valid_global[None], -torch.inf),
                             labels[valid], reduction='sum')
        ct = F.cross_entropy(scores[valid].masked_fill(~valid_global[None], -torch.inf),
                             labels[valid], reduction='sum')
    else:
        ci = ct = zero
    logits = pair_logits(module, visual, condition, paired=True)
    probability = logits.sigmoid()
    positive = hard_st(probability)
    sparse = positive[valid].abs().mean(-1).sum()
    selected = positive.detach() >= .5
    ps = torch.stack((selected.masked_fill(~valid[:, None], False).float().sum(),
                      valid.sum(dtype=torch.float32) * selected.shape[-1],
                      (selected.all(-1) & valid).float().sum(),
                      ((~selected).all(-1) & valid).float().sum()))
    totals, positives = global_sum(summary), global_sum(ps)
    negatives = totals - positives
    ce = global_sum(torch.stack((ci.detach(), ct.detach(), sparse.detach()))) / n
    logs = dict(i2t=ce[0], t2i=ce[1], sparse=ce[2], keep_ratio=ce[2], candidates=n,
                all_open=positives[2] / n, all_closed=positives[3] / n,
                positive_keep_ratio=positives[0] / positives[1].clamp_min(1),
                negative_keep_ratio=negatives[0] / negatives[1].clamp_min(1),
                negative_all_open=negatives[2] / (negatives[1] / selected.shape[-1]).clamp_min(1),
                negative_all_closed=negatives[3] / (negatives[1] / selected.shape[-1]).clamp_min(1))
    if diagnostics:
        d = global_sum(diagnostic)
        mean = d[0] / d[3].clamp_min(1)
        logs.update(logit_mean=mean, logit_variance=d[1] / d[3].clamp_min(1) - mean.square(),
                    sigmoid_saturation=d[2] / d[3].clamp_min(1))
        if module.fusion == 'stack_pool':
            logs.update(text_pool_weight=d[5] / d[4].clamp_min(1),
                        visual_pool_weight=1 - d[5] / d[4].clamp_min(1))
        elif module.fusion in ('crossscore_flat', 'cosine_crossscore'):
            qk_mean = d[5] / d[7].clamp_min(1)
            logs.update(qk_mean=qk_mean, qk_variance=d[6] / d[7].clamp_min(1) - qk_mean.square())
            bound = diagnostic[8].detach().clone()
            if world > 1:
                torch.distributed.all_reduce(bound, op=torch.distributed.ReduceOp.MAX)
            logs['qk_abs_max'] = bound
        elif module.fusion == 'balanced_stack':
            gate = module.fusion_branch.balanced_gate(visual[1], condition[1], paired=True).detach()
            global_gate, enabled = gather(gate, False), gather(valid, False)
            selected_gates = global_gate[enabled].flatten()
            logs.update(g_mean=selected_gates.mean(), g_variance=selected_gates.var(unbiased=False))
            quantiles = torch.quantile(selected_gates, gate.new_tensor([0., .05, .25, .5, .75, .95, 1.]))
            logs.update({f'g_q{q}': value for q, value in zip(('00', '05', '25', '50', '75', '95', '100'), quantiles)})
            logs['_diagnostic_gate'] = gate
        with torch.no_grad():
            replacement = tuple(x.roll(1, 0) for x in global_visual)
            replacement_local = tuple(x[rank * len(z):(rank + 1) * len(z)] for x in replacement)
            alternate = pair_logits(module, replacement_local, condition, paired=True) >= 0
            switch = ((positive.detach() >= .5) != alternate).float().mean(-1)
            logs['replaced_image_mask_switch'] = global_sum(switch[valid].sum()) / n
    return world / n * (ci + ct + zero), world / n * sparse, positive, probability, logs


class NestedFusionMask(nn.Module):
    def __init__(self, clip, arm='A3', checkpoint_encoders=True, image_chunk=128,
                 text_chunk=128, condition_mode='dual_branch', shuffle_seed=0,
                 checkpoint_pair_blocks=False, fusion='stack_pool', visual='cls', text_tokens=248):
        super().__init__()
        assert arm == 'A3' and condition_mode == 'dual_branch'
        assert fusion in ('stack_pool', 'crossscore_flat', 'balanced_stack', 'cosine_crossscore') and visual in ('cls', 'patch')
        assert fusion != 'balanced_stack' or visual == 'patch'
        assert fusion != 'cosine_crossscore' or visual == 'cls'
        self.clip, self.arm, self.condition_mode = clip, arm, condition_mode
        self.fusion, self.visual = fusion, visual
        self.image_chunk, self.text_chunk = int(image_chunk), int(text_chunk)
        self.checkpoint_pair_blocks = bool(checkpoint_pair_blocks)
        self.checkpoint_encoders = bool(checkpoint_encoders)
        self.shuffle_seed = shuffle_seed
        width = int(clip.text_projection.shape[0])
        positions = getattr(clip.visual, 'positional_embedding', None)
        patches = int(positions.shape[0])-1 if positions is not None else 196
        assert int(clip.text_projection.shape[1]) == width, 'Mask channels must match native embedding dimension'
        self.fusion_branch = FusionBranch(clip.mask_net.resblocks, int(clip.visual.proj.shape[0]),
                                          width, fusion, 1 if visual == 'cls' else patches, text_tokens)
        if fusion in ('crossscore_flat', 'cosine_crossscore'):
            clip.mask_net.attn_pool.requires_grad_(False)
        if checkpoint_encoders:
            enable_encoder_checkpointing(clip)

    def mask_parameters(self):
        yield from self.clip.mask_net.resblocks.parameters()
        yield from self.fusion_branch.visual_blocks.parameters()
        if self.fusion in ('stack_pool', 'balanced_stack'):
            yield from self.clip.mask_net.attn_pool.parameters()

    def encode_visual(self, images):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=images.is_cuda):
            z, hidden = self.clip.encode_image(images, return_token_hidden=True)
        source = hidden[:, :1] if self.visual == 'cls' else hidden[:, 1:]
        transformed = self.fusion_branch.encode_visual(source)
        if self.fusion in ('stack_pool', 'balanced_stack'):
            pool = self.clip.mask_net.attn_pool.attention
            condition = pool_summary(transformed, pool.weight, pool.bias)
        else:
            condition = (self.fusion_branch.projected_queries(transformed),)
        return z.float(), condition

    def encode_view(self, tokens):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=tokens.is_cuda):
            text, hidden = self.clip.encode_text(tokens, return_full=True)
        transformed = self.clip.mask_net.resblocks(hidden.detach().float().permute(1, 0, 2)).permute(1, 0, 2)
        if self.fusion in ('stack_pool', 'balanced_stack'):
            pool = self.clip.mask_net.attn_pool.attention
            condition = pool_summary(transformed, pool.weight, pool.bias)
        else:
            keys = self.fusion_branch.projected_keys(transformed)
            condition = (self.fusion_branch.contract_keys(keys), keys)
        return text.float(), condition

    def forward(self, images, tokens_f, tokens_o, tokens_e, valid, completed=0):
        valid_global = gather(valid, False)
        valid_count = int(valid_global.sum())
        z, visual = self.encode_visual(images)
        global_z = gather(z)
        global_visual = tuple(gather(x) for x in visual)
        diagnostics = completed in (0, 99, 199, 299, 399, 499)
        def terms(tokens, enabled, enabled_global):
            text, condition = self.encode_view(tokens)
            return fusion_view_terms(self, z, text, visual, condition, enabled, enabled_global,
                                     global_z, global_visual, diagnostics)
        af, sf, mf, pf, lf = terms(tokens_f, torch.ones_like(valid), torch.ones_like(valid_global))
        gate_f = lf.pop('_diagnostic_gate', None)
        logs = {'F_' + key: value for key, value in lf.items()}
        weight = inclusion_weight(self.arm, completed) if valid_count >= 2 else 0.
        if valid_count >= 2:
            ao, so, mo, po, lo = terms(tokens_o, valid, valid_global)
            ae, se, me, pe, le = terms(tokens_e, valid, valid_global)
            gate_o, gate_e = lo.pop('_diagnostic_gate', None), le.pop('_diagnostic_gate', None)
            if gate_f is not None:
                differences = torch.stack(((gate_f - gate_o).abs().mean(-1)[valid].sum(),
                                           (gate_f - gate_e).abs().mean(-1)[valid].sum(),
                                           (gate_o - gate_e).abs().mean(-1)[valid].sum()))
                differences = global_sum(differences) / valid_count
                logs.update(g_F_P_abs_difference=differences[0], g_F_R_abs_difference=differences[1],
                            g_P_R_abs_difference=differences[2])
            inc_sum = inclusion(pf, po, pe)[valid].sum()
            loss = 10 / 3 * (af + ao + ae) + (sf + 2 * so + 2 * se) / 3 + weight * world_rank()[0] / valid_count * inc_sum
            violation = .5 * ((mo.detach() > mf.detach()).float().mean(-1) +
                              (me.detach() > mf.detach()).float().mean(-1))
            iou = (mo.detach() * me.detach()).sum(-1) / ((mo.detach() + me.detach()) > 0).sum(-1).clamp_min(1)
            extra = global_sum(torch.stack((inc_sum.detach(), violation[valid].sum(), iou[valid].sum()))) / valid_count
            logs.update({'O_' + key: value for key, value in lo.items()})
            logs.update({'E_' + key: value for key, value in le.items()})
            logs.update(inc=extra[0], hard_inclusion_violation=extra[1], oe_iou=extra[2])
        else:
            loss = 10 * af + sf
            logs.update(inc=0., hard_inclusion_violation=0., oe_iou=0., O_candidates=0, E_candidates=0)
        logs.update(loss=global_sum(loss) / world_rank()[0], inc_weight=weight, valid_global=valid_count,
                    fusion=self.fusion, visual=self.visual, condition_mode=self.condition_mode,
                    shuffle_shift=0, nonfinite=global_sum((~torch.isfinite(loss)).sum()))
        return loss, {key: value.detach() if torch.is_tensor(value) else value for key, value in logs.items()}
