"""Isolated ViT-B/16 component ablation modules.

These classes intentionally live outside the production HNS trainer.  They
reuse the frozen data path and score primitives, while making the three
authorized ablation objectives explicit and auditable.
"""
import torch
from torch import nn
from torch.nn import functional as F

from model.nested_fusion_mask import NestedFusionMask, fusion_view_terms
from model.balanced_hparam_search import BalancedSearch
from model.nested_semantic_mask import gather, global_sum, world_rank


def native_clip_logits(image_features, text_features):
    """CLIP cosine logits used by the native alignment-only ablation."""
    image_features = F.normalize(image_features.float(), dim=-1, eps=1e-6)
    text_features = F.normalize(text_features.float(), dim=-1, eps=1e-6)
    return 100 * image_features @ text_features.T


def _native_terms(module, z, tokens, valid, valid_global, global_z):
    """Native CLIP bidirectional CE over the complete global candidate pool."""
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=tokens.is_cuda):
        text = module.clip.encode_text(tokens)
    text = F.normalize(text.float(), dim=-1, eps=1e-6)
    z = F.normalize(z.float(), dim=-1, eps=1e-6)
    global_z = F.normalize(global_z.float(), dim=-1, eps=1e-6)
    global_text = gather(text)
    labels = world_rank()[1] * len(z) + torch.arange(len(z), device=z.device)
    qi = native_clip_logits(z, global_text)
    qt = native_clip_logits(global_z, text).T
    zero = (z.sum() + global_z.sum() + text.sum() + global_text.sum()) * 0
    if bool(valid.any()):
        ci = F.cross_entropy(qi[valid].masked_fill(~valid_global[None], -torch.inf),
                             labels[valid], reduction='sum')
        ct = F.cross_entropy(qt[valid].masked_fill(~valid_global[None], -torch.inf),
                             labels[valid], reduction='sum')
    else:
        ci = ct = zero
    n = int(valid_global.sum())
    world = world_rank()[0]
    logs = global_sum(torch.stack((ci.detach(), ct.detach()))) / max(n, 1)
    return (world / max(n, 1)) * (ci + ct + zero), logs


class JointMaskOnly(BalancedSearch):
    """B: full joint visual/text mask, F view only, no HNS or detail views."""

    ablation_mode = 'joint_mask_only'

    def forward(self, images, tokens_f, tokens_o, tokens_e, valid, completed=0):
        valid_global = gather(valid, False)
        z, visual = self.encode_visual(images)
        global_z, global_visual = gather(z), tuple(gather(x) for x in visual)
        text, condition = self.encode_view(tokens_f)
        af, sf, mf, pf, logs_f = fusion_view_terms(
            self, z, text, visual, condition,
            torch.ones_like(valid), torch.ones_like(valid_global),
            global_z, global_visual, completed in (0, 99, 199, 299, 399, 499))
        loss = 10 * af + sf
        logs = {'F_' + key: value for key, value in logs_f.items()
                if key != '_diagnostic_gate'}
        logs.update(loss=global_sum(loss) / world_rank()[0], valid_global=int(valid_global.sum()),
                    inc=0., inc_weight=0., hard_inclusion_violation=0., oe_iou=0.,
                    fusion=self.fusion, visual=self.visual, condition_mode=self.condition_mode,
                    ablation_mode=self.ablation_mode,
                    nonfinite=global_sum((~torch.isfinite(loss)).sum()))
        return loss, {key: value.detach() if torch.is_tensor(value) else value
                      for key, value in logs.items()}


class NestedAlignmentOnly(nn.Module):
    """C: native CLIP F/Dall/D3 alignment with every mask/fusion module frozen."""

    ablation_mode = 'nested_alignment_only'

    def __init__(self, clip, checkpoint_encoders=True):
        super().__init__()
        self.clip = clip
        self.arm = 'A3'
        self.image_chunk = 128
        self.text_chunk = 128
        self.checkpoint_pair_blocks = False
        self.condition_mode = 'native'
        self.checkpoint_encoders = bool(checkpoint_encoders)
        # The common step0 contains the random mask network.  It is retained
        # for state identity but is not part of this objective.
        self.clip.mask_net.requires_grad_(False)
        if checkpoint_encoders:
            from model.nested_semantic_mask import enable_encoder_checkpointing
            enable_encoder_checkpointing(clip)

    def optimizer_groups(self):
        params = [p for p in self.clip.parameters() if p.requires_grad]
        return [dict(params=params, name='backbone', lr=1e-6, peak_lr=1e-6,
                     weight_decay=.01,
                     param_names=[n for n, p in self.clip.named_parameters()
                                  if p.requires_grad])]

    def forward(self, images, tokens_f, tokens_o, tokens_e, valid, completed=0):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=images.is_cuda):
            z = self.clip.encode_image(images)
        z = F.normalize(z.float(), dim=-1, eps=1e-6)
        global_z = gather(z)
        all_valid = torch.ones_like(valid)
        vf, lf = _native_terms(self, z, tokens_f, all_valid,
                               torch.ones_like(gather(valid, False)), global_z)
        valid_global = gather(valid, False)
        if bool(valid_global.any()):
            vo, lo = _native_terms(self, z, tokens_o, valid, valid_global, global_z)
            ve, le = _native_terms(self, z, tokens_e, valid, valid_global, global_z)
            loss = (10 / 3) * (1.35 * vf + 1.35 * vo + .30 * ve)
        else:
            vo = ve = z.sum() * 0
            lo = le = torch.zeros(2, device=z.device)
            loss = (10 / 3) * 1.35 * vf
        logs = dict(F_i2t=lf[0], F_t2i=lf[1], O_i2t=lo[0], O_t2i=lo[1],
                    E_i2t=le[0], E_t2i=le[1], F_sparse=0., O_sparse=0., E_sparse=0.,
                    loss=global_sum(loss) / world_rank()[0], valid_global=int(valid_global.sum()),
                    inc=0., inc_weight=0., hard_inclusion_violation=0., oe_iou=0.,
                    fusion='none', visual='none', condition_mode='native',
                    ablation_mode=self.ablation_mode,
                    nonfinite=global_sum((~torch.isfinite(loss)).sum()))
        return loss, {key: value.detach() if torch.is_tensor(value) else value
                      for key, value in logs.items()}


class TextOnlySmartMask(BalancedSearch):
    """D: full HNS objective with mask logits conditioned only on text."""

    ablation_mode = 'text_mask_full'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_only_mask = True
