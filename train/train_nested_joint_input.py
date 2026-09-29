"""Shared optimizer/schedule helpers for NEST input-level JointInput candidates.

Formal training is intentionally gated by the resource study. This module keeps the
optimizer definition testable without introducing a second copy of any model parameter.
"""
import math

import torch


def build_optimizer(module):
    mask_ids = {id(p) for p in module.clip.mask_net.parameters()}
    projection_ids = {id(p) for p in module.visual_input_projection.parameters()}
    backbone, mask, projection = [], [], []
    for parameter in module.parameters():
        if not parameter.requires_grad:
            continue
        target = (mask if id(parameter) in mask_ids else
                  projection if id(parameter) in projection_ids else backbone)
        target.append(parameter)
    params = backbone + mask + projection
    assert len(params) == len({id(p) for p in params})
    assert len(mask) == len(mask_ids) and len(projection) == len(projection_ids)
    return torch.optim.AdamW([
        dict(params=backbone, lr=1e-6, weight_decay=1e-2, name='backbone'),
        dict(params=mask, lr=1e-3, weight_decay=0., name='shared_mask'),
        dict(params=projection, lr=1e-4, weight_decay=0., name='visual_input_projection'),
    ], betas=(.9, .999), eps=1e-8)


def learning_rates(step, horizon):
    backbone = (1e-6 * (step + 1) / 200 if step < 200 else
                .5e-6 * (1 + math.cos(math.pi * (step - 200) / (horizon - 200))))
    return (backbone,
            .5e-3 * (1 + math.cos(math.pi * step / horizon)),
            .5e-4 * (1 + math.cos(math.pi * step / horizon)))
