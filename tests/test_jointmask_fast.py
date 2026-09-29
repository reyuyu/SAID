"""Acceptance tests for the fixed NEST JointMask fast experiment configuration."""
import json
from pathlib import Path

import torch
from torch import nn

from model.nested_semantic_mask import (NestedSemanticMask, _checkpoint_blocks,
                                        positive_masks)
from tests.test_nested_jointmask import TinyJointCLIP


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    'T-fast': ('configs/nest_jointmask_t_fast.json', 'text_only'),
    'TI-fast': ('configs/nest_jointmask_ti_fast.json', 'joint_image'),
    'TI-Shuffle-fast': ('configs/nest_jointmask_ti_shuffle_fast.json',
                        'joint_shuffled_image'),
}


class DummyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.resblocks = nn.ModuleList([nn.Identity()])

    def forward(self, value):
        return value


def checkpoint_capable_tiny_clip():
    clip = TinyJointCLIP()
    clip.visual.transformer = DummyTransformer()
    clip.transformer = DummyTransformer()
    return clip


def test_fast_configs_reach_actual_model_instances():
    for _, (path, condition_mode) in CONFIGS.items():
        config = json.loads((ROOT / path).read_text())
        module = NestedSemanticMask(
            checkpoint_capable_tiny_clip(), arm=config['arm'],
            checkpoint_encoders=config['checkpoint_encoders'],
            image_chunk=config['image_chunk'], text_chunk=config['text_chunk'],
            condition_mode=config['condition_mode'], shuffle_seed=config['shuffle_seed'],
            checkpoint_pair_blocks=config['checkpoint_pair_blocks'])
        assert module.image_chunk == 128 and module.text_chunk == 128
        assert module.checkpoint_pair_blocks is False
        assert module.checkpoint_encoders is True
        assert module.condition_mode == condition_mode
        assert all(getattr(transformer.forward, '__func__', None) is _checkpoint_blocks
                   for transformer in (module.clip.visual.transformer,
                                       module.clip.transformer))


def test_nonzero_joint_adapter_uses_visual_condition_without_changing_formal_init():
    torch.manual_seed(73)
    module = NestedSemanticMask(TinyJointCLIP(), checkpoint_encoders=False,
                                condition_mode='joint_image')
    assert module.joint_adapter.output.weight.count_nonzero() == 0
    with torch.no_grad():
        module.joint_adapter.output.weight.normal_(std=.1)
    logits = torch.randn(1, 8)
    text_condition = torch.randn(1, 64)
    image_condition = torch.stack((torch.ones(64), -torch.ones(64)))
    _, probabilities, delta = positive_masks(
        logits.expand(2, -1), image_condition,
        text_condition.expand(2, -1), module.joint_adapter)
    assert delta.abs().max() > 0
    assert not torch.equal(delta[0], delta[1])
    assert not torch.equal(probabilities[0], probabilities[1])
