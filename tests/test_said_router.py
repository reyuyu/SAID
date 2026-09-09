"""Phase 2 unit tests: caption-conditioned Said Router (Said-only).

Only the router / wrapper are tested here -- no prototype bank, no unsaid
explorer, no training loop.
"""
import os
import sys

import pytest
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.salu_model import SALUModel  # noqa: E402
from model.salu_modules import SaidRouter  # noqa: E402

B, N, D = 2, 196, 512
TAU = 0.07


def _inputs(seed=0, requires_grad=False):
    torch.manual_seed(seed)
    text_feature = torch.randn(B, D, requires_grad=requires_grad)
    patch_features = torch.randn(B, N, D, requires_grad=requires_grad)
    return text_feature, patch_features


@pytest.fixture(scope='module')
def router():
    torch.manual_seed(0)
    return SaidRouter(dim=D, tau_said=TAU)


def test_a_shape(router):
    text_feature, patch_features = _inputs()
    A_s, z_s = router(text_feature, patch_features)
    print('A_s', tuple(A_s.shape), 'z_s', tuple(z_s.shape))
    assert tuple(A_s.shape) == (B, N)
    assert tuple(z_s.shape) == (B, D)


def test_b_attention_normalization(router):
    text_feature, patch_features = _inputs(seed=1)
    A_s, _ = router(text_feature, patch_features)
    sums = A_s.sum(dim=-1)
    print('attention sums', sums.tolist())
    assert torch.allclose(sums, torch.ones(B), atol=1e-5)
    assert bool((A_s >= 0).all())


def test_c_finite(router):
    text_feature, patch_features = _inputs(seed=2)
    A_s, z_s = router(text_feature, patch_features)
    assert bool(torch.isfinite(A_s).all())
    assert bool(torch.isfinite(z_s).all())


def test_d_zs_normalized(router):
    text_feature, patch_features = _inputs(seed=3)
    _, z_s = router(text_feature, patch_features)
    norms = z_s.norm(dim=-1)
    print('z_s norms', norms.tolist())
    assert torch.allclose(norms, torch.ones(B), atol=1e-5)


def test_e_gradients(router):
    """Said loss must reach the router *and* the vision/text features."""
    text_feature, patch_features = _inputs(seed=4, requires_grad=True)
    _, z_s = router(text_feature, patch_features)
    loss = z_s.sum()
    loss.backward()

    assert router.q_proj.weight.grad is not None, 'q_proj has no gradient'
    assert router.k_proj.weight.grad is not None, 'k_proj has no gradient'
    assert patch_features.grad is not None, 'patch_features have no gradient'
    assert text_feature.grad is not None, 'text_feature has no gradient'

    for name, tensor in (
        ('q_proj.weight', router.q_proj.weight.grad),
        ('k_proj.weight', router.k_proj.weight.grad),
        ('patch_features', patch_features.grad),
        ('text_feature', text_feature.grad),
    ):
        assert bool(torch.isfinite(tensor).all()), '%s gradient is not finite' % name
        assert float(tensor.abs().sum()) > 0.0, '%s gradient is all zeros' % name
        print(name, 'grad_abs_sum', float(tensor.abs().sum()))


def test_f_caption_conditioning(router):
    """Same patches + different captions must give different attention."""
    text_a, patch_features = _inputs(seed=5)
    text_b = torch.randn(B, D)
    A_a, _ = router(text_a, patch_features)
    A_b, _ = router(text_b, patch_features)
    max_abs_diff = (A_a - A_b).abs().max().item()
    print('caption_conditioning_max_abs_diff', max_abs_diff)
    assert max_abs_diff > 0.0
    assert not torch.allclose(A_a, A_b, atol=1e-8, rtol=0.0)


class _DummyCLIP(nn.Module):
    """Minimal stand-in for CLIP: only what SALUModel touches."""

    def __init__(self, dim=16):
        super().__init__()
        self.text_projection = nn.Parameter(torch.randn(dim, dim))
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.encode_image_calls = 0
        self.encode_text_calls = 0

    def encode_image(self, image):
        self.encode_image_calls += 1
        return image

    def encode_text(self, text):
        self.encode_text_calls += 1
        return text


def test_g_no_prototype_or_unsaid():
    """Phase 2 must not contain prototype-bank / unsaid-explorer machinery."""
    model = SALUModel(_DummyCLIP(dim=16), tau_said=TAU)

    for forbidden in ('prototype_bank', 'prototype', 'unsaid_explorer', 'unsaid_queries', 'unsaid'):
        assert not hasattr(model, forbidden), 'unexpected attribute %r' % forbidden
    bad_modules = [
        name for name, _ in model.named_modules()
        if 'prototype' in name or 'unsaid' in name
    ]
    assert bad_modules == [], 'unexpected modules: %r' % bad_modules

    keys = set(model.said_router.state_dict().keys())
    print('said_router state_dict keys', sorted(keys))
    assert keys == {'q_proj.weight', 'q_proj.bias', 'k_proj.weight', 'k_proj.bias'}


def test_h_inference_delegates_to_clip():
    """Standard inference must go through CLIP, never the Said router."""
    clip = _DummyCLIP(dim=16)
    model = SALUModel(clip, tau_said=TAU)
    image = torch.randn(2, 3, 4, 4)
    text = torch.randn(2, 5)
    out_image = model.encode_image(image)
    out_text = model.encode_text(text)
    assert clip.encode_image_calls == 1 and clip.encode_text_calls == 1
    assert torch.equal(out_image, image) and torch.equal(out_text, text)


def test_i_mask_net_frozen_not_in_optimizer_groups():
    """mask_net stays in the model but must be frozen and excluded from groups."""
    clip = _DummyCLIP(dim=16)
    clip.mask_net = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16))
    model = SALUModel(clip, tau_said=TAU)

    assert all(not p.requires_grad for p in model.clip.mask_net.parameters())
    backbone_ids = {id(p) for p in model.backbone_parameters()}
    head_ids = {id(p) for p in model.said_head_parameters()}
    mask_ids = {id(p) for p in model.clip.mask_net.parameters()}
    assert mask_ids.isdisjoint(backbone_ids)
    assert mask_ids.isdisjoint(head_ids)
    assert len(head_ids) == 4  # q_proj + k_proj (weight, bias)
