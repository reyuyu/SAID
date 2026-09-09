"""Phase 2.2 unit tests: identifiable Said routing objective.

Covers
  A. generic-saliency degeneracy: caption-independent routing cannot beat log(B)
  B. pairwise routing shapes / normalisation / caption dependence
  C. chunked vs unchunked numerical equivalence + gradients
  D. gradients reach router, text features, patch features, vision/text encoders
"""
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.salu_model import SALUModel, identifiable_said_loss  # noqa: E402
from model.salu_modules import SaidRouter  # noqa: E402


def _random_texts(batch, dim, seed=0, requires_grad=False):
    torch.manual_seed(seed)
    return F.normalize(torch.randn(batch, dim, requires_grad=requires_grad), dim=-1)


# --------------------------------------------------------------------------- #
# A. degeneracy test (STEP 11)
# --------------------------------------------------------------------------- #
def test_a_caption_independent_routing_cannot_beat_log_b():
    torch.manual_seed(0)
    batch, dim, scale = 8, 16, torch.tensor(14.2784)

    texts = _random_texts(batch, dim, seed=1)

    # Case A: every caption routes image i to the SAME z (caption ignored)
    z_shared = F.normalize(torch.randn(batch, dim), dim=-1)
    z_pair_ignoring_caption = z_shared.unsqueeze(1).repeat(1, batch, 1)
    stats_a = identifiable_said_loss(z_pair_ignoring_caption, texts, scale)

    # Case B: the correct caption route lands on t_i, others are random
    z_pair_identifiable = F.normalize(torch.randn(batch, batch, dim), dim=-1)
    for i in range(batch):
        z_pair_identifiable[i, i] = texts[i]
    stats_b = identifiable_said_loss(z_pair_identifiable, texts, scale)

    log_b = float(torch.log(torch.tensor(float(batch))))
    print('case A loss_route', float(stats_a['loss_route']), 'top1', float(stats_a['route_top1_acc']))
    print('case B loss_route', float(stats_b['loss_route']), 'top1', float(stats_b['route_top1_acc']))

    # caption-independent routing sits exactly at the chance level log(B)
    assert abs(float(stats_a['loss_route']) - log_b) < 1e-4
    assert float(stats_a['route_top1_acc']) <= 1.0 / batch + 1e-6
    # a truly caption-dependent routing must be much better
    assert float(stats_b['loss_route']) < float(stats_a['loss_route']) - 0.5
    assert float(stats_b['route_top1_acc']) == 1.0
    assert float(stats_b['route_margin']) > float(stats_a['route_margin'])


# --------------------------------------------------------------------------- #
# B. pairwise routing (STEP 12)
# --------------------------------------------------------------------------- #
def test_b_pairwise_routing_shapes_and_caption_dependence():
    torch.manual_seed(0)
    batch, n_patches, dim = 4, 16, 32
    router = SaidRouter(dim=dim, tau_said=0.07)
    texts = _random_texts(batch, dim, seed=2)
    patches = torch.randn(batch, n_patches, dim)

    z_pair, A_pair = router.route_pairwise(texts, patches, chunk_size=None, return_attention=True)

    assert tuple(z_pair.shape) == (batch, batch, dim)
    assert tuple(A_pair.shape) == (batch, batch, n_patches)
    assert bool(torch.isfinite(z_pair).all()) and bool(torch.isfinite(A_pair).all())
    assert torch.allclose(A_pair.sum(dim=-1), torch.ones(batch, batch), atol=1e-5)
    assert bool((A_pair >= 0).all())
    assert torch.allclose(z_pair.norm(dim=-1), torch.ones(batch, batch), atol=1e-5)

    # same image, different captions -> different attention
    diffs = [(A_pair[0, j] - A_pair[0, 0]).abs().max().item() for j in range(1, batch)]
    print('attention difference for fixed image across captions', diffs)
    assert max(diffs) > 0.0


# --------------------------------------------------------------------------- #
# C. chunking equivalence (STEP 13)
# --------------------------------------------------------------------------- #
def test_c_chunked_equals_unchunked_and_has_gradients():
    torch.manual_seed(0)
    batch, n_patches, dim = 6, 12, 16
    scale = torch.tensor(14.2784)

    router = SaidRouter(dim=dim, tau_said=0.07)
    texts_leaf = torch.randn(batch, dim, requires_grad=True)
    patches_leaf = torch.randn(batch, n_patches, dim, requires_grad=True)
    texts = F.normalize(texts_leaf, dim=-1)
    patches = patches_leaf

    z_full, A_full = router.route_pairwise(texts, patches, chunk_size=None, return_attention=True)
    z_chunk, A_chunk = router.route_pairwise(texts, patches, chunk_size=2, return_attention=True)
    assert torch.allclose(z_full, z_chunk, atol=1e-5)
    assert torch.allclose(A_full, A_chunk, atol=1e-5)

    stats_full = identifiable_said_loss(z_full, texts, scale)
    stats_chunk = identifiable_said_loss(z_chunk, texts, scale)
    assert torch.allclose(stats_full['loss_route'], stats_chunk['loss_route'], atol=1e-5)
    assert torch.allclose(stats_full['loss_evidence'], stats_chunk['loss_evidence'], atol=1e-5)

    (stats_full['loss_said'] + stats_chunk['loss_said']).backward()
    assert router.q_proj.weight.grad is not None
    assert router.k_proj.weight.grad is not None
    assert texts_leaf.grad is not None
    assert patches_leaf.grad is not None
    for name, grad in (('q_proj', router.q_proj.weight.grad), ('k_proj', router.k_proj.weight.grad),
                       ('texts', texts_leaf.grad), ('patches', patches_leaf.grad)):
        assert bool(torch.isfinite(grad).all()), '%s grad not finite' % name
        assert float(grad.abs().sum()) > 0.0, '%s grad is zero' % name


# --------------------------------------------------------------------------- #
# D. gradients reach encoders / router / features (STEP 14)
# --------------------------------------------------------------------------- #
class _TinyCLIP(nn.Module):
    """Minimal stand-in exposing the CLIP surface SALUModel uses."""

    def __init__(self, dim=32, n_patches=16):
        super().__init__()
        self.dim = dim
        self.n_patches = n_patches
        self.text_projection = nn.Parameter(torch.eye(dim))
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.vision_backbone = nn.Linear(dim, dim)
        self.text_encoder = nn.Linear(dim, dim)
        self.mask_net = nn.Linear(dim, dim)

    def encode_image_with_patches(self, images, use_checkpoint=False):
        global_feature = self.vision_backbone(images)
        scales = torch.linspace(0.5, 1.5, self.n_patches, device=images.device)
        patches = global_feature.unsqueeze(1) * scales.view(1, -1, 1)
        return global_feature, patches

    def encode_text(self, texts):
        return self.text_encoder(texts)

    def encode_image(self, images):
        return self.vision_backbone(images)


def test_d_identifiable_gradients_reach_all_parts():
    torch.manual_seed(0)
    batch, dim = 6, 32
    clip = _TinyCLIP(dim=dim)
    model = SALUModel(clip, dim=dim, said_loss_mode='identifiable',
                      pair_chunk_size=None, fp32_master_weights=False)

    images = torch.randn(batch, dim, requires_grad=True)
    texts = torch.randn(batch, dim, requires_grad=True)
    out = model.forward_train(images, texts)
    out['loss_total'].backward()

    checks = {
        'said_q_proj': model.said_router.q_proj.weight.grad,
        'said_k_proj': model.said_router.k_proj.weight.grad,
        'vision_backbone': clip.vision_backbone.weight.grad,
        'text_encoder': clip.text_encoder.weight.grad,
        'images': images.grad,
        'texts': texts.grad,
    }
    for name, grad in checks.items():
        assert grad is not None, '%s has no gradient' % name
        assert bool(torch.isfinite(grad).all()), '%s gradient not finite' % name
        assert float(grad.abs().sum()) > 0.0, '%s gradient is zero' % name
    # mask_net stays frozen / unused
    assert clip.mask_net.weight.grad is None
    # new diagnostics are present
    for key in ('loss_route', 'loss_evidence', 'route_top1_acc', 'evidence_top1_acc',
                'route_margin', 'evidence_margin'):
        assert key in out, 'missing diagnostic %s' % key


def test_e_positive_mode_is_preserved_as_ablation():
    torch.manual_seed(0)
    batch, dim = 6, 32
    clip = _TinyCLIP(dim=dim)
    model = SALUModel(clip, dim=dim, said_loss_mode='positive', fp32_master_weights=False)
    images = torch.randn(batch, dim)
    texts = torch.randn(batch, dim)
    out = model.forward_train(images, texts)
    assert float(out['loss_route']) == 0.0
    assert float(out['loss_evidence']) == 0.0
    assert float(out['loss_said']) > 0.0


def test_f_invalid_mode_rejected():
    with pytest.raises(ValueError):
        SALUModel(_TinyCLIP(), said_loss_mode='prototype')
