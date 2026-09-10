"""Phase 2.8A tests: minimal Unsaid V1 core.

Covered here: anti-routing (A), the explicit "not ``1 - A_said``" guarantee (B),
complement-target orthogonality (C), stop-gradient / prediction-gradient split (D),
degenerate residual and all-invalid batches (E), attention overlap (F), the
``lambda_unsaid == 0`` backward-compatibility gate (G) and the state-dict /
legacy-checkpoint compatibility gate (H).
"""
import math
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, 'train')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import unsaid_core  # noqa: E402
from model.salu_model import (  # noqa: E402
    SALUModel,
    contrastive_loss,
    gather_features_with_grad,
    identifiable_said_loss,
)
from model.salu_modules import SaidRouter  # noqa: E402

DIM = 8
PATCHES = 5
TOKENS = 6
BATCH = 4
LEGACY_CHECKPOINT = os.path.join(REPO_ROOT, 'runs_salu', 'phase22', 'salu_said_only_last.pt')


class _StubClip(nn.Module):
    """Small differentiable stand-in for the wrapped CLIP model (no weights needed).

    The global embedding and the patch features come from different projections and
    the per-patch offsets differ, so ``z_G`` is not parallel to ``z_S`` (otherwise
    every complement residual would be degenerate) and routing actually changes the
    pooled vector.
    """

    def __init__(self, dim=DIM, patches=PATCHES):
        super().__init__()
        self.dim = dim
        self.patches = patches
        self.text_projection = nn.Parameter(torch.zeros(dim, dim))   # SALUModel reads shape[1]
        self.logit_scale = nn.Parameter(torch.tensor(math.log(20.0)))
        self.global_proj = nn.Linear(dim, dim)                       # global image embedding
        self.patch_proj = nn.Linear(dim, dim)                        # per-patch features
        self.text_proj = nn.Linear(dim, dim)
        generator = torch.Generator().manual_seed(11)
        self.register_buffer('patch_offsets', torch.randn(patches, dim, generator=generator))

    def _patches(self, images):
        base = self.patch_proj(images.flatten(1)[:, :self.dim])
        return base.unsqueeze(1) + self.patch_offsets.unsqueeze(0)

    def encode_image(self, images):
        return self.global_proj(images.flatten(1)[:, :self.dim])

    def encode_image_with_patches(self, images, use_checkpoint=False):
        return self.encode_image(images), self._patches(images)

    def encode_text(self, tokens):
        pooled = tokens.float().mean(dim=1, keepdim=True).repeat(1, self.dim)
        return self.text_proj(pooled)


def build_model(**kwargs):
    kwargs.setdefault('pair_chunk_size', None)
    return SALUModel(_StubClip(), tau_said=0.07, **kwargs)


def make_batch(batch=BATCH):
    torch.manual_seed(3)
    images = torch.randn(batch, 3, 4, 4)
    texts = torch.randint(0, 10, (batch, TOKENS))
    return images, texts


def gradients_of(model, names):
    parameters = dict(model.named_parameters())
    return {name: parameters[name].grad for name in names}


# --------------------------------------------------------------------------- #
# A. anti-routing
# --------------------------------------------------------------------------- #
def test_anti_routing_puts_mass_on_low_scores():
    scores = torch.tensor([[2.0, 0.0, -2.0]])
    A_said = F.softmax(scores / 0.07, dim=-1)
    A_unsaid = unsaid_core.complement_attention(scores, 0.07)

    assert A_said.argmax(dim=-1).item() == 0
    assert A_unsaid.argmax(dim=-1).item() == 2
    assert A_said[0, 0] > A_said[0, 1] > A_said[0, 2]
    assert A_unsaid[0, 2] > A_unsaid[0, 1] > A_unsaid[0, 0]
    assert A_unsaid.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert torch.isfinite(A_unsaid).all()
    assert (A_unsaid >= 0).all()
    # temperature: a smaller tau sharpens the complement selection
    mild = torch.tensor([[0.4, 0.0, -0.4]])
    softer = unsaid_core.complement_attention(mild, 0.07)
    sharper = unsaid_core.complement_attention(mild, 0.01)
    assert sharper[0, 2] > softer[0, 2]
    assert unsaid_core.attention_entropy(sharper).item() < unsaid_core.attention_entropy(softer).item()
    with pytest.raises(ValueError):
        unsaid_core.complement_attention(scores, 0.0)


# --------------------------------------------------------------------------- #
# B. anti-softmax, never 1 - A_said
# --------------------------------------------------------------------------- #
def test_unsaid_is_negative_softmax_not_one_minus_said():
    scores = torch.tensor([[1.5, 0.2, -0.7, -2.0]])
    A_said = F.softmax(scores / 0.07, dim=-1)
    A_unsaid = unsaid_core.complement_attention(scores, 0.07)
    one_minus = 1.0 - A_said

    # exactly softmax(-s / tau): same values, same order of operations
    assert torch.equal(A_unsaid, F.softmax(-scores / 0.07, dim=-1))
    assert A_unsaid.sum().item() == pytest.approx(1.0, abs=1e-6)
    # 1 - A_said is not even a distribution when N > 2 (its sum is N - 1)
    assert one_minus.sum().item() == pytest.approx(3.0, abs=1e-6)
    assert abs(one_minus.sum().item() - 1.0) > 0.5
    # and it is numerically far from the anti-routing distribution
    assert (A_unsaid - one_minus).abs().max().item() > 0.3
    assert not torch.allclose(A_unsaid, one_minus, atol=1e-3)


# --------------------------------------------------------------------------- #
# C. complement target
# --------------------------------------------------------------------------- #
def test_complement_target_is_unit_norm_and_orthogonal_to_said():
    torch.manual_seed(0)
    z_global = torch.randn(BATCH, DIM)
    z_said = torch.randn(BATCH, DIM)
    target, residual_norm, valid = unsaid_core.complement_target(z_global, z_said, 1e-4)

    assert valid.all()
    assert torch.allclose(target.norm(dim=-1), torch.ones(BATCH), atol=1e-5)
    cosines = (target * F.normalize(z_said, dim=-1)).sum(dim=-1)
    assert cosines.abs().max().item() < 1e-5
    # the target is the global direction with the Said direction removed
    g = F.normalize(z_global, dim=-1)
    s = F.normalize(z_said, dim=-1)
    expected_norm = (g - (g * s).sum(-1, keepdim=True) * s).norm(dim=-1)
    assert torch.allclose(residual_norm, expected_norm, atol=1e-6)
    assert torch.isfinite(target).all() and torch.isfinite(residual_norm).all()


# --------------------------------------------------------------------------- #
# D. stop-gradient on the target, gradient on the prediction
# --------------------------------------------------------------------------- #
def test_target_branch_is_stop_gradient_and_prediction_is_differentiable():
    z_global = torch.randn(3, DIM, requires_grad=True)
    z_said = torch.randn(3, DIM, requires_grad=True)
    target, residual_norm, valid = unsaid_core.complement_target(z_global, z_said, 1e-4)
    assert target.requires_grad is False
    assert residual_norm.requires_grad is False

    attention = torch.rand(3, PATCHES)
    attention = attention / attention.sum(dim=-1, keepdim=True)
    patches = torch.randn(3, PATCHES, DIM, requires_grad=True)
    z_unsaid = unsaid_core.complement_representation(attention, patches)
    loss = unsaid_core.complement_loss(z_unsaid, target, valid)
    loss.backward()

    assert patches.grad is not None and torch.isfinite(patches.grad).all()
    assert patches.grad.abs().sum().item() > 0
    assert z_global.grad is None and z_said.grad is None   # target side is frozen


def test_unsaid_gradients_reach_q_proj_k_proj_and_the_visual_backbone():
    model = build_model()
    images, texts = make_batch()
    out = model.forward_train(images, texts, 1.0, 1.0, lambda_unsaid=0.25)
    assert out['unsaid_enabled'] is True
    assert torch.isfinite(out['loss_unsaid'])

    out['loss_unsaid'].backward()
    grads = gradients_of(model, ['said_router.q_proj.weight', 'said_router.k_proj.weight',
                                 'clip.patch_proj.weight'])
    for name, grad in grads.items():
        assert grad is not None, name
        assert torch.isfinite(grad).all(), name
        assert grad.abs().sum().item() > 0, name
    # the target branch is stop-gradient: the global projection gets nothing from
    # L_unsaid (it only feeds the frozen target, not the prediction)
    assert dict(model.named_parameters())['clip.global_proj.weight'].grad is None


def test_unsaid_attention_is_not_identical_to_said_attention():
    model = build_model()
    images, texts = make_batch()
    out = model.forward_train(images, texts, 1.0, 1.0, lambda_unsaid=0.25)
    overlap = float(out['said_unsaid_attention_overlap'])
    jsd = float(out['said_unsaid_attention_jsd'])
    assert 0.0 <= overlap <= 1.0
    assert overlap < 1.0 and jsd > 0.0
    assert torch.isfinite(out['loss_unsaid']) and torch.isfinite(out['loss_total'])


# --------------------------------------------------------------------------- #
# E. degenerate residual / all-invalid batch
# --------------------------------------------------------------------------- #
def test_degenerate_residual_is_invalid_and_loss_is_zero_without_nan():
    z = F.normalize(torch.randn(BATCH, DIM), dim=-1)
    target, residual_norm, valid = unsaid_core.complement_target(z, z.clone(), 1e-4)

    assert not valid.any()
    assert torch.isfinite(residual_norm).all() and residual_norm.max().item() == pytest.approx(0.0, abs=1e-6)
    assert torch.isfinite(target).all() and target.abs().max().item() == 0.0

    z_unsaid = F.normalize(torch.randn(BATCH, DIM), dim=-1).requires_grad_(True)
    loss = unsaid_core.complement_loss(z_unsaid, target, valid)
    assert loss.item() == 0.0
    assert torch.isfinite(loss)
    loss.backward()                                   # backward stays legal
    assert z_unsaid.grad is not None and torch.isfinite(z_unsaid.grad).all()
    assert z_unsaid.grad.abs().sum().item() == 0.0


def test_mixed_validity_averages_only_over_valid_samples():
    z_global = torch.randn(3, DIM)
    z_said = z_global.clone()
    z_said[0] = torch.randn(DIM)
    target, residual_norm, valid = unsaid_core.complement_target(z_global, z_said, 1e-4)
    assert valid.tolist() == [True, False, False]

    z_unsaid = F.normalize(torch.randn(3, DIM), dim=-1)
    loss = unsaid_core.complement_loss(z_unsaid, target, valid)
    expected = 1.0 - (z_unsaid[0] * target[0]).sum()
    assert loss.item() == pytest.approx(expected.item(), abs=1e-6)
    assert torch.isfinite(unsaid_core.masked_mean(residual_norm, valid))


def test_unsaid_is_finite_under_bf16_autocast():
    model = build_model()
    images, texts = make_batch()
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        out = model.forward_train(images, texts, 1.0, 1.0, lambda_unsaid=0.25)
    assert torch.isfinite(out['loss_unsaid'])
    assert torch.isfinite(out['loss_total'])
    for key in unsaid_core.DIAGNOSTIC_KEYS:
        assert torch.isfinite(out[key]), key


# --------------------------------------------------------------------------- #
# F. attention overlap
# --------------------------------------------------------------------------- #
def test_attention_overlap_bounds_and_separation():
    identical = torch.tensor([[0.5, 0.3, 0.2]])
    assert unsaid_core.attention_overlap(identical, identical).item() == pytest.approx(1.0, abs=1e-6)
    disjoint = torch.tensor([[1.0, 0.0, 0.0]])
    other = torch.tensor([[0.0, 0.0, 1.0]])
    assert unsaid_core.attention_overlap(disjoint, other).item() == pytest.approx(0.0, abs=1e-9)

    scores = torch.tensor([[2.0, 0.0, -2.0]])
    A_said = F.softmax(scores / 0.07, dim=-1)
    A_unsaid = unsaid_core.complement_attention(scores, 0.07)
    overlap = unsaid_core.attention_overlap(A_said, A_unsaid)
    assert 0.0 <= overlap.item() <= 1.0
    assert overlap.item() < unsaid_core.attention_overlap(A_said, A_said).item()

    # entropy / effective patch count stay in range, and JSD is a real divergence
    entropy = unsaid_core.attention_entropy(A_unsaid)
    assert 0.0 <= entropy.item() <= math.log(3.0) + 1e-6
    assert 1.0 <= entropy.exp().item() <= 3.0 + 1e-6
    assert unsaid_core.attention_jsd(A_said, A_said).item() == pytest.approx(0.0, abs=1e-9)
    assert unsaid_core.attention_jsd(A_said, A_unsaid).item() > 0.0


# --------------------------------------------------------------------------- #
# G. lambda_unsaid = 0 backward compatibility
# --------------------------------------------------------------------------- #
def test_lambda_zero_skips_the_unsaid_path_entirely(monkeypatch):
    model = build_model()
    images, texts = make_batch()

    def forbidden(*args, **kwargs):
        raise AssertionError('the Unsaid path must not run when lambda_unsaid == 0')

    monkeypatch.setattr(unsaid_core, 'complement_attention', forbidden)
    monkeypatch.setattr(SaidRouter, 'forward_with_details', forbidden)

    out = model.forward_train(images, texts, 1.0, 1.0, lambda_unsaid=0.0)
    assert out['unsaid_enabled'] is False
    assert float(out['loss_unsaid']) == 0.0
    assert all(out[key] is None for key in unsaid_core.DIAGNOSTIC_KEYS)
    assert torch.isfinite(out['loss_total'])

    # the hooks are effective: with Unsaid enabled the patched call is reached
    with pytest.raises(AssertionError):
        model.forward_train(images, texts, 1.0, 1.0, lambda_unsaid=0.25)


def test_lambda_zero_reproduces_the_said_only_objective_exactly():
    model = build_model()
    images, texts = make_batch()
    out = model.forward_train(images, texts, 0.7, 1.3, lambda_unsaid=0.0)

    # independent re-implementation of the Phase 2.7C Said-only forward
    z_global, patch_features = model.encode_router_input(images)
    text_features = model.clip.encode_text(texts)
    z_g = F.normalize(z_global, dim=-1)
    t = F.normalize(text_features, dim=-1)
    A_own, z_s_own = model.said_router(t, patch_features)
    scale = model.clip.logit_scale.exp().clamp(max=100)
    ref_global = contrastive_loss(gather_features_with_grad(z_g), gather_features_with_grad(t), scale)
    z_pair, _ = model.said_router.route_pairwise(t, patch_features, chunk_size=None)
    ref_said = identifiable_said_loss(z_pair, t, scale)

    assert torch.equal(out['loss_global'], ref_global)
    assert torch.equal(out['loss_said'], ref_said['loss_said'])
    assert torch.equal(out['loss_route'], ref_said['loss_route'].detach())
    assert torch.equal(out['loss_evidence'], ref_said['loss_evidence'].detach())
    assert torch.equal(out['route_top1_acc'], ref_said['route_top1_acc'].detach())
    assert torch.equal(out['evidence_top1_acc'], ref_said['evidence_top1_acc'].detach())
    assert torch.equal(out['route_margin'], ref_said['route_margin'].detach())
    assert torch.equal(out['evidence_margin'], ref_said['evidence_margin'].detach())
    assert torch.equal(out['loss_total'], 0.7 * ref_global + 1.3 * ref_said['loss_said'])

    entropy = SaidRouter.attention_entropy(A_own.detach().float())
    assert torch.equal(out['said_attention_entropy'], entropy.mean())
    assert torch.equal(out['said_effective_patch_count'], entropy.exp().mean())
    assert torch.equal(out['said_feature_norm'], z_s_own.detach().float().norm(dim=-1).mean())


def test_enabled_and_disabled_said_attention_are_bit_identical():
    """Enabling Unsaid may not change a single Said number (details reuse q/k/s)."""
    model = build_model()
    images, texts = make_batch()
    disabled = model.forward_train(images, texts, 1.0, 1.0, lambda_unsaid=0.0)
    enabled = model.forward_train(images, texts, 1.0, 1.0, lambda_unsaid=0.25)

    for key in ('loss_global', 'loss_said', 'loss_route', 'loss_evidence',
                'route_top1_acc', 'evidence_top1_acc', 'route_margin', 'evidence_margin',
                'said_attention_entropy', 'said_effective_patch_count', 'said_attention_max',
                'said_attention_min', 'said_feature_norm', 'pair_gap_full', 'pair_gap_said',
                'balancing_gain'):
        assert torch.equal(disabled[key], enabled[key]), key


def test_unsaid_adds_no_pairwise_routing():
    """Minimal Unsaid uses the own caption only: no Unsaid(I_i, C_j) pairwise pass."""
    model = build_model()
    images, texts = make_batch()
    calls = []
    original = SaidRouter.route_pairwise

    def counting(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    SaidRouter.route_pairwise = counting
    try:
        model.forward_train(images, texts, 1.0, 1.0, lambda_unsaid=0.25)
    finally:
        SaidRouter.route_pairwise = original
    assert len(calls) == 1          # the identifiable Said path, and nothing else


def test_loss_total_includes_lambda_unsaid_only_when_enabled():
    model = build_model()
    images, texts = make_batch()
    out = model.forward_train(images, texts, 1.0, 0.0, lambda_unsaid=0.25)
    expected = out['loss_global'] + 0.25 * out['loss_unsaid']
    assert torch.allclose(out['loss_total'], expected, atol=1e-6)
    assert not torch.allclose(out['loss_total'], out['loss_global'], atol=1e-9)


# --------------------------------------------------------------------------- #
# H. parameters / state dict / legacy checkpoint
# --------------------------------------------------------------------------- #
def test_no_new_parameters_or_state_dict_keys():
    model = build_model()
    keys = sorted(model.state_dict())
    assert not any('unsaid' in key for key in keys)
    assert sorted(name for name, p in model.named_parameters() if p.requires_grad) == [
        'clip.global_proj.bias', 'clip.global_proj.weight', 'clip.logit_scale',
        'clip.patch_proj.bias', 'clip.patch_proj.weight', 'clip.text_proj.bias',
        'clip.text_proj.weight', 'clip.text_projection',
        'said_router.k_proj.bias', 'said_router.k_proj.weight',
        'said_router.q_proj.bias', 'said_router.q_proj.weight',
    ]
    # the router is still exactly q_proj + k_proj
    assert sorted(model.said_router.state_dict()) == [
        'k_proj.bias', 'k_proj.weight', 'q_proj.bias', 'q_proj.weight']


def test_standard_inference_paths_are_untouched():
    model = build_model()
    images, _ = make_batch()
    with torch.no_grad():
        assert torch.equal(model.encode_image(images), model.clip.encode_image(images))
        z, patches = model.encode_router_input(images)
        assert torch.equal(z, model.clip.encode_image(images))
        assert torch.equal(patches, model.clip.encode_image_with_patches(images)[1])
        tokens = torch.randint(0, 10, (BATCH, TOKENS))
        assert torch.equal(model.encode_text(tokens), model.clip.encode_text(tokens))


@pytest.mark.skipif(not os.path.exists(LEGACY_CHECKPOINT),
                    reason='Phase 2.2 Said-only checkpoint is not present in this checkout')
def test_legacy_said_only_checkpoint_loads_with_empty_key_diff():
    """The reviewed Said-only checkpoint must still load strictly (no key diff)."""
    from model import longclip
    from train_salu import parse_args

    checkpoint = torch.load(LEGACY_CHECKPOINT, map_location='cpu', weights_only=False)
    args = parse_args([])
    args.base_model = 'ViT-B/16'
    try:
        clip_model, _ = longclip.load_from_clip(args.base_model, device='cpu',
                                                download_root=args.download_root, args=args)
    except Exception as exc:                                  # weights not cached here
        pytest.skip('CLIP weights unavailable: %s' % exc)

    model = SALUModel(clip_model, tau_said=args.tau_said, said_loss_mode=args.said_loss_mode,
                      pair_chunk_size=args.pair_chunk_size,
                      said_feature_source=args.said_feature_source)
    incompatible = model.load_state_dict(checkpoint['model'], strict=False)
    assert list(incompatible.missing_keys) == []
    assert list(incompatible.unexpected_keys) == []
    model.load_state_dict(checkpoint['model'], strict=True)   # must not raise

    assert set(model.state_dict()) == set(checkpoint['model'])
    assert sum(p.numel() for p in model.parameters()) == \
        sum(value.numel() for value in checkpoint['model'].values())
