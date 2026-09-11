"""SAID-ExGAP v1 tests.

Covers the five required first-stage tests (forward correctness, mask correctness, explanatory
gap, loss monotonicity, gradient isolation) plus the degenerate-mask contract and the
objective-mode / CLI validation.

The gradient-isolation tests are the ones that matter most: ``L_ExGAP`` must move only ``z_U``
(hence the surviving patches of the shared backbone) and must never be able to shrink the gap by
moving the Said router, the global pooling or the text encoder.
"""
import inspect
import math
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, 'train'), os.path.join(REPO_ROOT, 'tests')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import exgap  # noqa: E402
from model.salu_model import SALUModel  # noqa: E402
from test_gap_completion import build_model, make_batch  # noqa: E402

DIM = 8
PATCHES = 6
BATCH = 5


def exgap_forward(model, images, texts, **kwargs):
    kwargs.setdefault('objective_mode', 'said_exgap')
    kwargs.setdefault('lambda_global', 0.0)
    kwargs.setdefault('lambda_unsaid', 0.0)
    return model.forward_train(images, texts, **kwargs)


# --------------------------------------------------------------------------- #
# Test 1 -- forward correctness
# --------------------------------------------------------------------------- #
def test_forward_shapes_ranges_and_finiteness():
    model = build_model()
    images, texts = make_batch()
    out = exgap_forward(model, images, texts)

    assert set(out) >= {'loss_said', 'loss_route', 'loss_evidence', 'loss_exgap',
                        'loss_total', 'said_relevance', 'mask'}
    assert out['objective_mode'] == 'said_exgap'
    assert out['loss_global'] is None and out['loss_unsaid'] is None
    assert out['global_text_alignment_enabled'] is False

    # every returned tensor is finite
    for key, value in out.items():
        if torch.is_tensor(value):
            assert torch.isfinite(value).all(), key

    # relevance is a proper independent probability per patch
    relevance = out['said_relevance']
    assert relevance.shape == (BATCH, PATCHES)
    assert float(relevance.min()) >= 0.0 and float(relevance.max()) <= 1.0
    attention = out['said_attention']
    assert torch.allclose(attention.sum(dim=-1), torch.ones(BATCH), atol=1e-5)
    # the sigmoid relevance is NOT a softmax: it does not sum to 1 by construction
    assert not torch.allclose(relevance.sum(dim=-1), torch.ones(BATCH), atol=1e-3)

    # gap weight is in [0, 1] and carries no gradient
    assert float(out['explanatory_gap_norm_mean']) >= 0.0
    assert float(out['explanatory_gap_norm_mean']) <= 1.0
    assert out['explanatory_gap_raw_mean'].requires_grad is False

    # the three feature norms are ~1 (they are normalised representations)
    for key in ('global_feature_norm', 'said_feature_norm', 'unsaid_feature_norm'):
        assert float(out[key]) == pytest.approx(1.0, abs=1e-4), key
    # unit-norm patches are the precondition for the 1 - S_gc normalisation
    assert float(out['patch_norm_deviation']) < 1e-4


def test_forward_works_for_both_pooling_modes():
    for pool in exgap.GLOBAL_POOL_MODES:
        model = build_model()
        images, texts = make_batch()
        out = exgap_forward(model, images, texts, exgap_global_pool=pool)
        assert out['exgap_global_pool'] == pool
        for key in ('loss_exgap', 'loss_total', 'explanatory_gap_norm_mean',
                    'global_feature_norm', 'unsaid_feature_norm'):
            assert torch.isfinite(out[key]).all(), (pool, key)
        assert float(out['global_feature_norm']) == pytest.approx(1.0, abs=1e-4)
    # attention mode really does add parameters, mean mode really does not
    mean_model = build_model()
    exgap_forward(mean_model, *make_batch(), exgap_global_pool='mean')
    assert mean_model.exgap_module('mean') is None
    assert not any(name == 'exgap' or name.startswith('exgap.') for name, _ in
                   mean_model.named_parameters())
    attention_model = build_model()
    exgap_forward(attention_model, *make_batch(), exgap_global_pool='attention')
    assert any(name.startswith('exgap.') for name, _ in attention_model.named_parameters())


def test_attention_pooling_starts_uniform_and_ignores_the_caption():
    """z_G must be f(I): the global pooling never sees the caption."""
    model = build_model()
    images, texts = make_batch()
    first = model.encode_said_exgap(images, texts, exgap_global_pool='attention',
                                    return_details=True)
    second = model.encode_said_exgap(images, texts.flip(0), exgap_global_pool='attention',
                                     return_details=True)
    # different captions leave z_G bit-identical
    assert torch.equal(first['global_feature'], second['global_feature'])
    assert torch.equal(first['global_attention'], second['global_attention'])
    # ... while the Said side does react to the caption
    assert not torch.allclose(first['said_feature'], second['said_feature'])
    # zero-init output projection => uniform attention at initialisation
    uniform = torch.full_like(first['global_attention'], 1.0 / PATCHES)
    assert torch.allclose(first['global_attention'], uniform, atol=1e-5)


def test_said_relevance_is_alive_at_the_router_scale():
    """Regression: the mask must actually be able to fire.

    The literal ``(tW_Q)(h_pW_K)^T / sqrt(d)`` form was measured on the 4x A800 smoke run to be
    inert: with default-initialised projections the logit is ``~1e-4``, so
    ``r^S_p = sigmoid(1e-4) = 0.5 +- 4e-5``, ``r^S_p > tau_M = 0.6`` is never true and the whole
    ExGAP objective silently does nothing (``mask_keep_ratio`` 1.0, ``loss_exgap`` 0.0 at every
    step). The relevance therefore uses the Said router's own logit scale (``/ tau_said``).
    """
    torch.manual_seed(0)
    dim, patches, batch = 32, 16, 6
    # default nn.Linear init, exactly like ``SaidRouter.q_proj`` / ``k_proj``
    w_query, w_key = nn.Linear(dim, dim), nn.Linear(dim, dim)
    t = F.normalize(torch.randn(batch, dim), dim=-1)
    h = F.normalize(torch.randn(batch, patches, dim), dim=-1)

    logits = exgap.said_relevance_logits(t, h, w_query, w_key,
                                         tau_said=exgap.DEFAULT_TAU_SAID)
    relevance = torch.sigmoid(logits)
    assert float(relevance.std()) > 0.05, 'relevance must not be a constant 0.5'
    mask, diagnostics = exgap.compute_said_mask(relevance, threshold=0.6)
    assert float(diagnostics['said_above_threshold_fraction']) > 0.0
    assert float(diagnostics['mask_keep_ratio']) < 1.0, 'the hard mask must drop some patches'
    assert float(diagnostics['mask_drop_ratio']) == pytest.approx(
        1.0 - float(diagnostics['mask_keep_ratio']), abs=1e-6)
    assert float(mask.sum(dim=-1).min()) >= 1.0

    # the rejected variant, for the record: / sqrt(d) on unnormalised projections is inert
    inert = torch.sigmoid(torch.einsum('bd,bnd->bn', w_query(t), w_key(h)) / (dim ** 0.5))
    assert float(inert.std()) < 0.01
    assert float(inert.std()) < float(relevance.std()) / 20.0
    assert float(exgap.compute_said_mask(inert, threshold=0.6)[1]['mask_keep_ratio']) == 1.0


def test_full_model_mask_is_alive_end_to_end():
    """End-to-end on the stub model: the Said router really does claim patches.

    Without this the smoke could pass while the hard mask -- and therefore the whole ExGAP
    objective -- is silently inert, which is exactly what the first 100-step run measured.
    """
    model = build_model()
    images, texts = make_batch()
    encoded = model.encode_said_exgap(images, texts, exgap_global_pool='mean')
    relevance = encoded['said_relevance']
    mask = encoded['mask']
    assert float(relevance.std()) > 1e-4, 'relevance must vary across patches'
    assert float(mask.mean()) < 1.0, 'the hard mask must drop at least one patch'
    assert float(mask.sum(dim=-1).min()) >= 1.0, 'every sample must keep something (no NaN path)'
    # the mask is what makes the masked representation differ from the global one
    assert not torch.allclose(encoded['unsaid_feature'], encoded['global_feature'])
    # and the Said representation is the relevance-weighted pool, not the plain mean
    assert not torch.allclose(encoded['said_feature'], encoded['global_feature'])


def test_mean_global_equals_plain_mean_pooling():
    model = build_model()
    images, texts = make_batch()
    encoded = model.encode_said_exgap(images, texts, exgap_global_pool='mean')
    with torch.no_grad():
        _, patches = model.encode_router_input(images)
        # the module normalises the patches first, so it is the mean of the *normalised* patches
        expected = F.normalize(F.normalize(patches, dim=-1).mean(dim=1), dim=-1)
    assert torch.allclose(encoded['global_feature'], expected, atol=1e-5)


# --------------------------------------------------------------------------- #
# Test 2 -- mask correctness
# --------------------------------------------------------------------------- #
def test_mask_selects_patches_that_are_not_said():
    relevance = torch.tensor([[0.9, 0.8, 0.2, 0.1]])
    mask, diagnostics = exgap.compute_said_mask(relevance, threshold=0.6)
    assert mask.tolist() == [[0.0, 0.0, 1.0, 1.0]]
    assert mask.requires_grad is False
    assert float(diagnostics['mask_keep_ratio']) == pytest.approx(0.5)
    assert float(diagnostics['mask_drop_ratio']) == pytest.approx(0.5)
    assert float(diagnostics['said_above_threshold_fraction']) == pytest.approx(0.5)
    assert float(diagnostics['masked_patch_count_mean']) == pytest.approx(2.0)


def test_masked_pooling_ignores_the_said_dominant_patches():
    """The masked pool must be exactly the mean of the kept patches."""
    patches = torch.zeros(1, 4, DIM)
    patches[0, 0, 0] = 10.0                      # Said-dominant, must be ignored
    patches[0, 1, 0] = 20.0                      # Said-dominant, must be ignored
    patches[0, 2, 1] = 1.0
    patches[0, 3, 1] = 3.0
    attention = torch.full((1, 4), 0.25)
    mask = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    z_u, weights = exgap.compute_masked_unsaid_representation(patches, mask, attention)
    # attention mode renormalises the masked global attention
    assert torch.allclose(weights, torch.tensor([[0.0, 0.0, 0.5, 0.5]]), atol=1e-6)
    expected = F.normalize(torch.tensor([[0.0, 2.0] + [0.0] * (DIM - 2)]), dim=-1)
    assert torch.allclose(z_u, expected, atol=1e-6)
    # changing a dropped patch must not change z_U at all
    other = patches.clone()
    other[0, 0, 0] = -999.0
    other[0, 1, 0] = 999.0
    z_other, _ = exgap.compute_masked_unsaid_representation(other, mask, attention)
    assert torch.allclose(z_u, z_other, atol=1e-6)


def test_mean_mode_masked_pooling_is_the_masked_mean():
    patches = torch.randn(2, PATCHES, DIM)
    mask = torch.zeros(2, PATCHES)
    mask[0, :3] = 1.0
    mask[1, 3:] = 1.0
    uniform = exgap.mean_global_weights(patches)
    z_u, weights = exgap.compute_masked_unsaid_representation(patches, mask, uniform)
    with torch.no_grad():
        expected = F.normalize(patches * mask.unsqueeze(-1), dim=-1)
        expected = torch.stack([
            F.normalize(patches[i][mask[i].bool()].mean(dim=0), dim=-1) for i in range(2)])
    assert torch.allclose(z_u, expected, atol=1e-5)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_all_patches_masked_falls_back_and_never_produces_nan():
    """``r^S_p > tau`` for every patch: fall back to the lowest-relevance patch."""
    relevance = torch.full((2, PATCHES), 0.99)
    mask, diagnostics = exgap.compute_said_mask(relevance, threshold=0.6)
    assert float(mask.sum(dim=-1).min()) == 1.0
    assert bool(diagnostics['masked_patch_count_is_zero_anywhere']) is False
    assert float(diagnostics['fallback_fraction']) == 1.0
    # and the resulting representation is finite
    patches = torch.randn(2, PATCHES, DIM)
    z_u, weights = exgap.compute_masked_unsaid_representation(
        patches, mask, exgap.mean_global_weights(patches))
    assert torch.isfinite(z_u).all() and torch.isfinite(weights).all()
    # the fallback keeps the single least-Said patch, deterministically the first on a tie
    assert mask[0].argmax().item() == 0


def test_mask_threshold_validation_and_extremes():
    relevance = torch.rand(3, PATCHES)
    keep_all, _ = exgap.compute_said_mask(relevance, threshold=1.0)
    assert float(keep_all.sum(dim=-1).min()) == PATCHES          # everything kept
    keep_none, diagnostics = exgap.compute_said_mask(relevance, threshold=-0.0)
    assert float(keep_none.sum(dim=-1).max()) >= 1.0             # fallback engaged
    with pytest.raises(ValueError):
        exgap.compute_said_mask(relevance, threshold=1.5)
    with pytest.raises(ValueError):
        exgap.compute_said_mask(relevance, threshold=0.5, min_keep=0)
    with pytest.raises(ValueError):
        exgap.compute_said_mask(torch.rand(3), threshold=0.5)


# --------------------------------------------------------------------------- #
# Test 3 -- explanatory gap
# --------------------------------------------------------------------------- #
def test_explanatory_gap_values_and_normalisation():
    s_sc = torch.tensor([0.75])
    s_gc = torch.tensor([0.50])
    gap = exgap.compute_explanatory_gap(s_sc, s_gc)
    assert float(gap['gap_raw']) == pytest.approx(0.25, abs=1e-5)
    assert float(gap['gap_weight']) == pytest.approx(0.50, abs=1e-5)

    # S_sc < S_gc must give exactly zero, never an absolute value
    negative = exgap.compute_explanatory_gap(torch.tensor([0.45]), torch.tensor([0.55]))
    assert float(negative['gap_raw']) == 0.0
    assert float(negative['gap_weight']) == 0.0

    # wider positive gap -> larger weight
    small = exgap.compute_explanatory_gap(torch.tensor([0.60]), torch.tensor([0.50]))
    large = exgap.compute_explanatory_gap(torch.tensor([0.90]), torch.tensor([0.50]))
    assert float(large['gap_weight']) > float(small['gap_weight'])
    # normalisation reduces to the raw value when S_gc = 0
    zero_ref = exgap.compute_explanatory_gap(torch.tensor([0.30]), torch.tensor([0.0]))
    assert float(zero_ref['gap_weight']) == pytest.approx(0.30, abs=1e-5)
    # without normalisation the raw positive part is used
    raw_mode = exgap.compute_explanatory_gap(torch.tensor([0.75]), torch.tensor([0.50]),
                                             normalize=False)
    assert float(raw_mode['gap_weight']) == pytest.approx(0.25, abs=1e-5)
    with pytest.raises(ValueError):
        exgap.compute_explanatory_gap(torch.rand(3), torch.rand(4))


def test_explanatory_gap_is_detached_and_clamped():
    s_sc = torch.tensor([0.9, 0.2], requires_grad=True)
    s_gc = torch.tensor([0.1, 0.5], requires_grad=True)
    gap = exgap.compute_explanatory_gap(s_sc, s_gc)
    assert gap['gap_weight'].requires_grad is False
    assert gap['gap_raw'].requires_grad is False
    assert float(gap['gap_weight'].min()) >= 0.0
    assert float(gap['gap_weight'].max()) <= 1.0
    # an extreme positive difference cannot exceed the clamp
    huge = exgap.compute_explanatory_gap(torch.tensor([2.0]), torch.tensor([0.0]))
    assert float(huge['gap_weight']) == 1.0


# --------------------------------------------------------------------------- #
# Test 4 -- loss monotonicity
# --------------------------------------------------------------------------- #
def test_exgap_loss_is_monotone_in_the_violation():
    gap_weight = torch.tensor([0.5])
    s_gc = torch.tensor([0.5])
    worse = exgap.compute_exgap_loss(torch.tensor([0.7]), s_gc, gap_weight,
                                     temperature=0.05)['loss']
    better = exgap.compute_exgap_loss(torch.tensor([0.4]), s_gc, gap_weight,
                                      temperature=0.05)['loss']
    assert float(worse) > float(better)
    # S_uc below S_gc still costs something (softplus), but much less
    assert float(better) > 0.0


def test_exgap_loss_scales_with_the_gap_weight():
    s_uc = torch.tensor([0.7])
    s_gc = torch.tensor([0.5])
    small = exgap.compute_exgap_loss(s_uc, s_gc, torch.tensor([0.1]),
                                     temperature=0.05)['loss']
    large = exgap.compute_exgap_loss(s_uc, s_gc, torch.tensor([0.9]),
                                     temperature=0.05)['loss']
    assert float(large) > float(small)
    # a zero gap weight contributes exactly nothing
    zero = exgap.compute_exgap_loss(s_uc, s_gc, torch.tensor([0.0]), temperature=0.05)['loss']
    assert float(zero) == 0.0


def test_exgap_loss_respects_the_valid_mask():
    s_uc = torch.tensor([10.0, 0.0])
    s_gc = torch.tensor([0.0, 0.0])
    gap_weight = torch.tensor([1.0, 1.0])
    valid = torch.tensor([True, False])
    masked = exgap.compute_exgap_loss(s_uc, s_gc, gap_weight, temperature=0.1, valid=valid)
    all_valid = exgap.compute_exgap_loss(s_uc, s_gc, gap_weight, temperature=0.1)
    assert float(masked['loss']) > float(all_valid['loss'])       # the huge sample dominates
    assert float(masked['valid_fraction']) == pytest.approx(0.5)
    # all-invalid returns a finite, differentiable zero rather than NaN
    none_valid = exgap.compute_exgap_loss(s_uc, s_gc, gap_weight, temperature=0.1,
                                          valid=torch.tensor([False, False]))
    assert torch.isfinite(none_valid['loss'])
    assert float(none_valid['loss']) == 0.0
    with pytest.raises(ValueError):
        exgap.compute_exgap_loss(s_uc, s_gc, gap_weight, temperature=0.0)


# --------------------------------------------------------------------------- #
# Test 5 -- gradient isolation
# --------------------------------------------------------------------------- #
def _zero_or_none(grad):
    return grad is None or float(grad.abs().sum()) == 0.0


def test_exgap_only_backward_touches_only_the_masked_path():
    """In MEAN mode ``A^G`` is a constant, so L_ExGAP reaches only z_U -> the backbone.

    NOTE: on this tiny stub the explanatory gap is usually zero on a random batch
    (``S_sc <= S_gc``), which makes every ExGAP gradient exactly zero. That is the *correct*
    behaviour -- the objective is inactive when Said does not beat Global -- so the assertions
    below are about the gradient *route* (which parameters are touched at all), not about
    magnitude. ``test_exgap_gradient_route_with_an_active_gap`` checks magnitudes.
    """
    model = build_model()
    images, texts = make_batch()
    out = exgap_forward(model, images, texts, exgap_global_pool='mean')
    assert out['explanatory_gap_norm_mean'].requires_grad is False
    out['loss_exgap'].backward()

    # the Said router must never be reachable from L_ExGAP
    for name, parameter in model.said_router.named_parameters():
        assert _zero_or_none(parameter.grad), name
    # nor the CLS/global branch, nor the text encoder
    assert model.clip.global_proj.weight.grad is None
    assert _zero_or_none(model.clip.text_proj.weight.grad)
    # the masked patch path is the designed target
    assert model.clip.patch_proj.weight.grad is not None


def test_exgap_gradient_route_with_an_active_gap():
    """With a positive explanatory gap the ExGAP gradient is live: check where it goes.

    Built directly from the module functions so the gap is guaranteed non-zero, then the same
    isolation contract is asserted: the masked patch path receives gradient, the gap weight and
    global reference do not, and a live text embedding WOULD have received gradient -- which is
    exactly why the model detaches it.
    """
    torch.manual_seed(0)
    batch, patches, dim = 4, 6, 8
    h = F.normalize(torch.randn(batch, patches, dim), dim=-1).requires_grad_(True)
    t_live = F.normalize(torch.randn(batch, dim), dim=-1).requires_grad_(True)

    logits = torch.zeros(batch, patches)
    logits[:, 0] = 6.0                                   # make the Said selection selective
    z_s = F.normalize(torch.einsum('bn,bnd->bd', F.softmax(logits, dim=-1), h), dim=-1)
    z_g = F.normalize(h.mean(dim=1), dim=-1)
    mask = torch.zeros(batch, patches)
    mask[:, 1:] = 1.0
    z_u, _ = exgap.compute_masked_unsaid_representation(h, mask, exgap.mean_global_weights(h))

    s_gc = (z_g * t_live.detach()).sum(dim=-1)
    s_sc = (z_s * t_live.detach()).sum(dim=-1)
    gap = exgap.compute_explanatory_gap(s_sc, s_gc)
    assert float(gap['gap_weight'].max()) > 0.0, 'the probe must produce an active gap'
    s_uc = (z_u * t_live.detach()).sum(dim=-1)           # the model detaches the text here
    loss = exgap.compute_exgap_loss(s_uc, s_gc, gap['gap_weight'], temperature=0.05)['loss']
    assert float(loss) > 0.0
    loss.backward()

    assert float(h.grad.abs().sum()) > 0.0               # masked patch path moves
    assert _zero_or_none(t_live.grad)                    # detached text is not moved
    # and the gap weight is a constant, so it cannot be gamed
    assert gap['gap_weight'].requires_grad is False

    # a LIVE text embedding in the same position would receive gradient -- the leak we forbid
    h2 = h.detach().requires_grad_(True)
    z_u2, _ = exgap.compute_masked_unsaid_representation(h2, mask,
                                                         exgap.mean_global_weights(h2))
    leak = torch.autograd.grad(
        ((z_u2 * t_live).sum(dim=-1) * gap['gap_weight']).sum(), t_live,
        retain_graph=True, allow_unused=True)[0]
    assert leak is not None and float(leak.abs().sum()) > 0.0
    detached = torch.autograd.grad(
        ((z_u2 * t_live.detach()).sum(dim=-1) * gap['gap_weight']).sum(), t_live,
        retain_graph=True, allow_unused=True)[0]
    assert detached is None


def test_attention_pooling_hits_a_dead_saddle_if_both_sides_are_zero():
    """Regression: ``A^G`` must be *exactly* uniform at init but must not be frozen there.

    With both bilinear sides zero the logit gradient is zero as well (``d logit / d W_query ~ k
    = 0`` and ``d logit / d W_key ~ q = 0``), so no ExGAP signal could ever move the pooling and
    the attention arm would be a silent copy of the mean arm. The shipped module keeps the query
    *projection* zero (exactly uniform logits at step 0) while the query *token* and the key side
    stay alive, which leaves ``W_query`` a non-zero gradient at step 0.
    """
    torch.manual_seed(0)
    patches, batch = 6, 4
    h = F.normalize(torch.randn(batch, patches, DIM), dim=-1)
    target = F.normalize(h[:, 0], dim=-1)

    live = exgap.ExGapModule(dim=DIM)
    z_g, a_g = live(h)
    assert torch.allclose(a_g, torch.full_like(a_g, 1.0 / patches), atol=1e-7), \
        'attention must be exactly uniform at initialisation'
    assert float(live.w_key.weight.abs().sum()) > 0.0, 'the key side must not be zero'
    assert float(live.query.abs().sum()) > 0.0, 'the query token must not be zero'
    (-(z_g * target).sum()).backward()
    assert live.w_query.weight.grad is not None
    assert float(live.w_query.weight.grad.abs().sum()) > 0.0, \
        'W_query cannot move: the pooling is stuck at the uniform solution'

    # for the record: with BOTH projections zero the gradient is structurally zero
    dead = exgap.ExGapModule(dim=DIM)
    nn.init.zeros_(dead.w_query.weight)
    nn.init.zeros_(dead.w_query.bias)
    nn.init.zeros_(dead.w_key.weight)
    nn.init.zeros_(dead.w_key.bias)
    nn.init.zeros_(dead.query)
    z_dead, a_dead = dead(h)
    assert torch.allclose(a_dead, torch.full_like(a_dead, 1.0 / patches), atol=1e-7)
    (-(z_dead * target).sum()).backward()
    for name, parameter in dead.named_parameters():
        assert _zero_or_none(parameter.grad), name


def test_exgap_backward_in_attention_mode_does_not_touch_the_router():
    """In ATTENTION mode ``A^G`` is used by z_U, so the pooling agent legitimately trains --
    but the Said router and the text encoder still must not be reachable from L_ExGAP."""
    model = build_model()
    images, texts = make_batch()
    out = exgap_forward(model, images, texts, exgap_global_pool='attention')
    out['loss_exgap'].backward()

    pooling_grads = {name: parameter.grad for name, parameter in model.named_parameters()
                     if name.startswith('exgap.')}
    assert pooling_grads, 'attention pooling parameters missing'
    for name, parameter in model.said_router.named_parameters():
        assert _zero_or_none(parameter.grad), name
    assert _zero_or_none(model.clip.text_proj.weight.grad)
    assert model.clip.patch_proj.weight.grad is not None
    assert _zero_or_none(model.clip.global_proj.weight.grad)


def test_inference_api_does_not_leak_exgap_gradient_into_the_text_encoder():
    """The inference API must report the same detached-text ``s_uc`` as the training path.

    Regression: ``encode_said_exgap(return_details=True)`` used to build ``s_uc`` from the live
    text embedding, so the reported gap term carried a text-encoder gradient that the training
    path exists to forbid (measured with ``probe_exgap_gradient_route.py``:
    ``d s_uc / d clip.text_proj = 0.67`` against ``0`` in training).
    """
    model = build_model()
    images, texts = make_batch()
    details = model.encode_said_exgap(images, texts, return_details=True)
    leak = torch.autograd.grad(details['s_uc'].sum(), model.clip.text_proj.weight,
                               allow_unused=True)[0]
    assert _zero_or_none(leak), 's_uc must not be able to move the text encoder'
    # z_U is the only live input, so the masked patch path does receive gradient
    reach = torch.autograd.grad(details['s_uc'].sum(), model.clip.patch_proj.weight,
                                allow_unused=True)[0]
    assert reach is not None and float(reach.abs().sum()) > 0.0


def test_gap_weight_cannot_be_used_to_shrink_the_gap():
    """The whole point: d L_ExGAP / d S_sc must be zero, so ExGAP cannot game the gap."""
    s_sc = torch.tensor([0.8], requires_grad=True)
    s_gc = torch.tensor([0.4], requires_grad=True)
    s_uc = torch.tensor([0.9], requires_grad=True)
    gap = exgap.compute_explanatory_gap(s_sc, s_gc)
    loss = exgap.compute_exgap_loss(s_uc, s_gc, gap['gap_weight'], temperature=0.05)['loss']
    loss.backward()
    assert s_sc.grad is None or float(s_sc.grad.abs().sum()) == 0.0
    assert s_gc.grad is None or float(s_gc.grad.abs().sum()) == 0.0
    assert s_uc.grad is not None and float(s_uc.grad.abs().sum()) > 0.0


def test_full_objective_moves_the_router_only_through_the_said_loss():
    """With lambda_exgap = 0 the router is trained by L_S; ExGAP adds nothing to it."""
    model = build_model()
    images, texts = make_batch()
    said_only = exgap_forward(model, images, texts, lambda_said=1.0, lambda_exgap=0.0)
    said_only['loss_total'].backward()
    router_grad = model.said_router.q_proj.weight.grad.clone()
    assert float(router_grad.abs().sum()) > 0.0

    model.zero_grad(set_to_none=True)
    both = exgap_forward(model, images, texts, lambda_said=1.0, lambda_exgap=1.0)
    both['loss_total'].backward()
    combined = model.said_router.q_proj.weight.grad
    # the router gradient is exactly the L_S contribution: ExGAP adds nothing
    assert torch.allclose(combined, router_grad, atol=1e-6)


def test_mask_is_detached_from_the_relevance_graph():
    relevance = torch.rand(3, PATCHES, requires_grad=True)
    mask, _ = exgap.compute_said_mask(relevance, threshold=0.5)
    assert mask.requires_grad is False
    assert mask.grad_fn is None


# --------------------------------------------------------------------------- #
# objective / CLI validation
# --------------------------------------------------------------------------- #
def test_said_exgap_rejects_mixed_objectives():
    model = build_model()
    images, texts = make_batch()
    with pytest.raises(ValueError):
        model.forward_train(images, texts, 1.0, 1.0, lambda_unsaid=0.0,
                            objective_mode='said_exgap')
    with pytest.raises(ValueError):
        model.forward_train(images, texts, 0.0, 1.0, lambda_unsaid=0.5,
                            objective_mode='said_exgap')
    with pytest.raises(ValueError):
        model.forward_train(images, texts, 0.0, 1.0, lambda_unsaid=0.0,
                            objective_mode='said_exgap', global_caption_view='full',
                            texts_full=texts)
    with pytest.raises(ValueError):
        model.forward_train(images, texts, 0.0, 0.0, lambda_unsaid=0.0,
                            objective_mode='said_exgap')


def test_encode_said_exgap_api():
    model = build_model()
    images, texts = make_batch()
    plain = model.encode_said_exgap(images, texts)
    assert set(plain) >= {'global_feature', 'said_feature', 'unsaid_feature',
                          'said_relevance', 'said_attention', 'mask'}
    for key, value in plain.items():
        assert torch.isfinite(value).all(), key
    detailed = model.encode_said_exgap(images, texts, return_details=True)
    assert {'s_gc', 's_sc', 's_uc', 'gap_raw', 'gap_weight'} <= set(detailed)
    for key in ('s_gc', 's_sc', 's_uc', 'gap_raw', 'gap_weight'):
        assert detailed[key].shape == (BATCH,), key
    assert detailed['gap_weight'].requires_grad is False
    # the signature must not accept a withheld / candidate text
    parameters = list(inspect.signature(SALUModel.encode_said_exgap).parameters)
    for banned in ('unsaid_texts', 'candidate', 'uss', 'hidden_text', 'texts_full'):
        assert not any(banned in name for name in parameters), banned


def test_checkpoint_config_round_trip_and_validation():
    config = exgap.checkpoint_config('mean', 0.6, 0.05, 1.0, 1.0, True)
    assert config == {'global_pool': 'mean', 'mask_threshold': 0.6, 'temperature': 0.05,
                      'lambda_said': 1.0, 'lambda_exgap': 1.0, 'normalize_gap': True}

    class _Args:
        exgap_global_pool = 'mean'
        exgap_mask_threshold = 0.6
        exgap_temperature = 0.05
        lambda_said = 1.0
        lambda_exgap = 1.0
        lambda_global = 0.0
        lambda_unsaid = 0.0

    exgap.validate_exgap_config(_Args())                   # legal configuration passes

    for attribute, value in (('exgap_global_pool', 'max'), ('exgap_mask_threshold', 1.5),
                             ('exgap_temperature', 0.0), ('lambda_said', 0.0),
                             ('lambda_global', 1.0), ('lambda_unsaid', 0.5)):
        broken = _Args()
        setattr(broken, attribute, value)
        with pytest.raises(ValueError):
            exgap.validate_exgap_config(broken)


def test_no_forbidden_ingredients_in_the_new_module():
    """v1 must not contain reconstruction, prototypical, absorption or teacher machinery.

    Docstrings and comments are stripped first: the module *documents* what it is not, and prose
    must not be mistaken for executable code.
    """
    import tokenize

    path = os.path.join(REPO_ROOT, 'model', 'exgap.py')
    pieces = []
    with open(path, encoding='utf-8') as handle:
        for token in tokenize.generate_tokens(handle.readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            pieces.append(token.string)
    code = ' '.join(pieces).lower()
    for banned in ('reconstruct', 'prototype', 'absorb', 'closure', 'dino', 'teacher',
                   'lambda_uss', 'texts_full', 'decoder', 'cross_entropy_patch'):
        assert banned not in code, banned
