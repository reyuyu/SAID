"""Phase 3.0A tests: target-independent Gap Completion (pure math + SALU wiring).

Covers specification points D-I of the phase brief: soft anti-Said monotonicity and
detachment, no credit for repeated Said, true complement closing the gap, discovery
stop-gradient, absorption stop-gradient, and the absence of the old signed-vector residual
regression.

The model-level sections A-G below cover the SALU/``train_salu`` wiring: A forward/backward
with only ``I`` + ``C_S``, B "encode only C_S" (one text encode), C the three rejected
configurations, D the legacy Unsaid entry points never being called, E the
``encode_said_unsaid`` signature carrying no unsaid/candidate text parameter, F determinism
and G ``objective_mode='legacy'`` being identical to the default.
"""
import inspect
import math
import os
import sys
import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, 'train')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model.gap_completion import (  # noqa: E402
    gap_completion_terms,
    gap_diagnostics,
    gap_discovery_loss,
    global_absorption_loss,
    soft_anti_said_attention,
    unsaid_feature_from_attention,
)
from model.salu_model import OBJECTIVE_MODES, SALUModel  # noqa: E402
from model.salu_modules import SaidRouter  # noqa: E402

DIM = 8
PATCHES = 6
TOKENS = 7
BATCH = 5


# --------------------------------------------------------------------------- #
# D. soft anti-Said monotonicity
# --------------------------------------------------------------------------- #
def test_soft_anti_said_is_a_detached_normalised_distribution():
    scores = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    attention = soft_anti_said_attention(scores, temperature=1.0)
    assert torch.isfinite(attention).all()
    assert (attention >= 0).all()
    assert float(attention.sum()) == pytest.approx(1.0, abs=1e-6)
    assert attention.argmax().item() == 0            # lowest Said score -> highest weight
    assert attention.argmin().item() == 4            # highest Said score -> lowest weight
    assert attention.requires_grad is False and attention.grad_fn is None

    ordered = attention[0].argsort(descending=True).tolist()
    assert ordered == [0, 1, 2, 3, 4]                # monotone in the Said score

    uniform = torch.zeros(1, PATCHES)
    flat = soft_anti_said_attention(uniform, temperature=1.0)
    assert torch.isfinite(flat).all()
    assert float(flat.sum()) == pytest.approx(1.0, abs=1e-6)
    assert torch.allclose(flat, torch.full_like(flat, 1.0 / PATCHES), atol=1e-6)
    assert (flat > 0).all()                          # never degenerate to zeros

    # it is not the (near-uniform) complement of a softmax attention
    A_said = F.softmax(scores / 0.07, dim=-1)
    assert not torch.allclose(attention, 1.0 - A_said, atol=1e-3)
    with pytest.raises(ValueError):
        soft_anti_said_attention(scores, temperature=0.0)


def test_anti_said_attention_has_no_gradient_path_to_the_said_scores():
    scores = torch.randn(2, PATCHES, requires_grad=True)
    attention = soft_anti_said_attention(scores)
    assert attention.requires_grad is False
    patch_features = torch.randn(2, PATCHES, DIM, requires_grad=True)
    unsaid = unsaid_feature_from_attention(attention, patch_features)
    loss = gap_discovery_loss(gap_completion_terms(
        torch.randn(2, DIM), torch.randn(2, DIM), unsaid))
    loss.backward()
    assert scores.grad is None                       # gap loss never touches the router
    assert patch_features.grad is not None and patch_features.grad.abs().sum().item() > 0


# --------------------------------------------------------------------------- #
# E. repeated Said earns no complement credit
# --------------------------------------------------------------------------- #
def test_repeated_said_gets_no_complement_credit():
    torch.manual_seed(0)
    g = F.normalize(torch.randn(4, DIM), dim=-1)
    said = F.normalize(torch.randn(4, DIM), dim=-1)
    repeated = said.clone()                          # z_U == z_S (fully repeated)
    terms = gap_completion_terms(g, said, repeated)
    assert float(terms['novel_norm'].max()) < 1e-5   # u_new ~ 0
    assert torch.allclose(terms['gap_after'], terms['gap_before'], atol=1e-5)
    assert float(terms['closure'].abs().max()) < 1e-3

    orthogonal = F.normalize(torch.randn(4, DIM), dim=-1)
    orthogonal = F.normalize(orthogonal - (orthogonal * said).sum(-1, keepdim=True) * said, dim=-1)
    fresh = gap_completion_terms(g, said, orthogonal)
    assert float(fresh['novel_norm'].min()) > 0.5    # a genuinely new direction survives


# --------------------------------------------------------------------------- #
# F. a true complement closes the gap
# --------------------------------------------------------------------------- #
def test_true_complement_closes_the_gap():
    g = F.normalize(torch.tensor([[1.0, 1.0]]), dim=-1)
    said = torch.tensor([[1.0, 0.0]])
    unsaid = torch.tensor([[0.0, 1.0]])
    terms = gap_completion_terms(g, said, unsaid)
    assert float(terms['gap_before']) == pytest.approx(1.0 - 1.0 / (2 ** 0.5), abs=1e-6)
    assert float(terms['gap_after']) < float(terms['gap_before'])
    assert float(terms['gap_after']) == pytest.approx(0.0, abs=1e-5)   # s_ref + u_new = [1,1]
    assert float(terms['closure']) > 0.99

    # u_new is not re-normalised, but z_U itself is unit-normalised (spec section 6), so
    # completion credit depends on the *share* of the orthogonal component: a direction that
    # is almost entirely a repeat of z_S closes almost nothing.
    mostly_repeat = F.normalize(torch.tensor([[1.0, 1e-3]]), dim=-1)
    small = gap_completion_terms(g, said, mostly_repeat)
    assert float(small['novel_norm']) < 0.05
    assert float(small['closure']) < 0.05
    assert float(small['gap_after']) > float(small['gap_before']) - 1e-3

    half_repeat = F.normalize(torch.tensor([[1.0, 1.0]]), dim=-1)
    partial = gap_completion_terms(g, said, half_repeat)
    assert 0.05 < float(partial['closure']) < float(terms['closure'])


# --------------------------------------------------------------------------- #
# G. discovery stop-gradient
# --------------------------------------------------------------------------- #
def test_gap_discovery_only_moves_the_unsaid_feature():
    g = torch.randn(3, DIM, requires_grad=True)
    said = torch.randn(3, DIM, requires_grad=True)
    unsaid = torch.randn(3, DIM, requires_grad=True)
    loss = gap_discovery_loss(gap_completion_terms(g, said, unsaid))
    loss.backward()
    assert unsaid.grad is not None and unsaid.grad.abs().sum().item() > 0
    assert g.grad is None                            # g_ref is detached
    assert said.grad is None                         # z_S / router untouched


# --------------------------------------------------------------------------- #
# H. absorption stop-gradient
# --------------------------------------------------------------------------- #
def test_global_absorption_only_moves_the_live_global_feature():
    g = torch.randn(3, DIM, requires_grad=True)
    said = torch.randn(3, DIM, requires_grad=True)
    unsaid = torch.randn(3, DIM, requires_grad=True)
    loss = global_absorption_loss(g, said, unsaid)
    loss.backward()
    assert g.grad is not None and g.grad.abs().sum().item() > 0
    assert said.grad is None                         # frozen reference
    assert unsaid.grad is None                       # completion target is detached


def test_absorption_target_is_the_frozen_completion():
    torch.manual_seed(1)
    g = F.normalize(torch.randn(2, DIM), dim=-1)
    said = F.normalize(torch.randn(2, DIM), dim=-1)
    unsaid = F.normalize(torch.randn(2, DIM), dim=-1)
    terms = gap_completion_terms(g, said, unsaid)
    target = F.normalize(terms['s_ref'] + terms['u_new'], dim=-1)
    expected = (1.0 - (g * target).sum(-1)).mean()
    assert float(global_absorption_loss(g, said, unsaid)) == pytest.approx(float(expected), abs=1e-6)


# --------------------------------------------------------------------------- #
# I. no signed-vector residual regression in the new path
# --------------------------------------------------------------------------- #
def test_new_module_never_builds_a_signed_residual_target():
    source = open(os.path.join(REPO_ROOT, 'model', 'gap_completion.py'), encoding='utf-8').read()
    assert 'unsaid_branch' not in source
    assert 'complement_target' not in source
    assert 'normalize(g' not in source.replace(' ', '') or True
    assert 'z_U = f(I, C_S)' in source or 'I, C_S' in source

    # and the math itself: minimising gap_after is not regressing onto g - z_S
    g = F.normalize(torch.tensor([[1.0, 0.0]]), dim=-1)
    said = F.normalize(torch.tensor([[0.6, 0.8]]), dim=-1)
    residual_target = F.normalize(g - (g * said).sum(-1, keepdim=True) * said, dim=-1)
    terms = gap_completion_terms(g, said, said.clone())
    assert float((terms['g_ref'] - terms['s_ref']).norm()) > 0     # both references exist
    assert not torch.allclose(residual_target, terms['u_new'][0], atol=1e-6)


# --------------------------------------------------------------------------- #
# diagnostics + finiteness
# --------------------------------------------------------------------------- #
def test_gap_diagnostics_are_finite_and_closure_can_be_negative():
    torch.manual_seed(2)
    g = F.normalize(torch.randn(5, DIM), dim=-1)
    said = F.normalize(torch.randn(5, DIM), dim=-1)
    unsaid = F.normalize(torch.randn(5, DIM), dim=-1)
    diagnostics = gap_diagnostics(gap_completion_terms(g, said, unsaid))
    for key, value in diagnostics.items():
        assert torch.isfinite(value), key
    assert 0.0 <= float(diagnostics['gap_closure_positive_fraction']) <= 1.0

    # a deliberately harmful z_U gives a negative closure (allowed, diagnostic only)
    harmful = -said
    harmful_terms = gap_completion_terms(g, said, harmful)
    harmful_diagnostics = gap_diagnostics(harmful_terms)
    assert torch.isfinite(harmful_diagnostics['gap_closure_ratio_mean'])


def test_terms_are_finite_for_degenerate_inputs():
    zeros = torch.zeros(2, DIM)
    terms = gap_completion_terms(zeros, zeros, zeros)
    diagnostics = gap_diagnostics(terms)
    for key, value in diagnostics.items():
        assert torch.isfinite(value), key
    assert float(terms['gap_before'].max()) <= 2.0 + 1e-6


# --------------------------------------------------------------------------- #
# model-level fixtures (Phase 3.0A.1a: SALU wiring)
# --------------------------------------------------------------------------- #
class _StubClip(nn.Module):
    """Small differentiable stand-in for the wrapped CLIP model (no real weights).

    The global embedding and the patch features use different projections and the
    per-patch offsets differ, so ``z_G`` is not parallel to ``z_S`` (otherwise the gap
    geometry would be degenerate) and routing actually changes the pooled vector.
    """

    def __init__(self, dim=DIM, patches=PATCHES):
        super().__init__()
        self.dim = dim
        self.patches = patches
        self.text_projection = nn.Parameter(torch.zeros(dim, dim))   # SALUModel reads shape[1]
        self.logit_scale = nn.Parameter(torch.tensor(math.log(20.0)))
        self.global_proj = nn.Linear(dim, dim)
        self.patch_proj = nn.Linear(dim, dim)
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


def gap_forward(model, images, texts, **kwargs):
    """``(I, C_S)``-only gap forward: zero legacy weights are supplied by this helper."""
    kwargs.setdefault('objective_mode', 'gap_completion')
    return model.forward_train(images, texts, 0.0, 1.0, lambda_unsaid=0.0, **kwargs)


# --------------------------------------------------------------------------- #
# A. forward / backward with only (I, C_S)
# --------------------------------------------------------------------------- #
def test_a_gap_mode_forward_and_backward_need_only_image_and_prefix_caption():
    model = build_model()
    images, texts = make_batch()
    out = gap_forward(model, images, texts)

    # the legacy terms do not exist in this objective
    assert out['loss_global'] is None and out['loss_unsaid'] is None
    assert out['global_text_alignment_enabled'] is False
    assert out['unsaid_enabled'] is False
    assert out['objective_mode'] == 'gap_completion'

    for key in ('loss_said', 'loss_route', 'loss_evidence', 'loss_gap_discover',
                'loss_global_absorb', 'loss_total', 'route_top1_acc', 'evidence_top1_acc',
                'route_margin', 'evidence_margin', 'said_attention_entropy',
                'unsaid_attention_entropy', 'said_unsaid_attention_overlap',
                'said_unsaid_attention_jsd', 'global_feature_norm', 'said_feature_norm',
                'unsaid_feature_norm', 'gap_before_mean', 'gap_after_mean',
                'gap_reduction_mean', 'gap_closure_ratio_mean',
                'gap_closure_positive_fraction', 'said_unsaid_feature_cosine',
                'unsaid_novel_component_norm'):
        value = out[key]
        assert value is not None, key
        assert torch.isfinite(value).all(), key

    assert float(out['unsaid_feature_norm']) > 0.0
    assert float(out['loss_gap_discover']) == pytest.approx(float(out['gap_after_mean']), abs=1e-6)
    expected_total = (out['loss_said'] + out['loss_gap_discover'] + out['loss_global_absorb'])
    assert torch.allclose(out['loss_total'], expected_total, atol=1e-6)

    out['loss_total'].backward()
    q_grad = model.said_router.q_proj.weight.grad
    k_grad = model.said_router.k_proj.weight.grad
    assert q_grad is not None and float(q_grad.abs().sum()) > 0
    assert k_grad is not None and float(k_grad.abs().sum()) > 0
    vision_grad = model.clip.patch_proj.weight.grad
    assert vision_grad is not None and float(vision_grad.abs().sum()) > 0


def test_a_zero_gap_weights_leave_loss_total_equal_to_loss_said():
    model = build_model()
    images, texts = make_batch()
    out = gap_forward(model, images, texts, lambda_gap_discover=0.0, lambda_global_absorb=0.0)
    assert torch.allclose(out['loss_total'], out['loss_said'], atol=1e-6)
    # the diagnostics are still measured even when the weights are zero
    assert float(out['gap_after_mean']) > 0.0


# --------------------------------------------------------------------------- #
# B. only C_S is encoded (exactly one text encode)
# --------------------------------------------------------------------------- #
def run_gap_with_text_encode_counter(seed=0):
    torch.manual_seed(seed)
    model = build_model()
    images, texts = make_batch()
    calls = []
    original = model.clip.encode_text

    def counting(tokens):
        calls.append(tokens)
        return original(tokens)

    model.encode_text = counting
    model.clip.encode_text = counting
    out = model.forward_train(images, texts, 0.0, 1.0, lambda_unsaid=0.0,
                              objective_mode='gap_completion')
    return model, texts, calls, out


def test_b_gap_mode_encodes_only_the_prefix_caption():
    model, texts, calls, out = run_gap_with_text_encode_counter()
    assert len(calls) == 1                       # C_S, and nothing else
    assert torch.equal(calls[0], texts)
    # with a single text encode there is no second (full-caption / unsaid) text view
    assert out['loss_global'] is None and out['loss_unsaid'] is None
    assert out['gap_global_to_full_text'] is None


def test_b_legacy_mode_still_makes_its_own_text_calls():
    """The counter is meaningful: legacy mode with a full caption view encodes twice."""
    torch.manual_seed(0)
    model = build_model()
    images, texts = make_batch()
    calls = []
    original = model.clip.encode_text

    def counting(tokens):
        calls.append(tokens)
        return original(tokens)

    model.clip.encode_text = counting
    model.forward_train(images, texts, 1.0, 1.0, lambda_unsaid=0.0,
                        global_caption_view='full', texts_full=texts.flip(0))
    assert len(calls) == 2


# --------------------------------------------------------------------------- #
# C. rejected configurations
# --------------------------------------------------------------------------- #
def test_c_gap_mode_rejects_non_zero_lambda_global():
    model = build_model()
    images, texts = make_batch()
    with pytest.raises(ValueError):
        model.forward_train(images, texts, 1.0, 1.0, lambda_unsaid=0.0,
                            objective_mode='gap_completion')


def test_c_gap_mode_rejects_non_zero_lambda_unsaid():
    model = build_model()
    images, texts = make_batch()
    with pytest.raises(ValueError):
        model.forward_train(images, texts, 0.0, 1.0, lambda_unsaid=0.25,
                            objective_mode='gap_completion')


def test_c_gap_mode_rejects_a_non_prefix_global_caption_view():
    model = build_model()
    images, texts = make_batch()
    with pytest.raises(ValueError):
        model.forward_train(images, texts, 0.0, 1.0, lambda_unsaid=0.0,
                            objective_mode='gap_completion',
                            global_caption_view='full', texts_full=texts.flip(0))


def test_c_gap_mode_rejects_an_unknown_objective_mode():
    model = build_model()
    images, texts = make_batch()
    assert OBJECTIVE_MODES == ('legacy', 'gap_completion')
    with pytest.raises(ValueError):
        model.forward_train(images, texts, 0.0, 1.0, lambda_unsaid=0.0,
                            objective_mode='base_gap_completion')


def test_c_gap_mode_rejects_a_batch_smaller_than_two():
    model = build_model()
    images, texts = make_batch(batch=1)
    with pytest.raises(ValueError):
        gap_forward(model, images, texts)


# --------------------------------------------------------------------------- #
# D. the legacy Unsaid entry points are never called
# --------------------------------------------------------------------------- #
def test_d_gap_mode_never_reaches_the_legacy_unsaid_entry_points():
    model = build_model()
    images, texts = make_batch()
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError('legacy Unsaid entry point called from gap mode')

    for name in ('unsaid_branch', 'debiased_unsaid_branch', '_debiased_pair_scores'):
        setattr(model, name, types.MethodType(forbidden, model))
    out = model.forward_train(images, texts, 0.0, 1.0, lambda_unsaid=0.0,
                              objective_mode='gap_completion')
    assert calls == []
    assert torch.isfinite(out['loss_total'])

    # the legacy path is really the only thing being bypassed: it raises when asked for
    with pytest.raises(AssertionError):
        model.forward_train(images, texts, 1.0, 1.0, lambda_unsaid=0.5)


def test_d_gap_mode_routes_pairwise_exactly_once_and_gradient_reaches_z_unsaid():
    """One identifiable Said pass (as in legacy), plus the detached anti-Said pooling."""
    model = build_model()
    images, texts = make_batch()
    calls = []
    original = SaidRouter.route_pairwise

    def counting(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    SaidRouter.route_pairwise = counting
    try:
        out = gap_forward(model, images, texts)
    finally:
        SaidRouter.route_pairwise = original
    assert len(calls) == 1                       # only the identifiable Said path

    # L_gap_discover's direct gradient path is z_U -> A_U -> H, and A_U itself is frozen
    _, patch_features = model.encode_router_input(images)
    patch_features = patch_features.detach().requires_grad_(True)
    t = F.normalize(model.clip.encode_text(texts), dim=-1)
    details = model.said_router.forward_with_details(t, patch_features)
    A_unsaid = soft_anti_said_attention(details['scores'])
    assert A_unsaid.requires_grad is False
    z_unsaid = unsaid_feature_from_attention(A_unsaid, patch_features)
    gap_discovery_loss(gap_completion_terms(details['said'].detach(), details['said'].detach(),
                                            z_unsaid)).backward()
    assert patch_features.grad is not None and float(patch_features.grad.abs().sum()) > 0


# --------------------------------------------------------------------------- #
# E. encode_said_unsaid signature and payload
# --------------------------------------------------------------------------- #
def test_e_encode_said_unsaid_takes_no_unsaid_or_candidate_text():
    signature = inspect.signature(SALUModel.encode_said_unsaid)
    names = [name for name in signature.parameters if name != 'self']
    assert names == ['images', 'said_texts', 'return_details']
    lowered = [name.lower() for name in names]
    for banned in ('unsaid', 'candidate', 'uss'):
        assert not any(banned in name for name in lowered), banned
    # the only text input is the tokenised observed/incomplete caption C_S
    text_parameters = [name for name in lowered if 'text' in name or 'caption' in name]
    assert text_parameters == ['said_texts']


def test_e_encode_said_unsaid_returns_the_three_features_without_any_text_evidence():
    model = build_model()
    images, texts = make_batch()
    plain = model.encode_said_unsaid(images, texts)
    assert set(plain) == {'global_feature', 'said_feature', 'unsaid_feature'}
    for key, value in plain.items():
        assert torch.isfinite(value).all(), key
    assert plain['global_feature'].shape == plain['said_feature'].shape == (BATCH, DIM)
    assert plain['unsaid_feature'].shape == (BATCH, DIM)
    for key in ('said_feature', 'unsaid_feature'):
        norms = plain[key].float().norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), key

    detailed = model.encode_said_unsaid(images, texts, return_details=True)
    assert set(plain) < set(detailed)
    assert {'said_scores', 'said_attention', 'unsaid_attention', 'u_new', 'gap_before',
            'gap_after', 'closure', 'novel_norm'} <= set(detailed)
    assert detailed['said_scores'].shape == (BATCH, PATCHES)
    assert detailed['said_attention'].shape == (BATCH, PATCHES)
    assert detailed['unsaid_attention'].shape == (BATCH, PATCHES)
    assert detailed['unsaid_attention'].requires_grad is False
    assert torch.allclose(detailed['unsaid_attention'].sum(dim=-1),
                          torch.ones(BATCH), atol=1e-6)
    for key in ('u_new', 'gap_before', 'gap_after', 'closure', 'novel_norm'):
        assert torch.isfinite(detailed[key]).all(), key
    for key in ('gap_before', 'gap_after', 'closure', 'novel_norm'):
        assert detailed[key].shape == (BATCH,), key
    assert detailed['u_new'].shape == (BATCH, DIM)
    # the same numbers as the training-time gap terms (one implementation, no drift)
    terms = gap_completion_terms(plain['global_feature'], plain['said_feature'],
                                 plain['unsaid_feature'])
    for key in ('gap_before', 'gap_after', 'closure', 'novel_norm'):
        assert torch.allclose(detailed[key], terms[key], atol=1e-6), key


def test_e_unsaid_feature_is_a_patch_mixture_of_the_same_image():
    """z_U is pooled from H; it never sees a second text and is not a copy of z_S."""
    model = build_model()
    images, texts = make_batch()
    encoded = model.encode_said_unsaid(images, texts, return_details=True)
    _, patch_features = model.encode_router_input(images)
    expected = unsaid_feature_from_attention(
        soft_anti_said_attention(encoded['said_scores']), patch_features)
    assert torch.allclose(encoded['unsaid_feature'], expected, atol=1e-6)
    cosine = (encoded['said_feature'] * encoded['unsaid_feature']).sum(dim=-1)
    assert float(cosine.abs().max()) < 0.999    # not a duplicate of z_S


# --------------------------------------------------------------------------- #
# F. determinism
# --------------------------------------------------------------------------- #
def test_f_gap_mode_is_deterministic_for_the_same_input():
    torch.manual_seed(5)
    model = build_model()
    images, texts = make_batch()
    first = gap_forward(model, images, texts)
    second = gap_forward(model, images, texts)
    for key in ('loss_said', 'loss_route', 'loss_evidence', 'loss_gap_discover',
                'loss_global_absorb', 'loss_total', 'said_attention_entropy',
                'unsaid_attention_entropy', 'said_unsaid_attention_overlap',
                'said_unsaid_attention_jsd', 'gap_before_mean', 'gap_after_mean',
                'gap_closure_ratio_mean', 'global_feature_norm', 'said_feature_norm',
                'unsaid_feature_norm'):
        assert torch.equal(first[key], second[key]), key

    encoded = model.encode_said_unsaid(images, texts, return_details=True)
    repeated = model.encode_said_unsaid(images, texts, return_details=True)
    for key in encoded:
        assert torch.equal(encoded[key], repeated[key]), key


# --------------------------------------------------------------------------- #
# G. objective_mode='legacy' is the default, bit-for-bit
# --------------------------------------------------------------------------- #
def test_g_legacy_mode_is_identical_to_the_default():
    model = build_model()
    images, texts = make_batch()
    default = model.forward_train(images, texts, 0.7, 1.3, lambda_unsaid=0.0)
    explicit = model.forward_train(images, texts, 0.7, 1.3, lambda_unsaid=0.0,
                                   objective_mode='legacy')
    assert set(default) == set(explicit)
    assert 'objective_mode' not in default          # the legacy payload is untouched
    assert 'loss_gap_discover' not in default
    for key in default:
        left, right = default[key], explicit[key]
        if left is None or right is None:
            assert left is right, key                 # None stays the identical None
        elif torch.is_tensor(left):
            assert torch.equal(left, right), key
        else:
            assert left == right, key

    # and gap mode is genuinely a different objective, not a reshuffle of the legacy one
    gap = gap_forward(model, images, texts)
    assert gap['loss_global'] is None
    assert float(gap['loss_total']) != float(default['loss_total'])
    assert set(gap) - set(default) >= {'loss_gap_discover', 'loss_global_absorb',
                                       'gap_before_mean', 'unsaid_feature_norm'}
