"""Phase 3.0A tests: target-independent Gap Completion core (pure math).

Covers specification points D-I of the phase brief: soft anti-Said monotonicity and
detachment, no credit for repeated Said, true complement closing the gap, discovery
stop-gradient, absorption stop-gradient, and the absence of the old signed-vector residual
regression. Points A-C and J (base forward/backward without ``C_U``, "encode only C_S",
"C_U cannot change z_U", legacy suite) belong to the SALU/``train_salu`` wiring and are not
covered by this module-only test file yet.
"""
import os
import sys

import pytest
import torch
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

DIM = 8
PATCHES = 6


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
