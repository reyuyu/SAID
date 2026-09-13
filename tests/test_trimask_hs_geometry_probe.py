"""Tests for the read-only S0-TriMask-HS geometry probe.

They cover the five things the probe's claims depend on, all on synthetic data on the CPU, with no
GPU, no checkpoint and no training:

1. equal keep counts but different coordinates must give ``V_level == 0`` with
   ``V_caption_dependent > 0`` -- the case that stops a keep-rate statistic from being read as the
   whole caption-adaptive share;
2. identical masks give ``V_caption_dependent == 0``, ``V_profile > 0`` and a clean identity;
3. variant indexing: changing the text gate must leave the image features, the text features, the
   visual mask and the labels untouched, and the all-ones gate must give ``Q3 == Q1`` and
   ``Q2 == 100 * cos(v, t)``;
4. the intersection identity on identical / overlapping / empty / zero-energy / eps-boundary pairs,
   with invalid pairs neither NaN nor silently dropped;
5. the CE / margin relations and the fixed tie rule.
"""
import math
import os
import sys

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools.diag.trimask_hs_geometry_probe import (  # noqa: E402
    FIXED_SCALE, NORM_EPS, derangement, factory_form, quantiles, reconcile_forms,
    reference_form, retrieval_metrics, score_matrix, variance_decomposition)


def _hard_logits(width, keep):
    """A straight-through-free hard mask with exactly ``keep`` ones in the first coordinates."""
    mask = torch.zeros(1, width)
    mask[0, :keep] = 1.0
    return mask


# --------------------------------------------------------------------- 1 + 2: variance algebra
def test_equal_keep_counts_different_coordinates_are_caption_dependent():
    """Same number of kept coordinates, different coordinates: V_level = 0 but V_caption > 0."""
    width = 8
    matrix = torch.zeros(4, width)
    # every caption keeps exactly 3 coordinates, but a different set each time
    for row, start in enumerate((0, 2, 4, 6)):
        matrix[row, [start % width, (start + 1) % width, (start + 2) % width]] = 1.0
    result = variance_decomposition(matrix.numpy())

    assert result['row_keep_count']['min'] == result['row_keep_count']['max'] == 3
    assert result['v_level'] == pytest.approx(0.0, abs=1e-12)
    assert result['v_caption_dependent'] > 0.0
    assert result['v_interaction'] == pytest.approx(result['v_caption_dependent'], abs=1e-12)
    # the historical field (total - mean per-caption variance) is V_level, i.e. exactly 0 here
    assert result['historical_field_check']['equals_v_level'] is True
    assert result['historical_field_check']['value'] == pytest.approx(0.0, abs=1e-12)


def test_identical_masks_have_profile_only():
    width = 6
    row = torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0, 0.0]])
    matrix = row.repeat(5, 1)
    result = variance_decomposition(matrix.numpy())

    assert result['v_caption_dependent'] == pytest.approx(0.0, abs=1e-12)
    assert result['v_level'] == pytest.approx(0.0, abs=1e-12)
    assert result['v_interaction'] == pytest.approx(0.0, abs=1e-12)
    assert result['v_profile'] > 0.0
    assert result['v_total'] == pytest.approx(result['v_profile'], abs=1e-12)
    assert result['identity_v_total_minus_parts'] == pytest.approx(0.0, abs=1e-12)


def test_variance_identity_holds_for_a_random_matrix():
    generator = torch.Generator().manual_seed(0)
    matrix = (torch.rand(17, 11, generator=generator) > 0.4).float()
    result = variance_decomposition(matrix.numpy())
    parts = result['v_level'] + result['v_profile'] + result['v_interaction']
    assert result['v_total'] == pytest.approx(parts, abs=1e-12)
    assert result['v_caption_dependent'] == pytest.approx(
        result['v_level'] + result['v_interaction'], abs=1e-12)
    assert result['value_counts']['other_values'] == 0


def test_zero_matrix_reports_null_shares():
    result = variance_decomposition(np.zeros((4, 4)))
    assert result['v_total'] == 0.0
    assert result['share_of_total'] is None
    assert result['share_of_caption_dependent'] is None


# --------------------------------------------------------------------- 3: variant indexing
def _toy(n=5, width=7, seed=0):
    generator = torch.Generator().manual_seed(seed)
    v = torch.randn(n, width, generator=generator)
    t = torch.randn(n, width, generator=generator)
    m_i = (torch.rand(n, width, generator=generator) > 0.3).float()
    m_t = (torch.rand(n, width, generator=generator) > 0.3).float()
    return v, t, m_i, m_t


def test_ones_gate_identities_and_invariance_of_the_untouched_tensors():
    v, t, m_i, m_t = _toy()
    q1, q2, q3 = score_matrix(v, t, m_i, m_t)
    ones = torch.ones_like(m_t)
    ones_q1, ones_q2, ones_q3 = score_matrix(v, t, m_i, ones)

    # Q1 does not depend on the text gate at all
    assert torch.allclose(ones_q1, q1, atol=1e-6)
    # all-ones text gate: Q3 must collapse to Q1 and Q2 to the raw global cosine
    assert torch.allclose(ones_q3, q1, atol=1e-6)
    raw = FIXED_SCALE * (torch.nn.functional.normalize(v, dim=-1, eps=NORM_EPS)
                         @ torch.nn.functional.normalize(t, dim=-1, eps=NORM_EPS).t())
    assert torch.allclose(ones_q2, raw, atol=1e-6)


def test_shuffling_only_permutes_the_text_side():
    v, t, m_i, _ = _toy(n=6, width=9)
    # distinct masks per caption, so a derangement must move every caption to a different mask
    m_t = torch.zeros(6, 9)
    for row in range(6):
        m_t[row, row] = 1.0
        m_t[row, (row + 1) % 9] = 1.0
    permutation = derangement(6, 0)
    shuffled = m_t[torch.tensor(permutation)]

    assert not torch.any(torch.tensor(permutation) == torch.arange(6))
    assert sorted(permutation) == list(range(6))
    # the same permutation reproduced twice must be identical (seeds are saved, not re-drawn)
    assert permutation == derangement(6, 0)
    # a permutation preserves the multiset of masks and therefore the overall keep rate
    assert shuffled.mean() == pytest.approx(m_t.mean().item(), abs=1e-6)
    q1, _, q3 = score_matrix(v, t, m_i, m_t)
    shuffled_q1, _, q3_shuffled = score_matrix(v, t, m_i, shuffled)
    # the text gate is the only argument that changed, so Q1 must be untouched and Q3 must move
    assert torch.allclose(shuffled_q1, q1, atol=1e-6)
    assert not torch.allclose(q3, q3_shuffled)


def test_reconciliation_between_the_cosine_and_reference_forms():
    v, t, m_i, m_t = _toy(n=8, width=13, seed=3)
    reconciliation = reconcile_forms(v, t, m_i, m_t)
    for key, value in reconciliation.items():
        if key.endswith('max_abs_diff'):
            # scores live on the fixed 100x scale, so the tolerance is on that scale
            assert value < 1e-3, (key, value)


def test_image_side_mask_is_indexed_by_the_candidate_not_the_query():
    """The candidate-indexed mask is a correctness requirement, not a style choice.

    With ``v_i * mI_i`` a whole row would be scored with the query's own correct-pair mask, which is
    exactly the leakage the probe must not have; the two must therefore differ in general.
    """
    v, t, m_i, m_t = _toy(n=6, width=11, seed=8)
    _, _, correct = score_matrix(v, t, m_i, m_t)
    _, _, leaked = score_matrix(v, t, torch.stack([m_i[0]] * 6), m_t)   # one mask for every row
    assert not torch.allclose(correct, leaked)
    # the buggy form used mI[i] for row i; that is a different matrix by a wide margin
    wrong = torch.stack([torch.nn.functional.normalize(v[i] * m_i[i], dim=-1, eps=NORM_EPS)
                         for i in range(6)])
    right = torch.nn.functional.normalize(v.unsqueeze(1) * m_i.unsqueeze(0), dim=-1, eps=NORM_EPS)
    assert float((wrong[2] - right[2, 2]).abs().max()) < 1e-6         # the positive pair agrees
    assert float((wrong[2] - right[2, 0]).abs().max()) > 1e-3         # the negatives do not


# --------------------------------------------------------------------- 4: intersection identity
def test_intersection_identity_on_valid_pairs_and_graceful_invalid_pairs():
    v, t, m_i, m_t = _toy(n=6, width=9, seed=1)
    geometry = factory_form(v, t, m_i, m_t)
    q3 = geometry['q3']
    recomputed = geometry['q_cap'] * geometry['factor']
    valid = geometry['a'] > NORM_EPS ** 2
    valid &= geometry['b'].unsqueeze(0) > NORM_EPS ** 2
    valid &= geometry['c'] > NORM_EPS ** 2
    valid &= geometry['d'].unsqueeze(0) > NORM_EPS ** 2
    assert valid.any()
    assert torch.allclose(recomputed[valid], q3[valid], atol=1e-2, rtol=1e-5)
    assert torch.isfinite(q3).all()


def test_identical_masks_make_the_factor_exactly_one():
    v, t, m_i, m_t = _toy(n=5, width=6, seed=2)
    geometry = factory_form(v, t, m_i, m_i)          # c = mI * mI = mI and mT = mI
    assert torch.allclose(geometry['factor'], torch.ones_like(geometry['factor']), atol=1e-6)
    assert torch.allclose(geometry['q_cap'], geometry['q3'], atol=1e-2, rtol=1e-5)


def test_empty_intersection_is_zero_energy_but_not_nan_and_still_scored():
    """Disjoint masks: c is all zeros, so ||v*c|| = 0 -- the score must stay finite."""
    v = torch.randn(3, 5, generator=torch.Generator().manual_seed(4))
    t = torch.randn(3, 5, generator=torch.Generator().manual_seed(5))
    m_i = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]] * 3)
    m_t = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0]] * 3)
    geometry = factory_form(v, t, m_i, m_t)

    assert torch.all(geometry['c'] == 0)
    assert torch.all(geometry['d'] == 0)
    assert torch.isfinite(geometry['q_cap']).all()
    assert torch.isfinite(geometry['q3']).all()
    assert torch.all(geometry['factor'] == 0)
    # invalid by the eps test, but the pair still exists and is still scored
    assert not (geometry['c'] > NORM_EPS ** 2).any()


def test_partial_overlap_and_eps_boundary():
    v, t, m_i, m_t = _toy(n=4, width=8, seed=6)
    overlap = m_i * m_t
    assert (overlap.sum(dim=1) <= m_i.sum(dim=1)).all()
    geometry = factory_form(v, t, m_i, m_t)
    assert (geometry['c'] <= geometry['a'] + 1e-9).all()
    assert (geometry['d'] <= geometry['b'] + 1e-9).all()
    # rho <= 1, so the factor never exceeds 1
    assert (geometry['factor'] <= 1.0 + 1e-6).all()
    # a coordinate kept by neither mask can never be part of the intersection
    assert torch.all((m_i * m_t) <= m_i) and torch.all((m_i * m_t) <= m_t)


def test_tiny_features_do_not_produce_nan():
    v = torch.full((3, 4), 1e-9)
    t = torch.full((3, 4), 1e-9)
    m_i = torch.ones(3, 4)
    m_t = torch.ones(3, 4)
    geometry = factory_form(v, t, m_i, m_t)
    assert torch.isfinite(geometry['q3']).all()
    assert torch.isfinite(geometry['q_cap']).all()


# --------------------------------------------------------------------- 5: CE, margins, ties
def test_ce_identity_and_margin_signs():
    generator = torch.Generator().manual_seed(7)
    q = torch.randn(6, 6, generator=generator)
    metrics = retrieval_metrics(q)
    assert metrics['ce_identity_max_abs_diff'] < 1e-5
    diagonal = torch.diagonal(q)
    off = q.clone()
    off.fill_diagonal_(float('-inf'))
    expected_ce = float(torch.nn.functional.softplus(
        -(diagonal - torch.logsumexp(off, dim=1))).mean())
    assert metrics['ce'] == pytest.approx(expected_ce, abs=1e-6)


def test_negative_lse_margin_can_coexist_with_positive_max_margin():
    """A query can beat *every* single negative and still have more total negative mass than mass of
    its own: M_max > 0 while M_lse < 0. The two margins are not interchangeable."""
    q = torch.full((3, 3), -5.0)
    for row in range(2):
        q[row, row] = 0.0
        for other in range(3):
            if other != row:
                q[row, other] = -0.5          # each negative is below the positive ...
    q[2, 2] = 0.0
    q[2, 0] = q[2, 1] = -3.0                 # ... and here the negatives are far below

    metrics = retrieval_metrics(q)
    # no single negative beats its positive, so M_max is positive everywhere
    assert metrics['m_max_negative_count'] == 0
    # yet for the first two rows log(exp(-0.5) + exp(-0.5)) > 0, so M_lse < 0 there
    assert metrics['m_lse_negative_count'] == 2
    assert metrics['R@1'] == pytest.approx(1.0)
    lse = math.log(2.0 * math.exp(-0.5))
    assert lse > 0.0
    assert metrics['m_lse_mean'] < metrics['m_max_mean']


def test_tie_rule_does_not_let_a_negative_pass_the_positive():
    q = torch.zeros(3, 3)
    q[0, 0] = 1.0
    q[0, 1] = 1.0                        # exact tie with the positive
    q[1, 1] = 1.0
    q[1, 0] = 1.0                        # exact tie on another row
    q[2, 2] = 1.0
    q[2, 0] = 1.5                        # strictly better on the third row
    metrics = retrieval_metrics(q)
    assert metrics['m_max_tie_count'] == 2
    assert metrics['m_max_negative_count'] == 1
    assert metrics['R@1'] == pytest.approx(2.0 / 3.0)


def test_quantiles_helper_handles_empty_input():
    assert quantiles([]) is None
    assert quantiles([1.0])['q0.5'] == pytest.approx(1.0)
