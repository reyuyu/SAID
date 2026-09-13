"""Tests for the read-only CLIP text-nuisance probe: token/length handling, delta algebra,
coordinate concentration, split-half rule, SVD shares, the tie rule and the masked readout.

CPU-only, no GPU, no checkpoint and no dataset: every case is built from synthetic arrays and a stub
tokenizer, so a failure here is a formula or indexing bug and never a resource problem.
"""
import os
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'tools', 'diag'))

from clip_text_nuisance_probe import (BASE_CAPTIONS, CHANGE_DESCRIPTIONS, LENGTH_MATCHED_PAIRS,  # noqa: E402
                                      PARAPHRASES, R_ORDER, R_SUFFIXES, TOP_K, VISUAL_CHANGES,
                                      coordinate_report, ranking_metrics, split_half_report,
                                      suffix_check, svd_report, token_report)


class StubTokenizer:
    """A deterministic tokenizer: one id per word plus SOT/EOT, padded to the context length.

    Ids are a fixed function of the word (never ``hash()``, which is randomised per process), the EOT
    id is the largest one -- the convention ``encode_text`` relies on via ``text.argmax(dim=-1)`` --
    and the stub can emit id 0 for a content word so the "do not count token_id != 0" rule is
    actually exercised.
    """

    def __init__(self, context_length=16, zero_words=()):
        self.context_length = context_length
        self.zero_words = set(zero_words)

    def _id(self, word):
        if word in self.zero_words:
            return 0
        return 1 + (sum(ord(ch) for ch in word) % 900)

    def __call__(self, texts, context_length=None, truncate=False):
        width = context_length or self.context_length
        rows = []
        for text in texts:
            ids = [49406]
            for word in text.split():
                ids.append(self._id(word))
            ids.append(49407)
            if len(ids) > width:
                raise AssertionError('stub tokenizer does not truncate: %r' % text)
            rows.append(ids + [0] * (width - len(ids)))
        return torch.tensor(rows, dtype=torch.long)


# ------------------------------------------------------------------ token / length handling
def test_token_report_uses_the_argmax_eot_rule_not_nonzero_counting():
    tokenizer = StubTokenizer(zero_words=('zero',))
    report = token_report(tokenizer, 'a zero word here')
    # [SOT, a, zero, word, here, EOT]: the third content token is id 0 and must still be counted
    assert report['token_ids'][2] == 0
    assert report['effective_length'] == 6
    assert report['eot_index'] == 5
    assert sum(1 for t in report['token_ids'] if t != 0) == 5      # would undercount, hence argmax


def test_suffix_check_detects_a_present_and_an_absent_suffix():
    tokenizer = StubTokenizer()
    base = token_report(tokenizer, 'a dog runs')
    suffix = token_report(tokenizer, 'i like this')
    combined = token_report(tokenizer, 'a dog runs i like this')
    present = suffix_check(base, suffix, combined)
    assert present['entered'] is True
    assert present['shared_content_tokens'] == present['suffix_content_tokens'] == 3
    missing = token_report(tokenizer, 'a dog runs')
    assert suffix_check(base, suffix, missing)['entered'] is False
    # a suffix whose content is cut off must fail even though the base is intact
    cut = token_report(tokenizer, 'a dog runs i')
    assert suffix_check(base, suffix, cut)['entered'] is False
    empty = token_report(tokenizer, '')
    assert suffix_check(base, empty, combined)['entered'] is True


def test_every_hand_written_template_is_present_and_paired():
    assert len(BASE_CAPTIONS) == len(PARAPHRASES) == len(VISUAL_CHANGES) == 12
    assert len(CHANGE_DESCRIPTIONS) == 12
    assert set(R_SUFFIXES) == set(R_ORDER) == {'R1', 'R2', 'R3', 'R4'}
    assert len(set(BASE_CAPTIONS)) == 12 and len(set(PARAPHRASES)) == 12
    assert len(set(VISUAL_CHANGES)) == 12
    for base, change in zip(BASE_CAPTIONS, VISUAL_CHANGES):
        assert base != change, base
    for base, paraphrase in zip(BASE_CAPTIONS, PARAPHRASES):
        assert base != paraphrase
    assert all(len(left) == 2 for left in LENGTH_MATCHED_PAIRS)


# ------------------------------------------------------------------ delta algebra
def test_unit_delta_identities():
    generator = torch.Generator().manual_seed(0)
    base = torch.randn(512, generator=generator)
    variant = base + 0.01 * torch.randn(512, generator=generator)
    base_unit = torch.nn.functional.normalize(base, dim=0, eps=1e-6)
    variant_unit = torch.nn.functional.normalize(variant, dim=0, eps=1e-6)
    delta = variant_unit - base_unit
    # cos and L2 of the unit vectors satisfy ||d||^2 = 2 - 2 cos
    cos = float(torch.dot(base_unit, variant_unit))
    assert float(delta.norm() ** 2) == pytest.approx(2.0 - 2.0 * cos, abs=1e-6)
    # the mean delta cancelling out is not the same as there being no change
    mean_delta = delta.mean()
    assert abs(float(mean_delta)) < float(delta.abs().mean())


# ------------------------------------------------------------------ concentration / split half / SVD
def test_coordinate_concentration_matches_a_hand_computed_case():
    deltas = np.zeros((4, 8))
    deltas[:, 0] = 1.0            # all energy in coordinate 0
    deltas[0, 1] = 1.0            # a quarter of the pairs also move coordinate 1
    report = coordinate_report(deltas)
    energy = (deltas ** 2).mean(axis=0)
    total = energy.sum()
    assert report['energy_total'] == pytest.approx(total)
    assert report['topk_share']['top1_share'] == pytest.approx(energy[0] / total)
    assert report['topk_share']['top4_share'] == pytest.approx(
        (energy[0] + energy[1]) / total)
    assert report['topk_share']['top8_share'] == pytest.approx(1.0)
    assert report['top_coordinates_by_index'][:2] == [0, 1]
    assert len(report['per_coordinate_energy']) == 8


def test_split_half_uses_the_first_half_ordering_on_the_second_half():
    deltas = np.zeros((8, 16))
    deltas[:4, 0] = 1.0            # first half: coordinate 0 dominates
    deltas[4:, 1] = 1.0            # second half: coordinate 1 dominates instead
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    report = split_half_report(deltas, labels)
    assert report['coordinate_order_from_first_half'][0] == 0
    # with the order fixed on the first half, the second half's top-1 share must be 0 here
    assert report['second_half_topk_share_using_first_half_order']['top1_share'] == pytest.approx(0.0)
    assert report['topk_jaccard_between_halves']['top1_jaccard'] == 0.0
    # the two halves disagree about the profile, which is exactly what the split-half view reports
    assert report['energy_profile_correlation'] is not None
    assert report['n_pairs_first'] == 4 and report['n_pairs_second'] == 4


def test_svd_shares_sum_to_one_and_include_the_common_offset():
    generator = np.random.default_rng(0)
    deltas = generator.normal(size=(16, 6)) * 0.1
    deltas[:, 3] += 2.0                                   # one direction carries most energy
    report = svd_report(deltas)
    first = report['component_share']['pc1']
    assert 0.0 < first <= 1.0
    assert first > 0.5
    assert report['mean_delta_energy_share'] is not None
    assert report['squared_singular_values'][0] >= report['squared_singular_values'][1]
    # the uncentred first component is not the mean direction only: removing the mean leaves energy
    centred = deltas - deltas.mean(axis=0)
    assert np.linalg.svd(centred, compute_uv=False)[0] > 0


# ------------------------------------------------------------------ ranking metrics and ties
def test_ranking_metrics_tie_rule_and_counts():
    scores = torch.zeros(3, 3)
    scores[0, 0] = 1.0
    scores[0, 1] = 1.0            # exact tie with the positive: must not beat it
    scores[1, 1] = 1.0
    scores[1, 0] = 1.5            # strictly better: the positive is beaten
    scores[2, 2] = 1.0
    metrics = ranking_metrics(scores)
    assert metrics['R@1'] == pytest.approx(2.0 / 3.0)
    assert metrics['tie_pairs_total'] == 1
    assert metrics['tie_queries'] == 1
    assert metrics['ce_identity_max_abs_diff'] < 1e-6
    assert metrics['per_query_rank'] == [1, 2, 1]


def test_margins_and_ce_identity():
    generator = torch.Generator().manual_seed(3)
    scores = torch.randn(5, 5, generator=generator)
    metrics = ranking_metrics(scores)
    diagonal = torch.diagonal(scores)
    off = scores.clone()
    off.fill_diagonal_(float('-inf'))
    assert metrics['m_max_mean'] == pytest.approx(
        float((diagonal - off.max(dim=1).values).mean()), abs=1e-6)
    assert metrics['m_lse_mean'] == pytest.approx(
        float((diagonal - torch.logsumexp(off, dim=1)).mean()), abs=1e-6)
    assert metrics['ce'] == pytest.approx(
        float(torch.nn.functional.softplus(
            -(diagonal - torch.logsumexp(off, dim=1))).mean()), abs=1e-6)
    assert set(TOP_K) == {1, 4, 8, 16, 32, 64}
