"""Phase 3.0A.1e tests: common-mode semantic decomposition and paired statistics.

Two things are pinned down here. First, the decomposition algebra and the fact that *every*
query feature is built from ``(I, C_S)`` before the candidate pool exists, so the centroid /
centred / contrast scorers cannot be candidate-conditioned. Second, that the statistics used
for the attribution verdicts are correct and deterministic: a paired bootstrap that gives
``delta = 0`` and ``CI = [0, 0]`` for identical methods, an exact McNemar test, and finite
permutation-free Spearman correlations.
"""
import inspect
import math
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, 'train'), os.path.join(REPO_ROOT, 'tests')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval import paired_statistics as ps  # noqa: E402
from eval import phase30a_fixed_cohort as fc  # noqa: E402
from eval.unsaid_retrieval import retrieval_report  # noqa: E402
from model.gap_completion import (  # noqa: E402
    gap_completion_terms,
    soft_anti_said_attention,
    unsaid_feature_from_attention,
)
from test_gap_completion import build_model, gap_forward, make_batch  # noqa: E402

DIM = 8
PATCHES = 6
BATCH = 5


def reference_decomposition(patches, attention_said, attention_unsaid, global_feature,
                            complete_feature):
    """Independent, literal implementation of the decomposition used as ground truth."""
    centroid = patches.mean(dim=1)
    centered = patches - centroid.unsqueeze(1)
    delta_said = torch.einsum('bp,bpd->bd', attention_said, centered)
    delta_unsaid = torch.einsum('bp,bpd->bd', attention_unsaid, centered)
    pooled_said = torch.einsum('bp,bpd->bd', attention_said, patches)
    pooled_unsaid = torch.einsum('bp,bpd->bd', attention_unsaid, patches)
    return {
        'centroid': F.normalize(centroid, dim=-1),
        'delta_said': delta_said,
        'delta_unsaid': delta_unsaid,
        'pooled_said': pooled_said,
        'pooled_unsaid': pooled_unsaid,
        'contrast': pooled_unsaid - pooled_said,
        'global': F.normalize(global_feature, dim=-1),
        'complete_existing': F.normalize(complete_feature, dim=-1),
    }


# --------------------------------------------------------------------------- #
# decomposition algebra
# --------------------------------------------------------------------------- #
def test_pooling_equals_centroid_plus_centered_deviation():
    torch.manual_seed(0)
    patches = torch.randn(4, PATCHES, DIM)
    a_said = torch.softmax(torch.randn(4, PATCHES), dim=-1)
    a_unsaid = torch.softmax(torch.randn(4, PATCHES), dim=-1)
    z_global = torch.randn(4, DIM)
    z_said = torch.randn(4, DIM)
    complete = torch.randn(4, DIM)

    pieces = fc.decomposition_features(patches, a_said, a_unsaid, z_global, z_said, complete)
    reference = reference_decomposition(patches, a_said, a_unsaid, z_global, complete)

    # p = mu + delta, for both poolings
    assert torch.allclose(pieces['pooled_said_vector'],
                          pieces['centroid_vector'] + pieces['delta_said_vector'], atol=1e-5)
    assert torch.allclose(pieces['pooled_unsaid_vector'],
                          pieces['centroid_vector'] + pieces['delta_unsaid_vector'], atol=1e-5)
    # and the literal implementation agrees with the module
    assert torch.allclose(pieces['centroid_vector'], patches.mean(dim=1), atol=1e-5)
    assert torch.allclose(pieces['delta_said_vector'], reference['delta_said'], atol=1e-5)
    assert torch.allclose(pieces['delta_unsaid_vector'], reference['delta_unsaid'], atol=1e-5)
    assert torch.allclose(pieces['pooled_said_vector'], reference['pooled_said'], atol=1e-5)
    assert torch.allclose(pieces['pooled_unsaid_vector'], reference['pooled_unsaid'], atol=1e-5)


def test_contrast_equals_delta_unsaid_minus_delta_said():
    torch.manual_seed(1)
    patches = torch.randn(3, PATCHES, DIM)
    a_said = torch.softmax(torch.randn(3, PATCHES), dim=-1)
    a_unsaid = torch.softmax(torch.randn(3, PATCHES), dim=-1)
    pieces = fc.decomposition_features(patches, a_said, a_unsaid, torch.randn(3, DIM),
                                       torch.randn(3, DIM), torch.randn(3, DIM))
    # the centroid cancels: p_U - p_S = delta_U - delta_S
    assert torch.allclose(pieces['contrast_vector'],
                          pieces['delta_unsaid_vector'] - pieces['delta_said_vector'],
                          atol=1e-5)
    assert torch.allclose(pieces['contrast_vector'],
                          pieces['pooled_unsaid_vector'] - pieces['pooled_said_vector'],
                          atol=1e-5)
    # the normalised contrast scorer is parallel to the raw contrast
    cosine = (F.normalize(pieces['contrast_vector'], dim=-1)
              * pieces['anti_minus_said']).sum(dim=-1)
    assert torch.allclose(cosine, torch.ones_like(cosine), atol=1e-5)


def test_all_eight_scorers_are_present_and_unit_norm():
    torch.manual_seed(2)
    patches = torch.randn(2, PATCHES, DIM)
    pieces = fc.decomposition_features(patches, torch.softmax(torch.randn(2, PATCHES), -1),
                                       torch.softmax(torch.randn(2, PATCHES), -1),
                                       torch.randn(2, DIM), torch.randn(2, DIM),
                                       torch.randn(2, DIM))
    assert fc.DECOMPOSITION_SCORERS == ('global', 'said_raw', 'unsaid_raw', 'centroid',
                                        'said_centered', 'unsaid_centered', 'anti_minus_said',
                                        'complete_existing')
    for name in fc.DECOMPOSITION_SCORERS:
        norms = pieces[name].norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), name
        assert pieces[name].requires_grad is False, name


def test_decomposition_rejects_wrong_shapes():
    with pytest.raises(ValueError):
        fc.decomposition_features(torch.randn(2, DIM), torch.randn(2, PATCHES),
                                  torch.randn(2, PATCHES), torch.randn(2, DIM),
                                  torch.randn(2, DIM), torch.randn(2, DIM))


def test_centroid_scorer_uses_only_the_patch_mean():
    """The centroid scorer must not depend on the attention distributions at all."""
    torch.manual_seed(3)
    patches = torch.randn(4, PATCHES, DIM)
    a_said = torch.softmax(torch.randn(4, PATCHES), dim=-1)
    a_unsaid = torch.softmax(torch.randn(4, PATCHES), dim=-1)
    other = torch.softmax(torch.randn(4, PATCHES), dim=-1)
    first = fc.decomposition_features(patches, a_said, a_unsaid, torch.randn(4, DIM),
                                      torch.randn(4, DIM), torch.randn(4, DIM))
    second = fc.decomposition_features(patches, other, other.flip(1), torch.randn(4, DIM),
                                       torch.randn(4, DIM), torch.randn(4, DIM))
    assert torch.allclose(first['centroid'], second['centroid'], atol=1e-6)
    # while the centred scorers do change
    assert not torch.allclose(first['said_centered'], second['said_centered'])


# --------------------------------------------------------------------------- #
# the decomposition scorers are candidate-independent
# --------------------------------------------------------------------------- #
def test_precompute_has_no_candidate_argument_and_scoring_keeps_the_pool():
    parameters = list(inspect.signature(fc.decomposition_features).parameters)
    for banned in ('candidate', 'candidates', 'target', 'unsaid_texts', 'pool'):
        assert not any(banned in name for name in parameters), banned
    scorer_parameters = list(inspect.signature(fc.decomposition_scorer_matrices).parameters)
    assert scorer_parameters == ['features', 'candidate_text_features', 'logit_scale']

    torch.manual_seed(4)
    features = {name: F.normalize(torch.randn(5, DIM), dim=-1)
                for name in fc.DECOMPOSITION_SCORERS}
    candidates = torch.randn(5, DIM)
    matrices = fc.decomposition_scorer_matrices(features, candidates, logit_scale=2.0)
    assert set(matrices) == set(fc.DECOMPOSITION_SCORERS)
    for name, matrix in matrices.items():
        assert matrix.shape == (5, 5), name
        assert torch.allclose(matrix,
                              fc.scorer_matrix(features[name], candidates, logit_scale=2.0),
                              atol=1e-6), name
    report = retrieval_report(matrices['centroid'])
    assert report['candidate_pool'] == 5 and report['query_count'] == 5


def test_changing_the_candidate_pool_leaves_every_query_feature_untouched():
    """The query features are recomputed identically whatever pool the scoring uses.

    ``decomposition_features`` has no candidate input, so the only way a pool could leak in is
    through the model state; this checks that the feature vectors are reproducible and that
    the pool only changes the number of score columns.
    """
    model = build_model()
    images, texts = make_batch()

    def build_features():
        with torch.no_grad():
            z_global, patches = model.encode_router_input(images)
            t = F.normalize(model.encode_text(texts), dim=-1)
            details = model.said_router.forward_with_details(t, patches)
            attention_unsaid = soft_anti_said_attention(details['scores'])
            z_unsaid = unsaid_feature_from_attention(attention_unsaid, patches)
            terms = gap_completion_terms(z_global, details['said'], z_unsaid)
            return fc.decomposition_features(patches, details['attention'], attention_unsaid,
                                             z_global, details['said'],
                                             F.normalize(details['said'] + terms['u_new'],
                                                         dim=-1))

    pieces = build_features()
    large = torch.randn(3 * BATCH, DIM)
    small = large[:BATCH]                          # a strict sub-pool of the large one
    scores_small = fc.decomposition_scorer_matrices(pieces, small)
    scores_large = fc.decomposition_scorer_matrices(pieces, large)
    assert scores_small['centroid'].shape == (BATCH, BATCH)
    assert scores_large['centroid'].shape == (BATCH, 3 * BATCH)

    # the query side is bit-identical across pool sizes and across repeated construction
    again = build_features()
    for name in fc.DECOMPOSITION_SCORERS:
        assert torch.equal(pieces[name], again[name]), name
        assert torch.allclose(scores_small[name], scores_large[name][:, :BATCH], atol=1e-6), name


def test_decomposition_features_are_detached_on_a_real_forward():
    model = build_model()
    images, texts = make_batch()
    out = gap_forward(model, images, texts)
    assert out['loss_total'].requires_grad
    # the 3.0A.1c diagnostics merged into the gap output carry no graph
    for key in ('raw_pool_cosine',):
        assert key in out, key
        assert torch.is_tensor(out[key]), key
        assert out[key].requires_grad is False, key
        assert torch.isfinite(out[key]), key
    # the 1e common-mode metrics are computed by the evaluation runner, not by training, so
    # the training output must not have grown new keys for them
    for key in ('patch_centroid_norm', 'centered_pool_cosine', 'common_mode_norm_ratio_said',
                'common_mode_norm_ratio_unsaid'):
        assert key not in out, key
    # and the eight decomposition scorers built from the same forward are grad-free too
    with torch.no_grad():
        z_global, patches = model.encode_router_input(images)
        t = F.normalize(model.encode_text(texts), dim=-1)
        details = model.said_router.forward_with_details(t, patches)
        attention_unsaid = soft_anti_said_attention(details['scores'])
        z_unsaid = unsaid_feature_from_attention(attention_unsaid, patches)
        terms = gap_completion_terms(z_global, details['said'], z_unsaid)
        pieces = fc.decomposition_features(patches, details['attention'], attention_unsaid,
                                           z_global, details['said'],
                                           F.normalize(details['said'] + terms['u_new'], dim=-1))
    for name in fc.DECOMPOSITION_SCORERS:
        assert pieces[name].requires_grad is False, name


# --------------------------------------------------------------------------- #
# norm_ratio terminology
# --------------------------------------------------------------------------- #
def test_norm_ratio_can_legally_exceed_one_and_is_aliased():
    """``||mu|| / ||p||`` is a ratio of norms, not a fraction: it can exceed 1.

    ``p`` is a signed combination ``mu + delta``, so when the centroid and the attended patch
    point the same way the pooled norm shrinks well below the centroid norm (and, as a
    separate aligned example below shows, it can also exceed it). Either way the quantity is
    an unbounded ratio, so the old "fraction" name was wrong.
    """
    patches = torch.zeros(1, PATCHES, DIM)
    patches[0, 0, 0] = -19.0                       # p = h_0, same direction as mu
    patches[0, 1:, 0] = 1.0
    attention = torch.zeros(1, PATCHES)
    attention[0, 0] = 1.0
    centroid_norm = float(patches.mean(dim=1).norm(dim=-1))
    pooled_norm = float(patches[0, 0].norm())
    assert centroid_norm > 0.0 and pooled_norm > 0.0
    metrics = fc.common_mode_metrics(patches, attention, attention)
    assert metrics['common_mode_norm_ratio_said'] == pytest.approx(centroid_norm / pooled_norm,
                                                                   abs=1e-5)
    assert metrics['common_mode_norm_ratio_said'] < 1.0
    assert metrics['common_mode_norm_ratio_said'] == pytest.approx(
        metrics['common_mode_fraction_said'], abs=0.0)

    # an opposed construction goes above 1: the attended patch cancels the centroid's
    # direction, so ||p|| < ||mu||
    opposed = torch.zeros(1, PATCHES, DIM)
    opposed[0, 0, 0] = -1.0
    opposed[0, 1:, 0] = 3.0
    opposed_metrics = fc.common_mode_metrics(opposed, attention, attention)
    opposed_centroid = float(opposed.mean(dim=1).norm(dim=-1))
    opposed_pooled = float(opposed[0, 0].norm())
    assert opposed_centroid > opposed_pooled
    assert opposed_metrics['common_mode_norm_ratio_said'] == pytest.approx(
        opposed_centroid / opposed_pooled, abs=1e-5)
    assert opposed_metrics['common_mode_norm_ratio_said'] > 1.0


def test_common_mode_metrics_report_both_norm_ratios():
    torch.manual_seed(5)
    patches = torch.randn(3, PATCHES, DIM)
    metrics = fc.common_mode_metrics(patches, torch.softmax(torch.randn(3, PATCHES), -1),
                                     torch.softmax(torch.randn(3, PATCHES), -1))
    for key in ('common_mode_norm_ratio_said', 'common_mode_norm_ratio_unsaid',
                'common_mode_norm_ratio_said_max', 'common_mode_norm_ratio_unsaid_max'):
        assert key in metrics and math.isfinite(metrics[key]), key


# --------------------------------------------------------------------------- #
# paired bootstrap / McNemar
# --------------------------------------------------------------------------- #
def test_paired_bootstrap_is_deterministic_and_zero_for_identical_methods():
    values = [float(v) for v in np.random.default_rng(0).random(200)]
    first = ps.paired_bootstrap(values, values, replicates=2000, seed=7)
    second = ps.paired_bootstrap(values, values, replicates=2000, seed=7)
    assert first == second                                  # deterministic for a fixed seed
    assert first['delta'] == pytest.approx(0.0, abs=0.0)
    assert first['ci_low'] == pytest.approx(0.0, abs=0.0)
    assert first['ci_high'] == pytest.approx(0.0, abs=0.0)
    assert first['bootstrap_se'] == pytest.approx(0.0, abs=0.0)
    assert first['excludes_zero'] is False
    assert first['replicates'] == 2000 and first['query_count'] == 200

    different = ps.paired_bootstrap(values, [1.0 - v for v in values], replicates=2000, seed=7)
    assert different['delta'] != 0.0
    # a different seed gives a slightly different interval but the same point estimate
    other_seed = ps.paired_bootstrap(values, [1.0 - v for v in values], replicates=2000, seed=8)
    assert other_seed['delta'] == different['delta']
    assert other_seed['ci_low'] != different['ci_low']


def test_paired_bootstrap_rejects_bad_input():
    with pytest.raises(ValueError):
        ps.paired_bootstrap([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        ps.paired_bootstrap([1.0, 2.0], [1.0, 2.0], replicates=0)
    with pytest.raises(ValueError):
        ps.paired_bootstrap([1.0, float('nan')], [1.0, 2.0])


def test_mcnemar_exact_counts_and_p_value():
    left = [1, 1, 1, 0, 0, 0, 1, 1]
    right = [0, 0, 0, 0, 0, 0, 1, 1]
    result = ps.mcnemar_exact(left, right)
    assert result['n10_left_only'] == 3
    assert result['n01_right_only'] == 0
    assert result['discordant'] == 3
    assert result['p_value_exact_two_sided'] == pytest.approx(0.25, abs=1e-9)
    # identical methods have no discordant pairs
    same = ps.mcnemar_exact(left, left)
    assert same['discordant'] == 0 and same['p_value_exact_two_sided'] == 1.0
    # symmetric discordance gives p = 1
    symmetric = ps.mcnemar_exact([1, 0, 1, 0], [0, 1, 1, 0])
    assert symmetric['n10_left_only'] == 1 and symmetric['n01_right_only'] == 1
    assert symmetric['p_value_exact_two_sided'] == pytest.approx(1.0, abs=1e-9)


def test_compare_methods_uses_the_same_queries():
    torch.manual_seed(6)
    n = 40
    left = torch.randn(n, n)
    right = left.clone()
    right[:, 0] = right[:, 0] + 5.0                  # a deliberate, systematic change
    labels = torch.arange(n)
    result = ps.compare_methods(left, right, replicates=500, seed=3, labels=labels)
    assert result['R@1']['query_count'] == n
    assert result['R@1']['replicates'] == 500
    assert 'mcnemar_R@1' in result and 'discordant' in result['mcnemar_R@1']
    identical = ps.compare_methods(left, left, replicates=500, seed=3, labels=labels)
    assert identical['R@1']['delta'] == 0.0
    assert identical['MRR']['delta'] == 0.0
    assert identical['mcnemar_R@1']['discordant'] == 0


# --------------------------------------------------------------------------- #
# Spearman / proxy correlations
# --------------------------------------------------------------------------- #
def test_spearman_matches_scipy_free_expectations():
    perfect = ps.spearman([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
    assert perfect['rho'] == pytest.approx(1.0, abs=1e-9)
    inverted = ps.spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
    assert inverted['rho'] == pytest.approx(-1.0, abs=1e-9)
    assert inverted['n'] == 5
    # ties are handled by average ranks, so a constant input is degenerate rather than NaN
    degenerate = ps.spearman([1, 1, 1, 1], [1, 2, 3, 4])
    assert degenerate['rho'] == 0.0 and math.isfinite(degenerate['p_value'])
    # a mildly noisy monotone relation stays positive and finite
    rng = np.random.default_rng(1)
    x = rng.random(200)
    y = x + 0.1 * rng.random(200)
    noisy = ps.spearman(x.tolist(), y.tolist())
    assert 0.8 < noisy['rho'] <= 1.0
    assert 0.0 <= noisy['p_value'] <= 1.0


def test_spearman_rejects_degenerate_input():
    with pytest.raises(ValueError):
        ps.spearman([1.0, 2.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        ps.spearman([1.0, 2.0, 3.0], [1.0, float('inf'), 3.0])


def test_margin_per_query_is_positive_minus_negatives():
    scores = torch.tensor([[5.0, 1.0, 2.0], [0.0, 4.0, 3.0], [1.0, 2.0, 6.0]])
    mean_margin = ps.margin_per_query(scores, 'mean')
    best_margin = ps.margin_per_query(scores, 'best')
    assert torch.allclose(mean_margin, torch.tensor([5.0 - 1.5, 4.0 - 1.5, 6.0 - 1.5]),
                          atol=1e-6)
    assert torch.allclose(best_margin, torch.tensor([5.0 - 2.0, 4.0 - 3.0, 6.0 - 2.0]),
                          atol=1e-6)
    with pytest.raises(ValueError):
        ps.margin_per_query(scores, 'nonsense')


def test_margin_per_query_requires_labels_for_rectangular_scores():
    scores = torch.randn(3, 5)
    with pytest.raises(ValueError):
        ps.margin_per_query(scores)
    labels = torch.tensor([0, 1, 2])
    assert ps.margin_per_query(scores, 'mean', labels).shape == (3,)


# --------------------------------------------------------------------------- #
# no training code
# --------------------------------------------------------------------------- #
def test_no_training_code_or_optimizer_step_was_introduced():
    import tokenize

    for relative in (os.path.join('eval', 'phase30a_fixed_cohort.py'),
                     os.path.join('eval', 'paired_statistics.py'),
                     os.path.join('tools', 'phase30a_fixed_cohort_eval.py')):
        pieces = []
        with open(os.path.join(REPO_ROOT, relative), encoding='utf-8') as handle:
            for token in tokenize.generate_tokens(handle.readline):
                if token.type in (tokenize.COMMENT, tokenize.STRING):
                    continue
                pieces.append(token.string)
        code = ' '.join(pieces)
        for banned in ('optimizer', 'backward', 'GradScaler', 'lr_scheduler', 'AdamW',
                       'loss_total'):
            assert banned not in code, (relative, banned)


def test_the_training_path_still_has_no_c_u_input():
    train_source = open(os.path.join(REPO_ROOT, 'train', 'train_salu.py'),
                        encoding='utf-8').read()
    for banned in ('texts_uss', 'lambda_uss', 'caption_uss', 'texts_unsaid_candidate'):
        assert banned not in train_source, banned
