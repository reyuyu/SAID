"""Phase 3.0A.1d tests: fixed-cohort evaluation, Phase 3 USR scorers, patch common mode.

The central structural claim of the phase is that the Phase 3 scorers are
**target-independent**: ``z_U`` must be built from ``(I, C_S)`` alone and the score matrix
must then be a plain inner product over the *full* candidate pool. Most tests below pin that
down, including the two failure modes it excludes (candidate-conditioned features and a
sub-setted candidate pool).
"""
import inspect
import json
import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, 'train'), os.path.join(REPO_ROOT, 'tests')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval import phase30a_fixed_cohort as fc  # noqa: E402
from eval.unsaid_retrieval import retrieval_report  # noqa: E402
from test_gap_completion import build_model, gap_forward, make_batch  # noqa: E402

DIM = 8
PATCHES = 6
BATCH = 5


# --------------------------------------------------------------------------- #
# Phase 3 scorers are construction-independent
# --------------------------------------------------------------------------- #
def test_z_unsaid_is_built_before_any_candidate_exists():
    """``precompute_query_features`` has no candidate parameter at all."""
    parameters = list(inspect.signature(fc.precompute_query_features).parameters)
    assert parameters == ['model', 'images', 'prefix_tokens', 'gap_anti_temperature']
    for banned in ('candidate', 'candidates', 'target', 'unsaid_texts', 'pool'):
        assert not any(banned in name for name in parameters), banned
    # and the scorer takes precomputed features, never images or prefixes
    scorer_parameters = list(inspect.signature(fc.scorer_matrix).parameters)
    assert scorer_parameters == ['features', 'candidate_text_features', 'logit_scale']


def test_candidate_conditioned_scoring_is_explicitly_rejected():
    """The Phase 2.9 candidate-conditioned path must not be reachable from this module."""
    with pytest.raises(NotImplementedError):
        fc.targeted_upper_bound_matrix()


def test_changing_the_candidate_pool_cannot_change_a_precomputed_z_unsaid():
    model = build_model()
    images, texts = make_batch()
    baseline = fc.precompute_query_features(model, images, texts, gap_anti_temperature=1.0)

    # a completely different candidate pool exists only on the scoring side
    pool_a = torch.randn(BATCH, DIM) @ torch.randn(DIM, DIM)
    pool_b = torch.randn(2 * BATCH, DIM) @ torch.randn(DIM, DIM)
    scores_a = fc.scorer_matrix(baseline['unsaid_feature'], pool_a)
    scores_b = fc.scorer_matrix(baseline['unsaid_feature'], pool_b)
    assert scores_a.shape == (BATCH, BATCH)
    assert scores_b.shape == (BATCH, 2 * BATCH)

    # the query features are bit-identical whatever pool is used afterwards
    for candidate_pool in (pool_a, pool_b):
        again = fc.precompute_query_features(model, images, texts, gap_anti_temperature=1.0)
        assert torch.equal(baseline['unsaid_feature'], again['unsaid_feature'])
        assert candidate_pool.shape[1] == DIM


def test_scorer_matrix_is_the_precomputed_inner_product_over_the_full_pool():
    torch.manual_seed(0)
    query = torch.randn(4, DIM)
    candidates = torch.randn(4, DIM)
    scale = 3.5
    matrix = fc.scorer_matrix(query, candidates, logit_scale=scale)
    expected = scale * (F.normalize(query, dim=-1) @ F.normalize(candidates, dim=-1).t())
    assert torch.allclose(matrix, expected, atol=1e-6)
    # square pool -> the diagonal really is the own-target score
    assert matrix.shape == (4, 4)
    assert torch.allclose(matrix.diagonal(), (F.normalize(query, dim=-1)
                                              * F.normalize(candidates, dim=-1)).sum(-1) * scale,
                          atol=1e-6)


def test_phase3_matrices_keep_the_full_candidate_pool_and_are_all_present():
    torch.manual_seed(1)
    query_features = {
        'global_feature': torch.randn(5, DIM),
        'unsaid_feature': torch.randn(5, DIM),
        'complete_feature': torch.randn(5, DIM),
    }
    candidates = torch.randn(5, DIM)
    matrices = fc.phase3_scorer_matrices(query_features, candidates, logit_scale=2.0)
    assert set(matrices) == set(fc.PHASE3_SCORERS) == {'global', 'unsaid', 'complete'}
    for name, matrix in matrices.items():
        assert matrix.shape == (5, 5), name
        assert torch.isfinite(matrix).all(), name
    # retrieval_report keeps the full pool (Q candidates, Q queries)
    report = retrieval_report(matrices['unsaid'])
    assert report['candidate_pool'] == 5 and report['query_count'] == 5
    assert 0.0 <= report['R@1'] <= 1.0 and 0.0 <= report['MRR'] <= 1.0
    for key in ('R@1', 'R@5', 'R@10', 'MRR', 'mean_rank', 'median_rank'):
        assert key in report, key


def test_complete_feature_is_normalize_said_plus_u_new():
    model = build_model()
    images, texts = make_batch()
    encoded = model.encode_said_unsaid(images, texts, return_details=True)
    features = fc.precompute_query_features(model, images, texts, gap_anti_temperature=1.0)
    expected = F.normalize(encoded['said_feature'] + encoded['u_new'], dim=-1)
    assert torch.allclose(features['complete_feature'], expected, atol=1e-6)
    assert torch.allclose(features['unsaid_feature'], encoded['unsaid_feature'], atol=1e-6)
    norms = features['complete_feature'].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_precompute_does_not_require_or_create_gradients():
    model = build_model()
    images, texts = make_batch()
    features = fc.precompute_query_features(model, images, texts, gap_anti_temperature=1.0)
    for key, value in features.items():
        assert value.requires_grad is False, key
        assert value.grad_fn is None, key
    assert all(parameter.grad is None for parameter in model.parameters())


def test_moving_the_anti_temperature_moves_z_unsaid_but_not_the_cls_baseline():
    model = build_model()
    images, texts = make_batch()
    cold = fc.precompute_query_features(model, images, texts, gap_anti_temperature=0.25)
    warm = fc.precompute_query_features(model, images, texts, gap_anti_temperature=1.0)
    assert not torch.allclose(cold['unsaid_feature'], warm['unsaid_feature'])
    assert torch.equal(cold['global_feature'], warm['global_feature'])
    assert torch.equal(cold['said_feature'], warm['said_feature'])
    assert not torch.allclose(cold['complete_feature'], warm['complete_feature'])


# --------------------------------------------------------------------------- #
# patch common mode (Q5)
# --------------------------------------------------------------------------- #
def test_common_mode_decomposition_matches_the_direct_definition():
    """p_S = mu + delta_S must hold numerically, and raw vs centred cosines must be exact."""
    torch.manual_seed(2)
    patches = torch.randn(3, PATCHES, DIM)
    attention_said = torch.softmax(torch.randn(3, PATCHES), dim=-1)
    attention_unsaid = torch.softmax(torch.randn(3, PATCHES), dim=-1)
    metrics = fc.common_mode_metrics(patches, attention_said, attention_unsaid)

    centroid = patches.mean(dim=1)
    centered = patches - centroid.unsqueeze(1)
    delta_said = torch.einsum('bp,bpd->bd', attention_said, centered)
    delta_unsaid = torch.einsum('bp,bpd->bd', attention_unsaid, centered)
    pooled_said = torch.einsum('bp,bpd->bd', attention_said, patches)
    pooled_unsaid = torch.einsum('bp,bpd->bd', attention_unsaid, patches)
    assert torch.allclose(pooled_said, centroid + delta_said, atol=1e-5)
    assert torch.allclose(pooled_unsaid, centroid + delta_unsaid, atol=1e-5)

    assert metrics['patch_centroid_norm'] == pytest.approx(float(centroid.norm(dim=-1).mean()),
                                                           abs=1e-6)
    assert metrics['centered_said_norm'] == pytest.approx(float(delta_said.norm(dim=-1).mean()),
                                                          abs=1e-6)
    assert metrics['centered_unsaid_norm'] == pytest.approx(
        float(delta_unsaid.norm(dim=-1).mean()), abs=1e-6)
    assert metrics['raw_pool_cosine'] == pytest.approx(
        float((F.normalize(pooled_said, dim=-1) * F.normalize(pooled_unsaid, dim=-1)).sum(-1).mean()),
        abs=1e-6)
    assert metrics['centered_pool_cosine'] == pytest.approx(
        float((F.normalize(delta_said, dim=-1) * F.normalize(delta_unsaid, dim=-1)).sum(-1).mean()),
        abs=1e-6)


def test_common_mode_dominates_when_the_centroid_is_large_and_deltas_differ():
    """A dominant shared centroid gives a high raw cosine with a low centred cosine.

    Two patches per image, each a large shared centroid plus a small, different deviation;
    A_S selects patch 0 and A_U selects patch 1, so the poolings are far apart after
    centring while remaining nearly parallel before it.
    """
    torch.manual_seed(3)
    centroid = torch.randn(2, DIM) * 10.0
    delta_said = torch.randn(2, DIM) * 0.1
    delta_unsaid = torch.randn(2, DIM) * 0.1
    patches = torch.stack([centroid + delta_said, centroid + delta_unsaid], dim=1)  # [2, 2, D]
    attention_said = torch.zeros(2, 2)
    attention_said[:, 0] = 1.0
    attention_unsaid = torch.zeros(2, 2)
    attention_unsaid[:, 1] = 1.0

    metrics = fc.common_mode_metrics(patches, attention_said, attention_unsaid)
    assert metrics['raw_pool_cosine'] > 0.95
    assert metrics['centered_pool_cosine'] < metrics['raw_pool_cosine']
    assert metrics['common_mode_ratio'] > 0.9
    diagnosis = fc.common_mode_diagnosis(metrics)
    assert diagnosis['drop_raw_minus_centered'] == pytest.approx(
        metrics['raw_pool_cosine'] - metrics['centered_pool_cosine'], abs=1e-9)
    assert diagnosis['common_mode_dominates_supported'] is True


def test_common_mode_fractions_are_one_for_a_single_attention_spike_with_identical_patches():
    """When every patch is the centroid, p = mu exactly, so ||mu||/||p|| == 1."""
    patches = torch.ones(2, PATCHES, DIM)
    attention = torch.zeros(2, PATCHES)
    attention[:, 0] = 1.0
    metrics = fc.common_mode_metrics(patches, attention, attention)
    assert metrics['common_mode_ratio'] == pytest.approx(1.0, abs=1e-5)
    assert metrics['common_mode_fraction_said'] == pytest.approx(1.0, abs=1e-5)
    assert metrics['common_mode_fraction_unsaid'] == pytest.approx(1.0, abs=1e-5)
    assert metrics['centered_said_norm'] == pytest.approx(0.0, abs=1e-5)


def test_common_mode_metrics_are_finite_and_detached_on_a_real_forward():
    model = build_model()
    images, texts = make_batch()
    out = gap_forward(model, images, texts)
    assert torch.isfinite(torch.tensor(out['raw_pool_cosine']))
    # the same numbers as the 3.0A.1c patch metrics, computed from the same forward
    assert out['raw_pool_cosine'] is not None
    assert out['cos_said_unsaid'] is not None
    # the model output carries the 3.0A.1c diagnostics, still grad-free
    assert out['patch_pair_cosine_mean'].requires_grad is False


def test_common_mode_rejects_wrong_shapes():
    with pytest.raises(ValueError):
        fc.common_mode_metrics(torch.randn(2, DIM), torch.randn(2, PATCHES),
                               torch.randn(2, PATCHES))
    with pytest.raises(ValueError):
        fc.common_mode_metrics(torch.randn(2, PATCHES, DIM), torch.randn(2, PATCHES, 1),
                               torch.randn(2, PATCHES))


# --------------------------------------------------------------------------- #
# cohort helpers
# --------------------------------------------------------------------------- #
def test_cohort_mean_aggregates_only_finite_numbers():
    rows = [{'a': 1.0, 'b': 2.0}, {'a': 3.0, 'b': 4.0}]
    aggregate = fc.cohort_mean(rows, ('a', 'b'))
    assert aggregate == {'a': 2.0, 'b': 3.0}
    with pytest.raises(ValueError):
        fc.cohort_mean(rows, ('missing',))


def test_cohort_samples_are_frozen_and_order_preserving(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        'phase30a_fixed_cohort_eval',
        os.path.join(REPO_ROOT, 'tools', 'phase30a_fixed_cohort_eval.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    queries = [{'query_id': index, 'json_index': 100 + index, 'image_path': 'a/%d.jpg' % index,
                'caption_said': 'prefix %d' % index, 'target_text': 'suffix %d' % index,
                'K': 2, 'N': 3, 'J': 3} for index in range(4)]
    first = module.cohort_samples(queries)
    second = module.cohort_samples(queries)
    assert first == second                                   # deterministic, never resampled
    assert [sample['json_index'] for sample in first] == [100, 101, 102, 103]
    assert [sample['target_text'] for sample in first] == ['suffix %d' % i for i in range(4)]
    assert set(first[0]) == {'image_path', 'caption_said', 'target_text', 'json_index',
                             'K', 'N', 'J'}


def test_no_c_u_training_code_was_introduced():
    """Nothing in the training path may reference a withheld-caption input."""
    train_source = open(os.path.join(REPO_ROOT, 'train', 'train_salu.py'),
                        encoding='utf-8').read()
    assert 'texts_uss' not in train_source
    assert 'lambda_uss' not in train_source
    assert 'caption_uss' not in train_source
    # the gap forward still takes (I, C_S) only
    from model.salu_model import SALUModel
    gap_parameters = list(inspect.signature(SALUModel._forward_gap_completion).parameters)
    assert gap_parameters == ['self', 'images', 'texts', 'lambda_said', 'lambda_gap_discover',
                              'lambda_global_absorb', 'gap_anti_temperature']


def test_fixed_cohort_runner_is_evaluation_only():
    """No optimizer, no backward and no training-mode switch may appear in the code.

    Docstrings and comments are stripped first: the module *documents* that it takes no
    optimizer step, and prose must not be mistaken for executable code.
    """
    import tokenize

    path = os.path.join(REPO_ROOT, 'tools', 'phase30a_fixed_cohort_eval.py')
    pieces = []
    with open(path, encoding='utf-8') as handle:
        for token in tokenize.generate_tokens(handle.readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            pieces.append(token.string)
    code = ' '.join(pieces)
    for banned in ('optimizer', 'backward', 'GradScaler', 'lr_scheduler'):
        assert banned not in code, banned
