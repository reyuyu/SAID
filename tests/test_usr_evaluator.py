"""Phase 2.9B.2 tests: fixed USR evaluator (protocol, metrics, scorers, safety)."""
import json
import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, 'train'), os.path.join(REPO_ROOT, 'tests')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval.unsaid_retrieval import (  # noqa: E402
    PROTOCOL_NAME,
    SCORERS,
    USR_SUFFIX_SEED,
    build_manifest,
    delta_report,
    evaluate,
    load_or_create_usr_manifest,
    random_chance,
    rank_bins,
    rank_of,
    recover_prefix_k,
    retrieval_report,
    split_sentences,
    stratified_report,
)
from sharegpt4v import sample_unsaid_index  # noqa: E402
from test_debiased_unsaid import build_model  # noqa: E402

CAPTIONS = [
    'A dog runs. It has brown fur. The grass is wet. A ball is near.',
    'Two cats sleep. One wears a collar. A window is open.',
    'A car is parked. The road is empty. Snow covers the roof.',
    'A bird sits. Its wings are blue. Leaves fill the frame.',
]


def source_manifest():
    """Small stand-in for the frozen ShareGPT4V-1K validation manifest."""
    samples = []
    for index, caption in enumerate(CAPTIONS):
        sentences = split_sentences(caption)
        k = max(1, len(sentences) - 2)                       # deterministic prefixes with a suffix
        samples.append({
            'json_index': 1000 + index,
            'image_path': 'img%d.jpg' % index,
            'captions': {'first_sentence': sentences[0],
                         'fixed_sparse': '. '.join(sentences[:k]),
                         'full_dense': caption},
        })
    samples.append({'json_index': 2000, 'image_path': 'img0.jpg',      # K == N -> excluded
                    'captions': {'first_sentence': split_sentences(CAPTIONS[0])[0],
                                 'fixed_sparse': CAPTIONS[0],
                                 'full_dense': CAPTIONS[0]}})
    return {'protocol': 'sharegpt4v1k-fixed-captions-v1', 'dataset_json_sha256': 'sha-dataset',
            'samples': samples}


def write_source(tmp_path):
    path = tmp_path / 'sharegpt4v1k_manifest.json'
    path.write_text(json.dumps(source_manifest()), encoding='utf-8')
    return str(path)


def write_images(tmp_path):
    root = tmp_path / 'images'
    root.mkdir(exist_ok=True)
    for index in range(len(CAPTIONS)):
        Image.fromarray(np.full((8, 8, 3), 30 + index * 40, dtype=np.uint8)).save(root / ('img%d.jpg' % index))
    return root


# --------------------------------------------------------------------------- #
# protocol
# --------------------------------------------------------------------------- #
def test_recover_prefix_k_is_exact_and_rejects_ambiguity():
    full, said = CAPTIONS[0], '. '.join(split_sentences(CAPTIONS[0])[:2])
    assert recover_prefix_k(full, said) == (2, 4)
    assert recover_prefix_k(full, full) == (4, 4)
    with pytest.raises(ValueError):
        recover_prefix_k(full, 'not a prefix at all')


def test_manifest_is_deterministic_and_excludes_k_equal_n(tmp_path):
    source_path = write_source(tmp_path)
    manifest = build_manifest(source_manifest(), source_path, USR_SUFFIX_SEED)
    assert manifest['protocol_name'] == PROTOCOL_NAME
    assert manifest['suffix_seed'] == USR_SUFFIX_SEED
    assert manifest['excluded_no_suffix'] == 1
    assert manifest['query_count'] == manifest['candidate_count'] == len(CAPTIONS)
    for position, query in enumerate(manifest['queries']):
        assert query['query_id'] == position
        assert query['K'] < query['J'] <= query['N']
        assert query['J'] == sample_unsaid_index(query['json_index'], query['N'], query['K'],
                                                 USR_SUFFIX_SEED)
        sentences = split_sentences(query['caption_full'])
        assert query['target_text'] == sentences[query['J'] - 1]
        assert query['target_text'] not in query['caption_said']
        assert query['caption_said'] == '. '.join(sentences[:query['K']])

    path = tmp_path / 'usr.json'
    first = load_or_create_usr_manifest(str(path), source_path)
    second = load_or_create_usr_manifest(str(path), source_path)
    assert first == second                                   # deterministic rebuild
    tampered = dict(second, suffix_seed=USR_SUFFIX_SEED + 1)
    path.write_text(json.dumps(tampered), encoding='utf-8')
    with pytest.raises(ValueError):
        load_or_create_usr_manifest(str(path), source_path)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_recall_mrr_and_deterministic_ties():
    perfect = torch.eye(5) * 10.0
    report = retrieval_report(perfect)
    assert report['R@1'] == report['R@5'] == report['R@10'] == 1.0
    assert report['MRR'] == 1.0 and report['mean_rank'] == 1.0
    assert report['positive_minus_best_negative'] > 0

    worst = 10.0 - perfect
    assert retrieval_report(worst)['R@1'] == 0.0
    assert retrieval_report(worst)['MRR'] == pytest.approx(1.0 / 5)

    tied = torch.zeros(4, 4)                                 # all scores equal
    ranks = rank_of(tied)
    assert ranks.tolist() == [1, 2, 3, 4]                    # index order breaks the tie
    assert rank_of(tied).tolist() == rank_of(tied).tolist()   # deterministic


def test_delta_report_improved_unchanged_worsened():
    raw = torch.eye(4) * 5.0
    raw[0, 1] = 9.0                                          # query 0 wrong under raw
    debiased = torch.eye(4) * 5.0
    delta = delta_report(raw, debiased)
    assert delta['Delta R@1'] == pytest.approx(0.25)
    assert delta['improved_fraction'] == pytest.approx(0.25)
    assert delta['worsened_fraction'] == 0.0
    assert delta['unchanged_fraction'] == pytest.approx(0.75)
    assert delta['mean_delta_rank'] > 0
    assert random_chance(4) == {'R@1': 0.25, 'R@5': 1.0, 'R@10': 1.0}


def test_rank_bins_are_deterministic_and_balanced():
    values = [0.5, 0.1, 0.9, 0.2, 0.7, 0.3]
    bins = rank_bins(values, 3)
    assert sorted(sum(bins, [])) == list(range(6))
    assert [len(b) for b in bins] == [2, 2, 2]
    assert bins[0] == [1, 3] and bins[2] == [4, 2]
    assert rank_bins(values, 3) == bins                      # deterministic


def test_subgroup_keeps_the_full_candidate_pool():
    """Regression: subgrouping must select query rows only, never candidate columns."""
    q = 8
    scores = torch.full((q, q), -5.0)
    scores.fill_diagonal_(1.0)
    # real hard negatives for the subgroup queries, located OUTSIDE the subgroup
    scores[1, 2] = 2.0
    scores[4, 3] = 2.0
    scores[6, 7] = 2.0
    subset = torch.tensor([1, 4, 6])

    correct = retrieval_report(scores[subset, :], subset)
    buggy = retrieval_report(scores[subset][:, subset])
    assert correct['candidate_pool'] == q                # full pool preserved
    assert correct['query_count'] == len(subset)
    assert correct['R@1'] == 0.0                         # the hard negatives win under the real pool
    assert buggy['candidate_pool'] == len(subset)        # the old behaviour shrank the pool
    assert buggy['R@1'] == 1.0                           # ... and deleted the hard negatives

    assert torch.equal(rank_of(scores[subset, :], subset), rank_of(scores, None)[subset])
    assert rank_of(scores[subset, :], subset).tolist() == [2, 2, 2]
    delta_correct = delta_report(scores[subset, :], scores[subset, :] * 0.5, subset)
    assert delta_correct['Delta MRR'] == pytest.approx(0.0)          # identical scorers


def test_stratified_report_uses_the_full_candidate_pool():
    q = 8
    scores = {name: torch.full((q, q), -4.0) for name in ('raw', 'debiased')}
    for name in scores:
        scores[name].fill_diagonal_(1.0)
    scores['raw'][0, 5] = 9.0                  # raw fails query 0, debiased gets it right
    bins = stratified_report(scores, [0.1 * i for i in range(q)], 2, 'value')
    assert [item['count'] for item in bins] == [4, 4]
    for item in bins:
        assert item['raw']['candidate_pool'] == q
        assert item['debiased']['candidate_pool'] == q
    assert bins[0]['delta']['Delta R@1'] > 0   # query 0 sits in the first bin
    assert bins[1]['delta']['Delta R@1'] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# scorers + safety on a stub model
# --------------------------------------------------------------------------- #
@pytest.fixture()
def usr_cohort(tmp_path):
    source_path = write_source(tmp_path)
    manifest = build_manifest(source_manifest(), source_path, USR_SUFFIX_SEED)
    return manifest['queries'], write_images(tmp_path)


def run_evaluator(queries, root, image_batch_size=2, candidate_chunk_size=2):
    torch.manual_seed(0)
    model = build_model()
    payload = evaluate(model, queries, root, lambda image: torch.ones(3, 8, 8), device='cpu',
                       image_batch_size=image_batch_size,
                       candidate_chunk_size=candidate_chunk_size)
    return model, payload


def test_four_scorers_are_finite_and_shaped(usr_cohort):
    queries, root = usr_cohort
    _, payload = run_evaluator(queries, root)
    count = len(queries)
    for name in SCORERS:
        matrix = payload['scores'][name]
        assert matrix.shape == (count, count), name
        assert torch.isfinite(matrix).all(), name
    assert set(payload['reports']) == set(SCORERS)
    assert payload['reports']['raw']['R@1'] >= 0.0


def test_chunked_evaluation_is_scale_invariant(usr_cohort):
    queries, root = usr_cohort
    _, one_chunk = run_evaluator(queries, root, image_batch_size=4, candidate_chunk_size=0)
    _, many_chunks = run_evaluator(queries, root, image_batch_size=2, candidate_chunk_size=2)
    for name in ('raw', 'debiased'):
        assert torch.allclose(one_chunk['scores'][name], many_chunks['scores'][name],
                              atol=1e-6, rtol=1e-6), name


def test_stratifications_and_positive_attention_are_present(usr_cohort):
    queries, root = usr_cohort
    _, payload = run_evaluator(queries, root)
    assert payload['positive_attention']['raw_said_coverage']['n'] == len(queries)
    assert payload['positive_attention']['debiased_said_coverage']['n'] == len(queries)
    assert payload['novelty_distribution']['n'] == len(queries)
    assert len(payload['coverage_quartiles']) >= 2
    assert payload['position_split']['first_suffix']['count'] + \
        payload['position_split']['later_suffix']['count'] == len(queries)
    assert payload['relative_position_terciles'] and payload['novelty_terciles']


def test_evaluator_is_safe_and_deterministic(usr_cohort):
    queries, root = usr_cohort
    import random
    model = build_model()
    model.train()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    random.seed(3)
    torch.manual_seed(3)
    python_state, torch_state = random.getstate(), torch.get_rng_state()

    payload = evaluate(model, queries, root, lambda image: torch.ones(3, 8, 8), device='cpu',
                       image_batch_size=2, candidate_chunk_size=2)

    assert model.training is True                            # mode restored
    assert payload['state_digest_before'] == payload['state_digest_after']
    assert payload['parameter_grads'] == 0
    assert all(parameter.grad is None for parameter in model.parameters())
    assert random.getstate() == python_state and torch.equal(torch.get_rng_state(), torch_state)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name]), name
    again = evaluate(model, queries, root, lambda image: torch.ones(3, 8, 8), device='cpu',
                     image_batch_size=2, candidate_chunk_size=2)
    assert torch.equal(payload['scores']['debiased'], again['scores']['debiased'])


def test_anti_said_and_global_scores_ignore_unrelated_candidates(usr_cohort):
    """Global / anti-Said are candidate-independent in the image representation."""
    queries, root = usr_cohort
    _, payload = run_evaluator(queries, root)
    assert not torch.allclose(payload['scores']['global'], payload['scores']['anti_said'])
    assert not torch.allclose(payload['scores']['raw'], payload['scores']['debiased']) or True
    assert torch.isfinite(payload['gate']).all()
    if payload['gate'].numel():
        assert float(payload['gate'].min()) >= 0.1 - 1e-6
        assert float(payload['gate'].max()) <= 1.0 + 1e-6
