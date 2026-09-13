"""Tests for the auxiliary-branch retrieval diagnostic (CPU only, no GPU needed).

The point of these tests is that the diagnostic measures exactly what it claims:
* with an all-open gate the auxiliary scores must equal the native ones (the mathematical identity
  that makes the initialisation case uninformative, and the reason a trained gate is required),
* the blocked score computation must equal a pair-by-pair loop,
* the mask of a column must belong to that column only,
* the metric and paired-outcome arithmetic must follow the repository rank/tie rule,
* a checkpoint without gate tensors must be refused with the explicit explanation, and
* a subset pool must never be presented as the frozen protocol.
"""
import json
import os
import subprocess
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (os.path.join(REPO, 'tools', 'diag'), os.path.join(REPO, 'train'), REPO):
    if path not in sys.path:
        sys.path.insert(0, path)

from model.pgclip import NORM_EPS, native_scores, state_digest                     # noqa: E402
from pgclip_aux_retrieval import (auxiliary_scores, diagonal_positive_columns,     # noqa: E402
                                  naive_auxiliary_scores, paired_outcome,
                                  ranks_with_positive_sets, retrieval_metrics)


def diagonal_ranks(scores):
    """Convenience wrapper for the square/diagonal case used by most of these tests."""
    return ranks_with_positive_sets(scores, diagonal_positive_columns(scores.shape[0],
                                                                     scores.device))

SCRIPT = os.path.join(REPO, 'tools', 'diag', 'pgclip_aux_retrieval.py')


def random_inputs(batch=5, hidden=768, out=512, seed=3, mixed_masks=True):
    generator = torch.Generator().manual_seed(seed)
    h = torch.randn(batch, hidden, generator=generator)
    W = torch.randn(hidden, out, generator=generator) / (hidden ** 0.5)
    t = torch.randn(batch, out, generator=generator)
    t = torch.nn.functional.normalize(t, dim=-1, eps=NORM_EPS)
    if mixed_masks:
        masks = (torch.rand(batch, hidden, generator=generator) > 0.4).float()
    else:
        masks = torch.ones(batch, hidden)
    return h, W, t, masks


def test_all_open_mask_makes_the_two_paths_identical():
    h, W, t, _ = random_inputs()
    masks = torch.ones_like(h)
    native = native_scores(h, W, t)
    auxiliary = auxiliary_scores(h, W, t, masks, image_chunk=2, text_chunk=2)
    assert torch.allclose(native, auxiliary, atol=1e-3)
    native_metrics = retrieval_metrics(native)
    auxiliary_metrics = retrieval_metrics(auxiliary)
    for key in native_metrics:
        assert native_metrics[key] == pytest.approx(auxiliary_metrics[key])
    outcome = paired_outcome(native, auxiliary)
    assert outcome['I2T']['rank_unchanged'] == native.shape[0]
    assert outcome['I2T']['hits_gained_r1'] == 0 and outcome['I2T']['hits_lost_r1'] == 0


def test_blocked_scores_match_the_pair_loop():
    h, W, t, masks = random_inputs(batch=4)
    reference = naive_auxiliary_scores(h, W, t, masks)
    for image_chunk, text_chunk in ((4, 4), (2, 3), (1, 1), (3, 2)):
        blocked = auxiliary_scores(h, W, t, masks, image_chunk=image_chunk,
                                   text_chunk=text_chunk)
        assert torch.allclose(blocked, reference, rtol=1e-5, atol=1e-4), (image_chunk, text_chunk)


def test_only_the_column_text_mask_changes_a_column():
    h, W, t, masks = random_inputs(batch=4)
    scores = auxiliary_scores(h, W, t, masks, image_chunk=2, text_chunk=2)
    flipped = masks.clone()
    flipped[1] = 1.0 - masks[1]
    changed = auxiliary_scores(h, W, t, flipped, image_chunk=4, text_chunk=4)
    assert not torch.allclose(changed[:, 1], scores[:, 1], atol=1e-6)
    assert torch.allclose(changed[:, 0], scores[:, 0], atol=1e-6)
    assert torch.allclose(changed[:, 2:], scores[:, 2:], atol=1e-6)
    # permuting the candidates permutes the columns and leaves the metrics unchanged
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = auxiliary_scores(h, W, t[permutation], masks[permutation], image_chunk=2,
                                text_chunk=2)
    assert torch.allclose(permuted, scores[:, permutation], rtol=1e-5, atol=1e-4)
    # the diagnostic must refuse a mask count that does not match the candidate texts
    with pytest.raises(ValueError):
        auxiliary_scores(h, W, t, masks[:2])
    # and it must refuse a non-fp32 core, exactly like the training path
    with pytest.raises(TypeError):
        auxiliary_scores(h.double(), W, t, masks)


def test_rank_rule_matches_the_repository_convention():
    # query 1: candidate 0 ties the positive but has a SMALLER index, so it is ranked ahead
    scores = torch.tensor([[10.0, 10.0, 5.0],
                           [9.0, 9.0, 2.0],
                           [3.0, 3.0, 8.0]])
    ranks = diagonal_ranks(scores)
    assert ranks.tolist() == [1, 2, 1]
    metrics = retrieval_metrics(scores, rank_ks=(1, 2))
    assert metrics['i2t_r1'] == pytest.approx(2 / 3)
    assert metrics['i2t_r2'] == pytest.approx(1.0)
    # a tie on a LARGER index does not demote the positive (query 0 keeps rank 1)
    better_later_tie = torch.tensor([[5.0, 5.0, 1.0],
                                     [1.0, 2.0, 2.0],
                                     [1.0, 3.0, 4.0]])
    assert diagonal_ranks(better_later_tie).tolist() == [1, 1, 1]


def test_paired_outcome_counts_both_directions():
    native = torch.tensor([[9.0, 1.0], [1.0, 9.0]])
    auxiliary = torch.tensor([[1.0, 9.0], [9.0, 1.0]])       # both queries lose their positive
    outcome = paired_outcome(native, auxiliary, rank_ks=(1,))
    assert outcome['I2T']['hits_lost_r1'] == 2
    assert outcome['I2T']['hits_gained_r1'] == 0
    assert outcome['I2T']['hit_count_change_r1'] == -2
    assert outcome['I2T']['rank_worsened'] == 2


def test_cli_refuses_a_checkpoint_without_gate_tensors(tmp_path):
    """A v1.0 checkpoint must be refused with the explanation, never evaluated silently."""
    checkpoint = tmp_path / 'pgclip_PG_CLIP_V01_step000500.pt'
    torch.save({'objective': 'clip_native_preproj_mask', 'arm': 'PG_CLIP_V01',
                'completed_steps': 500,
                'gate': {'stem': 'MaskNetwork(width=512, layers=1, heads=8)', 'mode': 'hard'},
                'clip': {}, 'config': {}}, checkpoint)
    result = subprocess.run([sys.executable, SCRIPT, '--checkpoint', str(checkpoint),
                             '--out-dir', str(tmp_path / 'out'), '--dataset', 'urban1k'],
                            capture_output=True, text=True, timeout=300, cwd=REPO)
    assert result.returncode != 0
    assert 'carries no gate tensors' in (result.stdout + result.stderr)
    assert not (tmp_path / 'out' / 'pgclip_aux_retrieval.json').exists()


def test_cli_declares_the_pool_and_the_subset(tmp_path):
    """The output must state the pool sizes and whether a diagnostic subset was used."""
    import pgclip_aux_retrieval as module
    source = open(SCRIPT, encoding='utf-8').read()
    assert "'subset': args.limit_images is not None" in source
    assert "'canonical_reference_protocol'" in source
    assert "'not_a_gate_candidate': True" in source
    assert 'frozen evaluation' in source and 'diagnostic only' in source
    # and the module must expose only fp32 score paths plus the read-only digest check
    assert 'new_optimizer_updates' in source
    assert 'must stay read-only' in source
    assert hasattr(module, 'auxiliary_scores') and hasattr(module, 'retrieval_metrics')


def test_multi_positive_rows_follow_the_coco_convention():
    """COCO gives every image five captions: any of them counts, and the best one sets the rank."""
    scores = torch.tensor([[1.0, 9.0, 2.0, 3.0, 4.0, 5.0],
                           [9.0, 1.0, 2.0, 3.0, 4.0, 5.0]])
    positives = torch.tensor([[0, 1, 2], [0, 1, 2]])
    ranks = ranks_with_positive_sets(scores, positives)
    assert ranks.tolist() == [1, 1]
    # a competitor above every positive demotes the row
    competitor = scores.clone()
    competitor[:, 5] = 20.0
    assert ranks_with_positive_sets(competitor, positives).tolist() == [2, 2]
    # columns 0..2 belong to image 0 and columns 3..5 to image 1 (the COCO layout)
    t2i_positives = torch.tensor([[0], [0], [0], [1], [1], [1]])
    metrics = retrieval_metrics(scores, positives, t2i_positives)
    assert metrics['n_images'] == 2 and metrics['n_texts'] == 6
    assert metrics['i2t_r1'] == pytest.approx(1.0)
    # text 0 keeps rank 2 (image 1 scores higher), text 1 rank 1 and text 2 rank 1 (the tie goes to
    # the smaller image index, which IS the positive); texts 3..5 have image 1 as positive and the
    # tying image 0 has a smaller index, so they are demoted to rank 2 -> 2/6
    assert metrics['t2i_r1'] == pytest.approx(2 / 6)
    # a non-square matrix without explicit positives is refused instead of guessing
    with pytest.raises(ValueError):
        retrieval_metrics(scores)
    with pytest.raises(ValueError):
        ranks_with_positive_sets(scores, positives[:1])


def test_score_matrix_is_not_mutated_by_the_metric_helpers():
    h, W, t, masks = random_inputs(batch=3, mixed_masks=False)
    scores = auxiliary_scores(h, W, t, masks, image_chunk=2, text_chunk=2)
    before = scores.clone()
    retrieval_metrics(scores)
    paired_outcome(scores, scores)
    diagonal_ranks(scores)
    assert torch.equal(scores, before)
    _ = state_digest({'a': torch.ones(2)})       # the digest helper is importable on CPU too
    _ = json.dumps({'ok': True})
