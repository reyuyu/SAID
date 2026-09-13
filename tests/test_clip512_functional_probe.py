"""Formula and indexing tests for the CLIP-512 functional probe.

CPU-only, no GPU, no checkpoint and no dataset: every case is small synthetic tensors, so a failure
is an algebra or index bug. They cover exactly the identities the probe relies on: both retrieval
directions, the DeltaQ identity, the fixed-negative margin contributions, the I2T common-offset
invariance, the candidate-covariance identity, the crop four-score identity, the single-mask rule,
the view geometry (Normalize exactly once) and the tie/Jaccard/near-zero conventions.
"""
import os
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'tools', 'diag'))

from clip512_functional_probe import (BASE_CAPTIONS, CHANGE_DESCRIPTIONS,  # noqa: E402
                                      COLOUR_NUMBERS, FIXED_SCALE, PARAPHRASES, R_ORDER, R_SUFFIXES,
                                      SEMANTIC_LOCK, VISUAL_CHANGES, box_area,
                                      candidate_covariance, canvas_transform,
                                      common_offset_invariance, grid_control_crop,
                                      intersection_area, jaccard, map_box_to_canvas, overlap_at_k,
                                      ranked_metrics, retained_fraction, square_from_box,
                                      template_for, template_pair_for, tensor_transform,
                                      unit_rows, view_tensor)


def _toy(n_images=6, n_texts=6, dim=16, seed=0):
    generator = torch.Generator().manual_seed(seed)
    images = unit_rows(torch.randn(n_images, dim, generator=generator))
    texts = unit_rows(torch.randn(n_texts, dim, generator=generator))
    return images, texts


# ------------------------------------------------------------------ both directions
def test_i2t_and_t2i_are_different_rankings_not_one_row_direction():
    images, texts = _toy(5, 5, 8, seed=1)
    q = FIXED_SCALE * (images @ texts.t())
    i2t = ranked_metrics(q)
    t2i = ranked_metrics(q.t().contiguous())
    assert i2t['n_queries'] == t2i['n_queries'] == 5
    asymmetric = torch.randn(6, 6, generator=torch.Generator().manual_seed(2))
    assert (ranked_metrics(asymmetric)['per_query_rank']
            != ranked_metrics(asymmetric.t().contiguous())['per_query_rank'])
    # a symmetric matrix is the degenerate case where both directions coincide, and only there
    symmetric = asymmetric + asymmetric.t()
    assert (ranked_metrics(symmetric)['per_query_rank']
            == ranked_metrics(symmetric.t().contiguous())['per_query_rank'])


def test_delta_q_identity():
    images, texts = _toy(4, 4, 8, seed=3)
    delta = unit_rows(texts + 0.05 * torch.randn(4, 8,
                                                 generator=torch.Generator().manual_seed(4))) - texts
    q = FIXED_SCALE * (images @ texts.t())
    q_new = FIXED_SCALE * (images @ (texts + delta).t())
    delta_q = FIXED_SCALE * (images @ delta.t())
    assert float((q_new - q - delta_q).abs().max()) < 1e-4


# ------------------------------------------------------------------ margin contribution at a fixed negative
def test_fixed_negative_margin_contribution_identity_both_directions():
    images, texts = _toy(5, 5, 12, seed=5)
    delta = 0.02 * torch.randn(5, 12, generator=torch.Generator().manual_seed(6))
    delta_q = FIXED_SCALE * (images @ delta.t())
    index, negative = 2, 4
    contribution = FIXED_SCALE * images[index] * (delta[index] - delta[negative])
    actual = float((delta_q[index, index] - delta_q[index, negative]))
    assert abs(float(contribution.sum()) - actual) < 1e-4
    # T2I: query caption j, correct image j, fixed negative image i
    j, i = 1, 3
    contribution_t = FIXED_SCALE * delta[j] * (images[j] - images[i])
    actual_t = float(delta_q[j, j] - delta_q[i, j])
    assert abs(float(contribution_t.sum()) - actual_t) < 1e-4


def test_a_switched_negative_is_not_a_fixed_negative():
    """The identity only holds for one fixed candidate: mixing candidates does not add up."""
    images, texts = _toy(6, 6, 10, seed=7)
    delta = 0.02 * torch.randn(6, 10, generator=torch.Generator().manual_seed(8))
    delta_q = FIXED_SCALE * (images @ delta.t())
    index = 0
    old_negative, new_negative = 1, 5
    fixed = FIXED_SCALE * images[index] * (delta[index] - delta[old_negative])
    mixed = float((delta_q[index, index] - delta_q[index, old_negative])
                  - (delta_q[index, index] - delta_q[index, new_negative]))
    assert abs(float(fixed.sum()) - mixed) > 1e-6      # deliberately different quantities


# ------------------------------------------------------------------ common offset
def test_i2t_common_offset_adds_a_row_constant_and_leaves_ranking_alone():
    images, texts = _toy(8, 8, 12, seed=9)
    delta = 0.03 * torch.randn(8, 12, generator=torch.Generator().manual_seed(10))
    delta_q = FIXED_SCALE * (images @ delta.t())
    report = common_offset_invariance(delta_q, delta, images, 'R1')
    assert report['per_query_rank_identical'] is True
    assert report['argmax_negative_identical'] is True
    assert report['ce_max_abs_change'] < 1e-5
    assert report['entropy_max_abs_change'] < 1e-5
    assert abs(report['row_constant_mean']) > 0.0        # the offset is not a trivial zero


def test_t2i_has_no_common_row_offset():
    """For T2I the offset is per query because the image side differs, so the trick does not apply."""
    images, texts = _toy(6, 6, 10, seed=11)
    delta = 0.05 * torch.randn(6, 10, generator=torch.Generator().manual_seed(12))
    mu = delta.mean(dim=0)
    delta_q = FIXED_SCALE * (images @ delta.t())
    adjusted_i2t = delta_q - FIXED_SCALE * (images @ mu).unsqueeze(1)
    adjusted_t2i = delta_q.t() - FIXED_SCALE * (images @ mu).unsqueeze(1)
    q_transposed = delta_q.t()
    assert float((adjusted_i2t - (delta_q - FIXED_SCALE * (images @ mu).unsqueeze(1))).abs().max()) == 0.0
    assert float((adjusted_t2i - q_transposed).abs().max()) > 0.0


# ------------------------------------------------------------------ candidate covariance
def test_candidate_covariance_equals_score_change_variance():
    images, texts = _toy(12, 5, 9, seed=13)
    delta = 0.04 * torch.randn(5, 9, generator=torch.Generator().manual_seed(14))
    report = candidate_covariance(images, delta)
    assert report['max_abs_diff'] < 1e-4


# ------------------------------------------------------------------ crops and the four-score identity
def test_crop_four_score_identity_and_per_dimension_k():
    images, texts = _toy(4, 4, 10, seed=15)
    delta_q = FIXED_SCALE * (images @ texts.t())
    k_value = float(images[0] @ texts[0]) * FIXED_SCALE - float(images[0] @ texts[1]) * FIXED_SCALE \
        - float(images[1] @ texts[0]) * FIXED_SCALE + float(images[1] @ texts[1]) * FIXED_SCALE
    q_a_a = FIXED_SCALE * float(images[0] @ texts[0])
    q_b_b = FIXED_SCALE * float(images[1] @ texts[1])
    q_a_b = FIXED_SCALE * float(images[0] @ texts[1])
    q_b_a = FIXED_SCALE * float(images[1] @ texts[0])
    assert abs((q_a_a + q_b_b - q_a_b - q_b_a) - k_value) < 1e-3
    dv = images[0] - images[1]
    dt = texts[0] - texts[1]
    k_dim = FIXED_SCALE * dv * dt
    assert abs(float(k_dim.sum()) - k_value) < 1e-3


def test_crop_geometry_and_retention_rules():
    canvas = 224.0
    box_a = (10.0, 10.0, 60.0, 60.0)
    box_b = (150.0, 150.0, 210.0, 210.0)
    crop_a = square_from_box(box_a)
    crop_b = square_from_box(box_b)
    assert box_area(crop_a) > 0 and abs((crop_a[2] - crop_a[0]) - (crop_a[3] - crop_a[1])) < 1e-6
    for crop in (crop_a, crop_b):
        assert crop[0] >= -1e-6 and crop[1] >= -1e-6
        assert crop[2] <= canvas + 1e-6 and crop[3] <= canvas + 1e-6
    assert retained_fraction(crop_a, box_a) >= 0.95
    assert retained_fraction(crop_a, box_b) <= 0.01
    control = grid_control_crop([box_a, box_b])
    assert control is not None
    assert retained_fraction(control, box_a) >= 0.95
    assert retained_fraction(control, box_b) >= 0.95
    assert grid_control_crop([(0.0, 0.0, 224.0, 224.0)]) is None      # cannot keep the whole canvas
    mapped, clipped = map_box_to_canvas((100.0, 50.0, 300.0, 250.0), scale=0.5, offset=(10.0, 5.0))
    assert mapped == (40.0, 20.0, 140.0, 120.0)
    assert clipped == mapped
    assert intersection_area(box_a, box_b) == 0.0


def test_official_view_normalises_exactly_once():
    from PIL import Image
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711))])
    image = Image.fromarray((np.random.default_rng(0).random((300, 400, 3)) * 255)
                            .astype('uint8'))
    reference = preprocess(image)
    canvas, tensor = view_tensor(image, preprocess)
    assert canvas.size == (224, 224)
    assert torch.allclose(reference, tensor, atol=1e-6)
    # the canvas itself is un-normalised (0..1 range is not guaranteed, but Normalize was not applied)
    assert canvas_transform(preprocess)[0](image).size == (224, 224)


# ------------------------------------------------------------------ conventions
def test_true_jaccard_and_overlap_at_k_are_different_things():
    order_a = [1, 2, 3, 4, 5, 6]
    order_b = [1, 2, 9, 10, 11, 12]
    assert overlap_at_k(order_a, order_b, 2) == 2
    assert overlap_at_k(order_a, order_b, 4) == 2
    assert jaccard(order_a, order_b) == pytest.approx(2.0 / 10.0)      # intersection / union
    assert jaccard([], []) == 0.0


def test_tie_rule_is_by_candidate_id_not_by_label_priority():
    scores = torch.zeros(3, 3)
    scores[0, 0] = 1.0
    scores[0, 2] = 1.0        # tie with a HIGHER id: loses the tie break
    scores[1, 1] = 1.0
    scores[1, 0] = 1.0        # tie with a LOWER id: wins the tie break
    scores[2, 2] = 2.0        # a clear winner: rank 1 regardless of ties
    metrics = ranked_metrics(scores)
    assert metrics['per_query_rank'] == [1, 2, 1]
    assert metrics['R@1'] == pytest.approx(2.0 / 3.0)
    assert metrics['tie_pairs'] == 2


def test_near_zero_and_empty_inputs_do_not_produce_nan():
    tiny = torch.full((3, 4), 1e-9)
    normalised = unit_rows(tiny)
    assert torch.isfinite(normalised).all()
    assert float(normalised.norm(dim=1).min()) <= 1.0 + 1e-6
    scores = torch.zeros(3, 3)
    metrics = ranked_metrics(scores)
    assert np.isfinite(metrics['ce']) and np.isfinite(metrics['entropy'])
    assert metrics['tie_pairs'] == 6


def test_templates_and_article_mapping():
    assert template_for('dog') == 'A photo of a dog.'
    assert template_for('elephant') == 'A photo of an elephant.'
    assert template_pair_for('dog', 'elephant') == 'A photo of a dog and an elephant.'
    assert template_pair_for('apple', 'orange') == 'A photo of an apple and an orange.'


# ------------------------------------------------------------------ template audit (the fixed bug)
def test_paraphrases_keep_object_colour_count_and_relation():
    assert len(BASE_CAPTIONS) == len(PARAPHRASES) == len(VISUAL_CHANGES) == len(SEMANTIC_LOCK) == 12
    for index, base in enumerate(BASE_CAPTIONS):
        groups = SEMANTIC_LOCK[index]
        paraphrase = PARAPHRASES[index].lower()
        missing = [group[0] for group in groups
                   if not any(word in paraphrase for word in group)]
        introduced = [word for word in COLOUR_NUMBERS
                      if word in paraphrase and word not in base.lower()]
        assert not missing, (index, missing)
        assert not introduced, (index, introduced)
        # the visual change must drop at least one locked concept, or change the relation/action
        # (caption 2 changes "beside" -> "behind", which keeps every locked noun but is still a
        #  single visual change)
        dropped = [group[0] for group in groups
                   if not any(word in VISUAL_CHANGES[index].lower() for word in group)]
        relation_change = any(word in CHANGE_DESCRIPTIONS[index].lower()
                              for word in ('spatial', 'running', 'walking'))
        assert dropped or relation_change, index
    # the specific bug: the brown horse must stay brown in the paraphrase
    assert 'brown' in PARAPHRASES[8].lower()
    assert 'white' not in PARAPHRASES[8].lower()
    assert VISUAL_CHANGES[8].lower() != BASE_CAPTIONS[8].lower()
    assert len(CHANGE_DESCRIPTIONS) == 12
    assert set(R_SUFFIXES) == set(R_ORDER) == {'R1', 'R2', 'R3', 'R4'}
