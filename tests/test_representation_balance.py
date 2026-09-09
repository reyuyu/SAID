"""Analytical and integration checks for diagnostic-only representation balance."""
import numpy as np
import pytest
import torch
import json
import random
from pathlib import Path

from model.representation_metrics import batch_representation_gaps
from eval.salu.representation_probe import (
    DETAIL_LEVELS, build_manifest, caption_ladder, load_or_create_manifest, write_json)
from eval.salu.representation_balance_metrics import (
    cyclic_indices, gap_comparison, joint_pca, modality_metrics)


def test_detached_batch_gaps_and_gain():
    full = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
    text = full.flip(0)
    result = batch_representation_gaps(full, text, text)
    assert result['pair_gap_full'].item() == pytest.approx(1.)
    assert result['pair_gap_said'].item() == pytest.approx(0.)
    assert result['balancing_gain'].item() == pytest.approx(1.)
    assert result['relative_balancing_gain'].item() == pytest.approx(1.)
    assert all(not x.requires_grad and x.grad_fn is None for x in result.values())
    assert all(torch.isfinite(x) for x in batch_representation_gaps(full, full, full).values())


def test_manual_pair_l2m_rmg():
    visual = np.eye(2)
    text = visual[::-1]
    result = modality_metrics(visual, text)
    assert result['pair_gap'] == pytest.approx(1.)
    assert result['l2m'] == pytest.approx(0.)
    assert result['within_visual'] == pytest.approx(1.)
    assert result['within_text'] == pytest.approx(1.)
    assert result['rmg'] == pytest.approx(.5)
    # Unequal centroids: visual=e1,e1 and text=e2,e2. P=1, W=0.
    result = modality_metrics([[1, 0], [1, 0]], [[0, 1], [0, 1]])
    assert result['l2m'] == pytest.approx(np.sqrt(2))
    assert result['rmg'] == pytest.approx(1.)
    assert modality_metrics(visual, visual)['rmg'] == pytest.approx(0.)
    assert modality_metrics([[1, 0], [1, 0]], [[1, 0], [1, 0]])['rmg'] is None
    with pytest.raises(ValueError):
        modality_metrics([[0, 0], [1, 0]], visual)
    comparison = gap_comparison(visual, visual, text, text, visual)
    assert comparison['balancing_gain'] == pytest.approx(1.)
    assert comparison['conditioning_gap_margin'] == pytest.approx(1.)


def test_cyclic_control_has_no_fixed_points():
    np.testing.assert_array_equal(cyclic_indices(5), [1, 2, 3, 4, 0])
    with pytest.raises(ValueError):
        cyclic_indices(1)


def test_detail_ceil_order_and_deduplication():
    ladder = caption_ladder('一. 二. 三. 四. 五')
    assert [ladder['variants'][ladder['levels'][k]]['sentence_count'] for k in DETAIL_LEVELS] == [1, 2, 3, 4, 5]
    assert ladder['variants'][-1]['caption'] == '一. 二. 三. 四. 五'
    short = caption_ladder('one. two')
    assert len(short['variants']) == 2
    assert [short['levels'][k] for k in DETAIL_LEVELS] == [0, 0, 0, 1, 1]


def test_manifest_immutable_unicode_and_rng_isolation(tmp_path):
    records = [{'image': name, 'conversations': [{}, {'value': '第一句. 第二句. 第三句'}]}
               for name in ['a.jpg', 'a.jpg', 'b.jpg', 'c.jpg']]
    source = tmp_path / 'source.json'
    path = tmp_path / 'manifest.json'
    write_json(source, records)
    state = random.getstate()
    first = load_or_create_manifest(path, source, n=3)
    raw = path.read_bytes()
    assert load_or_create_manifest(path, source, n=3) == first
    assert path.read_bytes() == raw
    assert random.getstate() == state
    assert [s['dataset_index'] for s in first['samples']] == [0, 2, 3]
    assert '固定表征' in raw.decode('utf-8') and b'\\u56fa' not in raw
    with pytest.raises(ValueError, match='refusing'):
        load_or_create_manifest(path, source, n=2)


def test_diagnostic_variants_cannot_enter_training_path():
    root = Path(__file__).resolve().parents[1]
    for path in (root / 'train').glob('*.py'):
        source = path.read_text(encoding='utf-8')
        assert 'representation_probe' not in source
        assert 'caption_ladder' not in source
    # Production caption sampler is still the existing randomly selected prefix.
    source = (root / 'train/sharegpt4v.py').read_text(encoding='utf-8')
    assert 'random.randint(1,' in source


def test_joint_basis_shared_axes_and_original_dimension():
    rng = np.random.default_rng(6)
    arrays = [rng.normal(size=(36, 8)) for _ in range(4)]
    result = joint_pca(*arrays)
    mean, basis = np.array(result['mean']), np.array(result['basis'])
    for name, array in zip(('base', 'full', 'said', 'text'), arrays):
        normalized = array / np.linalg.norm(array, axis=1, keepdims=True)
        np.testing.assert_allclose(result['points'][name], (normalized - mean) @ basis)
    assert all(p['axis_limits'] == result['axis_limits'] and p['basis_id'] == 'joint'
               for p in result['panels'].values())
    assert result['paired_line_indices'] == list(range(30))
    assert modality_metrics(arrays[0], arrays[3])['dimension'] == 8
    assert joint_pca(*arrays) == result
