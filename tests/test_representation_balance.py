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


def make_dashboard_artifacts(root):
    rng = np.random.default_rng(4)
    base, full, said, text = [rng.normal(size=(8, 6)) for _ in range(4)]
    metrics = gap_comparison(base, full, said, text, full)
    pca = joint_pca(base, full, said, text)
    levels = [{'level': level, 'n': 8, **{k: metrics[k] for k in ('base', 'full', 'said')},
               'balancing_gain': metrics['balancing_gain'], 'mean_used_tokens': 20,
               'truncated_count': 0} for level in DETAIL_LEVELS]
    summary = {'status': 'complete', 'n': 8, 'checkpoints': {}}
    for tag, step in [('initial', 0), ('step100', 100), ('final', 659)]:
        write_json(root / tag / 'gap_metrics.json', metrics)
        write_json(root / tag / 'caption_detail.json', {'levels': levels})
        write_json(root / tag / 'pca.json', pca)
        summary['checkpoints'][tag] = {'step': step, 'epoch': 0, 'training': {},
                                        'metrics': metrics, 'coco': None}
    write_json(root / 'summary.json', summary)
    write_json(root / 'batch_history.json', {'note': '独立 smoke', 'run': 'test',
               'records': [{'step': 9, 'completed_steps': 10, 'epoch': 0,
                            'pair_gap_full': .6, 'pair_gap_said': .7,
                            'balancing_gain': -.1, 'relative_balancing_gain': -1/6}]})


def test_chinese_default_without_heatmaps_and_missing_coco(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    root = Path(__file__).resolve().parents[1]
    make_dashboard_artifacts(tmp_path)
    monkeypatch.setenv('REPRESENTATION_BALANCE_ROOT', str(tmp_path))
    monkeypatch.setenv('SAID_DASHBOARD_ROOT', str(tmp_path / 'nonexistent_heatmaps'))
    app = AppTest.from_file(str(root / 'tools/said_dashboard/app.py')).run(timeout=30)
    assert app.sidebar.radio[0].value == '表征平衡监控'
    assert app.title[0].value == '表征平衡监控'
    for tag in ('initial', 'step100', 'final'):
        app.sidebar.selectbox[0].set_value(tag).run(timeout=30)
        assert not app.exception and not app.error and not app.warning
        assert any(x.value == '当前 checkpoint 尚未运行 COCO Retrieval 评估。' for x in app.info)
    assert len(app.get('vega_lite_chart')) == 8
    app.sidebar.radio[0].set_value('训练状态').run(timeout=30)
    assert not app.exception and not app.error
    assert app.title[0].value == '训练状态'
    assert any('训练 Batch 诊断' in x.value for x in app.caption)


def test_missing_balance_artifact_is_friendly(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv('REPRESENTATION_BALANCE_ROOT', str(tmp_path))
    app = AppTest.from_file(str(root / 'tools/said_dashboard/app.py')).run(timeout=30)
    assert not app.exception and not app.error
    assert any('表征诊断暂不可用' in x.value for x in app.warning)


def test_production_default_legacy_strict_load_and_ablation():
    import inspect
    from model.salu_model import SALUModel
    from tests.test_local_evidence_router import TinyCLIP
    assert inspect.signature(SALUModel).parameters['said_feature_source'].default == 'residual'
    for source in ('residual', 'attention_delta'):
        model = SALUModel(TinyCLIP(), said_feature_source=source)
        # Metadata-free old state dict has no diagnostic parameters or buffers.
        state = model.state_dict()
        restored = SALUModel(TinyCLIP(), said_feature_source=source)
        result = restored.load_state_dict(state, strict=True)
        assert not result.missing_keys and not result.unexpected_keys
        with torch.no_grad():
            global_features, patches = restored.encode_router_input(torch.randn(2, 3, 32, 32))
        assert global_features.shape[0] == patches.shape[0] == 2


def test_dashboard_artifact_only_and_font_fallback():
    root = Path(__file__).resolve().parents[1]
    source = (root / 'tools/said_dashboard/representation_balance_page.py').read_text(encoding='utf-8')
    assert 'import torch' not in source and 'torch.load' not in source
    for font in ('Microsoft YaHei', 'PingFang SC', 'Noto Sans CJK SC', 'Source Han Sans SC', 'Arial Unicode MS', 'sans-serif'):
        assert font in source


def test_pca_rendering_uses_artifact_axes_and_independent_centroid_symbols():
    from tools.said_dashboard.representation_balance_page import pca_chart
    rng = np.random.default_rng(9)
    arrays = [rng.normal(size=(8, 6)) for _ in range(4)]
    pca = joint_pca(*arrays)
    for name in ('base', 'full', 'said'):
        spec = pca_chart(pca, name).to_dict()
        assert spec['resolve']['scale']['shape'] == 'independent'
        for layer in spec['layer']:
            for axis in ('x', 'y'):
                assert layer['encoding'][axis]['scale']['domain'] == pca['axis_limits'][axis]
