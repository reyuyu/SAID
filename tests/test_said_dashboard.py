"""Phase 2.2 tests for the Said attention dashboard data layer (no streamlit, no torch).

Builds a tiny fake artifact tree and checks the loader contract that the
dashboard relies on: manifest/metrics reading, sample + checkpoint selectors,
14x14 attention with sum ~ 1, shared-scale rendering, clear errors for missing
artifacts, and no NaN/Inf leakage.
"""
import json
import os
import sys

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASHBOARD_DIR = os.path.join(REPO_ROOT, 'tools', 'said_dashboard')
if DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, DASHBOARD_DIR)

import data as dash  # noqa: E402

TAGS = ['initial', 'final']
SAMPLES = [0, 1]


def _make_attention(seed, grid=14):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(grid * grid,)) * 3.0
    logits -= logits.max()
    attn = np.exp(logits)
    return (attn / attn.sum()).reshape(grid, grid)


@pytest.fixture()
def artifact_root(tmp_path):
    root = tmp_path / 'salu_dashboard'
    root.mkdir()
    manifest = {
        'num_samples': len(SAMPLES),
        'variants': list(dash.VARIANTS),
        'checkpoints': TAGS,
        'samples': [{'index': i, 'image_id': 'coco/train2017/%012d.jpg' % (i + 1), 'shuffled_with_index': 1 - i}
                    for i in SAMPLES],
        'captions': {
            str(i): {
                'own': 'own caption %d' % i,
                'shuffled': 'shuffled caption %d' % i,
                'short': 'short caption %d' % i,
                'long': 'long caption %d' % i,
                'image_id': 'coco/train2017/%012d.jpg' % (i + 1),
                'shuffled_with_index': 1 - i,
                'shuffled_image_id': 'coco/train2017/%012d.jpg' % (2 - i),
            } for i in SAMPLES
        },
    }
    (root / 'manifest.json').write_text(json.dumps(manifest))

    metrics = {'variants': list(dash.VARIANTS), 'checkpoints': {}}
    for k, tag in enumerate(TAGS):
        tag_dir = root / tag
        tag_dir.mkdir()
        per_sample = {}
        for i in SAMPLES:
            per_sample[str(i)] = {
                'image_id': 'coco/train2017/%012d.jpg' % (i + 1),
                'zs': {v: [1.0 if d == 0 else 0.0 for d in range(4)] for v in dash.VARIANTS},
            }
            Image.fromarray(np.full((224, 224, 3), 128 + i, dtype=np.uint8)).save(
                tag_dir / ('sample%02d_image.png' % i))
            for v in dash.VARIANTS:
                np.save(tag_dir / ('sample%02d_%s.npy' % (i, v)), _make_attention(seed=10 * k + i))
        (tag_dir / 'per_sample_metrics.json').write_text(json.dumps(per_sample))
        metrics['checkpoints'][tag] = {
            'step': 0 if tag == 'initial' else 659,
            'checkpoint': None,
            'said_attention': {'mean_entropy': 5.0 - k, 'mean_effective_patch_count': 150.0 - 10 * k,
                               'mean_attention_max': 0.02 + 0.1 * k},
            'caption_shuffle': {'mean_abs_diff': 0.001, 'mean_js_divergence': 0.01 + 0.01 * k,
                                'mean_zs_cosine': 0.999 - 0.001 * k},
            'route_identification': {'top1_acc': 0.05 * k, 'chance': 0.015625,
                                     'top1_over_chance': 3.2 * k, 'margin': 0.1 * k},
            'evidence_identification': {'top1_acc': 0.4 + 0.1 * k, 'chance': 0.015625,
                                        'top1_over_chance': 25.6 * (k + 1), 'margin': 5.0 * k},
            'precision_noise': {'mean_js_divergence': 0.005, 'mean_abs_diff': 0.0002},
            'caption_conditioning_ratio': (0.01 + 0.01 * k) / 0.005,
        }
    (root / 'metrics.json').write_text(json.dumps(metrics))
    return str(root)


def test_manifest_and_selectors(artifact_root):
    manifest = dash.load_manifest(artifact_root)
    assert dash.checkpoints(manifest) == TAGS
    assert [s['index'] for s in dash.samples(manifest)] == SAMPLES
    assert dash.caption_for(manifest, 0, 'own') == 'own caption 0'
    assert dash.caption_for(manifest, 1, 'shuffled') == 'shuffled caption 1'
    assert dash.sample_meta(manifest, 0)['image_id'].endswith('000000000001.jpg')


def test_attention_shape_and_normalisation(artifact_root):
    for tag in TAGS:
        for i in SAMPLES:
            for variant in dash.VARIANTS:
                attn = dash.load_attention(artifact_root, tag, i, variant)
                assert attn.shape == (14, 14)
                stats = dash.attention_stats(attn)
                assert abs(stats['sum'] - 1.0) < 1e-6
                assert stats['finite'] is True
                assert stats['entropy'] > 0.0


def test_shared_scale_rendering(artifact_root):
    a = dash.load_attention(artifact_root, 'final', 0, 'own')
    b = dash.load_attention(artifact_root, 'final', 0, 'shuffled')
    shared = max(a.max(), b.max())
    img_a = dash.heatmap_rgb(a, scale_max=shared)
    img_b = dash.heatmap_rgb(b, scale_max=shared)
    assert img_a.shape == (224, 224, 3) and img_a.dtype == np.uint8
    # shared scale: a map rendered with its own max must be at least as bright as with the shared max
    img_a_own_scale = dash.heatmap_rgb(a, scale_max=a.max())
    assert img_a_own_scale.astype(int).sum() >= img_a.astype(int).sum()
    overlay = dash.overlay_rgb(dash.load_image(artifact_root, 'final', 0), a, scale_max=shared)
    assert overlay.shape == (224, 224, 3)


def test_difference_map_and_js(artifact_root):
    a = dash.load_attention(artifact_root, 'final', 0, 'own')
    b = dash.load_attention(artifact_root, 'final', 0, 'shuffled')
    diff = dash.difference_map(a, b)
    assert diff.shape == (14, 14)
    assert diff.min() >= 0.0
    jsd = dash.js_divergence(a, b)
    assert 0.0 <= jsd < 1.0
    assert jsd == pytest.approx(0.0, abs=1e-9) if np.allclose(a, b) else True


def test_missing_artifact_raises_clear_error(artifact_root):
    with pytest.raises(dash.ArtifactError) as exc:
        dash.load_attention(artifact_root, 'step999', 0, 'own')
    assert 'missing artifact' in str(exc.value)
    with pytest.raises(dash.ArtifactError):
        dash.load_attention(artifact_root, 'final', 99, 'own')
    with pytest.raises(dash.ArtifactError):
        dash.checkpoint_metrics(dash.load_metrics(artifact_root), 'nope')
    with pytest.raises(ValueError):
        dash.caption_for(dash.load_manifest(artifact_root), 0, 'not_a_variant')


def test_metric_series_is_ordered_and_finite(artifact_root):
    metrics = dash.load_metrics(artifact_root)
    series = dash.metric_series(metrics)
    assert series['checkpoint'] == TAGS
    for key, values in series.items():
        if key == 'checkpoint':
            continue
        assert len(values) == len(TAGS)
        assert all(np.isfinite(values)), '%s has non-finite values' % key
    # attention sharpens: entropy decreases
    assert series['attention_entropy'][-1] < series['attention_entropy'][0]
    assert series['route_top1_acc'][-1] > series['route_top1_acc'][0]


def test_data_layer_has_no_torch_or_streamlit_dependency():
    source = open(os.path.join(DASHBOARD_DIR, 'data.py')).read()
    assert 'import torch' not in source
    assert 'import streamlit' not in source


def test_dashboard_app_does_not_load_checkpoints():
    source = open(os.path.join(DASHBOARD_DIR, 'app.py')).read()
    assert 'torch.load' not in source
    assert 'longclip' not in source
