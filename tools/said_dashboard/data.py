"""Pure artifact loaders for the Said attention dashboard.

No ``streamlit`` and no ``torch`` here: the dashboard only reads precomputed
artifacts, so it can start instantly and be unit-tested without a checkpoint.
"""
import json
import os

import numpy as np
from PIL import Image

VARIANTS = ('own', 'shuffled', 'short', 'long')
GRID = 14


class ArtifactError(FileNotFoundError):
    """Raised when a required dashboard artifact is missing."""


def _require(path):
    if not os.path.exists(path):
        raise ArtifactError('missing artifact: %s' % path)
    return path


def load_manifest(root):
    with open(_require(os.path.join(root, 'manifest.json'))) as fp:
        return json.load(fp)


def load_metrics(root):
    with open(_require(os.path.join(root, 'metrics.json'))) as fp:
        return json.load(fp)


def load_sample_metrics(root, tag):
    """Per-sample metrics / z_s vectors for one checkpoint."""
    path = os.path.join(root, tag, 'per_sample_metrics.json')
    with open(_require(path)) as fp:
        return json.load(fp)


def checkpoints(manifest):
    tags = list(manifest.get('checkpoints', []))
    if not tags:
        raise ArtifactError('manifest.json has no checkpoints')
    return tags


def samples(manifest):
    entries = list(manifest.get('samples', []))
    if not entries:
        raise ArtifactError('manifest.json has no samples')
    return entries


def caption_for(manifest, sample_index, variant):
    if variant not in VARIANTS:
        raise ValueError('unknown caption variant %r (expected one of %r)' % (variant, VARIANTS))
    captions = manifest.get('captions', {})
    key = str(sample_index)
    if key not in captions:
        raise ArtifactError('manifest has no captions for sample %s' % key)
    entry = captions[key]
    if variant not in entry:
        raise ArtifactError('manifest caption entry for sample %s has no %r variant' % (key, variant))
    return entry[variant]


def sample_meta(manifest, sample_index):
    captions = manifest.get('captions', {})
    key = str(sample_index)
    if key not in captions:
        raise ArtifactError('manifest has no entry for sample %s' % key)
    return captions[key]


def load_attention(root, tag, sample_index, variant, grid=GRID, require_normalised=True):
    path = os.path.join(root, tag, 'sample%02d_%s.npy' % (sample_index, variant))
    _require(path)
    attn = np.load(path).astype(np.float64)
    if attn.shape != (grid, grid):
        raise ValueError('attention at %s has shape %r, expected (%d, %d)' % (path, attn.shape, grid, grid))
    if not np.isfinite(attn).all():
        raise ValueError('attention at %s contains NaN/Inf' % path)
    total = float(attn.sum())
    if require_normalised and abs(total - 1.0) > 1e-3:
        raise ValueError('attention at %s does not sum to 1 (sum=%.6f)' % (path, total))
    return attn


def load_image(root, tag, sample_index):
    path = os.path.join(root, tag, 'sample%02d_image.png' % sample_index)
    _require(path)
    return Image.open(path).convert('RGB')


def attention_stats(attn):
    flat = np.asarray(attn, dtype=np.float64).reshape(-1)
    entropy = float(-(np.clip(flat, 1e-12, None) * np.log(np.clip(flat, 1e-12, None))).sum())
    return {
        'sum': float(flat.sum()),
        'entropy': entropy,
        'effective_patch_count': float(np.exp(entropy)),
        'max': float(flat.max()),
        'min': float(flat.min()),
        'finite': bool(np.isfinite(flat).all()),
    }


def colorize(values01):
    """Blue -> green -> red colour map for a [0, 1] array (no matplotlib)."""
    h = np.clip(np.asarray(values01, dtype=np.float64), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * h - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * h - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * h - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def upsample(values01, size=224):
    image = Image.fromarray((np.clip(np.asarray(values01, dtype=np.float64), 0, 1) * 255).astype(np.uint8))
    return np.asarray(image.resize((size, size), Image.BILINEAR), dtype=np.float64) / 255.0


def heatmap_rgb(attn, scale_max=None, size=224):
    """Render an attention map with a SHARED colour scale (no per-tile min-max)."""
    attn = np.asarray(attn, dtype=np.float64)
    if scale_max is None:
        scale_max = float(attn.max())
    norm = np.clip(attn / max(scale_max, 1e-12), 0.0, 1.0)
    return colorize(upsample(norm, size))


def overlay_rgb(image, attn, scale_max=None, alpha=0.5, size=224):
    base = np.asarray(image.convert('RGB').resize((size, size), Image.BILINEAR), dtype=np.float64)
    heat = heatmap_rgb(attn, scale_max=scale_max, size=size).astype(np.float64)
    return np.clip(alpha * base + (1.0 - alpha) * heat, 0, 255).astype(np.uint8)


def difference_map(attn_a, attn_b):
    return np.abs(np.asarray(attn_a, dtype=np.float64) - np.asarray(attn_b, dtype=np.float64))


def js_divergence(p, q, eps=1e-12):
    p = np.clip(np.asarray(p, dtype=np.float64).reshape(-1), eps, None)
    q = np.clip(np.asarray(q, dtype=np.float64).reshape(-1), eps, None)
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def checkpoint_metrics(metrics, tag):
    checkpoints = metrics.get('checkpoints', {})
    if tag not in checkpoints:
        raise ArtifactError('metrics.json has no entry for checkpoint %r' % tag)
    return checkpoints[tag]


def metric_series(metrics):
    """Ordered series for the dashboard charts."""
    tags = list(metrics.get('checkpoints', {}).keys())
    series = {
        'checkpoint': tags,
        'attention_entropy': [],
        'effective_patch_count': [],
        'caption_shuffle_jsd': [],
        'zs_cosine': [],
        'route_top1_acc': [],
        'route_margin': [],
        'evidence_top1_acc': [],
        'precision_noise_jsd': [],
        'conditioning_ratio': [],
    }
    for tag in tags:
        entry = metrics['checkpoints'][tag]
        series['attention_entropy'].append(entry['said_attention']['mean_entropy'])
        series['effective_patch_count'].append(entry['said_attention']['mean_effective_patch_count'])
        series['caption_shuffle_jsd'].append(entry['caption_shuffle']['mean_js_divergence'])
        series['zs_cosine'].append(entry['caption_shuffle']['mean_zs_cosine'])
        series['route_top1_acc'].append(entry['route_identification']['top1_acc'])
        series['route_margin'].append(entry['route_identification']['margin'])
        series['evidence_top1_acc'].append(entry['evidence_identification']['top1_acc'])
        series['precision_noise_jsd'].append(entry['precision_noise']['mean_js_divergence'])
        series['conditioning_ratio'].append(entry['caption_conditioning_ratio'])
    return series
