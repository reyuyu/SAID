"""Saved-artifact spatial sanity checks. Evaluation only; no model imports."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .grounding_metrics import contains, patch_coverage


D4_NAMES = ('identity', 'horizontal_flip', 'vertical_flip', 'transpose',
            'rotate90', 'rotate180', 'rotate270', 'transpose_plus_flip')


def d4_transform(a, name):
    """Eight distinct square symmetries on the final two axes.

    transpose_plus_flip is anti-diagonal reflection (transpose then BOTH flips).
    Transpose plus only one flip would duplicate a 90-degree rotation.
    """
    if name == 'identity':
        return a.copy()
    if name == 'horizontal_flip':
        return np.flip(a, -1)
    if name == 'vertical_flip':
        return np.flip(a, -2)
    if name == 'transpose':
        return np.swapaxes(a, -1, -2)
    if name.startswith('rotate'):
        return np.rot90(a, int(name[6:])//90, axes=(-2, -1))
    if name == 'transpose_plus_flip':
        return np.flip(np.swapaxes(a, -1, -2), (-2, -1))
    raise ValueError('unknown D4 transform: '+name)


def shift_attention(a, delta_row, delta_col):
    """Positive offsets move content down/right; discard boundaries, no wrap."""
    if a.ndim < 2 or abs(delta_row) >= a.shape[-2] or abs(delta_col) >= a.shape[-1]:
        raise ValueError('shift outside grid')
    out = np.zeros_like(a)
    h, w = a.shape[-2:]
    sr = slice(max(0, -delta_row), min(h, h-delta_row))
    sc = slice(max(0, -delta_col), min(w, w-delta_col))
    dr = slice(max(0, delta_row), min(h, h+delta_row))
    dc = slice(max(0, delta_col), min(w, w+delta_col))
    out[..., dr, dc] = a[..., sr, sc]
    total = out.sum(axis=(-2, -1), keepdims=True)
    if np.any(total <= 0):
        raise ValueError('shift discarded all attention mass')
    return out/total


def spatial_sweep(root, output, include_shifts=False):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('source audit incomplete')
    phrases = json.loads((root/'per_phrase.json').read_text())
    keys = sorted(phrases)
    coverage, centres, cache = [], [], {}
    for key in keys:
        boxes = phrases[key]['boxes']
        cache_key = (phrases[key]['image_id'], json.dumps(boxes))
        if cache_key not in cache:
            cache[cache_key] = (patch_coverage(boxes),
                np.array([contains(boxes, (c+.5)*16, (r+.5)*16)
                          for r in range(14) for c in range(14)]))
        cov, points = cache[cache_key]
        coverage.append(cov)
        centres.append(points)
    coverage, centres = np.stack(coverage), np.stack(centres)
    area = coverage.mean(axis=(1, 2))
    records = []
    for variant in manifest['variants']:
        maps = np.stack([np.load(root/'attention'/variant/(key+'.npy')) for key in keys]).astype(np.float64)
        if maps.shape != coverage.shape or not np.isfinite(maps).all() or np.any(maps < 0):
            raise ValueError('invalid saved maps')
        np.testing.assert_allclose(maps.sum((1,2)), 1, atol=1e-6, rtol=0)
        transforms = [(name, d4_transform(maps, name)) for name in D4_NAMES]
        if include_shifts:
            transforms += [(f'shift_{dr:+d}_{dc:+d}', shift_attention(maps, dr, dc))
                           for dr in range(-2,3) for dc in range(-2,3)]
        base_point = centres[np.arange(len(keys)), maps.reshape(len(keys), -1).argmax(1)]
        base_mass = np.sum(maps*coverage, axis=(1,2))
        for name, transformed in transforms:
            peak = transformed.reshape(len(keys), -1).argmax(1)
            point = centres[np.arange(len(keys)), peak]
            mass = np.sum(transformed*coverage, axis=(1,2))
            row = {'model': variant, 'transform': name, 'phrases': len(keys),
                   'pointing': float(point.mean()), 'gt_mass': float(mass.mean()),
                   'gt_area_fraction': float(area.mean()), 'mass_gain': float((mass-area).mean()),
                   'pointing_delta': float(point.mean()-base_point.mean()),
                   'mass_delta': float((mass-base_mass).mean()),
                   'changed_to_correct': int(np.sum(point & ~base_point)),
                   'changed_to_wrong': int(np.sum(~point & base_point))}
            records.append(row)
        print(variant, 'complete', flush=True)
    result = {'source': root.name, 'source_code_commit': manifest['code_commit'],
              'images': manifest['num_images'], 'phrases': len(keys),
              'd4_convention': 'rotations counterclockwise; transpose_plus_flip is anti-diagonal reflection',
              'shift_convention': 'positive down/right; no wrap; surviving mass renormalized',
              'records': records}
    (output/'spatial_sweep.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with (output/'spatial_sweep.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print('SPATIAL_SWEEP_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='outputs/semantic_grounding')
    parser.add_argument('--output', default='outputs/grounding_sanity_probe')
    parser.add_argument('--include_shifts', action='store_true')
    args = parser.parse_args()
    spatial_sweep(args.root, args.output, args.include_shifts)
