"""Geometry and metrics for evaluation only; no training/model dependencies.

Boxes use continuous, zero-based, half-open xyxy coordinates after CLIP's
resize/center crop. Overlapping boxes are an exact rectangle union, never a
sum of overlaps and never a bounding rectangle around all instances.
"""
import math

import numpy as np


def boxes_array(boxes):
    b = np.asarray(boxes, dtype=np.float64)
    if b.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    if b.ndim != 2 or b.shape[1] != 4 or not np.isfinite(b).all():
        raise ValueError('boxes must be finite [K,4] xyxy coordinates')
    if np.any(b[:, 2:] <= b[:, :2]):
        raise ValueError('boxes must have positive width and height')
    return b


def clip_geometry(width, height, size=224):
    """Match torchvision Resize(int) and CenterCrop rounding exactly."""
    if min(width, height, size) <= 0:
        raise ValueError('image dimensions and crop size must be positive')
    if width <= height:
        rw, rh = size, int(size * height / width)
    else:
        rw, rh = int(size * width / height), size
    return {'original_size': [width, height], 'resized_size': [rw, rh],
            'crop_xy': [int(round((rw-size)/2)), int(round((rh-size)/2))],
            'crop_size': size}


def transform_boxes(boxes, geometry):
    b = boxes_array(boxes).copy()
    w, h = geometry['original_size']
    rw, rh = geometry['resized_size']
    left, top = geometry['crop_xy']
    b *= np.array([rw/w, rh/h, rw/w, rh/h])
    b -= np.array([left, top, left, top])
    b = b.clip(0, geometry['crop_size'])
    return b[np.all(b[:, 2:] > b[:, :2], axis=1)]


def union_area(boxes):
    b = boxes_array(boxes)
    if not len(b):
        return 0.0
    xs = np.unique(b[:, [0, 2]])
    total = 0.0
    for x0, x1 in zip(xs[:-1], xs[1:]):
        active = b[(b[:, 0] < x1) & (b[:, 2] > x0)]
        intervals = sorted((row[1], row[3]) for row in active)
        covered, end = 0.0, -math.inf
        for y0, y1 in intervals:
            covered += max(0.0, y1-max(y0, end))
            end = max(end, y1)
        total += (x1-x0)*covered
    return float(total)


def intersection_boxes(a, b):
    a, b = boxes_array(a), boxes_array(b)
    if not len(a) or not len(b):
        return np.empty((0, 4))
    lo = np.maximum(a[:, None, :2], b[None, :, :2]).reshape(-1, 2)
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:]).reshape(-1, 2)
    valid = np.all(hi > lo, axis=1)
    return np.concatenate([lo[valid], hi[valid]], axis=1)


def union_iou(a, b):
    intersection = union_area(intersection_boxes(a, b))
    return intersection / max(union_area(a)+union_area(b)-intersection, 1e-12)


def patch_coverage(boxes, grid=14, size=224):
    if grid <= 0 or size <= 0:
        raise ValueError('grid and size must be positive')
    b = boxes_array(boxes)
    side = size/grid
    w = np.zeros((grid, grid), dtype=np.float64)
    for row in range(grid):
        for col in range(grid):
            cell = [[col*side, row*side, (col+1)*side, (row+1)*side]]
            w[row, col] = union_area(intersection_boxes(b, cell))/(side*side)
    return w


def contains(boxes, x, y):
    b = boxes_array(boxes)
    return bool(np.any((b[:, 0] <= x) & (x < b[:, 2]) &
                       (b[:, 1] <= y) & (y < b[:, 3])))


def checked_attention(attention):
    a = np.asarray(attention, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError('attention must be a square patch grid')
    if not np.isfinite(a).all() or a.min() < 0 or abs(a.sum()-1) > 1e-4:
        raise ValueError('attention must be finite, nonnegative and sum to one')
    return a


def phrase_metrics(attention, target_boxes, other_boxes, size=224, iou_threshold=.1,
                   coverage=None, other_coverage=None):
    a = checked_attention(attention)
    if not len(boxes_array(target_boxes)):
        raise ValueError('target has no visible boxes')
    w = patch_coverage(target_boxes, len(a), size) if coverage is None else coverage
    row, col = np.unravel_index(a.argmax(), a.shape)
    x, y = (col+.5)*size/len(a), (row+.5)*size/len(a)
    pointing = contains(target_boxes, x, y)
    area = float(np.mean(w))
    mass = float(np.sum(a*w))
    distractors = [key for key, boxes in other_boxes.items()
                   if len(boxes) and union_iou(target_boxes, boxes) < iou_threshold]
    masses = {key: float(np.sum(a*(other_coverage[key] if other_coverage is not None
                                 else patch_coverage(other_boxes[key], len(a), size))))
              for key in distractors}
    maximum = max(masses.values()) if masses else None
    location = 'target' if pointing else (
        'other_object' if any(contains(b, x, y) for b in other_boxes.values()) else 'background')
    return {'pointing_correct': pointing, 'gt_mass': mass, 'gt_area_fraction': area,
            'mass_gain': mass-area, 'mass_lift': mass/max(area, 1e-12),
            'distractor_mass': maximum, 'distractor_count': len(masses),
            'localization_margin': mass-maximum if maximum is not None else None,
            'target_gt_distractor': mass > maximum if maximum is not None else None,
            'peak_row': int(row), 'peak_col': int(col), 'peak_xy': [x, y],
            'peak_location': location}


def phrase_switch(attention_a, attention_b, coverage_a, coverage_b):
    a, b = checked_attention(attention_a), checked_attention(attention_b)
    aa, ab = float(np.sum(a*coverage_a)), float(np.sum(a*coverage_b))
    bb, ba = float(np.sum(b*coverage_b)), float(np.sum(b*coverage_a))
    p, q = a.clip(1e-12), b.clip(1e-12)
    mid = .5*(p+q)
    return {'M_aa': aa, 'M_ab': ab, 'M_bb': bb, 'M_ba': ba,
            'switch_margin': .5*((aa-ab)+(bb-ba)),
            'both_prefer_target': aa > ab and bb > ba,
            'both_pointing_correct': None,
            'jsd': float(.5*np.sum(p*np.log(p/mid)+q*np.log(q/mid)))}


def distribution(values):
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    if not len(v):
        return {'n': 0, 'mean': None, 'median': None, 'p25': None, 'p75': None}
    if not np.isfinite(v).all():
        raise ValueError('nonfinite metric')
    return {'n': len(v), 'mean': float(v.mean()), 'median': float(np.median(v)),
            'p25': float(np.percentile(v, 25)), 'p75': float(np.percentile(v, 75))}


def summarize(records, pairs):
    out = {'phrases': len(records)}
    for key in ['pointing_correct', 'gt_mass', 'gt_area_fraction', 'mass_gain',
                'mass_lift', 'target_gt_distractor', 'localization_margin']:
        out[key] = distribution([r[key] for r in records])
    out['peak_location'] = {key: sum(r['peak_location'] == key for r in records)
                            for key in ['target', 'other_object', 'background']}
    out['switch_margin'] = distribution([p['switch_margin'] for p in pairs])
    out['switch_positive_fraction'] = distribution([p['switch_margin'] > 0 for p in pairs])
    out['switch_both_prefer_target'] = distribution([p['both_prefer_target'] for p in pairs])
    out['switch_both_pointing_correct'] = distribution([p['both_pointing_correct'] for p in pairs])
    return out
