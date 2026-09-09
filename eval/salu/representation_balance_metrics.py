"""Original-dimensional gap metrics and one shared qualitative PCA per checkpoint."""
import numpy as np


def unit_rows(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('finite [N,D] embeddings with N >= 2 required')
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError('zero embedding has undefined cosine distance')
    return x / norms


def modality_metrics(visual, text):
    visual, text = unit_rows(visual), unit_rows(text)
    if visual.shape != text.shape:
        raise ValueError('matching paired embedding shapes required')
    n = len(visual)
    pair = float(np.mean(1 - np.clip(np.sum(visual * text, axis=1), -1, 1)))
    # The sum of all pairwise dot products equals ||sum_i x_i||^2.
    # Subtract the N self similarities, then average ordered off-diagonal pairs.
    def within(x):
        return float(np.clip(1 - (np.square(x.sum(0)).sum() - n) / (n * (n - 1)), 0, 2))
    wv, wt = within(visual), within(text)
    denominator = pair + .5 * (wv + wt)
    return {'pair_gap': pair,
            'l2m': float(np.linalg.norm(visual.mean(0) - text.mean(0))),
            'rmg': pair / denominator if denominator > 1e-12 else None,
            'rmg_status': 'defined' if denominator > 1e-12 else 'undefined_zero_denominator',
            'within_visual': wv, 'within_text': wt,
            'distance': '1 - cosine similarity', 'n': n, 'dimension': visual.shape[1]}


def cyclic_indices(n):
    if n < 2:
        raise ValueError('shuffle control requires N >= 2')
    return (np.arange(n) + 1) % n


def gap_comparison(base, full, said, text, shuffled):
    metrics = {name: modality_metrics(values, text)
               for name, values in [('base', base), ('full', full), ('said', said)]}
    gain = metrics['full']['pair_gap'] - metrics['said']['pair_gap']
    own_gap = metrics['said']['pair_gap']
    shuffle_gap = modality_metrics(shuffled, text)['pair_gap']
    metrics.update({'balancing_gain': gain,
                    'relative_balancing_gain': gain / max(metrics['full']['pair_gap'], 1e-8),
                    'said_own_gap': own_gap, 'said_shuffle_gap': shuffle_gap,
                    'conditioning_gap_margin': shuffle_gap - own_gap})
    return metrics


def joint_pca(base, full, said, text):
    groups = {name: unit_rows(x) for name, x in
              [('base', base), ('full', full), ('said', said), ('text', text)]}
    if len({x.shape for x in groups.values()}) != 1:
        raise ValueError('all four PCA groups must have identical shapes')
    x = np.concatenate(list(groups.values()))
    mean = x.mean(0)
    centered = x - mean
    # Symmetric eigendecomposition is sufficient for the fixed, small 512D space.
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered)
    basis = eigenvectors[:, -2:][:, ::-1]
    for axis in range(2):
        if basis[np.argmax(np.abs(basis[:, axis])), axis] < 0:
            basis[:, axis] *= -1
    projected = {name: (values - mean) @ basis for name, values in groups.items()}
    ellipses, centers = {}, {}
    angles = np.linspace(0, 2 * np.pi, 101)
    circle = np.stack([np.cos(angles), np.sin(angles)])
    for name, points in projected.items():
        center = points.mean(0)
        vals, vecs = np.linalg.eigh(np.cov(points, rowvar=False))
        boundary = center + (vecs @ np.diag(np.sqrt(np.maximum(vals, 0) * 5.991464547)) @ circle).T
        centers[name] = center.tolist()
        ellipses[name] = boundary.tolist()
    all_points = np.concatenate(list(projected.values()) + [np.asarray(x) for x in ellipses.values()])
    low, high = all_points.min(0), all_points.max(0)
    padding = np.maximum((high - low) * .05, 1e-4)
    limits = {'x': [float(low[0] - padding[0]), float(high[0] + padding[0])],
              'y': [float(low[1] - padding[1]), float(high[1] + padding[1])]}
    variance = np.maximum(eigenvalues, 0)
    total = variance.sum()
    return {'fit': 'joint [Z_base; Z_full; Z_said; T]', 'qualitative_only': True,
            'basis': basis.tolist(), 'mean': mean.tolist(),
            'explained_variance_ratio': (variance[-2:][::-1] / total).tolist() if total > 0 else [0., 0.],
            'points': {name: p.tolist() for name, p in projected.items()},
            'centroids': centers, 'ellipses': ellipses,
            'ellipse_definition': '95% Gaussian covariance contour; chi-square(df=2)=5.991464547; not centroid CI',
            'axis_limits': limits,
            'panels': {name: {'basis_id': 'joint', 'axis_limits': limits} for name in ('base', 'full', 'said')},
            'paired_line_indices': list(range(min(30, len(groups['base']))))}
