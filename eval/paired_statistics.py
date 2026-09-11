"""Phase 3.0A.1e: paired statistics for the common-mode semantic decomposition.

Every method is scored on the **same 868 queries** with the **same frozen candidate pool**,
so all comparisons are paired. Independent binomial error bars are not used: they ignore that
the methods share queries and candidates and are therefore wrong for matched comparisons.

* :func:`paired_bootstrap` -- deterministic paired bootstrap over query indices with a fixed
  seed and a caller-chosen replicate count (>= 10000 in the phase runs).
* :func:`mcnemar_exact` -- exact two-sided McNemar test on the R@1 hits (the discordant
  ``n01`` / ``n10`` counts and a binomial p-value).
* :func:`spearman` -- rank correlation with a t-approximation p-value, used to test whether
  internal closure tracks withheld-semantic quality.

Nothing here touches a model or a loss.
"""
import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

DEFAULT_REPLICATES = 10000
DEFAULT_SEED = 20260911


def per_query_hits(scores: torch.Tensor, k: int = 1,
                   labels: Optional[torch.Tensor] = None) -> torch.Tensor:
    """1.0 where the correct candidate is ranked within ``k``, else 0.0, per query row."""
    from eval.unsaid_retrieval import rank_of
    ranks = rank_of(scores, labels)
    return (ranks <= int(k)).float()


def per_query_reciprocal_rank(scores: torch.Tensor,
                              labels: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``1 / rank`` per query row."""
    from eval.unsaid_retrieval import rank_of
    return 1.0 / rank_of(scores, labels).float()


def paired_bootstrap(left: Sequence[float], right: Sequence[float],
                     replicates: int = DEFAULT_REPLICATES, seed: int = DEFAULT_SEED,
                     confidence: float = 0.95) -> Dict[str, float]:
    """Deterministic paired bootstrap of ``mean(left) - mean(right)``.

    The same resampled index set is applied to both vectors, which is what makes it *paired*.
    Returns the observed delta, the percentile confidence interval, the bootstrap standard
    error and the fraction of replicates in which the delta is positive.
    """
    if len(left) != len(right):
        raise ValueError('paired inputs must have equal length, got %d and %d'
                         % (len(left), len(right)))
    if replicates < 1:
        raise ValueError('replicates must be >= 1, got %r' % (replicates,))
    a = np.asarray([float(v) for v in left], dtype=np.float64)
    b = np.asarray([float(v) for v in right], dtype=np.float64)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError('paired inputs must be finite')
    n = a.size
    if n == 0:
        raise ValueError('paired inputs must be non-empty')
    delta = float(a.mean() - b.mean())
    if n == 1:
        return {'delta': delta, 'ci_low': delta, 'ci_high': delta, 'bootstrap_se': 0.0,
                'positive_fraction': 1.0 if delta > 0 else 0.0, 'replicates': int(replicates),
                'query_count': int(n)}
    rng = np.random.default_rng(int(seed))
    # one index matrix for both vectors: the resampling is shared, i.e. paired
    index = rng.integers(0, n, size=(int(replicates), n))
    differences = a[index].mean(axis=1) - b[index].mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    low, high = np.percentile(differences, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return {
        'delta': delta,
        'ci_low': float(low),
        'ci_high': float(high),
        'bootstrap_se': float(differences.std(ddof=1)),
        'positive_fraction': float((differences > 0).mean()),
        'excludes_zero': bool(low > 0.0 or high < 0.0),
        'replicates': int(replicates),
        'confidence': float(confidence),
        'seed': int(seed),
        'query_count': int(n),
    }


def mcnemar_exact(left_hits: Sequence[float], right_hits: Sequence[float]) -> Dict[str, float]:
    """Exact two-sided McNemar test on paired binary outcomes (here: R@1 hits).

    ``n10`` counts queries the left method gets right and the right method misses; ``n01`` the
    reverse. The p-value is the exact two-sided binomial test on ``n10`` out of ``n10 + n01``
    with ``p = 0.5``.
    """
    if len(left_hits) != len(right_hits):
        raise ValueError('paired inputs must have equal length')
    a = np.asarray([1 if float(v) >= 0.5 else 0 for v in left_hits], dtype=np.int64)
    b = np.asarray([1 if float(v) >= 0.5 else 0 for v in right_hits], dtype=np.int64)
    n10 = int(np.sum((a == 1) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    discordant = n10 + n01
    if discordant == 0:
        return {'n10_left_only': n10, 'n01_right_only': n01, 'discordant': 0,
                'p_value_exact_two_sided': 1.0, 'note': 'no discordant pairs'}
    smaller = min(n10, n01)
    tail = sum(math.comb(discordant, i) for i in range(0, smaller + 1)) / float(2 ** discordant)
    return {'n10_left_only': n10, 'n01_right_only': n01, 'discordant': discordant,
            'p_value_exact_two_sided': float(min(1.0, 2.0 * tail))}


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared (the standard Spearman treatment)."""
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    position = 0
    while position < values.size:
        stop = position
        while stop + 1 < values.size and sorted_values[stop + 1] == sorted_values[position]:
            stop += 1
        average = 0.5 * (position + stop) + 1.0
        ranks[order[position:stop + 1]] = average
        position = stop + 1
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> Dict[str, float]:
    """Spearman rank correlation with a two-sided t-approximation p-value."""
    a = np.asarray([float(v) for v in left], dtype=np.float64)
    b = np.asarray([float(v) for v in right], dtype=np.float64)
    if a.size != b.size:
        raise ValueError('inputs must have equal length')
    if a.size < 3:
        raise ValueError('at least 3 paired values are required, got %d' % a.size)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError('inputs must be finite')
    rank_a = _rankdata(a)
    rank_b = _rankdata(b)
    if rank_a.std() == 0.0 or rank_b.std() == 0.0:
        return {'rho': 0.0, 'p_value': 1.0, 'n': int(a.size),
                'note': 'degenerate: one input is constant'}
    rho = float(np.corrcoef(rank_a, rank_b)[0, 1])
    n = int(a.size)
    if abs(rho) >= 1.0:
        return {'rho': rho, 'p_value': 0.0, 'n': n}
    t_stat = rho * math.sqrt((n - 2) / max(1e-12, 1.0 - rho * rho))
    # two-sided p from the Student-t survival function, approximated by the normal for n large
    p_value = math.erfc(abs(t_stat) / math.sqrt(2.0))
    return {'rho': rho, 'p_value': float(min(1.0, p_value)), 'n': n}


def compare_methods(left_scores: torch.Tensor, right_scores: torch.Tensor,
                    replicates: int = DEFAULT_REPLICATES, seed: int = DEFAULT_SEED,
                    labels: Optional[torch.Tensor] = None) -> Dict[str, Dict[str, float]]:
    """Full paired comparison of two score matrices on the same queries and candidate pool."""
    left_rr = per_query_reciprocal_rank(left_scores, labels).tolist()
    right_rr = per_query_reciprocal_rank(right_scores, labels).tolist()
    left_r1 = per_query_hits(left_scores, 1, labels).tolist()
    right_r1 = per_query_hits(right_scores, 1, labels).tolist()
    return {
        'R@1': paired_bootstrap(left_r1, right_r1, replicates=replicates, seed=seed),
        'MRR': paired_bootstrap(left_rr, right_rr, replicates=replicates, seed=seed),
        'mcnemar_R@1': mcnemar_exact(left_r1, right_r1),
    }


def margin_per_query(scores: torch.Tensor, kind: str = 'mean',
                     labels: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Per-query semantic margin: positive minus mean (or best) negative score."""
    if scores.dim() != 2:
        raise ValueError('scores must be 2-D')
    n_query, n_candidate = scores.shape
    if labels is None:
        if n_query != n_candidate:
            raise ValueError('labels are required for rectangular scores')
        labels = torch.arange(n_query)
    labels = labels.long()
    positive = scores.gather(1, labels.unsqueeze(1)).squeeze(1)
    mask = torch.ones_like(scores, dtype=torch.bool)
    mask.scatter_(1, labels.unsqueeze(1), False)
    if n_candidate <= 1:
        return torch.zeros(n_query)
    negatives = scores[mask].view(n_query, n_candidate - 1)
    if kind == 'mean':
        return positive - negatives.mean(dim=1)
    if kind == 'best':
        return positive - negatives.max(dim=1).values
    raise ValueError("kind must be 'mean' or 'best', got %r" % (kind,))
