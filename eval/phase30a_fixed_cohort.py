"""Phase 3.0A.1d: fixed-cohort internal diagnostics and target-independent USR scorers.

Two strictly separated things live here.

**Internal diagnostics** (``common_mode_metrics``, ``internal_gap_metrics``) describe the
representation the *model* builds from ``(I, C_S)`` alone. They never see ``C_U`` and never
enter a loss.

**Retrieval scorers** (``precompute_query_features``, ``scorer_matrix``) implement the
Phase 3 USR protocol ``sharegpt4v1k-usr-v1``. The rule that defines the whole phase is
enforced structurally:

    ``z_U_i`` is computed from ``(I_i, C_S_i)`` *before* any candidate text exists, and the
    score matrix is then the plain inner product ``S[i, j] = z_U_i · t_U_j``.

``precompute_query_features`` therefore takes only images and prefixes; ``scorer_matrix``
takes only the precomputed features and the candidate texts. There is no candidate argument
anywhere on the query side, so candidate-conditioned attention cannot be introduced by
accident, and the score matrix cannot depend on the candidate pool it is scored against.

The withheld suffix ``C_U`` appears only as an evaluation target/candidate.

Patch common mode (Q5): because attention weights sum to 1,

    p_S = sum_p A_S,p h_p = mu + delta_S,   delta_S = sum_p A_S,p (h_p - mu),   mu = mean_p h_p
    p_U = sum_p A_U,p h_p = mu + delta_U

so the raw cosine between the two pools can be high even when the *centred* pools are far
apart: the shared centroid ``mu`` dominates both convex combinations. ``common_mode_ratio``
and ``common_mode_fraction_*`` quantify exactly that.
"""
from typing import Dict, Sequence

import torch
import torch.nn.functional as F

DEFAULT_EPS = 1e-6

# the three Phase 3 scorers; the first is the CLS baseline, the other two are Phase 3
PHASE3_SCORERS = ('global', 'unsaid', 'complete')


# --------------------------------------------------------------------------- #
# Q5: patch common mode
# --------------------------------------------------------------------------- #
def common_mode_metrics(patch_features: torch.Tensor,
                        said_attention: torch.Tensor,
                        unsaid_attention: torch.Tensor,
                        eps: float = DEFAULT_EPS) -> Dict[str, float]:
    """Common-mode decomposition of the two convex poolings (detached, diagnostic only).

    All norms are reported on the *raw* patch features (no per-patch normalisation), so the
    numbers describe the geometry the pooling actually sees.
    """
    if patch_features.dim() != 3:
        raise ValueError('patch_features must be [B, P, D], got %r'
                         % (tuple(patch_features.shape),))
    if said_attention.dim() != 2 or unsaid_attention.dim() != 2:
        raise ValueError('attention must be [B, P]')
    with torch.no_grad():
        patches = patch_features.detach().float()
        a_said = said_attention.detach().float()
        a_unsaid = unsaid_attention.detach().float()
        centroid = patches.mean(dim=1)                                  # [B, D]
        centered = patches - centroid.unsqueeze(1)                      # [B, P, D]
        delta_said = torch.einsum('bp,bpd->bd', a_said, centered)
        delta_unsaid = torch.einsum('bp,bpd->bd', a_unsaid, centered)
        pooled_said = torch.einsum('bp,bpd->bd', a_said, patches)
        pooled_unsaid = torch.einsum('bp,bpd->bd', a_unsaid, patches)

        centroid_norm = centroid.norm(dim=-1)
        mean_patch_norm = patches.norm(dim=-1).mean(dim=-1)
        said_norm = pooled_said.norm(dim=-1).clamp_min(eps)
        unsaid_norm = pooled_unsaid.norm(dim=-1).clamp_min(eps)
        # ``||mu|| / ||p||`` is a ratio of norms, not a fraction of anything: it can legally
        # exceed 1 because ``p`` is a signed combination (``mu + delta``) and the cancellation
        # can make ``||p|| < ||mu||``. Hence the official names are ``*_norm_ratio_*``.
        norm_ratio_said = centroid_norm / said_norm
        norm_ratio_unsaid = centroid_norm / unsaid_norm
        return {
            'patch_centroid_norm': float(centroid_norm.mean()),
            'common_mode_ratio': float((centroid_norm / mean_patch_norm.clamp_min(eps)).mean()),
            'centered_said_norm': float(delta_said.norm(dim=-1).mean()),
            'centered_unsaid_norm': float(delta_unsaid.norm(dim=-1).mean()),
            'centered_pool_cosine': float((F.normalize(delta_said, dim=-1, eps=eps)
                                           * F.normalize(delta_unsaid, dim=-1, eps=eps)
                                           ).sum(dim=-1).mean()),
            'raw_pool_cosine': float((F.normalize(pooled_said, dim=-1, eps=eps)
                                      * F.normalize(pooled_unsaid, dim=-1, eps=eps)
                                      ).sum(dim=-1).mean()),
            'common_mode_norm_ratio_said': float(norm_ratio_said.mean()),
            'common_mode_norm_ratio_unsaid': float(norm_ratio_unsaid.mean()),
            'common_mode_norm_ratio_said_max': float(norm_ratio_said.max()),
            'common_mode_norm_ratio_unsaid_max': float(norm_ratio_unsaid.max()),
            # deprecated aliases: the old names are kept so existing artefacts stay readable,
            # but reports and docs use the norm_ratio names only
            'common_mode_fraction_said': float(norm_ratio_said.mean()),
            'common_mode_fraction_unsaid': float(norm_ratio_unsaid.mean()),
        }


def common_mode_diagnosis(metrics: Dict[str, float], margin: float = 0.02) -> Dict:
    """Read ``common_mode_metrics`` as the Q5 verdict, without over-claiming.

    Supports "the shared patch centroid dominates both convex poolings" only when the raw
    pool cosine is high *and* the centred pool cosine is materially lower.
    """
    raw = float(metrics['raw_pool_cosine'])
    centered = float(metrics['centered_pool_cosine'])
    return {
        'raw_pool_cosine': raw,
        'centered_pool_cosine': centered,
        'drop_raw_minus_centered': raw - centered,
        'centered_is_materially_lower': bool(raw - centered > margin),
        'common_mode_dominates_supported': bool(raw > 0.95 and (raw - centered) > margin),
    }


# --------------------------------------------------------------------------- #
# fixed-cohort internal gap metrics
# --------------------------------------------------------------------------- #
def mean_of(rows: Sequence[Dict[str, float]], key: str) -> float:
    """Mean of a metric over cohort samples (finite values only)."""
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        raise ValueError('no values for key %r' % (key,))
    return float(sum(values) / len(values))


def cohort_mean(rows: Sequence[Dict[str, float]], keys: Sequence[str]) -> Dict[str, float]:
    """Aggregate a list of per-sample metric dicts into one cohort-level dict."""
    return {key: mean_of(rows, key) for key in keys}


# --------------------------------------------------------------------------- #
# Phase 3 USR scorers: target-independent
# --------------------------------------------------------------------------- #
def precompute_query_features(model, images: torch.Tensor, prefix_tokens: torch.Tensor,
                              gap_anti_temperature: float) -> Dict[str, torch.Tensor]:
    """Build every query-side feature from ``(I, C_S)`` only.

    This function has **no candidate argument**. It returns the frozen query features that
    are later scored against whatever candidate pool exists:

    * ``global_feature`` -- ``encode_image(I)`` (the CLS baseline)
    * ``unsaid_feature`` -- ``z_U`` from ``encode_said_unsaid(I, C_S)``
    * ``complete_feature`` -- ``normalize(s_ref + u_new)`` from the gap terms

    ``z_U`` is therefore fixed before any ``U_j`` is seen, and the score matrix built from it
    cannot be candidate-conditioned.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            encoded = model.encode_said_unsaid(
                images, prefix_tokens, gap_anti_temperature=gap_anti_temperature,
                return_details=True)
    finally:
        if was_training:
            model.train()
    complete = F.normalize(encoded['said_feature'] + encoded['u_new'], dim=-1)
    return {
        'global_feature': encoded['global_feature'].detach().float(),
        'unsaid_feature': encoded['unsaid_feature'].detach().float(),
        'complete_feature': complete.detach().float(),
        'said_feature': encoded['said_feature'].detach().float(),
        'u_new': encoded['u_new'].detach().float(),
        'gap_before': encoded['gap_before'].detach().float(),
        'gap_after': encoded['gap_after'].detach().float(),
        'novel_norm': encoded['novel_norm'].detach().float(),
    }


def scorer_matrix(features: torch.Tensor, candidate_text_features: torch.Tensor,
                  logit_scale: float = 1.0) -> torch.Tensor:
    """``S[i, j] = scale * cos(feature_i, t_U_j)`` over the **full** candidate pool.

    ``features`` are precomputed query features; ``candidate_text_features`` are the encoded
    withheld suffixes. Nothing here can depend on which candidate is the target, so the
    candidate pool is always complete.
    """
    if features.dim() != 2 or candidate_text_features.dim() != 2:
        raise ValueError('expected [Q, D] query features and [C, D] candidate features')
    return float(logit_scale) * (F.normalize(features, dim=-1)
                                 @ F.normalize(candidate_text_features, dim=-1).t())


def phase3_scorer_matrices(query_features: Dict[str, torch.Tensor],
                           candidate_text_features: torch.Tensor,
                           logit_scale: float = 1.0) -> Dict[str, torch.Tensor]:
    """The three Phase 3 score matrices, all from the same frozen query features."""
    return {
        'global': scorer_matrix(query_features['global_feature'], candidate_text_features,
                                logit_scale),
        'unsaid': scorer_matrix(query_features['unsaid_feature'], candidate_text_features,
                                logit_scale),
        'complete': scorer_matrix(query_features['complete_feature'], candidate_text_features,
                                  logit_scale),
    }


def targeted_upper_bound_matrix(*args, **kwargs):
    """Placeholder guard: candidate-conditioned scoring is NOT part of this phase.

    Phase 2.9's ``score_unsaid_candidates`` / ``_debiased_pair_scores`` read the candidate
    pool while building the representation, which is exactly the "TARGET-CONDITIONED UPPER
    BOUND" that must never be conflated with the Phase 3 target-independent scorers. This
    helper exists so the constraint is explicit and testable rather than implicit.
    """
    raise NotImplementedError(
        'candidate-conditioned USR scoring is deliberately not implemented in Phase 3.0A.1e; '
        'use the target-independent phase3_scorer_matrices, and label any Phase 2.9 '
        'comparison explicitly as a target-conditioned upper bound')


# --------------------------------------------------------------------------- #
# Phase 3.0A.1e: common-mode semantic decomposition
# --------------------------------------------------------------------------- #
DECOMPOSITION_SCORERS = ('global', 'said_raw', 'unsaid_raw', 'centroid', 'said_centered',
                         'unsaid_centered', 'anti_minus_said', 'complete_existing')

# features that are L2-normalised before scoring (deterministic naming for tests)
DECOMPOSITION_FEATURE_KEYS = ('global', 'said_raw', 'unsaid_raw', 'centroid', 'said_centered',
                              'unsaid_centered', 'anti_minus_said', 'complete_existing')


def decomposition_features(patch_features: torch.Tensor, said_attention: torch.Tensor,
                           unsaid_attention: torch.Tensor, global_feature: torch.Tensor,
                           said_feature: torch.Tensor, complete_feature: torch.Tensor,
                           eps: float = DEFAULT_EPS) -> Dict[str, torch.Tensor]:
    """Centre/deviate/contrast decomposition of the two poolings (all detached).

    With ``mu = mean_p h_p``, ``p_S = sum A_S h``, ``p_U = sum A_U h``:

        delta_S = sum A_S (h - mu)      delta_U = sum A_U (h - mu)
        p_S = mu + delta_S              p_U = mu + delta_U
        contrast = p_U - p_S = delta_U - delta_S

    Returns the eight normalised scorers plus the raw pieces they are built from.
    """
    if patch_features.dim() != 3:
        raise ValueError('patch_features must be [B, P, D], got %r'
                         % (tuple(patch_features.shape),))
    if said_attention.dim() != 2 or unsaid_attention.dim() != 2:
        raise ValueError('attention must be [B, P]')
    with torch.no_grad():
        patches = patch_features.detach().float()
        a_said = said_attention.detach().float()
        a_unsaid = unsaid_attention.detach().float()
        centroid = patches.mean(dim=1)
        centered = patches - centroid.unsqueeze(1)
        delta_said = torch.einsum('bp,bpd->bd', a_said, centered)
        delta_unsaid = torch.einsum('bp,bpd->bd', a_unsaid, centered)
        pooled_said = torch.einsum('bp,bpd->bd', a_said, patches)
        pooled_unsaid = torch.einsum('bp,bpd->bd', a_unsaid, patches)
        contrast = pooled_unsaid - pooled_said
        return {
            'global': F.normalize(global_feature.detach().float(), dim=-1, eps=eps),
            'said_raw': F.normalize(pooled_said, dim=-1, eps=eps),
            'unsaid_raw': F.normalize(pooled_unsaid, dim=-1, eps=eps),
            'centroid': F.normalize(centroid, dim=-1, eps=eps),
            'said_centered': F.normalize(delta_said, dim=-1, eps=eps),
            'unsaid_centered': F.normalize(delta_unsaid, dim=-1, eps=eps),
            'anti_minus_said': F.normalize(contrast, dim=-1, eps=eps),
            'complete_existing': F.normalize(complete_feature.detach().float(), dim=-1, eps=eps),
            # raw pieces, kept for the identity checks and the common-mode metrics
            'centroid_vector': centroid,
            'delta_said_vector': delta_said,
            'delta_unsaid_vector': delta_unsaid,
            'pooled_said_vector': pooled_said,
            'pooled_unsaid_vector': pooled_unsaid,
            'contrast_vector': contrast,
        }


def decomposition_scorer_matrices(features: Dict[str, torch.Tensor],
                                  candidate_text_features: torch.Tensor,
                                  logit_scale: float = 1.0) -> Dict[str, torch.Tensor]:
    """One score matrix per decomposition scorer, all over the full candidate pool."""
    return {name: scorer_matrix(features[name], candidate_text_features, logit_scale)
            for name in DECOMPOSITION_SCORERS}
