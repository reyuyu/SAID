"""Phase 3.0A: target-independent Said-Unsaid Gap Completion (pure math, no parameters).

Concept
-------
Training input is only ``Image I`` + observed/incomplete caption ``C_S``. The full caption
``C_F`` and any unsaid text ``C_U`` are **not** required: the base method must forward and
backward with ``texts_uss=None`` (and does not even receive ``texts_full``).

    q_S = normalize(W_q t_S),  k_p = normalize(W_k h_p),  s^S_p = q_S^T k_p     (existing Said)
    A_U = normalize_p sigmoid(-standardize_p(s^S) / tau_complement)            (soft anti-Said)
    z_U = normalize(sum_p A_U_p h_p)                                           (I, C_S only)

``A_U`` is detached: the gap/absorption objectives must never back-propagate into the Said
router (the router is learned by ``L_said`` alone). ``1 - A_S`` is deliberately *not* used:
a softmax attention is near zero on most patches, so its complement is nearly uniform.

Geometry (why this is not the Phase 2.8 residual)
-------------------------------------------------
The Phase 2.8B study showed that regressing ``z_U`` onto a *signed* vector residual
``normalize(g - z_S)`` runs into non-negative patch-mixture reachability limits. Gap
Completion instead asks: *does the span/composition of Said + Unsaid better explain the
global reference?*::

    g_ref = normalize(g).detach(),  s_ref = normalize(z_S).detach()      (fixed references)
    u     = normalize(z_U),  u_new = u - (u . s_ref) s_ref               (repeat removal)
    gap_before = clamp(1 - g_ref . s_ref, 0, 2)                          (diagnostic reference)
    gap_after  = clamp(1 - g_ref . normalize(s_ref + u_new), 0, 2)
    L_gap_discover  = mean(gap_after)
    closure = (gap_before - gap_after) / (gap_before + eps)              (diagnostic only)

``u_new`` is intentionally **not** re-normalised: if ``z_U`` mostly repeats ``z_S`` then
``u_new ~ 0`` and it must earn no completion credit. ``L_global_absorb`` then teaches the
live CLS embedding to absorb the completion::

    complete_target = normalize(s_ref + u_new.detach())
    L_global_absorb = mean(1 - normalize(g_live) . complete_target)

No loss moves both sides: ``L_gap_discover``'s **direct graph gradient** flows through the
``z_U`` branch (the ``g_ref`` and ``z_S`` reference branches are detached), and
``L_global_absorb``'s direct graph gradient flows through the live ``g`` branch (the
completion target is detached). Because ``g``, ``z_S`` and ``z_U`` all come from the *same*
visual backbone, this is **not** parameter isolation: one optimizer update changes every
visual representation of the next forward pass. Only the direct gradient path of the loss
being differentiated is isolated.

Terminology (Phase 3.0A). ``C_S`` **does** condition the visual complement construction
(``C_S -> said raw scores -> soft anti-Said -> A_U -> z_U``), whereas the unsaid text
``C_U`` never participates in visual representation construction
(``C_U -> A_U``: no, ``C_U -> z_U``: no, ``C_U -> g``: no). This is
**Target-Independent Unsaid Semantic Supervision**, and its base form is
**Said-Conditioned Visual Complement Discovery**. It is *not* "Said-conditioned
complementary retrieval" (there is no candidate-retrieval architecture here), and it must
never be described as "C_S does not condition Unsaid". ``L_gap_discover`` +
``L_global_absorb`` form a stop-gradient bidirectional visual completion / self-distillation
mechanism that **encourages the global image embedding to absorb complementary visual
evidence**; it does not *guarantee* a complete global semantic representation, which has to
be verified experimentally.
"""
from typing import Dict

import torch
import torch.nn.functional as F

DEFAULT_ANTI_TEMPERATURE = 1.0
DEFAULT_EPS = 1e-6


def soft_anti_said_attention(said_scores: torch.Tensor,
                             temperature: float = DEFAULT_ANTI_TEMPERATURE,
                             eps: float = DEFAULT_EPS) -> torch.Tensor:
    """Soft, target-independent anti-Said attention ``A_U`` [B, P] (detached).

    ``standardize`` is per image over the patch dimension, then a sigmoid with negative
    sign assigns more weight to patches the current prefix uses *less*. Weights are
    normalised to sum to 1. The result is detached (no gradient into the Said router).
    """
    if said_scores.dim() != 2:
        raise ValueError('said_scores must be [B, P], got %r' % (tuple(said_scores.shape),))
    if float(temperature) <= 0.0:
        raise ValueError('temperature must be positive, got %r' % (temperature,))
    scores = said_scores.detach().float()
    centred = scores - scores.mean(dim=-1, keepdim=True)
    scale = scores.std(dim=-1, unbiased=False, keepdim=True) + float(eps)
    weights = torch.sigmoid(-centred / scale / float(temperature))
    return (weights / weights.sum(dim=-1, keepdim=True).clamp_min(float(eps))).detach()


def unsaid_feature_from_attention(attention: torch.Tensor,
                                  patch_features: torch.Tensor) -> torch.Tensor:
    """``z_U = normalize(sum_p A_U_p h_p)`` -- depends on ``H`` and ``C_S`` only."""
    if attention.dim() != 2 or patch_features.dim() != 3:
        raise ValueError('expected attention [B, P] and patches [B, P, D]')
    pooled = torch.einsum('bp,bpd->bd', attention.float(), patch_features.float())
    return F.normalize(pooled, dim=-1, eps=DEFAULT_EPS)


def gap_completion_terms(global_feature: torch.Tensor, said_feature: torch.Tensor,
                         unsaid_feature: torch.Tensor, eps: float = DEFAULT_EPS) -> Dict[str, torch.Tensor]:
    """Gap terms; only ``unsaid_feature`` (``z_U``) receives gradient.

    Returns ``gap_before``, ``gap_after``, ``closure``, ``u_new``, ``g_ref``, ``s_ref``,
    ``cos_said_unsaid`` and ``novel_norm`` (``||u_new||``). Fixed references are detached,
    so minimising ``gap_after`` cannot be gamed by moving ``g`` or ``z_S``.
    """
    g_ref = F.normalize(global_feature.detach().float(), dim=-1, eps=eps)
    s_ref = F.normalize(said_feature.detach().float(), dim=-1, eps=eps)
    u = F.normalize(unsaid_feature.float(), dim=-1, eps=eps)
    cos_us = (u * s_ref).sum(dim=-1)
    u_new = u - cos_us.unsqueeze(-1) * s_ref          # deliberately not re-normalised
    gap_before = (1.0 - (g_ref * s_ref).sum(dim=-1)).clamp(0.0, 2.0)
    complete = F.normalize(s_ref + u_new, dim=-1, eps=eps)
    gap_after = (1.0 - (g_ref * complete).sum(dim=-1)).clamp(0.0, 2.0)
    closure = (gap_before - gap_after) / (gap_before + float(eps))
    return {'gap_before': gap_before, 'gap_after': gap_after, 'closure': closure,
            'u_new': u_new, 'g_ref': g_ref, 's_ref': s_ref,
            'cos_said_unsaid': cos_us, 'novel_norm': u_new.norm(dim=-1)}


def gap_discovery_loss(terms: Dict[str, torch.Tensor]) -> torch.Tensor:
    """``L_gap_discover = mean(gap_after)`` (``gap_before`` is diagnostic only)."""
    return terms['gap_after'].mean()


def global_absorption_loss(global_feature: torch.Tensor, said_feature: torch.Tensor,
                           unsaid_feature: torch.Tensor,
                           eps: float = DEFAULT_EPS) -> torch.Tensor:
    """``L_global_absorb = mean(1 - cos(normalize(g_live), normalize(s_ref + u_new.detach())))``.

    Gradient reaches the live global CLS feature only; the completion target is frozen.
    """
    terms = gap_completion_terms(global_feature, said_feature, unsaid_feature, eps=eps)
    target = F.normalize(terms['s_ref'] + terms['u_new'].detach(), dim=-1, eps=eps).detach()
    g_live = F.normalize(global_feature.float(), dim=-1, eps=eps)
    return (1.0 - (g_live * target).sum(dim=-1)).mean()


def gap_diagnostics(terms: Dict[str, torch.Tensor], eps: float = DEFAULT_EPS) -> Dict[str, torch.Tensor]:
    """Detached summary numbers for logging (closure is never a loss term)."""
    with torch.no_grad():
        gap_before = terms['gap_before'].detach().float()
        gap_after = terms['gap_after'].detach().float()
        closure = terms['closure'].detach().float()
        defined = gap_before > 1e-4
        return {
            'gap_before_mean': gap_before.mean(),
            'gap_after_mean': gap_after.mean(),
            'gap_reduction_mean': (gap_before - gap_after).mean(),
            'gap_closure_ratio_mean': closure[defined].mean() if bool(defined.any())
            else torch.zeros((), device=gap_before.device),
            'gap_closure_positive_fraction': (closure[defined] > 0).float().mean()
            if bool(defined.any()) else torch.zeros((), device=gap_before.device),
            'said_unsaid_feature_cosine': terms['cos_said_unsaid'].detach().float().mean(),
            'unsaid_novel_component_norm': terms['novel_norm'].detach().float().mean(),
        }
