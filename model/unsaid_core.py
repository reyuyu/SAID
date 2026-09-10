"""Minimal Unsaid V1 core math -- parameter-free complement branch.

Unsaid reuses **exactly** the Said router's own-caption quantities; it adds no
projection, no module, no prototype bank and no parameter::

    s_i   = q^T k_i                        (Said router's raw cosine scores)
    A^U   = softmax(-s / tau_unsaid)       (anti-routing: the complement patches)
    z_U   = normalize(sum_i A^U_i h_i)     (the same patch features H as Said)

``1 - A_said`` is explicitly **not** used: ``A_said`` is a normalised softmax
distribution, so ``1 - A_said`` is nearly constant (its sum is ``N - 1``, not 1)
and cannot act as a complement selector. The anti-routing softmax is a proper
distribution that puts mass on the patches the caption does *not* match.

Complementary target (both branches stop-gradient), built from the current global
representation and the own-caption Said representation::

    g = sg(normalize(z_G)),  s = sg(normalize(z_S))
    r = g - (g . s) s                      (remove the Said direction from global)
    n = ||r||_2,  valid = n > unsaid_residual_eps
    target = sg(r / n)      (invalid samples never divide by zero)

Loss (valid samples only)::

    L_U = mean_{valid}[1 - cos(z_U, target)]

The gradient flows into ``z_U`` (hence into the Said router's ``q_proj`` /
``k_proj`` and into the visual backbone that produced ``H``); the target branch is
frozen. Attention overlap / entropy / cosine diagnostics are monitoring only and
never enter a loss.
"""
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

DEFAULT_TAU_UNSAID = 0.07
DEFAULT_RESIDUAL_EPS = 1e-4
# Logged only when Unsaid is enabled; ``None`` when ``lambda_unsaid == 0`` so a
# disabled branch can never be mistaken for a measurement.
DIAGNOSTIC_KEYS = (
    'unsaid_valid_ratio',
    'unsaid_residual_norm_mean',
    'unsaid_target_cosine',
    'said_unsaid_cosine',
    'unsaid_attention_entropy',
    'unsaid_effective_patch_count',
    'unsaid_attention_max',
    'said_unsaid_attention_overlap',
    'said_unsaid_attention_jsd',
)


def complement_attention(scores: torch.Tensor, tau_unsaid: float = DEFAULT_TAU_UNSAID) -> torch.Tensor:
    """``softmax(-s / tau_unsaid)`` -- the anti-routing distribution over patches.

    Args:
        scores: [B, N] raw cosine scores ``q^T k_i`` (no temperature applied).
        tau_unsaid: positive temperature; smaller = sharper complement selection.
    """
    tau = float(tau_unsaid)
    if tau <= 0.0:
        raise ValueError('tau_unsaid must be positive, got %r' % (tau_unsaid,))
    if scores.dim() != 2:
        raise ValueError('scores must be [B, N], got %r' % (tuple(scores.shape),))
    # fp32 softmax: the Unsaid branch is new, so it may be computed in fp32 without
    # touching the existing Global / Said numerical paths.
    return F.softmax(-scores.float() / tau, dim=-1)


def complement_representation(attention: torch.Tensor, patch_features: torch.Tensor) -> torch.Tensor:
    """``normalize(sum_i A_i h_i)`` in fp32, keeping the gradient into ``H``."""
    if attention.dim() != 2 or patch_features.dim() != 3:
        raise ValueError('expected attention [B, N] and patches [B, N, D], got %r and %r'
                         % (tuple(attention.shape), tuple(patch_features.shape)))
    if attention.shape[:2] != patch_features.shape[:2]:
        raise ValueError('attention / patch mismatch: %r vs %r'
                         % (tuple(attention.shape), tuple(patch_features.shape)))
    pooled = torch.einsum('bn,bnd->bd', attention.float(), patch_features.float())
    return F.normalize(pooled, dim=-1)


def complement_target(z_global: torch.Tensor, z_said: torch.Tensor,
                      unsaid_residual_eps: float = DEFAULT_RESIDUAL_EPS
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stop-gradient complement of the Said direction inside the global feature.

    Returns:
        target: [B, D] unit-norm residual direction (zeros for invalid samples).
        residual_norm: [B] ``||r||`` before normalisation.
        valid: [B] bool mask ``residual_norm > eps``.
    """
    eps = float(unsaid_residual_eps)
    if eps <= 0.0:
        raise ValueError('unsaid_residual_eps must be positive, got %r' % (unsaid_residual_eps,))
    if z_global.shape != z_said.shape or z_global.dim() != 2:
        raise ValueError('z_global and z_said must share shape [B, D], got %r and %r'
                         % (tuple(z_global.shape), tuple(z_said.shape)))
    g = F.normalize(z_global.detach().float(), dim=-1)
    s = F.normalize(z_said.detach().float(), dim=-1)
    residual = g - (g * s).sum(dim=-1, keepdim=True) * s
    norm = residual.norm(dim=-1)
    valid = norm > eps
    direction = residual / norm.clamp_min(eps).unsqueeze(-1)
    # invalid rows (residual below eps, e.g. z_G == z_S) stay exactly zero: no 0/0
    target = (direction * valid.float().unsqueeze(-1)).detach()
    return target, norm.detach(), valid


def complement_loss(z_unsaid: torch.Tensor, target: torch.Tensor,
                    valid: torch.Tensor) -> torch.Tensor:
    """Masked mean of ``1 - cos(z_U, target)``; finite 0 when nothing is valid."""
    if z_unsaid.shape != target.shape or z_unsaid.dim() != 2:
        raise ValueError('z_unsaid and target must share shape [B, D], got %r and %r'
                         % (tuple(z_unsaid.shape), tuple(target.shape)))
    if valid.shape != z_unsaid.shape[:1]:
        raise ValueError('valid must be [B], got %r' % (tuple(valid.shape),))
    cosine = (F.normalize(z_unsaid.float(), dim=-1) * target).sum(dim=-1)
    weights = valid.float()
    # the all-invalid batch keeps a valid autograd path: 0 / 1 = 0 with zero gradient
    return (1.0 - cosine).mul(weights).sum() / weights.sum().clamp_min(1.0)


def attention_overlap(a_said: torch.Tensor, a_unsaid: torch.Tensor) -> torch.Tensor:
    """``sum_i min(A^S_i, A^U_i)`` per sample: in [0, 1] for two distributions."""
    return torch.minimum(a_said.float(), a_unsaid.float()).sum(dim=-1)


def attention_jsd(a_said: torch.Tensor, a_unsaid: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Jensen-Shannon divergence (nats) between the Said and Unsaid attention."""
    p = a_said.float().clamp_min(eps)
    q = a_unsaid.float().clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    m = 0.5 * (p + q)

    def _kl(a, b):
        return (a * (a.log() - b.log())).sum(dim=-1)

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def attention_entropy(attention: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Shannon entropy (nats) per sample; ``exp(entropy)`` = effective patch count."""
    p = attention.float().clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)


def masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean over the valid rows; 0 when nothing is valid (never NaN)."""
    weights = valid.float()
    return (values.float() * weights).sum() / weights.sum().clamp_min(1.0)
