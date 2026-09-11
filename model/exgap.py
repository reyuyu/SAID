"""SAID-ExGAP v1 -- Explanatory-Gap Guided Masked Representation Learning.

This module is **not** gap completion, not reconstruction, and not "Said + Unsaid rebuilds
Global". It implements exactly two things:

1. **Said learning** -- identify the visual evidence the current caption ``C`` selects, via a
   caption-conditioned sigmoid relevance ``r^S_p = sigmoid((q_hat . k_hat_p) / tau_said)`` (the
   router's own caption/evidence logit, read through a sigmoid) and the relevance-normalised
   pooling ``z_S = Norm(sum_p r^S_p h_p / sum_q r^S_q)``.
2. **ExGAP learning** -- when the selective Said representation explains the current caption
   *better than* the whole image does, the representation built from the patches the Said
   selection did **not** claim must stop redundantly re-encoding that same caption.

Everything (Global, Said, Masked-Unsaid) is read from the **same** patch set ``H``; there is no
``CLS -> global`` / ``patch -> said`` dual space.

Math implemented here (see ``docs/said_exgap/`` for the report):

    p_G      = mean_p h_p                                (global_pool='mean')
    A^G_p    = softmax_p ( (q_G W_Q)(h_p W_K)^T / sqrt(d) )   (global_pool='attention')
    z_G      = Norm(sum_p A^G_p h_p)
    ell^S_p  = (q_hat . k_hat_p) / tau_said,  q_hat = Norm(t W_Q), k_hat_p = Norm(h_p W_K)
    r^S_p    = sigmoid(ell^S_p)
    A^S_p    = r^S_p / (sum_q r^S_q + eps)
    z_S      = Norm(sum_p A^S_p h_p)
    M_p      = 1(r^S_p <= tau_M)                         (detached)
    z_U      = Norm(sum_p M_p h_p / (sum_p M_p + eps))    (mean mode)
    A^U_p    = M_p A^G_p / (sum_q M_q A^G_q + eps)         (attention mode)
    z_U      = Norm(sum_p A^U_p h_p)                      (attention mode)
    S_gc     = cos(z_G, t),  S_sc = cos(z_S, t),  S_uc = cos(z_U, t)
    G_exp    = sg( clamp( [S_sc - S_gc]_+ / (1 - S_gc + eps) ) )
    L_ExGAP  = mean_valid( G_exp * tau * log(1 + exp((S_uc - sg(S_gc)) / tau)) )

``G_exp`` is a *caption-incompleteness / selective explanatory gain proxy*, NOT a percentage of
unsaid semantics.

Gradient isolation (enforced and unit-tested): inside ``L_ExGAP`` the gap weight
``G_exp``, the global reference ``S_gc``, the text embedding ``t`` and the mask ``M`` are all
detached, so the ExGAP gradient reaches only ``z_U`` and, through it, the surviving
(mask-selected) patches of the shared visual backbone. It can never shrink the gap by moving
the Said router, the global pooling, or the text encoder.
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_MASK_THRESHOLD = 0.6
DEFAULT_GAP_TEMPERATURE = 0.05
DEFAULT_EPS = 1e-6
GLOBAL_POOL_MODES = ('mean', 'attention')

# Temperature of the Said router's own caption/evidence logits (``SALUModel.tau_said``). The
# sigmoid relevance is that same logit read through a sigmoid, so it must use the same scale;
# see ``said_relevance_logits`` for the measured failure of the ``/ sqrt(d)`` variant.
DEFAULT_TAU_SAID = 0.07

# ``p_G`` is a mean of unit-norm patches, so ``|S_gc|`` is bounded by ~1 and the explanatory-gap
# normalisation ``1 - S_gc`` in the denominator is meaningful.
EXPECTED_PATCH_NORM = 1.0


def normalize_patches(patch_features: torch.Tensor, eps: float = DEFAULT_EPS) -> torch.Tensor:
    """L2-normalise patch tokens; reported as ``patch_norm_deviation`` if they are not unit."""
    return F.normalize(patch_features, dim=-1, eps=eps)


# --------------------------------------------------------------------------- #
# Global representation
# --------------------------------------------------------------------------- #
def mean_global_weights(patch_features: torch.Tensor) -> torch.Tensor:
    """Uniform pooling weights ``1/N`` -> ``[B, N]``."""
    if patch_features.dim() != 3:
        raise ValueError('patch_features must be [B, N, D], got %r'
                         % (tuple(patch_features.shape),))
    n = patch_features.shape[1]
    return patch_features.new_full((patch_features.shape[0], n), 1.0 / float(n))


def attention_global_weights(patch_features: torch.Tensor, query: torch.Tensor,
                             w_query: nn.Linear, w_key: nn.Linear,
                             eps: float = DEFAULT_EPS) -> torch.Tensor:
    """Image-only attention pooling weights ``A^G`` -> ``[B, N]``.

    ``query`` must be an image-only token: it never sees the caption, so ``z_G = f(I)`` and the
    image side stays captio-independent (required for standard retrieval precomputation).
    """
    if query.dim() != 2:
        raise ValueError('global query must be [B, D], got %r' % (tuple(query.shape),))
    dim = float(patch_features.shape[-1])
    q = w_query(query)                                   # [B, D]
    k = w_key(patch_features)                            # [B, N, D]
    logits = torch.einsum('bd,bnd->bn', q, k) / (dim ** 0.5)
    return F.softmax(logits, dim=-1)


def compute_global_representation(patch_features: torch.Tensor,
                                  pool: str = 'mean',
                                  query: Optional[torch.Tensor] = None,
                                  w_query: Optional[nn.Linear] = None,
                                  w_key: Optional[nn.Linear] = None,
                                  eps: float = DEFAULT_EPS
                                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(z_G, A^G)`` from ``H`` alone. ``mean`` mode returns uniform ``A^G``."""
    if pool not in GLOBAL_POOL_MODES:
        raise ValueError('global_pool must be one of %r, got %r' % (GLOBAL_POOL_MODES, pool))
    if pool == 'mean':
        weights = mean_global_weights(patch_features)
    else:
        if query is None or w_query is None or w_key is None:
            raise ValueError("global_pool='attention' requires query and both projections")
        weights = attention_global_weights(patch_features, query, w_query, w_key, eps=eps)
    pooled = torch.einsum('bn,bnd->bd', weights, patch_features)
    return F.normalize(pooled, dim=-1, eps=eps), weights


# --------------------------------------------------------------------------- #
# Said representation
# --------------------------------------------------------------------------- #
def said_relevance_logits(text_feature: torch.Tensor, patch_features: torch.Tensor,
                          w_query: nn.Linear, w_key: nn.Linear,
                          tau_said: float = DEFAULT_TAU_SAID) -> torch.Tensor:
    """``ell^S_p = (q_hat . k_hat_p) / tau_said`` with normalised projections -> ``[B, N]``.

    ``q_hat = Norm(W_Q t)`` and ``k_hat_p = Norm(W_K h_p)``, exactly like ``SaidRouter``: this is
    the **same logit the Said objective already optimises**
    (``logits = normalize(W_q t) . normalize(W_k h) / tau_said``), read through a sigmoid
    instead of a softmax. Sharing that logit is what makes ``r^S_p`` a per-patch probability of
    being claimed by the caption, and it keeps ``tau_M`` on a fixed, interpretable scale.

    Measured reason for not using the literal ``(tW_Q)(h_pW_K)^T / sqrt(d)`` form with
    unnormalised projections: with default-initialised projections that logit is ``~1e-4`` for
    every patch, so ``r^S_p = sigmoid(1e-4) = 0.5 +- 4e-5``, ``r^S_p > tau_M = 0.6`` is never
    true, and the hard mask -- therefore the whole ExGAP objective -- is silently inert
    (measured on the 4x A800 smoke run: ``mask_keep_ratio`` exactly ``1.0``,
    ``said_above_threshold_fraction`` exactly ``0.0`` and ``loss_exgap`` exactly ``0.0`` at
    every one of the 100 steps, with ``said_relevance_std = 4e-5``).
    """
    if float(tau_said) <= 0.0:
        raise ValueError('tau_said must be positive, got %r' % (tau_said,))
    q = F.normalize(w_query(text_feature), dim=-1)       # [B, D]
    k = F.normalize(w_key(patch_features), dim=-1)       # [B, N, D]
    return torch.einsum('bd,bnd->bn', q, k) / float(tau_said)


def compute_said_representation(text_feature: torch.Tensor, patch_features: torch.Tensor,
                                w_query: nn.Linear, w_key: nn.Linear,
                                tau_said: float = DEFAULT_TAU_SAID,
                                eps: float = DEFAULT_EPS
                                ) -> Dict[str, torch.Tensor]:
    """Sigmoid relevance, normalised attention, and the Said feature.

    The relevance ``r^S_p = sigmoid(ell^S_p)`` is an **independent per-patch probability**; the
    softmax attention is deliberately *not* compared against the mask threshold, because a
    softmax over 196 patches is tiny by construction and would make the threshold meaningless.
    """
    logits = said_relevance_logits(text_feature, patch_features, w_query, w_key,
                                   tau_said=tau_said)
    relevance = torch.sigmoid(logits)
    attention = relevance / (relevance.sum(dim=-1, keepdim=True) + eps)
    pooled = torch.einsum('bn,bnd->bd', attention, patch_features)
    return {
        'said_logits': logits,
        'said_relevance': relevance,
        'said_attention': attention,
        'said_feature': F.normalize(pooled, dim=-1, eps=eps),
    }


# --------------------------------------------------------------------------- #
# Said hard mask
# --------------------------------------------------------------------------- #
def compute_said_mask(said_relevance: torch.Tensor, threshold: float = DEFAULT_MASK_THRESHOLD,
                      min_keep: int = 1) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Hard mask ``M_p = 1(r^S_p <= tau_M)``, **detached**.

    ``M_p = 1`` marks a patch available to the masked complementary representation; ``M_p = 0``
    marks a Said-dominant patch that is removed. The mask is detached so the ExGAP objective can
    never influence the routing decision by moving patches across the threshold.

    Degenerate handling: if a sample would keep nothing (every ``r^S_p > tau_M``, i.e. the Said
    selection claims the whole image), the mask falls back to keeping the single lowest-relevance
    patch. That keeps the masked pooling well defined and NaN-free for **every** sample, so no
    sample has to be dropped; ``fallback_fraction`` reports how often it happens.
    """
    if said_relevance.dim() != 2:
        raise ValueError('said_relevance must be [B, N], got %r'
                         % (tuple(said_relevance.shape),))
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError('mask threshold must be in [0, 1], got %r' % (threshold,))
    if int(min_keep) < 1:
        raise ValueError('min_keep must be >= 1, got %r' % (min_keep,))
    with torch.no_grad():
        mask = (said_relevance <= float(threshold)).to(said_relevance.dtype)
        empty = mask.sum(dim=-1, keepdim=True) < 1
        if bool(empty.any()):
            lowest = said_relevance.argmin(dim=-1, keepdim=True)
            fallback = torch.zeros_like(mask).scatter_(1, lowest, 1.0)
            mask = torch.where(empty, fallback, mask)
        mask = mask.detach()
        counts = mask.sum(dim=-1)
        diagnostics = {
            'mask_keep_ratio': mask.mean(),
            'mask_drop_ratio': 1.0 - mask.mean(),
            'said_above_threshold_fraction': (said_relevance > float(threshold)).float().mean(),
            'masked_patch_count_mean': counts.mean(),
            'masked_patch_count_min': counts.min(),
            'masked_patch_count_max': counts.max(),
            'fallback_fraction': empty.float().mean(),
            'masked_patch_count_is_zero_anywhere': (counts < 1).any(),
        }
    return mask, diagnostics


# --------------------------------------------------------------------------- #
# Masked complementary representation
# --------------------------------------------------------------------------- #
def compute_masked_unsaid_representation(patch_features: torch.Tensor, mask: torch.Tensor,
                                         global_weights: torch.Tensor,
                                         eps: float = DEFAULT_EPS
                                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(z_U, A^U)`` -- the masked global evidence. No new query is introduced.

    ``mean`` mode renormalises the mask itself; ``attention`` mode renormalises the *masked*
    global attention ``A^U_p = M_p A^G_p / sum_q M_q A^G_q``. Both reduce to "the global
    evidence after removing the strongly Said patches".
    """
    if mask.shape != global_weights.shape:
        raise ValueError('mask %r and global weights %r must have the same shape'
                         % (tuple(mask.shape), tuple(global_weights.shape)))
    weights = mask * global_weights
    weights = weights / (weights.sum(dim=-1, keepdim=True) + eps)
    pooled = torch.einsum('bn,bnd->bd', weights, patch_features)
    return F.normalize(pooled, dim=-1, eps=eps), weights


# --------------------------------------------------------------------------- #
# Explanatory gap
# --------------------------------------------------------------------------- #
def compute_explanatory_gap(s_sc: torch.Tensor, s_gc: torch.Tensor,
                            normalize: bool = True,
                            eps: float = DEFAULT_EPS) -> Dict[str, torch.Tensor]:
    """Detached explanatory gap ``G_exp`` and its raw value.

    ``gap_raw = [S_sc - S_gc]_+`` (a positive part, never an absolute value: ``S_gc > S_sc``
    means the Said router is not yet doing its job, not that there is more unsaid content).
    ``gap_weight = sg(clamp(gap_raw / (1 - S_gc + eps), 0, 1))``.

    The value is a *caption-incompleteness / selective explanatory gain proxy*: how much of the
    remaining image-text explanatory headroom the Said selection recovers. It is deliberately
    detached, so ``L_ExGAP`` cannot shrink the gap by moving the readings that define it.
    """
    if s_sc.shape != s_gc.shape:
        raise ValueError('S_sc %r and S_gc %r must have the same shape'
                         % (tuple(s_sc.shape), tuple(s_gc.shape)))
    gap_raw = F.relu(s_sc - s_gc)
    if normalize:
        gap_weight = (gap_raw / (1.0 - s_gc + eps)).clamp(min=0.0, max=1.0)
    else:
        gap_weight = gap_raw
    return {'gap_raw': gap_raw.detach(), 'gap_weight': gap_weight.detach()}


def compute_exgap_loss(s_uc: torch.Tensor, s_gc: torch.Tensor, gap_weight: torch.Tensor,
                       temperature: float = DEFAULT_GAP_TEMPERATURE,
                       valid: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """``L_ExGAP = mean_valid( G_exp * tau * softplus((S_uc - sg(S_gc)) / tau) )``.

    ``S_uc`` is the only non-detached input: the gradient enters ``z_U`` (hence the surviving
    patches and the shared visual backbone) and nothing else.
    """
    if float(temperature) <= 0.0:
        raise ValueError('gap temperature must be positive, got %r' % (temperature,))
    s_gc_ref = s_gc.detach()
    per_sample = gap_weight.detach() * float(temperature) * F.softplus(
        (s_uc - s_gc_ref) / float(temperature))
    if valid is None:
        valid_weight = torch.ones_like(per_sample)
    else:
        valid_weight = valid.to(per_sample.dtype)
    total = (per_sample * valid_weight).sum()
    denominator = valid_weight.sum()
    if float(denominator) == 0.0:
        # every sample invalid: return a differentiable zero instead of NaN
        loss = (per_sample * valid_weight).sum() * 0.0
    else:
        loss = total / denominator
    return {'loss': loss, 'per_sample': per_sample, 's_gc_ref': s_gc_ref,
            'valid_fraction': valid_weight.mean() if valid is not None
            else torch.ones((), device=per_sample.device)}


# --------------------------------------------------------------------------- #
# Image-only global attention pooling agent
# --------------------------------------------------------------------------- #
class ExGapModule(nn.Module):
    """Image-only global attention pooling for SAID-ExGAP.

    The pooling query is a **learnable image-only token**, projected by ``w_query`` and compared
    against the projected patches by ``w_key``::

        q_G = normalize(w_query(q_token)),  k_p = normalize(w_key(h_p))
        A^G_p = softmax_p( q_G . k_p / sqrt(d) )

    It never receives the caption, so ``z_G = f(I)`` and the image side of standard retrieval
    stays precomputable. ``w_query`` is zero-initialised, so at step 0 every logit is exactly
    ``0`` and ``A^G`` is **exactly uniform**: the attention mode starts out identical to the
    mean-pooling baseline. That matters because v1 has no global-text loss -- the pooling only
    leaves the uniform solution if the ExGAP path (which uses ``A^G`` inside ``z_U``) actually
    rewards it.

    ``w_key`` **and** the query token are deliberately **not** zero: if the query side were zero
    too, the gradient of the logit would vanish as well (``d logit / d W_query ~ k`` and
    ``d logit / d W_key ~ q``), so the attention could never leave the uniform solution no
    matter how strongly ExGAP rewarded it -- a dead saddle. With a live query token the logits
    are still *exactly* zero at step 0 (``W_query`` is zero), but ``W_query`` itself now receives
    ``d L / d W_query ~ q_token != 0``, so the pooling can move as soon as ExGAP asks for it.
    ``global_attention_entropy`` in the training log is the monitor for exactly this: it stays
    ``log(N)`` while ``A^G`` is uniform.
    """

    def __init__(self, dim: int, zero_init: bool = True, key_init_std: float = 0.02):
        super().__init__()
        self.dim = int(dim)
        self.query = nn.Parameter(torch.zeros(1, self.dim))
        self.w_query = nn.Linear(self.dim, self.dim)
        self.w_key = nn.Linear(self.dim, self.dim)
        # the query token is a real (small) token: with an exactly-zero token the query
        # projection could never move either, see the class docstring
        nn.init.normal_(self.query, std=float(key_init_std))
        # deterministic small key side: gives the second escape route, and is irrelevant to the
        # initial logits because the query projection is exactly zero
        nn.init.normal_(self.w_key.weight, mean=0.0, std=float(key_init_std))
        nn.init.zeros_(self.w_key.bias)
        if zero_init:
            # zero query projection => zero logits => exactly uniform attention at initialisation.
            # (A merely *small* query projection would be amplified by the L2 normalisation below,
            # giving near-uniform-but-not-equal weights.)
            nn.init.zeros_(self.w_query.weight)
            nn.init.zeros_(self.w_query.bias)

    def forward(self, patch_features: torch.Tensor, eps: float = DEFAULT_EPS
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(z_G, A^G)`` for a batch of patch sets."""
        if patch_features.dim() != 3:
            raise ValueError('patch_features must be [B, N, D], got %r'
                             % (tuple(patch_features.shape),))
        batch = patch_features.shape[0]
        dim = patch_features.shape[-1]
        base = self.query.reshape(1, dim).expand(batch, dim)
        # the wrapped projections follow the same "normalise before the dot product" pipeline as
        # the rest of this codebase
        q = F.normalize(self.w_query(base), dim=-1, eps=eps)
        k = F.normalize(self.w_key(patch_features), dim=-1, eps=eps)
        logits = torch.einsum('bd,bnd->bn', q, k) / (dim ** 0.5)
        weights = F.softmax(logits, dim=-1)
        pooled = torch.einsum('bn,bnd->bd', weights, patch_features)
        return F.normalize(pooled, dim=-1, eps=eps), weights


def checkpoint_config(global_pool: str, mask_threshold: float, temperature: float,
                      lambda_said: float, lambda_exgap: float,
                      normalize_gap: bool = True) -> Dict[str, object]:
    """The ``exgap_config`` block written into checkpoints and compared on resume."""
    return {
        'global_pool': str(global_pool),
        'mask_threshold': float(mask_threshold),
        'temperature': float(temperature),
        'lambda_said': float(lambda_said),
        'lambda_exgap': float(lambda_exgap),
        'normalize_gap': bool(normalize_gap),
    }


def validate_exgap_config(args) -> None:
    """Reject an illegal SAID-ExGAP configuration before any model or data is built."""
    problems = []
    if args.exgap_global_pool not in GLOBAL_POOL_MODES:
        problems.append('--exgap-global-pool must be one of %r, got %r'
                        % (GLOBAL_POOL_MODES, args.exgap_global_pool))
    if not 0.0 <= float(args.exgap_mask_threshold) <= 1.0:
        problems.append('--exgap-mask-threshold must be in [0, 1], got %r'
                        % (args.exgap_mask_threshold,))
    if float(args.exgap_temperature) <= 0.0:
        problems.append('--exgap-temperature must be positive, got %r'
                        % (args.exgap_temperature,))
    if float(args.lambda_said) <= 0.0:
        problems.append('--lambda-said must be positive, got %r' % (args.lambda_said,))
    if not 0.0 <= float(args.lambda_exgap) < float('inf'):
        problems.append('--lambda-exgap must be finite and >= 0, got %r' % (args.lambda_exgap,))
    if float(args.lambda_global) != 0.0:
        problems.append('--lambda-global must be 0 for said_exgap (v1 has no global-text CLIP '
                        'loss), got %r' % (args.lambda_global,))
    if float(args.lambda_unsaid) != 0.0:
        problems.append('--lambda-unsaid must be 0 for said_exgap (no Phase 2.x Unsaid branch), '
                        'got %r' % (args.lambda_unsaid,))
    if problems:
        raise ValueError('illegal said_exgap configuration: ' + '; '.join(problems))
