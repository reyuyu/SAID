"""Phase 3.0A.1c: complement emergence diagnostics (pure measurement, never a loss).

These functions answer *why* the Said and Unsaid pooled representations stay close
(``cos(z_S, z_U) ~ 0.99`` after 3 steps). They are deliberately split into three
independent questions so a conclusion cannot be over-read:

* :func:`patch_homogeneity_metrics` -- **H3, patch intrinsic geometry only.** Does the
  final CLIP residual patch set already carry near-identical global semantics? Nothing
  about attention, the router or C_S enters here: a high ``patch_pair_cosine_mean`` with a
  low ``patch_centered_energy`` alone supports "the patch tokens are homogeneous".
* :func:`raw_pooling_metrics` -- **feature diversity of the two poolings.** ``z_S`` and
  ``z_U`` are L2-normalised (norm ~ 1 by construction), so they carry no magnitude
  information; the *raw* pools ``p_S``/``p_U`` do, and ``raw_pool_cosine`` shows how
  different the two attention distributions actually pool.
* :func:`global_relation_metrics` -- **relation to the global embedding.** ``cos(g, z_S)``
  vs ``cos(g, z_U)`` and ``cos(z_S, z_U)``.

Attention diversity and feature diversity are **not** useful complementarity: whether a
different ``z_U`` actually closes the gap is measured only by
``gap_completion.gap_diagnostics`` (``gap_before`` / ``gap_after`` / ``closure``).
Nothing in this module may ever enter a loss, and none of it may be read as "the method
works".
"""
from typing import Dict

import torch
import torch.nn.functional as F

DEFAULT_EPS = 1e-6


def patch_homogeneity_metrics(patch_features: torch.Tensor,
                              global_feature: torch.Tensor,
                              eps: float = DEFAULT_EPS) -> Dict[str, torch.Tensor]:
    """Geometry of one image's patch set, before any attention is applied.

    Args:
        patch_features: ``[B, P, D]`` projected patch tokens.
        global_feature: ``[B, D]`` global image embedding.

    Returns (all detached scalars, averaged over the batch):

        ``patch_pair_cosine_mean`` / ``patch_pair_cosine_std``
            Off-diagonal pairwise cosine ``cos(h_p, h_q)`` of the L2-normalised patches.
            Computed exactly, one image at a time, from that image's ``[P, P]`` cosine
            Gram matrix with the diagonal masked out (``unbiased=True`` std). The ``[P, P]``
            tensor is never held for the whole batch and no patch pairs are sampled, so the
            numbers are exact rather than estimated.

        ``patch_to_global_cosine_mean`` / ``patch_to_global_cosine_std``
            ``cos(h_p, g)`` over patches.
        ``patch_centered_energy``
            ``mean_p ||h_p - mean_q h_q||_2`` on the normalised patches: how far the
            patches spread around their own centroid.
    """
    if patch_features.dim() != 3:
        raise ValueError('patch_features must be [B, P, D], got %r'
                         % (tuple(patch_features.shape),))
    if global_feature.dim() != 2:
        raise ValueError('global_feature must be [B, D], got %r'
                         % (tuple(global_feature.shape),))
    with torch.no_grad():
        patches = F.normalize(patch_features.detach().float(), dim=-1, eps=eps)
        global_vector = F.normalize(global_feature.detach().float(), dim=-1, eps=eps)
        batch, n_patches = patches.shape[0], patches.shape[1]

        # Exact off-diagonal Gram statistics, one image at a time: the [P, P] matrix is
        # never held for the whole batch (P = 196 -> 150 KB per image in fp32). The
        # closed-form ||G||_F^2 - P version cancels catastrophically for the near-parallel
        # patch sets this diagnostic exists to detect, so the diagonal is masked instead.
        pair_mean = patches.new_zeros(batch)
        pair_std = patches.new_zeros(batch)
        for index in range(batch):
            gram = patches[index] @ patches[index].transpose(0, 1)         # [P, P]
            off_diagonal = gram.masked_select(
                ~torch.eye(n_patches, dtype=torch.bool, device=gram.device)).flatten()
            if off_diagonal.numel() > 1:
                pair_mean[index] = off_diagonal.mean()
                pair_std[index] = off_diagonal.std(unbiased=True)
            elif off_diagonal.numel() == 1:
                pair_mean[index] = off_diagonal[0]
                pair_std[index] = 0.0

        to_global = torch.einsum('bpd,bd->bp', patches, global_vector)  # [B, P]
        centroid = patches.mean(dim=1, keepdim=True)
        centered_energy = (patches - centroid).norm(dim=-1).mean()

        return {
            'patch_pair_cosine_mean': pair_mean.mean(),
            'patch_pair_cosine_std': pair_std.mean(),
            'patch_to_global_cosine_mean': to_global.mean(),
            'patch_to_global_cosine_std': to_global.std(dim=-1, unbiased=False).mean(),
            'patch_centered_energy': centered_energy,
            'patch_count': float(n_patches),
        }


def raw_pooling_metrics(said_attention: torch.Tensor,
                        unsaid_attention: torch.Tensor,
                        patch_features: torch.Tensor,
                        eps: float = DEFAULT_EPS) -> Dict[str, torch.Tensor]:
    """Magnitude and direction of the two *raw* (pre-normalisation) pools.

    ``p_S = sum_p A_S,p h_p`` and ``p_U = sum_p A_U,p h_p``. The official ``z_S`` / ``z_U``
    are ``normalize(p_S)`` / ``normalize(p_U)``, so their norms are ~1 by construction and
    say nothing; the raw norms and ``raw_pool_cosine`` carry the magnitude information.

    Returns ``said_raw_pool_norm``, ``unsaid_raw_pool_norm``, ``raw_pool_cosine``,
    ``raw_pool_norm_ratio``. Nothing here is normalised away or fed to a loss.
    """
    if said_attention.dim() != 2 or unsaid_attention.dim() != 2:
        raise ValueError('attention must be [B, P]')
    if patch_features.dim() != 3:
        raise ValueError('patch_features must be [B, P, D]')
    with torch.no_grad():
        patches = patch_features.detach().float()
        pooled_said = torch.einsum('bp,bpd->bd', said_attention.detach().float(), patches)
        pooled_unsaid = torch.einsum('bp,bpd->bd', unsaid_attention.detach().float(), patches)
        norm_said = pooled_said.norm(dim=-1)
        norm_unsaid = pooled_unsaid.norm(dim=-1)
        cosine = (F.normalize(pooled_said, dim=-1, eps=eps)
                  * F.normalize(pooled_unsaid, dim=-1, eps=eps)).sum(dim=-1)
        return {
            'said_raw_pool_norm': norm_said.mean(),
            'unsaid_raw_pool_norm': norm_unsaid.mean(),
            'raw_pool_cosine': cosine.mean(),
            'raw_pool_norm_ratio': (norm_unsaid / norm_said.clamp_min(eps)).mean(),
        }


def global_relation_metrics(global_feature: torch.Tensor,
                            said_feature: torch.Tensor,
                            unsaid_feature: torch.Tensor,
                            eps: float = DEFAULT_EPS) -> Dict[str, torch.Tensor]:
    """``cos(g, z_S)``, ``cos(g, z_U)`` and ``cos(z_S, z_U)`` (detached, monitoring only)."""
    with torch.no_grad():
        g = F.normalize(global_feature.detach().float(), dim=-1, eps=eps)
        s = F.normalize(said_feature.detach().float(), dim=-1, eps=eps)
        u = F.normalize(unsaid_feature.detach().float(), dim=-1, eps=eps)
        return {
            'cos_global_said': (g * s).sum(dim=-1).mean(),
            'cos_global_unsaid': (g * u).sum(dim=-1).mean(),
            'cos_said_unsaid': (s * u).sum(dim=-1).mean(),
        }


COMPLEMENT_DIAGNOSTIC_KEYS = (
    'patch_pair_cosine_mean',
    'patch_pair_cosine_std',
    'patch_to_global_cosine_mean',
    'patch_to_global_cosine_std',
    'patch_centered_energy',
    'said_raw_pool_norm',
    'unsaid_raw_pool_norm',
    'raw_pool_cosine',
    'raw_pool_norm_ratio',
    'cos_global_said',
    'cos_global_unsaid',
    'cos_said_unsaid',
)
