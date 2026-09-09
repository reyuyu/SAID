"""SALU modules -- Phase 2 "Align the Said" (Said-only).

This module contains only the caption-conditioned Said Router. It deliberately
contains no prototype bank, no unsaid explorer, no extra loss, no k-means/EMA and
no reconstruction/completeness/overlap terms.

Pipeline implemented here::

    patch_features H  [B, N, D]
        +  text_feature t  [B, D]
        -> q_proj / k_proj (lightweight, "where to look")
        -> A_s = softmax(q^T k / tau_said)          [B, N]
        -> z_s = sum_i A_s_i * H_i  (raw patch features, "what to take")
        -> z_s = L2-normalise(z_s)                  [B, D]
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SaidRouter(nn.Module):
    """Caption-conditioned patch attention router.

    Args:
        dim: feature dimension D (512 for ViT-B/16).
        tau_said: softmax temperature used on the q^T k logits.

    Inputs:
        text_feature:   [B, D]     global text embedding of the sampled sparse caption.
        patch_features: [B, N, D]  projected ViT patch tokens from the Phase 1 interface.

    Returns:
        A_s: [B, N]  Said attention over patches (non-negative, sums to 1).
        z_s: [B, D]  Said visual representation (L2-normalised).
    """

    def __init__(self, dim: int = 512, tau_said: float = 0.07):
        super().__init__()
        tau_said = float(tau_said)
        if tau_said <= 0.0:
            raise ValueError('tau_said must be positive, got %r' % (tau_said,))
        self.dim = int(dim)
        self.tau_said = tau_said
        # Only two lightweight projections; no multi-head / decoder / MLP stack.
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)

    def forward(
        self,
        text_feature: torch.Tensor,
        patch_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if text_feature.dim() != 2:
            raise ValueError('text_feature must be [B, D], got %r' % (tuple(text_feature.shape),))
        if patch_features.dim() != 3:
            raise ValueError('patch_features must be [B, N, D], got %r' % (tuple(patch_features.shape),))
        if text_feature.shape[0] != patch_features.shape[0]:
            raise ValueError(
                'batch mismatch: text_feature %r vs patch_features %r'
                % (tuple(text_feature.shape), tuple(patch_features.shape))
            )
        if text_feature.shape[-1] != patch_features.shape[-1]:
            raise ValueError(
                'dim mismatch: text_feature %r vs patch_features %r'
                % (tuple(text_feature.shape), tuple(patch_features.shape))
            )

        # "where to look": projections are normalised before the dot product.
        q = F.normalize(self.q_proj(text_feature), dim=-1)      # [B, D]
        k = F.normalize(self.k_proj(patch_features), dim=-1)    # [B, N, D]
        logits = torch.einsum('bd,bnd->bn', q, k) / self.tau_said
        A_s = F.softmax(logits, dim=-1)                         # [B, N]

        # "what to take": pooling uses the raw Phase 1 patch features, not k_proj(H).
        z_s = torch.einsum('bn,bnd->bd', A_s, patch_features)   # [B, D]
        z_s = F.normalize(z_s, dim=-1)
        return A_s, z_s

    @staticmethod
    def attention_entropy(A_s: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """Shannon entropy of the Said attention, per sample: [B]."""
        p = A_s.float().clamp_min(eps)
        return -(p * p.log()).sum(dim=-1)

    def extra_repr(self) -> str:
        return 'dim=%d, tau_said=%g' % (self.dim, self.tau_said)
