"""SALU modules -- Said routing.

Phase 2 ("Align the Said", positive-only) plus Phase 2.2 (identifiable Said
routing). Still contains **no** prototype bank, no unsaid explorer, no k-means/EMA,
no reconstruction/completeness/overlap term.

Positive-only pipeline (Phase 2, kept as an ablation)::

    patch_features H  [B, N, D]  +  text_feature t  [B, D]
        -> q_proj / k_proj (lightweight, "where to look")
        -> A_s = softmax(q^T k / tau_said)          [B, N]
        -> z_s = sum_i A_s_i * H_i  (raw patch features, "what to take")
        -> z_s = L2-normalise(z_s)                  [B, D]

Identifiable pipeline (Phase 2.2): every (image i, caption j) pair is routed, so
the router is forced to use the caption instead of collapsing to image saliency::

    q_j = normalize(Wq t_j),  k_i,n = normalize(Wk h_i,n)
    A_pair[i, j, n] = softmax_n(q_j^T k_i,n / tau_said)     [B, B, N]
    z_pair[i, j]    = normalize(sum_n A_pair[i,j,n] h_i,n)  [B, B, D]
"""
from typing import Optional, Tuple

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

    # ------------------------------------------------------------------ #
    # single caption (own caption): unchanged Phase 2 behaviour
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # pairwise routing: every (image i, caption j) pair (Phase 2.2)
    # ------------------------------------------------------------------ #
    def route_pairwise(
        self,
        text_features: torch.Tensor,
        patch_features: torch.Tensor,
        chunk_size: Optional[int] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Vectorised caption x image routing.

        Args:
            text_features: [B, D]      routing captions (queries).
            patch_features: [B, N, D]  images (keys and values).
            chunk_size: caption candidates per chunk; ``None`` = all at once.
                Chunking only changes the loop split, so results are numerically
                identical to the unchunked computation (each softmax row is
                independent).
            return_attention: also return A_pair [B, B, N] (diagnostics only).

        Returns:
            z_pair: [B, B, D], ``z_pair[i, j]`` = L2-normalised Said feature of
                image ``i`` routed by caption ``j``.
            A_pair: [B, B, N] or ``None``.
        """
        if text_features.dim() != 2 or patch_features.dim() != 3:
            raise ValueError(
                'expected text [B, D] and patches [B, N, D], got %r and %r'
                % (tuple(text_features.shape), tuple(patch_features.shape))
            )
        batch = patch_features.shape[0]
        if text_features.shape[0] != batch:
            raise ValueError('batch mismatch: %d captions vs %d images' % (text_features.shape[0], batch))

        q = F.normalize(self.q_proj(text_features), dim=-1)       # [B, D]  (captions)
        k = F.normalize(self.k_proj(patch_features), dim=-1)      # [B, N, D] (images)

        step = batch if chunk_size is None else int(chunk_size)
        if step <= 0:
            raise ValueError('chunk_size must be positive or None, got %r' % (chunk_size,))

        z_chunks, a_chunks = [], []
        for start in range(0, batch, step):
            q_chunk = q[start:start + step]                                   # [C, D]
            logits = torch.einsum('cd,ind->icn', q_chunk, k) / self.tau_said  # [B, C, N]
            A_chunk = F.softmax(logits, dim=-1)                               # [B, C, N]
            # value = raw patch features of image i (not k_proj(H))
            z_chunk = torch.einsum('icn,ind->icd', A_chunk, patch_features)   # [B, C, D]
            z_chunks.append(F.normalize(z_chunk, dim=-1))
            if return_attention:
                a_chunks.append(A_chunk)

        z_pair = torch.cat(z_chunks, dim=1)                                   # [B, B, D]
        A_pair = torch.cat(a_chunks, dim=1) if return_attention else None
        return z_pair, A_pair

    @staticmethod
    def attention_entropy(A_s: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """Shannon entropy of the Said attention, per sample: [B]."""
        p = A_s.float().clamp_min(eps)
        return -(p * p.log()).sum(dim=-1)

    def extra_repr(self) -> str:
        return 'dim=%d, tau_said=%g' % (self.dim, self.tau_said)
