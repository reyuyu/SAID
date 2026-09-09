"""SALU model wrapper -- Phase 2 "Align the Said" (Said-only).

Wraps an existing SmartCLIP / LongCLIP model and adds the caption-conditioned
Said router. Standard inference (retrieval, zero-shot) is *unchanged*: it
delegates to the wrapped CLIP model and never uses the Said router.

Training objective of this phase (and only this phase)::

    L_total = lambda_global * L_global + lambda_said * L_said

with two symmetric InfoNCE terms. No prototype bank, no unsaid explorer, no
sparsity / entropy / overlap / reconstruction / completeness losses.

Notes
-----
* ``z_s`` is a *training-time conditional* visual representation produced from
  ``image_i`` and ``caption_i``; it is never used for standard inference.
* ``mask_net`` (SmartCLIP) is kept for checkpoint compatibility but is frozen and
  never used here.
"""
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .salu_modules import SaidRouter


def gather_features_with_grad(features: torch.Tensor) -> torch.Tensor:
    """All-gather features along dim 0 while keeping the autograd graph.

    ``torch.distributed.all_gather`` detaches the remote features; the
    ``torch.distributed.nn`` variant is autograd-aware, so gradients from the
    contrastive loss reach every rank's local features.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return features
    if dist.get_world_size() == 1:
        return features
    import torch.distributed.nn as dist_nn

    gathered = dist_nn.functional.all_gather(features)
    if isinstance(gathered, (list, tuple)):
        return torch.cat(list(gathered), dim=0)
    return gathered


def contrastive_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Symmetric CLIP InfoNCE on already-gathered, L2-normalised features."""
    if image_features.shape != text_features.shape:
        raise ValueError(
            'image/text feature shape mismatch: %r vs %r'
            % (tuple(image_features.shape), tuple(text_features.shape))
        )
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    batch = image_features.shape[0]
    labels = torch.arange(batch, device=image_features.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return 0.5 * (loss_i + loss_t)


class SALUModel(nn.Module):
    """Said-only wrapper around a CLIP / SmartCLIP model.

    Args:
        clip_model: a ``model.model_longclip.CLIP`` instance (from
            ``longclip.load_from_clip`` or ``longclip.load``).
        tau_said: Said softmax temperature.
        dim: router feature dimension; defaults to the CLIP embedding dim.
        freeze_mask_net: freeze SmartCLIP's ``mask_net`` (kept, never trained).
    """

    def __init__(
        self,
        clip_model: nn.Module,
        tau_said: float = 0.07,
        dim: Optional[int] = None,
        freeze_mask_net: bool = True,
        fp32_master_weights: bool = True,
    ):
        super().__init__()
        self.clip = clip_model
        # SmartCLIP/OpenAI CLIP weights are stored as fp16 by `convert_weights`.
        # fp16 gradient buffers overflow under GradScaler, so SALU training keeps
        # fp32 master weights and relies on autocast for mixed precision.
        if fp32_master_weights:
            self.clip.float()
        if dim is None:
            dim = int(clip_model.text_projection.shape[1])
        self.said_router = SaidRouter(dim=dim, tau_said=tau_said)
        self.tau_said = float(tau_said)

        if freeze_mask_net and hasattr(self.clip, 'mask_net'):
            for param in self.clip.mask_net.parameters():
                param.requires_grad_(False)

    # ------------------------------------------------------------------ #
    # standard inference: identical to the wrapped CLIP model
    # ------------------------------------------------------------------ #
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.clip.encode_image(image)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        return self.clip.encode_text(text)

    def encode_image_with_patches(self, image: torch.Tensor, use_checkpoint: bool = False):
        return self.clip.encode_image_with_patches(image, use_checkpoint=use_checkpoint)

    @property
    def logit_scale(self) -> torch.Tensor:
        return self.clip.logit_scale

    def said_head_parameters(self):
        """Parameters of the Said router only (optimizer group B)."""
        return list(self.said_router.parameters())

    def backbone_parameters(self):
        """Trainable parameters outside the Said router (optimizer group A)."""
        head_ids = {id(p) for p in self.said_router.parameters()}
        return [p for p in self.parameters() if p.requires_grad and id(p) not in head_ids]

    # ------------------------------------------------------------------ #
    # training-time Said-only objective
    # ------------------------------------------------------------------ #
    def forward_train(
        self,
        images: torch.Tensor,
        texts: torch.Tensor,
        lambda_global: float = 1.0,
        lambda_said: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        z_global, patch_features = self.clip.encode_image_with_patches(images)
        text_features = self.clip.encode_text(texts)

        z_g = F.normalize(z_global, dim=-1)
        t = F.normalize(text_features, dim=-1)

        A_s, z_s = self.said_router(t, patch_features)

        scale = self.clip.logit_scale.exp().clamp(max=100)
        z_g_all = gather_features_with_grad(z_g)
        t_all = gather_features_with_grad(t)
        z_s_all = gather_features_with_grad(z_s)

        loss_global = contrastive_loss(z_g_all, t_all, scale)
        loss_said = contrastive_loss(z_s_all, t_all, scale)
        loss_total = lambda_global * loss_global + lambda_said * loss_said

        with torch.no_grad():
            attn = A_s.detach().float()
            entropy = SaidRouter.attention_entropy(attn)
            out = {
                'loss_global': loss_global,
                'loss_said': loss_said,
                'loss_total': loss_total,
                'said_attention_entropy': entropy.mean(),
                'said_effective_patch_count': entropy.exp().mean(),
                'said_attention_max': attn.max(dim=-1).values.mean(),
                'said_attention_min': attn.min(dim=-1).values.mean(),
                'said_feature_norm': z_s.detach().float().norm(dim=-1).mean(),
                'global_feature_norm': z_g.detach().float().norm(dim=-1).mean(),
            }
        return out

    def forward(self, images, texts, lambda_global: float = 1.0, lambda_said: float = 1.0):
        return self.forward_train(
            images, texts, lambda_global=lambda_global, lambda_said=lambda_said
        )

    def extra_repr(self) -> str:
        return 'tau_said=%g' % self.tau_said
