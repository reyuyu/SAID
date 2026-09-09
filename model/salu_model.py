"""SALU model wrapper -- Said-only training.

Two Said objectives are supported (``said_loss_mode``):

``positive`` (Phase 2, kept as an ablation)
    z_s = Said(I_i, C_i); symmetric InfoNCE between gathered z_s and gathered t.
    Diagnostic finding (Phase 2.1): this objective collapses to generic image
    saliency -- own-vs-shuffled caption attention JSD ~0.013, below the bf16/fp32
    precision noise floor, because the negative captions never re-enter the router.

``identifiable`` (Phase 2.2, default)
    Every (image i, caption j) pair is routed, and the *scoring* text is always the
    image's own caption::

        route_score[i, j]    = scale * cos(z_pair[i, j], t_i)   # routing caption varies
        evidence_score[i, j] = scale * cos(z_pair[j, i], t_i)   # routing image varies

    with ``L_said = 0.5 * (CE(route_score) + CE(evidence_score))``.
    If the router ignores the caption, every z_pair[i, j] is identical and
    route_score has identical rows, so L_route cannot fall below log(B). This
    directly penalises caption-independent (saliency) routing.

Standard inference (retrieval / zero-shot) still delegates to the wrapped CLIP
model and never uses the Said router. No prototype bank, no unsaid explorer, no
sparsity / entropy / overlap / reconstruction term.
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


def identifiable_said_loss(
    z_pair: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Same-image route identification + cross-image evidence identification.

    Args:
        z_pair: [B, B, D] L2-normalised Said features, ``z_pair[i, j]`` = image i
            routed by caption j.
        text_features: [B, D] L2-normalised captions of the local batch.
        logit_scale: scalar tensor.

    Returns:
        dict with ``loss_route``, ``loss_evidence``, ``loss_said`` and the
        detached diagnostics ``route_top1_acc``, ``evidence_top1_acc``,
        ``route_margin``, ``evidence_margin``.
    """
    if z_pair.dim() != 3 or z_pair.shape[0] != z_pair.shape[1]:
        raise ValueError('z_pair must be [B, B, D], got %r' % (tuple(z_pair.shape),))
    batch = z_pair.shape[0]
    if batch < 2:
        raise ValueError('identifiable Said loss needs batch >= 2, got %d' % batch)
    labels = torch.arange(batch, device=z_pair.device)

    # routing caption varies, scoring text fixed to t_i
    route_score = logit_scale * torch.einsum('ijd,id->ij', z_pair, text_features)
    # routing image varies, scoring text fixed to t_i
    evidence_score = logit_scale * torch.einsum('jid,id->ij', z_pair, text_features)

    loss_route = F.cross_entropy(route_score, labels)
    loss_evidence = F.cross_entropy(evidence_score, labels)
    loss_said = 0.5 * (loss_route + loss_evidence)

    with torch.no_grad():
        def _stats(score):
            top1 = (score.argmax(dim=1) == labels).float().mean()
            diag = score.diagonal()
            off_diag_mean = (score.sum(dim=1) - diag) / float(batch - 1)
            return top1, (diag - off_diag_mean).mean()

        route_top1, route_margin = _stats(route_score)
        evidence_top1, evidence_margin = _stats(evidence_score)

    return {
        'loss_route': loss_route,
        'loss_evidence': loss_evidence,
        'loss_said': loss_said,
        'route_top1_acc': route_top1,
        'evidence_top1_acc': evidence_top1,
        'route_margin': route_margin,
        'evidence_margin': evidence_margin,
    }


class SALUModel(nn.Module):
    """Said-only wrapper around a CLIP / SmartCLIP model.

    Args:
        clip_model: a ``model.model_longclip.CLIP`` instance (from
            ``longclip.load_from_clip`` or ``longclip.load``).
        tau_said: Said softmax temperature.
        dim: router feature dimension; defaults to the CLIP embedding dim.
        freeze_mask_net: freeze SmartCLIP's ``mask_net`` (kept, never trained).
        fp32_master_weights: cast the wrapped CLIP weights to fp32 (see note below).
        said_loss_mode: ``identifiable`` (default) or ``positive``.
        pair_chunk_size: caption candidates per chunk for pairwise routing
            (``None`` = all at once; numerically identical).
    """

    def __init__(
        self,
        clip_model: nn.Module,
        tau_said: float = 0.07,
        dim: Optional[int] = None,
        freeze_mask_net: bool = True,
        fp32_master_weights: bool = True,
        said_loss_mode: str = 'identifiable',
        pair_chunk_size: Optional[int] = 64,
    ):
        super().__init__()
        if said_loss_mode not in ('positive', 'identifiable'):
            raise ValueError("said_loss_mode must be 'positive' or 'identifiable', got %r" % (said_loss_mode,))
        self.clip = clip_model
        # SmartCLIP/OpenAI CLIP weights are stored as fp16 by ``convert_weights``.
        # fp16 gradient buffers overflow under GradScaler, so SALU training keeps
        # fp32 master weights and relies on autocast for mixed precision.
        if fp32_master_weights:
            self.clip.float()
        if dim is None:
            dim = int(clip_model.text_projection.shape[1])
        self.said_router = SaidRouter(dim=dim, tau_said=tau_said)
        self.tau_said = float(tau_said)
        self.said_loss_mode = said_loss_mode
        self.pair_chunk_size = pair_chunk_size

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
    # training-time Said objective
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

        # own-caption attention: diagnostics for every mode, and the whole
        # objective in 'positive' mode
        A_own, z_s_own = self.said_router(t, patch_features)

        scale = self.clip.logit_scale.exp().clamp(max=100)
        loss_global = contrastive_loss(
            gather_features_with_grad(z_g), gather_features_with_grad(t), scale
        )

        zero = torch.zeros((), device=images.device, dtype=loss_global.dtype)
        if self.said_loss_mode == 'identifiable':
            z_pair, _ = self.said_router.route_pairwise(
                t, patch_features, chunk_size=self.pair_chunk_size
            )
            said = identifiable_said_loss(z_pair, t, scale)
            loss_route = said['loss_route']
            loss_evidence = said['loss_evidence']
            loss_said = said['loss_said']
            route_top1 = said['route_top1_acc']
            evidence_top1 = said['evidence_top1_acc']
            route_margin = said['route_margin']
            evidence_margin = said['evidence_margin']
        else:  # 'positive' -- Phase 2 ablation, unchanged
            z_s_all = gather_features_with_grad(z_s_own)
            t_all = gather_features_with_grad(t)
            loss_said = contrastive_loss(z_s_all, t_all, scale)
            loss_route = zero
            loss_evidence = zero
            route_top1 = evidence_top1 = zero
            route_margin = evidence_margin = zero

        loss_total = lambda_global * loss_global + lambda_said * loss_said

        with torch.no_grad():
            attn = A_own.detach().float()
            entropy = SaidRouter.attention_entropy(attn)
            out = {
                'loss_global': loss_global,
                'loss_said': loss_said,
                'loss_total': loss_total,
                'loss_route': loss_route.detach(),
                'loss_evidence': loss_evidence.detach(),
                'route_top1_acc': route_top1.detach(),
                'evidence_top1_acc': evidence_top1.detach(),
                'route_margin': route_margin.detach(),
                'evidence_margin': evidence_margin.detach(),
                'said_attention_entropy': entropy.mean(),
                'said_effective_patch_count': entropy.exp().mean(),
                'said_attention_max': attn.max(dim=-1).values.mean(),
                'said_attention_min': attn.min(dim=-1).values.mean(),
                'said_feature_norm': z_s_own.detach().float().norm(dim=-1).mean(),
                'global_feature_norm': z_g.detach().float().norm(dim=-1).mean(),
            }
        return out

    def forward(self, images, texts, lambda_global: float = 1.0, lambda_said: float = 1.0):
        return self.forward_train(
            images, texts, lambda_global=lambda_global, lambda_said=lambda_said
        )

    def extra_repr(self) -> str:
        return 'tau_said=%g, said_loss_mode=%s, pair_chunk_size=%s' % (
            self.tau_said, self.said_loss_mode, self.pair_chunk_size)
