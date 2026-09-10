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
model and never uses the Said router. No prototype bank, no extra learnable module,
no sparsity / entropy / overlap / reconstruction term.

Phase 2.8A adds the minimal Unsaid branch, active only when ``lambda_unsaid > 0``:
it reuses the Said router's own-caption raw scores ``s`` and the same patch features
``H`` with **no new parameter** (``A^U = softmax(-s / tau_unsaid)``,
``z_U = normalize(sum_i A^U_i h_i)``) and regresses ``z_U`` onto the stop-gradient
complement of the Said direction inside the global representation. With
``lambda_unsaid == 0`` the branch is not executed at all, so the model stays exactly
the Said-only model (same losses, same RNG use, same speed).
"""
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .salu_modules import SaidRouter
from .representation_metrics import batch_representation_gaps
from . import unsaid_core


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
        said_feature_source: str = 'residual',
    ):
        super().__init__()
        if said_feature_source not in ('residual', 'attention_delta'):
            raise ValueError('said_feature_source must be residual or attention_delta')
        self.said_feature_source = said_feature_source
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

    def encode_image_with_local_evidence(self, image: torch.Tensor, use_checkpoint: bool = False):
        return self.clip.encode_image_with_local_evidence(image, use_checkpoint=use_checkpoint)

    def encode_router_input(self, images: torch.Tensor):
        if self.said_feature_source == 'attention_delta':
            return self.clip.encode_image_with_local_evidence(images)
        return self.clip.encode_image_with_patches(images)

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
        lambda_unsaid: float = 0.0,
        tau_unsaid: float = unsaid_core.DEFAULT_TAU_UNSAID,
        unsaid_residual_eps: float = unsaid_core.DEFAULT_RESIDUAL_EPS,
        texts_full: Optional[torch.Tensor] = None,
        texts_unsaid: Optional[torch.Tensor] = None,
        has_unsaid: Optional[torch.Tensor] = None,
        global_caption_view: str = 'prefix',
        unsaid_mode: str = 'residual',
        unsaid_gate_floor: float = unsaid_core.DEFAULT_GATE_FLOOR,
        unsaid_gate_temperature: float = unsaid_core.DEFAULT_GATE_TEMPERATURE,
        unsaid_suppression_beta: float = unsaid_core.DEFAULT_SUPPRESSION_BETA,
        unsaid_candidate_chunk_size: int = 0,
    ) -> Dict[str, torch.Tensor]:
        # lambda_unsaid == 0 must stay bit-for-bit the Said-only path: the branch is
        # skipped (not computed and multiplied by zero), so no extra q/k forward runs.
        unsaid_enabled = float(lambda_unsaid) != 0.0
        if global_caption_view not in ('prefix', 'full'):
            raise ValueError("global_caption_view must be 'prefix' or 'full'")
        if unsaid_mode not in unsaid_core.UNSAID_MODES:
            raise ValueError('unsaid_mode must be one of %r' % (unsaid_core.UNSAID_MODES,))

        z_global, patch_features = self.encode_router_input(images)
        text_features = self.clip.encode_text(texts)

        z_g = F.normalize(z_global, dim=-1)
        t = F.normalize(text_features, dim=-1)
        # Phase 2.9A: the global objective may align against the full caption while the
        # Said router keeps the sampled prefix. Default keeps the legacy prefix view.
        t_global = t
        if global_caption_view == 'full':
            if texts_full is None:
                raise ValueError("global_caption_view='full' requires texts_full")
            t_global = F.normalize(self.clip.encode_text(texts_full), dim=-1)

        # own-caption attention: diagnostics for every mode, and the whole
        # objective in 'positive' mode. With Unsaid enabled the details variant is
        # used so that Unsaid can reuse the very same q / k / s (same numbers as
        # ``self.said_router(t, patch_features)``, one projection instead of two).
        unsaid_scores = None
        if unsaid_enabled:
            details = self.said_router.forward_with_details(t, patch_features)
            A_own = details['attention']
            z_s_own = details['said']
            unsaid_scores = details['scores']
        else:
            A_own, z_s_own = self.said_router(t, patch_features)

        scale = self.clip.logit_scale.exp().clamp(max=100)
        loss_global = contrastive_loss(
            gather_features_with_grad(z_g), gather_features_with_grad(t_global), scale
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

        if unsaid_enabled:
            if unsaid_mode == 'debiased_suffix':
                if texts_unsaid is None or has_unsaid is None:
                    raise ValueError("unsaid_mode='debiased_suffix' requires texts_unsaid and has_unsaid")
                unsaid = self.debiased_unsaid_branch(
                    unsaid_scores, A_own, patch_features, self.clip.encode_text(texts_unsaid),
                    has_unsaid, scale, tau_unsaid=tau_unsaid,
                    gate_floor=unsaid_gate_floor, gate_temperature=unsaid_gate_temperature,
                    beta=unsaid_suppression_beta,
                    candidate_chunk_size=unsaid_candidate_chunk_size,
                )
            else:
                unsaid = self.unsaid_branch(
                    unsaid_scores, A_own, patch_features, z_g, z_s_own,
                    tau_unsaid=tau_unsaid, unsaid_residual_eps=unsaid_residual_eps,
                )
            loss_unsaid = unsaid['loss_unsaid']
            loss_total = (lambda_global * loss_global + lambda_said * loss_said
                          + float(lambda_unsaid) * loss_unsaid)
        else:
            unsaid = None
            loss_unsaid = zero
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
                'router_input_feature_norm': patch_features.detach().float().norm(dim=-1).mean(),
                'global_feature_norm': z_g.detach().float().norm(dim=-1).mean(),
                'unsaid_enabled': unsaid_enabled,
                'loss_unsaid': loss_unsaid,
            }
            # disabled branch: None (never a fabricated measurement)
            for key in unsaid_core.DIAGNOSTIC_KEYS:
                out[key] = None if unsaid is None else unsaid.get(key)
            out.update(batch_representation_gaps(z_g, z_s_own, t))
            # Phase 2.9B explicit gap diagnostics (1 - cosine, detached, never in a loss);
            # the legacy ``pair_gap_*`` fields stay untouched.
            def _gap(a, b):
                return (1.0 - (F.normalize(a.float(), dim=-1)
                               * F.normalize(b.float(), dim=-1)).sum(dim=-1)).mean()

            out['gap_global_to_said_text'] = _gap(z_g, t)
            out['gap_said_to_said_text'] = _gap(z_s_own, t)
            out['gap_global_to_full_text'] = (
                _gap(z_g, t_global) if global_caption_view == 'full' else None)
        return out

    # ------------------------------------------------------------------ #
    # Phase 2.9A: debiased suffix Unsaid (hidden-semantic positive attention)
    # ------------------------------------------------------------------ #
    def debiased_unsaid_branch(self, scores_said, attention_said, patch_features, text_unsaid,
                               has_unsaid, scale, tau_unsaid=unsaid_core.DEFAULT_TAU_UNSAID,
                               gate_floor=unsaid_core.DEFAULT_GATE_FLOOR,
                               gate_temperature=unsaid_core.DEFAULT_GATE_TEMPERATURE,
                               beta=unsaid_core.DEFAULT_SUPPRESSION_BETA,
                               candidate_chunk_size: int = 0) -> Dict[str, torch.Tensor]:
        """Unsaid alignment on the *withheld suffix sentence* of each sample.

        ``gate`` comes from the sample's own prefix Said scores and is detached; the
        direction of Unsaid attention is decided by the positive hidden-semantic score
        ``r^U`` (same ``W_q`` / ``W_k`` as Said, no new projection). Only samples with
        ``has_unsaid`` take part; a valid batch below 2 returns a differentiable zero.

        Phase 2.9B: the valid filter runs *before* any pairwise interaction, and the
        candidate dimension can be chunked (``candidate_chunk_size > 0``) so that the
        ``[B_u, B_u, P]`` tensor never exists. Diagnostics keep only the own-positive
        attention rows ``A^U_{i,i}`` ([B_u, P]).
        """
        gate = unsaid_core.said_suppression_gate(scores_said, gate_floor, gate_temperature)
        index = torch.nonzero(has_unsaid.reshape(-1).bool(), as_tuple=False).flatten()
        valid_batch = int(index.numel())
        empty = {key: None for key in unsaid_core.DIAGNOSTIC_KEYS}
        empty.update({'loss_unsaid': patch_features.sum() * 0.0, 'unsaid_valid_batch_size': valid_batch,
                      'unsaid_valid_ratio': has_unsaid.float().mean().detach()})
        if valid_batch < 2:
            return empty
        step = valid_batch if not candidate_chunk_size else min(int(candidate_chunk_size), valid_batch)
        if step <= 0:
            raise ValueError('candidate_chunk_size must be >= 0, got %r' % (candidate_chunk_size,))

        sub_patches = patch_features[index]                                      # [Bu, P, D]
        sub_gate = gate[index]
        sub_attention_said = attention_said[index]
        sub_text = F.normalize(text_unsaid[index], dim=-1)
        k_hidden = F.normalize(self.said_router.k_proj(sub_patches), dim=-1)      # [Bu, P, D]
        q_hidden = F.normalize(self.said_router.q_proj(text_unsaid[index]), dim=-1)  # [Bu, D]

        rows = torch.arange(valid_batch, device=sub_patches.device)
        score_chunks, own_attention, own_raw_attention = [], [], []
        for start in range(0, valid_batch, step):
            stop = min(start + step, valid_batch)
            logits = torch.einsum('cd,ipd->icp', q_hidden[start:stop], k_hidden)  # [Bu, C, P]
            attention = unsaid_core.debiased_unsaid_attention(logits, sub_gate, tau_unsaid, beta)
            pooled = torch.einsum('icp,ipd->icd', attention, sub_patches)
            z_chunk = F.normalize(pooled, dim=-1)
            score_chunks.append(scale * torch.einsum('icd,cd->ic', z_chunk, sub_text[start:stop]))
            inside = (rows >= start) & (rows < stop)      # diagonal candidate inside this chunk
            if bool(inside.any()):
                local = rows[inside] - start
                own_attention.append(attention[inside, local])
                own_raw_attention.append(
                    torch.softmax(logits / float(tau_unsaid), dim=-1)[inside, local])
        pair_scores = torch.cat(score_chunks, dim=1)                              # [Bu, Bu]
        A_own_pair = torch.cat(own_attention, dim=0)
        A_raw_pair = torch.cat(own_raw_attention, dim=0)
        retrieval = unsaid_core.pairwise_retrieval_loss(pair_scores)

        with torch.no_grad():
            entropy_said = unsaid_core.attention_entropy(sub_attention_said)
            entropy_unsaid = unsaid_core.attention_entropy(A_own_pair)
            diagnostics = {
                'unsaid_valid_ratio': has_unsaid.float().mean().detach(),
                'unsaid_valid_batch_size': torch.tensor(float(valid_batch), device=pair_scores.device),
                'unsaid_retrieval_top1_i2t': retrieval['top1_i2t'],
                'unsaid_retrieval_top1_t2i': retrieval['top1_t2i'],
                'unsaid_retrieval_margin_i2t': retrieval['margin_i2t'],
                'unsaid_retrieval_margin_t2i': retrieval['margin_t2i'],
                'unsaid_attention_entropy': entropy_unsaid.mean(),
                'unsaid_effective_patch_count': entropy_unsaid.exp().mean(),
                'unsaid_attention_max': A_own_pair.max(dim=-1).values.mean(),
                'said_attention_entropy': entropy_said.mean(),
                'said_effective_patch_count': entropy_said.exp().mean(),
                'said_unsaid_attention_overlap':
                    unsaid_core.attention_overlap(sub_attention_said, A_own_pair).mean(),
                'said_unsaid_attention_jsd':
                    unsaid_core.attention_jsd(sub_attention_said, A_own_pair).mean(),
                'unsaid_gate_mean': sub_gate.mean(),
                'unsaid_gate_min': sub_gate.min(),
                'unsaid_gate_max': sub_gate.max(),
                'said_coverage_under_raw': unsaid_core.said_coverage(A_raw_pair, sub_gate).mean(),
                'said_coverage_under_gated': unsaid_core.said_coverage(A_own_pair, sub_gate).mean(),
            }
        return {'loss_unsaid': retrieval['loss'], **diagnostics}

    # ------------------------------------------------------------------ #
    # scoring APIs (Phase 2.9A); standard encode_image / encode_text unchanged
    # ------------------------------------------------------------------ #
    def score_said_conditioned(self, patch_features: torch.Tensor,
                               texts: torch.Tensor) -> torch.Tensor:
        """[B, C] Said score matrix: image ``i`` routed by caption ``j``, scored on ``t_j``."""
        z_pair, _ = self.said_router.route_pairwise(texts, patch_features,
                                                    chunk_size=self.pair_chunk_size)
        scale = self.clip.logit_scale.exp().clamp(max=100)
        return scale * torch.einsum('ijd,jd->ij', z_pair, F.normalize(texts, dim=-1))

    def score_unsaid_candidates(self, patch_features: torch.Tensor, prefix_texts: torch.Tensor,
                               candidate_texts: torch.Tensor,
                               tau_unsaid: float = unsaid_core.DEFAULT_TAU_UNSAID,
                               gate_floor: float = unsaid_core.DEFAULT_GATE_FLOOR,
                               gate_temperature: float = unsaid_core.DEFAULT_GATE_TEMPERATURE,
                               beta: float = unsaid_core.DEFAULT_SUPPRESSION_BETA,
                               return_details: bool = False):
        """[B, C] pair scores for (image i, candidate missing text j).

        The gate is built from each image's own prefix (``prefix_texts``); the attention
        direction comes from the positive hidden-semantic score against the candidates.
        Returns the score matrix, or ``{'scores', 'attention', 'gate', ...}`` when
        ``return_details`` is set.
        """
        with torch.no_grad():
            details = self.said_router.forward_with_details(
                F.normalize(prefix_texts, dim=-1), patch_features)
        gate = unsaid_core.said_suppression_gate(details['scores'], gate_floor, gate_temperature)
        q_hidden = F.normalize(self.said_router.q_proj(F.normalize(candidate_texts, dim=-1)), dim=-1)
        k_hidden = F.normalize(self.said_router.k_proj(patch_features), dim=-1)
        hidden_logits = torch.einsum('jd,ipd->ijp', q_hidden, k_hidden)
        attention = unsaid_core.debiased_unsaid_attention(hidden_logits, gate, tau_unsaid, beta)
        z_unsaid = F.normalize(torch.einsum('ijp,ipd->ijd', attention, patch_features), dim=-1)
        scale = self.clip.logit_scale.exp().clamp(max=100)
        scores = scale * torch.einsum('ijd,jd->ij', z_unsaid,
                                      F.normalize(candidate_texts, dim=-1))
        if not return_details:
            return scores
        return {'scores': scores, 'attention': attention, 'gate': gate,
                'hidden_logits': hidden_logits}

    def unsaid_branch(
        self,
        scores: torch.Tensor,
        attention_said: torch.Tensor,
        patch_features: torch.Tensor,
        z_global: torch.Tensor,
        z_said_own: torch.Tensor,
        tau_unsaid: float = unsaid_core.DEFAULT_TAU_UNSAID,
        unsaid_residual_eps: float = unsaid_core.DEFAULT_RESIDUAL_EPS,
    ) -> Dict[str, torch.Tensor]:
        """Minimal Unsaid V1: complement routing + complement-target regression.

        Uses only quantities that already exist in the Said path (raw scores ``s``,
        the same patch features ``H``, ``z_G``, ``z_S``); adds no parameter. Only the
        own caption is used -- there is no ``Unsaid(I_i, C_j)`` pairwise variant and
        no B x B Unsaid loss.
        """
        A_u = unsaid_core.complement_attention(scores, tau_unsaid)
        z_u = unsaid_core.complement_representation(A_u, patch_features)
        target, residual_norm, valid = unsaid_core.complement_target(
            z_global, z_said_own, unsaid_residual_eps
        )
        loss_unsaid = unsaid_core.complement_loss(z_u, target, valid)

        with torch.no_grad():
            A_u_detached = A_u.detach().float()
            A_s_detached = attention_said.detach().float()
            z_u_detached = F.normalize(z_u.detach().float(), dim=-1)
            z_s_detached = F.normalize(z_said_own.detach().float(), dim=-1)
            entropy = unsaid_core.attention_entropy(A_u_detached)
            cosine = (z_u_detached * target).sum(dim=-1)
            diagnostics = {
                'unsaid_valid_ratio': valid.float().mean(),
                'unsaid_residual_norm_mean': unsaid_core.masked_mean(residual_norm, valid),
                'unsaid_target_cosine': unsaid_core.masked_mean(cosine, valid),
                'said_unsaid_cosine': (z_s_detached * z_u_detached).sum(dim=-1).mean(),
                'unsaid_attention_entropy': entropy.mean(),
                'unsaid_effective_patch_count': entropy.exp().mean(),
                'unsaid_attention_max': A_u_detached.max(dim=-1).values.mean(),
                'said_unsaid_attention_overlap':
                    unsaid_core.attention_overlap(A_s_detached, A_u_detached).mean(),
                'said_unsaid_attention_jsd':
                    unsaid_core.attention_jsd(A_s_detached, A_u_detached).mean(),
            }
        return {'loss_unsaid': loss_unsaid, **diagnostics}

    def forward(self, images, texts, lambda_global: float = 1.0, lambda_said: float = 1.0,
                lambda_unsaid: float = 0.0, tau_unsaid: float = unsaid_core.DEFAULT_TAU_UNSAID,
                unsaid_residual_eps: float = unsaid_core.DEFAULT_RESIDUAL_EPS, **kwargs):
        return self.forward_train(
            images, texts, lambda_global=lambda_global, lambda_said=lambda_said,
            lambda_unsaid=lambda_unsaid, tau_unsaid=tau_unsaid,
            unsaid_residual_eps=unsaid_residual_eps, **kwargs,
        )

    def extra_repr(self) -> str:
        return 'tau_said=%g, said_loss_mode=%s, pair_chunk_size=%s, said_feature_source=%s' % (
            self.tau_said, self.said_loss_mode, self.pair_chunk_size, self.said_feature_source)
