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

Phase 3.0A adds a second objective, selected by ``objective_mode='gap_completion'``
(default ``'legacy'``): **Target-Independent Unsaid Semantic Supervision**, whose base
form is **Said-Conditioned Visual Complement Discovery**. The training input is only
``Image I`` + observed/incomplete caption ``C_S``: the gap path never receives
``texts_full``, ``texts_unsaid`` or any candidate text, and the unsaid text ``C_U``
never participates in visual representation construction. The gap path is a completely
separate method (``_forward_gap_completion``) that re-runs the *same* reviewed math of
``model/gap_completion.py`` --:

    A_U = soft_anti_said_attention(s^S, temperature=gap_anti_temperature)     (detached)
    z_U = normalize(sum_p A_U_p h_p)
    L_gap_discover  = mean(gap_after)                     (direct graph gradient via z_U)
    L_global_absorb = mean(1 - cos(normalize(g_live), normalize(s_ref + u_new).detach()))
    L_total = lambda_said * L_S + lambda_gap_discover * L_gap + lambda_global_absorb * L_absorb

-- and never calls ``unsaid_branch``, ``debiased_unsaid_branch`` or
``_debiased_pair_scores``. With ``objective_mode='legacy'`` no gap code runs at all.

The gap payload reports the semantics explicitly: ``visual_complement_enabled=True``,
``legacy_unsaid_enabled=False``, ``global_text_alignment_enabled=False``. The legacy
``unsaid_enabled`` key survives only for old logger compatibility and must not be read as
"the Phase 3 visual complement is on".
"""
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .salu_modules import SaidRouter
from .representation_metrics import batch_representation_gaps
from . import unsaid_core
# Phase 3.0A.1c: pure complement-emergence diagnostics (never part of any loss).
from . import complement_diagnostics

# Phase 3.0A: reviewed parameter-free Gap Completion math (never re-implemented here).
from .gap_completion import (
    gap_completion_terms,
    gap_diagnostics,
    gap_discovery_loss,
    global_absorption_loss,
    soft_anti_said_attention,
    unsaid_feature_from_attention,
)

# Phase 3.0A.2: SAID-ExGAP (explanatory-gap guided masked representation learning).
from . import exgap
# SAID-ExGAP v1.5: pre-final evidence routing read out by the original final block.
from . import final_cls_routing

OBJECTIVE_MODES = ('legacy', 'gap_completion', 'said_exgap', 'said_exgap_finalcls')


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


def gradient_route_report(model, loss, measured=None):
    """Deterministic description of which variables SAID-ExGAP's loss is allowed to move.

    Returned as plain data so it can be logged and asserted in tests: ``detached`` lists the
    quantities that are stop-gradient by construction, ``flows_to`` describes the only direct
    graph path, and ``measured_grads`` records what the last backward actually produced.
    """
    return {
        'loss': float(loss.detach()) if torch.is_tensor(loss) else loss,
        'detached': ['gap_weight (G_exp)', 's_gc_reference', 't_gap (text embedding)', 'mask M'],
        'flows_to': 'z_U -> masked surviving patches -> shared visual backbone',
        'note': ("in attention mode A^G also appears inside z_U, so the global pooling agent is "
                 'a legitimate ExGAP target; the Said router never is'),
        'never_flows_to': ['said_router (the gap cannot be shrunk by re-routing)',
                           'text encoder', 'global CLS/global-feature reference'],
        'measured_grads': measured or {},
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

        # ---------------------------------------------------------------- #
        # SAID-ExGAP: image-only global attention pooling, built lazily by the said_exgap
        # forward so that legacy / gap_completion checkpoints keep their exact state dict.
        # The reference is stored through ``object.__setattr__`` on purpose: a plain
        # ``self._exgap_module = module`` would ALSO register the module under that private
        # name, registering it twice and making ``named_modules()`` de-duplicate the
        # canonical ``exgap`` entry away.
        # ---------------------------------------------------------------- #
        object.__setattr__(self, '_exgap_module', None)
        # SAID-ExGAP v1.5: the CLS-only readout of the ORIGINAL final visual block. It owns no
        # parameter (it only references the vision tower), so it is cached the same way and is
        # never registered as a submodule.
        object.__setattr__(self, '_final_cls_readout', None)

    def build_exgap_module(self):
        """Register (idempotently) the image-only attention pooling agent and return it.

        Called by the training script **before** DDP wrapping whenever the objective is
        ``said_exgap``, so that the agent is part of the process group and its parameters are
        identical across ranks and across the pooling-mode arms. Creating it lazily inside the
        first forward instead would leave it on the CPU (device mismatch) and outside DDP's
        reducer (silently unsynchronised gradients).
        """
        module = object.__getattribute__(self, '_exgap_module')
        if module is None:
            reference = next(self.said_router.parameters())
            module = exgap.ExGapModule(dim=int(self.said_router.dim)).to(
                device=reference.device, dtype=reference.dtype)
            self.add_module('exgap', module)
            object.__setattr__(self, '_exgap_module', module)
        return module

    def exgap_module(self, global_pool: str):
        """Return the image-only attention pooling agent for ``global_pool='attention'``.

        Only ``global_pool='attention'`` needs parameters; ``'mean'`` is parameter-free and
        returns ``None``. The agent is registered under the fixed submodule name ``exgap`` so the
        checkpoint is self-describing and ``state_dict`` / ``named_parameters`` see it exactly
        once.
        """
        if global_pool != 'attention':
            return None
        return self.build_exgap_module()

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

    def encode_exgap_global(self, images: torch.Tensor, global_pool: str = 'mean'):
        """SAID-ExGAP primary image representation ``z_G = Pool(H)`` -> ``[B, D]``.

        This is the *same* function the training forward calls, not a second implementation: the
        patch set comes from ``encode_router_input`` -- exactly as in ``_forward_said_exgap`` --
        and the pooling is ``exgap.compute_global_representation``, so an evaluator that reads
        ``encode_exgap_global`` cannot drift away from the representation the objective trained.

        ``mean`` is parameter-free (``z_G = Norm(mean_p h_p)``); ``attention`` uses the
        image-only attention pooling agent and therefore needs that agent to exist. ``z_G`` never
        sees a caption, so the image side of standard retrieval stays precomputable.
        """
        _, patch_features = self.encode_router_input(images)
        h = exgap.normalize_patches(patch_features)
        pooling = self.exgap_module(global_pool)
        if pooling is None:
            z_g, _ = exgap.compute_global_representation(h, pool='mean')
        else:
            z_g, _ = exgap.compute_global_representation(
                h, pool='attention',
                query=pooling.query.reshape(1, -1).expand(images.shape[0], -1),
                w_query=pooling.w_query, w_key=pooling.w_key)
        return z_g

    @property
    def logit_scale(self) -> torch.Tensor:
        return self.clip.logit_scale

    def said_head_parameters(self):
        """Parameters of the Said router plus the SAID-ExGAP pooling agent (optimizer group B).

        The ExGAP pooling agent exists only in ``said_exgap`` runs (it is registered on demand by
        ``build_exgap_module``), so this group is unchanged for every other objective. It is put
        here rather than in the backbone group because it is a new *head* of the objective and
        has to move at the head learning rate.
        """
        parameters = list(self.said_router.parameters())
        module = object.__getattribute__(self, '_exgap_module')
        if module is not None:
            parameters.extend(module.parameters())
        return parameters

    def backbone_parameters(self):
        """Trainable parameters outside the objective head (optimizer group A).

        Excludes every parameter of ``said_head_parameters()`` -- the Said router *and* the
        SAID-ExGAP pooling agent when it exists -- so that no parameter is ever placed in both
        optimizer groups (which would update it twice per step).
        """
        head_ids = {id(p) for p in self.said_head_parameters()}
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
        objective_mode: str = 'legacy',
        lambda_gap_discover: float = 1.0,
        lambda_global_absorb: float = 1.0,
        gap_anti_temperature: float = 1.0,
        lambda_exgap: float = 1.0,
        exgap_global_pool: str = 'mean',
        exgap_mask_threshold: float = exgap.DEFAULT_MASK_THRESHOLD,
        exgap_temperature: float = exgap.DEFAULT_GAP_TEMPERATURE,
        exgap_normalize_gap: bool = True,
        finalcls_pair_chunk_size: int = 32,
        collapse_cohort: Optional[torch.Tensor] = None,
        collapse_cohort_texts: Optional[torch.Tensor] = None,
        global_identity_check: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # Phase 3.0A: a new keyword keeps every existing positional call site valid.
        if objective_mode not in OBJECTIVE_MODES:
            raise ValueError('objective_mode must be one of %r, got %r'
                             % (OBJECTIVE_MODES, objective_mode))
        if objective_mode == 'gap_completion':
            # The gap objective is only Image I + C_S; the legacy global / Unsaid terms do
            # not exist in it, so a non-zero weight is a configuration error, not a no-op.
            if float(lambda_global) != 0.0:
                raise ValueError("objective_mode='gap_completion' requires lambda_global == 0, got %r"
                                 % (lambda_global,))
            if float(lambda_unsaid) != 0.0:
                raise ValueError("objective_mode='gap_completion' requires lambda_unsaid == 0, got %r"
                                 % (lambda_unsaid,))
            if global_caption_view != 'prefix':
                raise ValueError("objective_mode='gap_completion' requires global_caption_view == 'prefix', got %r"
                                 % (global_caption_view,))
            # Dispatched *before* any legacy code: no global contrastive loss, no Unsaid
            # branch, and ``encode_text`` runs exactly once (on C_S).
            return self._forward_gap_completion(
                images, texts,
                lambda_said=lambda_said,
                lambda_gap_discover=lambda_gap_discover,
                lambda_global_absorb=lambda_global_absorb,
                gap_anti_temperature=gap_anti_temperature,
            )
        if objective_mode == 'said_exgap_finalcls':
            # SAID-ExGAP v1.5 reads the SAME native final-block CLS for Global / Said / Unsaid and
            # has no global-text CLIP loss and no Phase 2.x Unsaid branch, so those weights are
            # configuration errors rather than no-ops.
            if float(lambda_global) != 0.0:
                raise ValueError("objective_mode='said_exgap_finalcls' requires lambda_global == 0, "
                                 'got %r' % (lambda_global,))
            if float(lambda_unsaid) != 0.0:
                raise ValueError("objective_mode='said_exgap_finalcls' requires lambda_unsaid == 0, "
                                 'got %r' % (lambda_unsaid,))
            if global_caption_view != 'prefix':
                raise ValueError("objective_mode='said_exgap_finalcls' requires "
                                 "global_caption_view == 'prefix', got %r" % (global_caption_view,))
            return self._forward_said_exgap_finalcls(
                images, texts,
                lambda_said=lambda_said,
                lambda_exgap=lambda_exgap,
                exgap_temperature=exgap_temperature,
                exgap_normalize_gap=exgap_normalize_gap,
                finalcls_pair_chunk_size=finalcls_pair_chunk_size,
                collapse_cohort=collapse_cohort,
                collapse_cohort_texts=collapse_cohort_texts,
                global_identity_check=global_identity_check,
            )
        if objective_mode == 'said_exgap':
            # SAID-ExGAP v1 reads Global / Said / Masked-Unsaid from the SAME patch set and has
            # no global-text CLIP loss and no Phase 2.x Unsaid branch, so those weights are
            # configuration errors rather than no-ops.
            if float(lambda_global) != 0.0:
                raise ValueError("objective_mode='said_exgap' requires lambda_global == 0, got %r"
                                 % (lambda_global,))
            if float(lambda_unsaid) != 0.0:
                raise ValueError("objective_mode='said_exgap' requires lambda_unsaid == 0, got %r"
                                 % (lambda_unsaid,))
            if global_caption_view != 'prefix':
                raise ValueError("objective_mode='said_exgap' requires global_caption_view == "
                                 "'prefix', got %r" % (global_caption_view,))
            return self._forward_said_exgap(
                images, texts,
                lambda_said=lambda_said,
                lambda_exgap=lambda_exgap,
                exgap_global_pool=exgap_global_pool,
                exgap_mask_threshold=exgap_mask_threshold,
                exgap_temperature=exgap_temperature,
                exgap_normalize_gap=exgap_normalize_gap,
                collapse_cohort=collapse_cohort,
            )

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
        rows = torch.arange(valid_batch, device=sub_patches.device)
        pair_scores, A_own_pair, A_raw_pair = self._debiased_pair_scores(
            sub_patches, text_unsaid[index], sub_gate, scale, tau_unsaid=tau_unsaid, beta=beta,
            candidate_chunk_size=candidate_chunk_size, own_rows=rows)
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

    def _debiased_pair_scores(self, patch_features, candidate_texts, gate, scale, *,
                              tau_unsaid=unsaid_core.DEFAULT_TAU_UNSAID,
                              beta=unsaid_core.DEFAULT_SUPPRESSION_BETA,
                              candidate_chunk_size: int = 0, own_rows=None):
        """Shared debiased-unsaid pairwise scoring used by training *and* inference.

        ``patch_features`` [I, P, D], ``candidate_texts`` [C, D], ``gate`` [I, P].
        Returns ``(scores [I, C], own_attention, own_raw_attention)``; the two attention
        outputs are ``None`` unless ``own_rows`` is given, in which case only the
        own-positive rows ``A^U_{i, own_rows[i]}`` are collected as ``[I, P]``.

        The candidate dimension is chunked when ``candidate_chunk_size > 0``; the full
        ``[I, C, P]`` attention is never materialised. ``candidate_chunk_size < 0``
        raises ``ValueError``.
        """
        n_candidates = int(candidate_texts.shape[0])
        step = n_candidates if not candidate_chunk_size else min(int(candidate_chunk_size),
                                                                 n_candidates)
        if int(candidate_chunk_size) < 0 or step <= 0:
            raise ValueError('candidate_chunk_size must be >= 0, got %r'
                             % (candidate_chunk_size,))
        k_hidden = F.normalize(self.said_router.k_proj(patch_features), dim=-1)   # [I, P, D]
        # Phase 2.9B.1a: the hidden text goes through the *same* pipeline as Said --
        # t_U = normalize(E_T(U)) first, then q_U = normalize(W_q t_U). The frozen
        # algorithm spec normalises before the shared affine projection, so the raw text
        # feature must never be fed to q_proj directly.
        candidate_unit = F.normalize(candidate_texts, dim=-1)                    # [C, D]
        q_hidden = F.normalize(self.said_router.q_proj(candidate_unit), dim=-1)  # [C, D]
        rows = torch.arange(patch_features.shape[0], device=patch_features.device)
        score_chunks, own_attention, own_raw_attention = [], [], []
        for start in range(0, n_candidates, step):
            stop = min(start + step, n_candidates)
            logits = torch.einsum('cd,ipd->icp', q_hidden[start:stop], k_hidden)  # [I, Cc, P]
            attention = unsaid_core.debiased_unsaid_attention(logits, gate, tau_unsaid, beta)
            pooled = torch.einsum('icp,ipd->icd', attention, patch_features)
            z_chunk = F.normalize(pooled, dim=-1)
            score_chunks.append(scale * torch.einsum('icd,cd->ic', z_chunk,
                                                     candidate_unit[start:stop]))
            if own_rows is not None:
                inside = (own_rows >= start) & (own_rows < stop)
                if bool(inside.any()):
                    local = own_rows[inside] - start
                    own_attention.append(attention[inside, local])
                    own_raw_attention.append(
                        torch.softmax(logits / float(tau_unsaid), dim=-1)[inside, local])
        scores = torch.cat(score_chunks, dim=1)
        if own_rows is None:
            return scores, None, None
        return scores, torch.cat(own_attention, dim=0), torch.cat(own_raw_attention, dim=0)

    def score_unsaid_candidates(self, patch_features: torch.Tensor, prefix_texts: torch.Tensor,
                               candidate_texts: torch.Tensor,
                               tau_unsaid: float = unsaid_core.DEFAULT_TAU_UNSAID,
                               gate_floor: float = unsaid_core.DEFAULT_GATE_FLOOR,
                               gate_temperature: float = unsaid_core.DEFAULT_GATE_TEMPERATURE,
                               beta: float = unsaid_core.DEFAULT_SUPPRESSION_BETA,
                               candidate_chunk_size: int = 0,
                               return_details: bool = False):
        """[B, C] pair scores for (image i, candidate missing text j).

        ``candidate_chunk_size == 0`` keeps the Phase 2.9A unchunked behaviour, including
        the legacy ``return_details`` payload (``scores`` / ``attention`` /
        ``hidden_logits`` / ``gate``). With ``candidate_chunk_size > 0`` the candidate
        dimension is chunked and the full ``[B, C, P]`` attention is **never** built:
        details then contain ``scores``, ``gate``, ``mode``/``n_chunks`` and — only when
        ``B == C`` (candidate j belongs to image j, so the identity is well defined) — the
        own-positive attention rows ``[B, P]``. No fabricated substitutes are returned.
        """
        with torch.no_grad():
            details = self.said_router.forward_with_details(
                F.normalize(prefix_texts, dim=-1), patch_features)
        gate = unsaid_core.said_suppression_gate(details['scores'], gate_floor, gate_temperature)
        scale = self.clip.logit_scale.exp().clamp(max=100)
        chunked = bool(candidate_chunk_size)
        own_rows = (torch.arange(patch_features.shape[0], device=patch_features.device)
                    if chunked and int(patch_features.shape[0]) == int(candidate_texts.shape[0])
                    else None)
        # Phase 2.9B.1: training and inference share one scoring helper (no drift). The
        # Phase 2.9A API also normalised the candidate text *before* ``q_proj``, which the
        # training path never did; that pre-normalisation is dropped here so both paths
        # use the same frozen math (`q_proj` is affine, so the two forms differ).
        scores, own_attention, own_raw_attention = self._debiased_pair_scores(
            patch_features, candidate_texts, gate, scale, tau_unsaid=tau_unsaid, beta=beta,
            candidate_chunk_size=candidate_chunk_size, own_rows=own_rows)
        if not return_details:
            return scores
        if not chunked:
            # legacy Phase 2.9A payload (unchunked: the full attention is affordable),
            # computed with the same normalised-text pipeline as the shared helper
            with torch.no_grad():
                q_hidden = F.normalize(self.said_router.q_proj(F.normalize(candidate_texts, dim=-1)),
                                       dim=-1)
                k_hidden = F.normalize(self.said_router.k_proj(patch_features), dim=-1)
                hidden_logits = torch.einsum('jd,ipd->ijp', q_hidden, k_hidden)
                attention = unsaid_core.debiased_unsaid_attention(hidden_logits, gate,
                                                                  tau_unsaid, beta)
            return {'scores': scores, 'attention': attention, 'gate': gate,
                    'hidden_logits': hidden_logits, 'mode': 'unchunked'}
        payload = {'scores': scores, 'gate': gate, 'mode': 'chunked',
                   'candidate_chunk_size': int(candidate_chunk_size),
                   'n_chunks': int(-(-int(candidate_texts.shape[0]) // int(candidate_chunk_size)))}
        if own_attention is not None:
            payload['own_attention'] = own_attention
            payload['own_raw_attention'] = own_raw_attention
        else:
            payload['own_attention'] = None           # B != C: no well-defined diagonal
            payload['own_raw_attention'] = None
        return payload

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

    # ------------------------------------------------------------------ #
    # Phase 3.0A: base Gap Completion objective (Image I + C_S only)
    # ------------------------------------------------------------------ #
    def _forward_gap_completion(
        self,
        images: torch.Tensor,
        texts: torch.Tensor,
        lambda_said: float = 1.0,
        lambda_gap_discover: float = 1.0,
        lambda_global_absorb: float = 1.0,
        gap_anti_temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Said-Conditioned Visual Complement Discovery: ``(I, C_S) -> L_total``.

        ``texts`` is the tokenised observed/incomplete caption ``C_S`` and is the *only*
        text this method ever sees: there is deliberately no ``texts_full``,
        ``texts_unsaid``, ``texts_uss`` or candidate-text parameter, because the gap
        objective must be constructible without the unsaid text ``C_U``.

        Information flow (all of it conditioned on ``C_S``)::

            z_global, patches = encode_router_input(images)   # one visual pass
            t = normalize(encode_text(texts))                 # one text pass
            said_scores, A_S, z_S = forward_with_details(t, patches)
            A_U = soft_anti_said_attention(said_scores, gap_anti_temperature)   # detached
            z_U = unsaid_feature_from_attention(A_U, patches)

        Gradient description (see ``model/gap_completion.py``): ``L_gap_discover`` has a
        **direct graph gradient** through the ``z_U`` branch (``z_S`` / ``g_ref`` are
        detached) and ``L_global_absorb`` has one through the live global ``g`` branch
        (the completion target is detached). This is *not* parameter isolation: ``g``,
        ``z_S`` and ``z_U`` are all produced by the same visual backbone.
        """
        if int(images.shape[0]) < 2:
            raise ValueError('gap_completion needs batch >= 2 for the identifiable Said loss, got %d'
                             % int(images.shape[0]))
        if images.shape[0] != texts.shape[0]:
            raise ValueError('batch mismatch: %d images vs %d texts'
                             % (int(images.shape[0]), int(texts.shape[0])))
        if abs(float(lambda_said)) < 1e-12:
            raise ValueError('lambda_said must be non-zero, otherwise L_said cannot train the '
                             'Said router, got %r' % (lambda_said,))

        z_global, patch_features = self.encode_router_input(images)
        t = F.normalize(self.encode_text(texts), dim=-1)

        details = self.said_router.forward_with_details(t, patch_features)
        said_scores = details['scores']
        A_own = details['attention']
        z_s_own = details['said']

        # soft anti-Said attention: target-independent, and detached by construction.
        A_unsaid = soft_anti_said_attention(said_scores, temperature=gap_anti_temperature)
        assert A_unsaid.requires_grad is False, 'A_U must never carry gradient'

        z_unsaid = unsaid_feature_from_attention(A_unsaid, patch_features)

        terms = gap_completion_terms(z_global, z_s_own, z_unsaid)
        loss_gap_discover = gap_discovery_loss(terms)
        loss_global_absorb = global_absorption_loss(z_global, z_s_own, z_unsaid)

        scale = self.clip.logit_scale.exp().clamp(max=100)
        if self.said_loss_mode == 'identifiable':
            z_pair, _ = self.said_router.route_pairwise(
                t, patch_features, chunk_size=self.pair_chunk_size
            )
            said = identifiable_said_loss(z_pair, t, scale)
            loss_said = said['loss_said']
            loss_route = said['loss_route'].detach()
            loss_evidence = said['loss_evidence'].detach()
            route_top1 = said['route_top1_acc'].detach()
            evidence_top1 = said['evidence_top1_acc'].detach()
            route_margin = said['route_margin'].detach()
            evidence_margin = said['evidence_margin'].detach()
        else:
            # 'positive' keeps the legacy Phase 2 ablation wording: the own-caption Said
            # feature (the one the router already produced above) against the gathered
            # captions. Reuses the already-encoded C_S -- no second text encode.
            loss_said = contrastive_loss(
                gather_features_with_grad(z_s_own), gather_features_with_grad(t), scale
            )
            zero = torch.zeros((), device=loss_said.device, dtype=loss_said.dtype)
            loss_route = loss_evidence = zero
            route_top1 = evidence_top1 = zero
            route_margin = evidence_margin = zero

        loss_total = (float(lambda_said) * loss_said
                      + float(lambda_gap_discover) * loss_gap_discover
                      + float(lambda_global_absorb) * loss_global_absorb)

        z_g = F.normalize(z_global, dim=-1)

        with torch.no_grad():
            A_s_detached = A_own.detach().float()
            A_u_detached = A_unsaid.detach().float()
            entropy_said = SaidRouter.attention_entropy(A_s_detached)
            entropy_unsaid = SaidRouter.attention_entropy(A_u_detached)
            out = {
                # the legacy global (CLIP contrastive) and Unsaid terms do not exist in
                # this objective: ``None``, never a fabricated zero measurement.
                'loss_global': None,
                'loss_unsaid': None,
                # Phase 3.0A logging semantics (never overload a flag name):
                #   visual_complement_enabled -- Said-Conditioned Visual Complement
                #     Discovery is the objective that is running.
                #   legacy_unsaid_enabled -- the Phase 2.8A / 2.9A Unsaid branches are not.
                #   global_text_alignment_enabled -- no Global-text InfoNCE here.
                # ``unsaid_enabled`` is kept only for legacy logger compatibility and must
                # not be read as "is the Phase 3 visual complement on".
                'visual_complement_enabled': True,
                'legacy_unsaid_enabled': False,
                'global_text_alignment_enabled': False,
                'unsaid_enabled': False,
                'loss_said': loss_said,
                'loss_total': loss_total,
                # The two gap losses stay live tensors in the output so that the direct
                # graph gradient can be checked at this boundary, exactly as the
                # train_salu logger detaches them (``.detach()`` is a no-op on a
                # detached tensor, so both callers are safe).
                'loss_gap_discover': loss_gap_discover,
                'loss_global_absorb': loss_global_absorb,
                'loss_route': loss_route,
                'loss_evidence': loss_evidence,
                'route_top1_acc': route_top1,
                'evidence_top1_acc': evidence_top1,
                'route_margin': route_margin,
                'evidence_margin': evidence_margin,
                'said_attention_entropy': entropy_said.mean(),
                'said_effective_patch_count': entropy_said.exp().mean(),
                'said_attention_max': A_s_detached.max(dim=-1).values.mean(),
                'said_attention_min': A_s_detached.min(dim=-1).values.mean(),
                'unsaid_attention_entropy': entropy_unsaid.mean(),
                'unsaid_effective_patch_count': entropy_unsaid.exp().mean(),
                'unsaid_attention_max': A_u_detached.max(dim=-1).values.mean(),
                'said_unsaid_attention_overlap':
                    unsaid_core.attention_overlap(A_s_detached, A_u_detached).mean(),
                'said_unsaid_attention_jsd':
                    unsaid_core.attention_jsd(A_s_detached, A_u_detached).mean(),
                'global_feature_norm': z_g.detach().float().norm(dim=-1).mean(),
                'said_feature_norm': z_s_own.detach().float().norm(dim=-1).mean(),
                'unsaid_feature_norm': z_unsaid.detach().float().norm(dim=-1).mean(),
                'router_input_feature_norm': patch_features.detach().float().norm(dim=-1).mean(),
                'gap_anti_temperature': float(gap_anti_temperature),
                'objective_mode': 'gap_completion',
                'said_loss_mode': self.said_loss_mode,
                # the effective weights are reported so a zero (matched control) is explicit
                'lambda_said': float(lambda_said),
                'lambda_gap_discover': float(lambda_gap_discover),
                'lambda_global_absorb': float(lambda_global_absorb),
                'legacy_unsaid_terms': False,
            }
            # detached gap diagnostics: closure is monitoring only and never in a loss.
            for key, value in gap_diagnostics(terms).items():
                out[key] = value
            # Phase 3.0A.1c complement-emergence diagnostics. Three strictly separated
            # questions, all detached and all outside every loss:
            #   patch homogeneity  -- H3, intrinsic patch geometry (no attention involved)
            #   raw pooling        -- the magnitude/direction the normalisation hides
            #   global relation    -- cos(g, z_S) vs cos(g, z_U)
            out.update(complement_diagnostics.patch_homogeneity_metrics(patch_features,
                                                                        z_global))
            out.update(complement_diagnostics.raw_pooling_metrics(A_own, A_unsaid,
                                                                 patch_features))
            out.update(complement_diagnostics.global_relation_metrics(z_global, z_s_own,
                                                                     z_unsaid))
            # the legacy 2.9B pair-gap diagnostics keep their names and stay available.
            out.update(batch_representation_gaps(z_g, z_s_own, t))

            def _gap(a, b):
                return (1.0 - (F.normalize(a.float(), dim=-1)
                               * F.normalize(b.float(), dim=-1)).sum(dim=-1)).mean()

            out['gap_global_to_said_text'] = _gap(z_g, t)
            out['gap_said_to_said_text'] = _gap(z_s_own, t)
            # C_S is the only text view in this objective, so there is no full-caption gap.
            out['gap_global_to_full_text'] = None
        return out

    # ------------------------------------------------------------------ #
    # SAID-ExGAP v1.5: pre-final (H11) evidence routing + native final CLS readout
    # ------------------------------------------------------------------ #
    def encode_visual_prefinal(self, images: torch.Tensor):
        """``(c11_raw, patch11_raw, x11_raw)`` -- the tokens entering the final visual block."""
        return self.clip.encode_visual_prefinal(images)

    def prefinal_route_features(self, patch11_raw: torch.Tensor) -> torch.Tensor:
        """``H11_route = Norm(LN_post(H11_raw) W_V)`` -- routing features, **no new parameter**.

        The Said router already consumed ``Norm(LN_post(H12) W_V)``; v1.5 keeps exactly that
        projection and only moves its *input* to the pre-final tokens, so the routing space is the
        one the router was trained in and no second projection is introduced.
        """
        visual = self.clip.visual
        features = visual.ln_post(patch11_raw)
        if visual.proj is not None:
            features = features @ visual.proj
        return F.normalize(features, dim=-1)

    def final_cls_readout(self):
        """The CLS-only readout of the original final block (parameter-free, lazily built).

        Cached through ``object.__setattr__`` and deliberately **not** registered as a submodule:
        it owns no parameter, and registering a module that only holds a reference to the vision
        tower would duplicate every visual parameter in ``named_parameters`` / ``state_dict``.
        """
        readout = object.__getattribute__(self, '_final_cls_readout')
        if readout is None:
            readout = final_cls_routing.FinalBlockCLSReadout(self.clip.visual)
            object.__setattr__(self, '_final_cls_readout', readout)
        return readout

    def said_final_cls(
        self,
        images: torch.Tensor,
        said_texts: torch.Tensor,
        pair_chunk_size: int = 32,
        return_details: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Diagnostic inference API for v1.5: ``g`` / ``z_S`` / ``z_U`` from the native final block.

        ``g`` is read with the *unmodified* native attention, so it equals ``clip.encode_image``
        (the standard inference path is untouched; this API exists for analysis only).
        """
        pre = self.encode_visual_prefinal(images)
        route = self.prefinal_route_features(pre['patch11_raw'])
        t = F.normalize(self.encode_text(said_texts), dim=-1)
        readout = self.final_cls_readout()
        prepared = readout.prepare(pre['x11_raw'])
        g = readout.read_global(prepared)
        logits = exgap.said_relevance_logits(t, route, self.said_router.q_proj,
                                             self.said_router.k_proj, tau_said=self.tau_said)
        relevance = final_cls_routing.said_relevance_from_logits(logits)
        z_s = F.normalize(readout.read_said(prepared, relevance), dim=-1)
        z_u = F.normalize(readout.read_unsaid(prepared, relevance.detach()), dim=-1)
        g = F.normalize(g, dim=-1)
        out = {'global_feature': g, 'said_feature': z_s, 'unsaid_feature': z_u,
               'said_relevance': relevance,
               'said_logits': logits.detach()}
        if return_details:
            s_gc = (g * t).sum(dim=-1)
            s_sc = (z_s * t).sum(dim=-1)
            s_uc = (z_u * t.detach()).sum(dim=-1)
            gap = exgap.compute_explanatory_gap(s_sc, s_gc)
            out.update({'s_gc': s_gc, 's_sc': s_sc, 's_uc': s_uc,
                        'gap_raw': gap['gap_raw'], 'gap_weight': gap['gap_weight']})
        return out

    def _forward_said_exgap_finalcls(
        self,
        images: torch.Tensor,
        texts: torch.Tensor,
        lambda_said: float = 1.0,
        lambda_exgap: float = 1.0,
        exgap_temperature: float = exgap.DEFAULT_GAP_TEMPERATURE,
        exgap_normalize_gap: bool = True,
        finalcls_pair_chunk_size: int = 32,
        collapse_cohort: Optional[torch.Tensor] = None,
        collapse_cohort_texts: Optional[torch.Tensor] = None,
        global_identity_check: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """``L = lambda_said * L_S + lambda_exgap * L_ExGAP`` on ``(I, C)`` only.

        H11 decides the evidence; the **original** final block produces the representation. The
        three representations are the CLS row of the same block under three attention gates
        (native / ``log r`` / ``log(1 - sg(r))``); no patch pooling, no reconstruction, no new
        projection head, no global-text CLIP loss.
        """
        if int(images.shape[0]) < 2:
            raise ValueError('said_exgap_finalcls needs batch >= 2 for the identifiable Said '
                             'loss, got %d' % int(images.shape[0]))
        if images.shape[0] != texts.shape[0]:
            raise ValueError('batch mismatch: %d images vs %d texts'
                             % (images.shape[0], texts.shape[0]))
        if float(lambda_said) <= 0.0:
            raise ValueError('lambda_said must be positive, got %r' % (lambda_said,))

        pre = self.encode_visual_prefinal(images)
        x11 = pre['x11_raw']
        route_patches = self.prefinal_route_features(pre['patch11_raw'])
        t = F.normalize(self.encode_text(texts), dim=-1)
        readout = self.final_cls_readout()
        prepared = readout.prepare(x11)

        # the native readout must be the CLIP global feature; reported so a mismatch is visible
        g_raw = readout.read_global(prepared)
        g = F.normalize(g_raw, dim=-1)

        # own-caption relevance (LIVE: L_S trains the router through it)
        own_logits = exgap.said_relevance_logits(t, route_patches, self.said_router.q_proj,
                                                 self.said_router.k_proj,
                                                 tau_said=self.tau_said)
        own_relevance = final_cls_routing.said_relevance_from_logits(own_logits)
        z_s = F.normalize(readout.read_said(prepared, own_relevance), dim=-1)
        # complementary branch: the complement is a CONSTANT, so ExGAP cannot move the router
        relevance_detached = own_relevance.detach()
        z_u = F.normalize(readout.read_unsaid(prepared, relevance_detached), dim=-1)

        # pairwise Said readout for the unchanged identifiable objective (chunked)
        pair_relevance = final_cls_routing.pairwise_said_relevance(
            t, route_patches, self.said_router.q_proj, self.said_router.k_proj,
            tau_said=self.tau_said, chunk_size=finalcls_pair_chunk_size)
        z_pair = F.normalize(readout.read_pairwise_said(prepared, pair_relevance,
                                                        chunk_size=finalcls_pair_chunk_size),
                             dim=-1)

        s_gc = (g * t).sum(dim=-1)
        s_sc = (z_s * t).sum(dim=-1)
        # ExGAP's own similarity uses a DETACHED text embedding: the objective may shape z_U but
        # must never push the text encoder (v1's contract, unchanged).
        s_uc = (z_u * t.detach()).sum(dim=-1)

        gap = exgap.compute_explanatory_gap(s_sc, s_gc, normalize=exgap_normalize_gap)
        gap_loss = exgap.compute_exgap_loss(s_uc, s_gc, gap['gap_weight'],
                                           temperature=exgap_temperature)

        scale = self.clip.logit_scale.exp().clamp(max=100)
        said_loss = identifiable_said_loss(z_pair, t, scale)
        loss_said = said_loss['loss_said']
        loss_total = float(lambda_said) * loss_said + float(lambda_exgap) * gap_loss['loss']

        with torch.no_grad():
            relevance = own_relevance.detach().float()
            complement = (1.0 - relevance).clamp(min=0.0)
            pair_percentiles = (0.10, 0.25, 0.50, 0.75, 0.90)
            gap_weight_flat = gap['gap_weight'].detach().float()
            gap_quantiles = torch.quantile(
                gap_weight_flat,
                torch.tensor(pair_percentiles, device=gap_weight_flat.device,
                             dtype=gap_weight_flat.dtype))
            rel_quantiles = torch.quantile(
                relevance.reshape(-1),
                torch.tensor((0.10, 0.25, 0.50, 0.75, 0.90), device=relevance.device,
                             dtype=relevance.dtype))
            z_s_detached = z_s.detach().float()
            z_u_detached = z_u.detach().float()
            g_detached = g.detach().float()
            out = {
                'loss_said': loss_said,
                'loss_route': said_loss['loss_route'].detach(),
                'loss_evidence': said_loss['loss_evidence'].detach(),
                'loss_exgap': gap_loss['loss'],
                'loss_total': loss_total,
                'loss_global': None,
                'loss_unsaid': None,
                'objective_mode': 'said_exgap_finalcls',
                'said_loss_mode': self.said_loss_mode,
                'global_text_alignment_enabled': False,
                'unsaid_enabled': False,
                'finalcls_pair_chunk_size': int(finalcls_pair_chunk_size),
                'exgap_temperature': float(exgap_temperature),
                'exgap_normalize_gap': bool(exgap_normalize_gap),
                'lambda_said': float(lambda_said),
                'lambda_exgap': float(lambda_exgap),
                'route_top1_acc': said_loss['route_top1_acc'].detach(),
                'evidence_top1_acc': said_loss['evidence_top1_acc'].detach(),
                'route_margin': said_loss['route_margin'].detach(),
                'evidence_margin': said_loss['evidence_margin'].detach(),
                # three similarities of the SAME representation family (final-block CLS)
                'S_gc_mean': s_gc.detach().mean(),
                'S_sc_mean': s_sc.detach().mean(),
                'S_uc_mean': s_uc.detach().mean(),
                'S_sc_minus_S_gc_mean': (s_sc - s_gc).detach().mean(),
                'S_uc_minus_S_gc_mean': (s_uc - s_gc).detach().mean(),
                'S_gc_minus_S_uc_mean': (s_gc - s_uc).detach().mean(),
                'D_U_mean': (s_gc - s_uc).detach().mean(),
                'S_gc_p50': torch.quantile(s_gc.detach().float(), 0.5),
                'S_sc_p50': torch.quantile(s_sc.detach().float(), 0.5),
                'S_uc_p50': torch.quantile(s_uc.detach().float(), 0.5),
                # explanatory gap (formula unchanged from v1)
                'explanatory_gap_raw_mean': gap['gap_raw'].mean(),
                'explanatory_gap_norm_mean': gap['gap_weight'].mean(),
                'gap_weight_mean': gap['gap_weight'].mean(),
                'explanatory_gap_positive_fraction': (gap['gap_raw'] > 0).float().mean(),
                'gap_p10': gap_quantiles[0],
                'gap_p25': gap_quantiles[1],
                'gap_p50': gap_quantiles[2],
                'gap_p75': gap_quantiles[3],
                'gap_p90': gap_quantiles[4],
                # soft-gate diagnostics (0.6 is a DIAGNOSTIC threshold only, never a train mask)
                'said_relevance_mean': relevance.mean(),
                'said_relevance_std': relevance.std(unbiased=False),
                'said_relevance_p10': rel_quantiles[0],
                'said_relevance_p25': rel_quantiles[1],
                'said_relevance_p50': rel_quantiles[2],
                'said_relevance_p75': rel_quantiles[3],
                'said_relevance_p90': rel_quantiles[4],
                'said_fraction_gt_0_6': (relevance > 0.6).float().mean(),
                'said_fraction_gt_0_7': (relevance > 0.7).float().mean(),
                'diagnostic_threshold': 0.6,
                # soft effective patch counts (participation ratio, NOT a hard mask count)
                'said_effective_patch_count': (
                    relevance.sum(dim=-1).pow(2)
                    / (relevance.pow(2).sum(dim=-1) + exgap.DEFAULT_EPS)).mean(),
                'unsaid_effective_patch_count': (
                    complement.sum(dim=-1).pow(2)
                    / (complement.pow(2).sum(dim=-1) + exgap.DEFAULT_EPS)).mean(),
                # final CLS intervention geometry
                'said_to_global_cos': (z_s_detached * g_detached).sum(dim=-1).mean(),
                'unsaid_to_global_cos': (z_u_detached * g_detached).sum(dim=-1).mean(),
                'said_to_unsaid_cos': (z_s_detached * z_u_detached).sum(dim=-1).mean(),
                'said_minus_global_l2': (z_s_detached - g_detached).norm(dim=-1).mean(),
                'unsaid_minus_global_l2': (z_u_detached - g_detached).norm(dim=-1).mean(),
                'said_to_global_cos_std': (z_s_detached * g_detached).sum(dim=-1).std(
                    unbiased=False),
                'global_feature_norm': g_detached.norm(dim=-1).mean(),
                'said_feature_norm': z_s_detached.norm(dim=-1).mean(),
                'unsaid_feature_norm': z_u_detached.norm(dim=-1).mean(),
                'router_input_feature_norm': route_patches.detach().float().norm(dim=-1).mean(),
                'patch11_norm': pre['patch11_raw'].detach().float().norm(dim=-1).mean(),
                'exgap_valid_fraction': gap_loss['valid_fraction'],
                'loss_said_only_reference': loss_said.detach(),
                # per-sample payload
                'said_relevance': relevance,
                'said_logits': own_logits.detach().float(),
                'mask': (relevance > 0.6).float(),      # diagnostics / visualisation only
            }
            if collapse_cohort is not None:
                out.update(self._finalcls_collapse_metrics(collapse_cohort,
                                                           collapse_cohort_texts))
            if global_identity_check:
                # Gate 0: the CLS-only readout of the ORIGINAL final block must reproduce the
                # native CLIP image feature. Measured on the same batch, in the same configuration,
                # on every rank (so the graph stays rank-independent).
                native = self.encode_image(images).detach().float()
                out['global_identity_max_abs_diff'] = (
                    native - g_raw.detach().float()).abs().max()
                out['global_identity_relative'] = (
                    (native - g_raw.detach().float()).abs().max()
                    / native.abs().max().clamp(min=1e-12))
        return out

    @torch.no_grad()
    def _finalcls_collapse_metrics(self, images: torch.Tensor,
                                   texts: Optional[torch.Tensor] = None
                                   ) -> Dict[str, torch.Tensor]:
        """Native-CLS geometry on the fixed cohort (v1.5 primary representation).

        The global representation here is exactly ``clip.encode_image`` -- not a patch pool -- so
        the numbers are directly comparable with the CLIP retrieval readout. When the cohort's own
        tokenised captions are supplied, the 64-way image-to-text R@1 is computed too.
        """
        was_training = self.training
        self.eval()
        try:
            g = F.normalize(self.encode_image(images).detach().float(), dim=-1)
            gram = g @ g.t()
            off = gram[~torch.eye(gram.shape[0], dtype=torch.bool, device=gram.device)]
            metrics = {
                'global_cls_pairwise_cos': off.mean(),
                'global_cls_pairwise_cos_max': off.max(),
                'global_cls_std': g.std(dim=0).mean(),
            }
            if texts is not None and int(texts.shape[0]) == int(images.shape[0]):
                t = F.normalize(self.encode_text(texts).detach().float(), dim=-1)
                similarity = g @ t.t()
                labels = torch.arange(similarity.shape[0], device=similarity.device)
                metrics['64way_i2t_at1'] = (similarity.argmax(dim=1) == labels).float().mean()
                metrics['64way_t2i_at1'] = (similarity.argmax(dim=0) == labels).float().mean()
        finally:
            if was_training:
                self.train()
        return metrics

    # ------------------------------------------------------------------ #
    # SAID-ExGAP v1: Explanatory-Gap Guided Masked Representation Learning
    # ------------------------------------------------------------------ #
    def _forward_said_exgap(
        self,
        images: torch.Tensor,
        texts: torch.Tensor,
        lambda_said: float = 1.0,
        lambda_exgap: float = 1.0,
        exgap_global_pool: str = 'mean',
        exgap_mask_threshold: float = exgap.DEFAULT_MASK_THRESHOLD,
        exgap_temperature: float = exgap.DEFAULT_GAP_TEMPERATURE,
        exgap_normalize_gap: bool = True,
        collapse_cohort: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """``L = lambda_said * L_S + lambda_exgap * L_ExGAP`` on ``(I, C)`` only.

        Global, Said and Masked-Unsaid all read the **same** projected patch set ``H``; there is
        no CLS/patch dual space, no global-text CLIP loss, no reconstruction and no Phase 2.x
        Unsaid branch. See ``model/exgap.py`` for the math and the gradient-isolation contract.
        """
        if int(images.shape[0]) < 2:
            raise ValueError('said_exgap needs batch >= 2 for the identifiable Said loss, got %d'
                             % int(images.shape[0]))
        if images.shape[0] != texts.shape[0]:
            raise ValueError('batch mismatch: %d images vs %d texts'
                             % (int(images.shape[0]), int(texts.shape[0])))
        if float(lambda_said) <= 0.0:
            raise ValueError('lambda_said must be positive, got %r' % (lambda_said,))

        # one visual pass; the CLS output of this same call is ignored on purpose -- Global comes
        # from the patch set so that all three representations live in one space
        _, patch_features = self.encode_router_input(images)
        t = F.normalize(self.encode_text(texts), dim=-1)
        h = exgap.normalize_patches(patch_features)

        pooling = self.exgap_module(exgap_global_pool)
        if pooling is None:
            z_g, a_g = exgap.compute_global_representation(h, pool='mean')
        else:
            z_g, a_g = exgap.compute_global_representation(
                h, pool='attention', query=pooling.query.expand(images.shape[0], -1),
                w_query=pooling.w_query, w_key=pooling.w_key)

        said = exgap.compute_said_representation(t, h, self.said_router.q_proj,
                                                 self.said_router.k_proj,
                                                 tau_said=self.tau_said)
        z_s = said['said_feature']
        mask, mask_diagnostics = exgap.compute_said_mask(
            said['said_relevance'], threshold=exgap_mask_threshold)
        z_u, a_u = exgap.compute_masked_unsaid_representation(h, mask, a_g)

        s_gc = (z_g * t).sum(dim=-1)
        s_sc = (z_s * t).sum(dim=-1)
        # The cosine that L_ExGAP actually optimises uses a DETACHED text embedding: ``s_uc`` is
        # a function of both ``z_U`` and ``t``, and without this detach the objective would also
        # push the text encoder to lower the similarity instead of only shaping ``z_U`` (spec
        # section 12: ``t_gap = t.detach()``). The forward value is unchanged -- the loss is
        # numerically identical -- only the text-encoder gradient path is removed.
        t_gap = t.detach()
        s_uc = (z_u * t_gap).sum(dim=-1)

        gap = exgap.compute_explanatory_gap(s_sc, s_gc, normalize=exgap_normalize_gap)
        gap_loss = exgap.compute_exgap_loss(s_uc, s_gc, gap['gap_weight'],
                                            temperature=exgap_temperature)

        scale = self.clip.logit_scale.exp().clamp(max=100)
        z_pair, _ = self.said_router.route_pairwise(t, patch_features,
                                                    chunk_size=self.pair_chunk_size)
        said_loss = identifiable_said_loss(z_pair, t, scale)
        loss_said = said_loss['loss_said']
        loss_total = float(lambda_said) * loss_said + float(lambda_exgap) * gap_loss['loss']

        with torch.no_grad():
            relevance = said['said_relevance'].detach().float()
            attention = said['said_attention'].detach().float()
            entropy_said = SaidRouter.attention_entropy(attention)
            entropy_global = SaidRouter.attention_entropy(a_g.detach().float())
            pair_percentiles = (0.25, 0.50, 0.75, 0.90)
            gap_weight_flat = gap['gap_weight'].detach().float()
            gap_quantiles = torch.quantile(
                gap_weight_flat,
                torch.tensor(pair_percentiles, device=gap_weight_flat.device,
                             dtype=gap_weight_flat.dtype))
            out = {
                'loss_said': loss_said,
                'loss_route': said_loss['loss_route'].detach(),
                'loss_evidence': said_loss['loss_evidence'].detach(),
                'loss_exgap': gap_loss['loss'],
                'loss_total': loss_total,
                'loss_global': None,
                'loss_unsaid': None,
                'objective_mode': 'said_exgap',
                'said_loss_mode': self.said_loss_mode,
                'global_text_alignment_enabled': False,
                'unsaid_enabled': False,
                'exgap_global_pool': exgap_global_pool,
                'exgap_mask_threshold': float(exgap_mask_threshold),
                'exgap_temperature': float(exgap_temperature),
                'exgap_normalize_gap': bool(exgap_normalize_gap),
                'lambda_said': float(lambda_said),
                'lambda_exgap': float(lambda_exgap),
                'route_top1_acc': said_loss['route_top1_acc'].detach(),
                'evidence_top1_acc': said_loss['evidence_top1_acc'].detach(),
                'route_margin': said_loss['route_margin'].detach(),
                'evidence_margin': said_loss['evidence_margin'].detach(),
                # three per-sample similarities and their differences
                'S_gc_mean': s_gc.detach().mean(),
                'S_sc_mean': s_sc.detach().mean(),
                'S_uc_mean': s_uc.detach().mean(),
                'S_sc_minus_S_gc_mean': (s_sc - s_gc).detach().mean(),
                'S_uc_minus_S_gc_mean': (s_uc - s_gc).detach().mean(),
                'S_gc_minus_S_uc_mean': (s_gc - s_uc).detach().mean(),
                # Phase ExGAP-1B causal separation: D_U = S_gc - S_uc (larger = the masked
                # representation repeats the current caption less)
                'D_U_mean': (s_gc - s_uc).detach().mean(),
                'S_gc_p50': torch.quantile(s_gc.detach().float(), 0.5),
                'S_sc_p50': torch.quantile(s_sc.detach().float(), 0.5),
                'S_uc_p50': torch.quantile(s_uc.detach().float(), 0.5),
                # explanatory gap
                'explanatory_gap_raw_mean': gap['gap_raw'].mean(),
                'explanatory_gap_norm_mean': gap['gap_weight'].mean(),
                'gap_weight_mean': gap['gap_weight'].mean(),
                'explanatory_gap_positive_fraction':
                    (gap['gap_raw'] > 0).float().mean(),
                'gap_p25': gap_quantiles[0],
                'gap_p50': gap_quantiles[1],
                'gap_p75': gap_quantiles[2],
                'gap_p90': gap_quantiles[3],
                # mask / relevance monitors
                'said_relevance_mean': relevance.mean(),
                'said_relevance_std': relevance.std(unbiased=False),
                'said_attention_entropy': entropy_said.mean(),
                'said_effective_patch_count': entropy_said.exp().mean(),
                # global-pooling monitors: entropy log(N) means A^G is still exactly uniform, so
                # the attention arm has not left the mean-pooling baseline. The L1 deviation is
                # the sensitive one (it is O(delta), the entropy change is O(delta^2)).
                'global_attention_entropy': entropy_global.mean(),
                'global_effective_patch_count': entropy_global.exp().mean(),
                'global_attention_l1_deviation':
                    (a_g.detach().float()
                     - 1.0 / float(h.shape[1])).abs().sum(dim=-1).mean(),
                'global_attention_max_deviation':
                    (a_g.detach().float()
                     - 1.0 / float(h.shape[1])).abs().max(),
                'said_feature_norm': z_s.detach().float().norm(dim=-1).mean(),
                'global_feature_norm': z_g.detach().float().norm(dim=-1).mean(),
                'unsaid_feature_norm': z_u.detach().float().norm(dim=-1).mean(),
                'router_input_feature_norm': patch_features.detach().float().norm(dim=-1).mean(),
                'exgap_valid_fraction': gap_loss['valid_fraction'],
                'loss_said_only_reference': loss_said.detach(),
                # per-sample mask / relevance payload (the return contract of the module API)
                'said_relevance': relevance,
                'said_attention': attention,
                'said_logits': said['said_logits'].detach().float(),
                'mask': mask.detach().float(),
                'global_attention': a_g.detach().float(),
                'unsaid_attention': a_u.detach().float(),
            }
            for key, value in mask_diagnostics.items():
                out[key] = (value.float() if torch.is_tensor(value) and value.dtype == torch.bool
                            else value)
            # raw (pre-normalisation) magnitude of the two pools, for monitoring
            pooled_g = torch.einsum('bn,bnd->bd', a_g.detach().float(),
                                    h.detach().float())
            pooled_u = torch.einsum('bn,bnd->bd', a_u.detach().float(),
                                    h.detach().float())
            out['global_raw_pool_norm'] = pooled_g.norm(dim=-1).mean()
            out['unsaid_raw_pool_norm'] = pooled_u.norm(dim=-1).mean()
            out['patch_norm_deviation'] = (
                h.detach().float().norm(dim=-1) - exgap.EXPECTED_PATCH_NORM).abs().mean()
            if collapse_cohort is not None:
                out.update(self._exgap_collapse_metrics(collapse_cohort, exgap_global_pool))
        return out

    @torch.no_grad()
    def _exgap_collapse_metrics(self, images: torch.Tensor,
                                exgap_global_pool: str) -> Dict[str, torch.Tensor]:
        """Representation-geometry monitor on a fixed cohort (the Full-Base arm collapsed).

        Reports the mean off-diagonal cosine of ``z_G`` / ``z_S`` / ``z_U`` over the cohort. A
        rising ``global_pairwise_cos`` (say beyond 0.9) is the collapse signature and must be read
        as a warning regardless of how well the losses look.
        """
        was_training = self.training
        self.eval()
        try:
            _, patch_features = self.encode_router_input(images)
            h = exgap.normalize_patches(patch_features)
            pooling = self.exgap_module(exgap_global_pool)
            if pooling is None:
                z_g, a_g = exgap.compute_global_representation(h, pool='mean')
            else:
                z_g, a_g = exgap.compute_global_representation(
                    h, pool='attention', query=pooling.query.expand(images.shape[0], -1),
                    w_query=pooling.w_query, w_key=pooling.w_key)
            # the collapse monitor is caption-independent: it uses a fixed neutral probe text so
            # the number can be compared across steps without confounding from the batch text
            probe = torch.zeros(images.shape[0], h.shape[-1], device=h.device)
            probe[:, 0] = 1.0
            said = exgap.compute_said_representation(probe, h, self.said_router.q_proj,
                                                     self.said_router.k_proj,
                                                     tau_said=self.tau_said)
            mask, _ = exgap.compute_said_mask(said['said_relevance'],
                                              threshold=exgap.DEFAULT_MASK_THRESHOLD)
            z_u, _ = exgap.compute_masked_unsaid_representation(h, mask, a_g)
            metrics = {}
            for name, feature in (('global', z_g), ('said', said['said_feature']),
                                  ('unsaid', z_u)):
                gram = feature @ feature.t()
                off = gram[~torch.eye(gram.shape[0], dtype=torch.bool, device=gram.device)]
                metrics['%s_pairwise_cos' % name] = off.mean()
                metrics['%s_pairwise_cos_max' % name] = off.max()
                # spread of the cohort embeddings: a mean cosine near 1 can come from a genuinely
                # collapsed code or from a low-variance one, and the std separates the two
                metrics['%s_embedding_std' % name] = feature.detach().float().std(dim=0).mean()
            return metrics
        finally:
            if was_training:
                self.train()

    def encode_said_exgap(
        self,
        images: torch.Tensor,
        said_texts: torch.Tensor,
        exgap_global_pool: str = 'mean',
        exgap_mask_threshold: float = exgap.DEFAULT_MASK_THRESHOLD,
        exgap_temperature: float = exgap.DEFAULT_GAP_TEMPERATURE,
        return_details: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Inference API for SAID-ExGAP: ``(I, C) -> z_G / z_S / z_U`` and the gap terms.

        ``said_texts`` is the tokenised caption; the caption never enters the global pooling, so
        ``z_G`` is a pure function of the image and the image side stays precomputable.
        """
        _, patch_features = self.encode_router_input(images)
        t = F.normalize(self.encode_text(said_texts), dim=-1)
        h = exgap.normalize_patches(patch_features)
        pooling = self.exgap_module(exgap_global_pool)
        if pooling is None:
            z_g, a_g = exgap.compute_global_representation(h, pool='mean')
        else:
            z_g, a_g = exgap.compute_global_representation(
                h, pool='attention', query=pooling.query.expand(images.shape[0], -1),
                w_query=pooling.w_query, w_key=pooling.w_key)
        said = exgap.compute_said_representation(t, h, self.said_router.q_proj,
                                                 self.said_router.k_proj,
                                                 tau_said=self.tau_said)
        mask, _ = exgap.compute_said_mask(said['said_relevance'],
                                          threshold=exgap_mask_threshold)
        z_u, a_u = exgap.compute_masked_unsaid_representation(h, mask, a_g)
        out = {
            'global_feature': z_g,
            'said_feature': said['said_feature'],
            'unsaid_feature': z_u,
            'said_relevance': said['said_relevance'],
            'said_attention': said['said_attention'],
            'global_attention': a_g,
            'unsaid_attention': a_u,
            'mask': mask,
        }
        if return_details:
            s_gc = (z_g * t).sum(dim=-1)
            s_sc = (said['said_feature'] * t).sum(dim=-1)
            # Same contract as the training path: the similarity that L_ExGAP optimises uses a
            # DETACHED text embedding, so this inference API reports the identical quantity and
            # cannot be used to route an ExGAP gradient into the text encoder.
            s_uc = (z_u * t.detach()).sum(dim=-1)
            gap = exgap.compute_explanatory_gap(s_sc, s_gc)
            out.update({'said_logits': said['said_logits'], 's_gc': s_gc, 's_sc': s_sc,
                        's_uc': s_uc, 'gap_raw': gap['gap_raw'],
                        'gap_weight': gap['gap_weight']})
        return out

    def encode_said_unsaid(
        self,
        images: torch.Tensor,
        said_texts: torch.Tensor,
        gap_anti_temperature: float = 1.0,
        return_details: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Encode ``(I, C_S)`` into the global / Said / Unsaid features (inference API).

        ``said_texts`` is the tokenised observed/incomplete caption ``C_S``. The unsaid
        feature ``z_U`` is built **without any unsaid text**: it is the soft anti-Said
        pooling ``normalize(sum_p A_U_p h_p)`` of the image's own patches, where ``A_U``
        comes from the Said raw scores of ``C_S`` alone.

        ``gap_anti_temperature`` is the soft anti-Said temperature: a numerical
        hyper-parameter of ``A_U``, never a text input, so this API sees exactly the same
        ``A_U`` function as training. At the training default ``1.0`` the numbers match
        ``_forward_gap_completion``.

        Always returned: ``global_feature``, ``said_feature``, ``unsaid_feature``.
        With ``return_details=True`` also: ``said_scores``, ``said_attention``,
        ``unsaid_attention``, ``u_new``, ``gap_before``, ``gap_after``, ``closure``
        and ``novel_norm``.
        """
        z_global, patch_features = self.encode_router_input(images)
        t = F.normalize(self.encode_text(said_texts), dim=-1)
        details = self.said_router.forward_with_details(t, patch_features)
        z_said = details['said']
        A_unsaid = soft_anti_said_attention(details['scores'],
                                            temperature=gap_anti_temperature)

        z_unsaid = unsaid_feature_from_attention(A_unsaid, patch_features)
        out = {
            'global_feature': z_global,
            'said_feature': z_said,
            'unsaid_feature': z_unsaid,
        }
        if return_details:
            terms = gap_completion_terms(z_global, z_said, z_unsaid)
            out.update({
                'said_scores': details['scores'],
                'said_attention': details['attention'],
                'unsaid_attention': A_unsaid,
                'u_new': terms['u_new'],
                'gap_before': terms['gap_before'],
                'gap_after': terms['gap_after'],
                'closure': terms['closure'],
                'novel_norm': terms['novel_norm'],
            })
        return out

    def extra_repr(self) -> str:
        return 'tau_said=%g, said_loss_mode=%s, pair_chunk_size=%s, said_feature_source=%s' % (
            self.tau_said, self.said_loss_mode, self.pair_chunk_size, self.said_feature_source)
