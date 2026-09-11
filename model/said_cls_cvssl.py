"""SAID-CLS-CVSSL v0.1 objective: SmartCLIP's verified CLS objective + complementary visual SSL.

This module is intentionally thin and faithful:

* the image-text part is the **reference SmartCLIP forward** (``CLIP.forward``) re-expressed at the
  feature level so that one image/text/mask forward can serve both terms. ``compute_smartclip_terms``
  reproduces ``CLIP.forward`` line by line:

      mask_logits = mask_net(text_hidden.detach())
      soft        = sigmoid(mask_logits)
      hard        = (soft >= 0.5).float() - soft.detach() + soft      # default (soft_mask=0)
      Q_ij        = 100 * cos(Norm(v_i * m_j), t_j)                   # SIDM: image rows
      DISM        = the same matrix with the text row classification
      loss        = lambda_align * (SIDM + DISM) + lambda_sparse * mean|m|

  The cross-rank gather stays the reference's autograd-aware ``torch.distributed.nn.all_gather``.

* the new part is ``complement_visual_contrastive_loss`` from
  :mod:`model.complement_visual_ssl`, whose candidates are gathered **detached**.

Forbidden here on purpose (and asserted by the tests): the old ``SaidRouter`` /
``identifiable_said_loss``, ``FinalBlockCLSReadout``, H11/H12 routing, any ExGAP term, any extra U
network, any projection head, any teacher/EMA/queue, any ``C_U`` / full-caption input.
"""
from typing import Dict, Optional, Tuple

import torch
import torch.distributed.nn as nn_dist
import torch.nn as nn
import torch.nn.functional as F

from . import complement_visual_ssl as cvssl

LAMBDA_ALIGN = 10.0
LAMBDA_SPARSE = 2.0
SMARTCLIP_FIXED_SCALE = 100.0
ARMS = ('S0_smartclip', 'G0_global_vssl', 'R0_random_vssl', 'C0_complement_vssl')
ARM_MASK = {'S0_smartclip': 'none', 'G0_global_vssl': 'ones',
            'R0_random_vssl': 'random', 'C0_complement_vssl': 'complement'}
ARM_LAMBDA_U = {'S0_smartclip': 0.0, 'G0_global_vssl': 1.0,
                'R0_random_vssl': 1.0, 'C0_complement_vssl': 1.0}


def effective_lambda_u(lambda_target: float, completed_steps: int, warmup_steps: int) -> float:
    """The U weight actually applied at this update.

    ``warmup_steps = 0`` reproduces the historical constant-weight behaviour exactly. Otherwise the
    weight ramps linearly over the first ``warmup_steps`` updates: step 1
    (``completed_steps = 0``) gets ``lambda_target / warmup_steps``, and update ``warmup_steps`` and
    everything after it gets ``lambda_target``.
    """
    target = float(lambda_target)
    warmup = int(warmup_steps)
    if warmup <= 0:
        return target
    return target * min((int(completed_steps) + 1) / float(warmup), 1.0)



def said_mask_from_hidden(mask_net, text_hidden: torch.Tensor, soft_mask: bool = False
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(m_s, soft_mask_tensor, mask_logits)`` -- the reference mask pipeline, unchanged.

    ``text_hidden`` is the full ``ln_final`` hidden state of the *current* ``C_S`` tokens (the
    reference passes ``full_text_embedding.detach()``); ``soft_mask=False`` selects the
    straight-through hard mask, which is the repository default.
    """
    mask_logits = mask_net(text_hidden.detach())
    soft = torch.sigmoid(mask_logits)
    hard = (soft >= 0.5).float() - soft.detach() + soft
    return (soft if soft_mask else hard), soft, mask_logits


def compute_smartclip_terms(v_a: torch.Tensor, t_raw: torch.Tensor, m_s: torch.Tensor,
                            rank: int, lambda_align: float = LAMBDA_ALIGN,
                            lambda_sparse: float = LAMBDA_SPARSE) -> Dict[str, torch.Tensor]:
    """The reference SmartCLIP SIDM / DISM / sparsity terms from already-computed features.

    ``v_a`` is the raw (unnormalised) image embedding, ``t_raw`` the unnormalised pooled text
    embedding and ``m_s`` the mask actually used for the Said forward. The arithmetic, the fixed
    ``100`` multiplier, the gather order, the target indices and the two cross-entropies are the
    reference's; only the features are passed in instead of being recomputed.
    """
    text_features = t_raw / t_raw.norm(dim=1, keepdim=True)
    text_features_all = torch.cat(nn_dist.all_gather(text_features), dim=0)
    use_mask = m_s
    loss_sparsity = torch.mean(torch.abs(use_mask))
    use_mask_all = torch.cat(nn_dist.all_gather(use_mask), dim=0)
    bs = v_a.size(0)
    targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=torch.long).to(v_a.device)

    # SIDM: same image, one mask per candidate text, scored against that candidate text
    rep_use_mask_all_sidm = use_mask_all.repeat(bs, 1)
    rep_image_features_sidm = v_a.repeat_interleave(len(use_mask_all), dim=0)
    rep_image_features_sidm = rep_image_features_sidm * rep_use_mask_all_sidm
    rep_image_features_sidm = rep_image_features_sidm / rep_image_features_sidm.norm(
        dim=1, keepdim=True)
    rep_text_features_all = text_features_all.repeat(bs, 1)
    sidm_sim = torch.sum(rep_image_features_sidm * rep_text_features_all, dim=1)
    sidm_sim = sidm_sim.view(bs, len(use_mask_all))
    sidm_sim = SMARTCLIP_FIXED_SCALE * sidm_sim
    loss_sidm = F.cross_entropy(sidm_sim, targets)

    # DISM: same text (with its own mask), candidate images compete
    image_features_all = torch.cat(nn_dist.all_gather(v_a), dim=0)
    ns = len(use_mask_all)
    rep_use_mask_dism = use_mask.repeat_interleave(ns, dim=0)
    rep_image_features_all_dism = image_features_all.repeat(bs, 1)
    rep_image_features_dism = rep_image_features_all_dism * rep_use_mask_dism
    rep_image_features_dism = rep_image_features_dism / rep_image_features_dism.norm(
        dim=1, keepdim=True)
    rep_text_features_dism = text_features.repeat_interleave(ns, 0)
    dism_sim = torch.sum(rep_image_features_dism * rep_text_features_dism, dim=1)
    dism_sim = dism_sim.view(bs, len(use_mask_all))
    dism_sim = SMARTCLIP_FIXED_SCALE * dism_sim
    loss_dism = F.cross_entropy(dism_sim, targets)

    with torch.no_grad():
        sidm_top1 = (sidm_sim.argmax(dim=1) == targets).float().mean()
        dism_top1 = (dism_sim.argmax(dim=1) == targets).float().mean()
        keep = (m_s.detach() > 0).float()
        energy = m_s.detach().float()
        total = energy.sum(dim=1).clamp_min(1e-12)
    return {
        'loss_sidm': loss_sidm,
        'loss_dism': loss_dism,
        'loss_sparsity': loss_sparsity,
        'loss_smart': lambda_align * (loss_sidm + loss_dism) + lambda_sparse * loss_sparsity,
        'sidm_top1': sidm_top1,
        'dism_top1': dism_top1,
        'actual_global_candidate_count': sidm_sim.shape[1],
        'mask_s_keep_ratio': keep.mean(),
        'mask_s_energy_fraction': (energy * keep).sum(dim=1).div(total).mean(),
        'mask_s_count_mean': keep.sum(dim=1).mean(),
    }


class SaidClsCvsslTrainModule(nn.Module):
    """The DDP-visible training module: owns the CLIP model (registered exactly once) and runs the
    whole objective inside ``forward`` so that ``DistributedDataParallel`` synchronises every
    parameter gradient, ``mask_net`` included.

    ``ddp_gradient_averaging=True`` multiplies the CVSSL term by the world size, which is required
    because DDP averages parameter gradients over ranks: the backward value then produces the true
    global-mean U gradient. The SmartCLIP term is *not* rescaled (its reference semantics already
    all-gather the features with autograd).

    Activation checkpointing on both view encoders uses ``use_reentrant=False`` so that it stays
    compatible with DDP's reducer and with diagnostic ``autograd.grad`` graphs.
    """

    def __init__(self, clip_model, rank: int = 0, arm: str = 'C0_complement_vssl',
                 lambda_u: float = 1.0, tau_u: float = cvssl.DEFAULT_TAU_U,
                 rho: float = cvssl.DEFAULT_RHO, soft_mask: bool = False,
                 lambda_align: float = LAMBDA_ALIGN, lambda_sparse: float = LAMBDA_SPARSE,
                 duplicate_policy: str = 'exclude', ddp_gradient_averaging: bool = True,
                 eps: float = cvssl.DEFAULT_EPS, grad_checkpoint_views: bool = False):
        super().__init__()
        self.clip = clip_model                       # registered once, never duplicated
        self.objective = SaidClsCvsslObjective(
            clip_model, rank=rank, arm=arm, lambda_u=lambda_u, tau_u=tau_u, rho=rho,
            soft_mask=soft_mask, lambda_align=lambda_align, lambda_sparse=lambda_sparse,
            duplicate_policy=duplicate_policy, ddp_gradient_averaging=ddp_gradient_averaging,
            eps=eps, grad_checkpoint_views=False)     # view encoding is done here, not there
        self.grad_checkpoint_views = bool(grad_checkpoint_views)
        self.ddp_gradient_averaging = bool(ddp_gradient_averaging)

    def _encode_view(self, images: torch.Tensor) -> torch.Tensor:
        if not self.grad_checkpoint_views:
            return self.clip.encode_image(images)
        from torch.utils.checkpoint import checkpoint
        return checkpoint(self.clip.visual, images.type(self.clip.dtype),
                          use_reentrant=False, preserve_rng_state=True)

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor, text: torch.Tensor,
                image_ids: torch.Tensor, random_generator=None, mask_seed=None,
                mask_override=None, effective_lambda_u=None):
        v_a = self._encode_view(image_a)
        v_b = self._encode_view(image_b)
        return self.objective.forward_from_features(v_a, v_b, text, image_ids,
                                                    random_generator=random_generator,
                                                    image_a=image_a, image_b=image_b,
                                                    mask_seed=mask_seed,
                                                    mask_override=mask_override,
                                                    effective_lambda_u=effective_lambda_u)


class SaidClsCvsslObjective:
    """``L = L_smart + lambda_U * L_U`` on ``(I_a, I_b, C_S)`` only.

    Both views are encoded by the same online vision tower **with the graph kept**: each view is
    an anchor in one of the two directions, so wrapping the b-encoder in ``no_grad`` would remove
    the b-anchor gradient (which is the whole point of the second direction).

    The class owns no parameters; it holds the CLIP model and the two objectives' constants.
    """

    def __init__(self, clip_model, rank: int = 0, arm: str = 'C0_complement_vssl',
                 lambda_u: float = 1.0, tau_u: float = cvssl.DEFAULT_TAU_U,
                 rho: float = cvssl.DEFAULT_RHO, soft_mask: bool = False,
                 lambda_align: float = LAMBDA_ALIGN, lambda_sparse: float = LAMBDA_SPARSE,
                 duplicate_policy: str = 'exclude', ddp_gradient_averaging: bool = False,
                 eps: float = cvssl.DEFAULT_EPS, grad_checkpoint_views: bool = False):
        if arm not in ARMS:
            raise ValueError('arm must be one of %r, got %r' % (ARMS, arm))
        self.clip = clip_model
        self.rank = int(rank)
        self.arm = arm
        self.mask_kind = ARM_MASK[arm]
        self.lambda_u = float(lambda_u)
        self.tau_u = float(tau_u)
        self.rho = float(rho)
        self.soft_mask = bool(soft_mask)
        self.lambda_align = float(lambda_align)
        self.lambda_sparse = float(lambda_sparse)
        self.duplicate_policy = duplicate_policy
        self.ddp_gradient_averaging = bool(ddp_gradient_averaging)
        self.eps = float(eps)
        self.grad_checkpoint_views = bool(grad_checkpoint_views)

    # -- one training step's forward ---------------------------------------- #
    def __call__(self, *args, **kwargs):
        """Alias so the objective reads like a module without owning any parameter."""
        return self.forward(*args, **kwargs)

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor, text: torch.Tensor,
                image_ids: torch.Tensor, random_generator: Optional[torch.Generator] = None,
                mask_seed: Optional[int] = None,
                effective_lambda_u: Optional[float] = None,
                ) -> Dict[str, torch.Tensor]:
        encoder = (self.clip.encode_image_with_checkpoint if self.grad_checkpoint_views
                   else self.clip.encode_image)
        v_a = encoder(image_a)
        v_b = encoder(image_b)
        return self.forward_from_features(v_a, v_b, text, image_ids,
                                          random_generator=random_generator,
                                          image_a=image_a, image_b=image_b,
                                          mask_seed=mask_seed,
                                          effective_lambda_u=effective_lambda_u)

    def forward_from_features(self, v_a: torch.Tensor, v_b: torch.Tensor, text: torch.Tensor,
                              image_ids: torch.Tensor,
                              random_generator: Optional[torch.Generator] = None,
                              image_a: Optional[torch.Tensor] = None,
                              image_b: Optional[torch.Tensor] = None,
                              mask_seed: Optional[int] = None,
                              mask_override: Optional[torch.Tensor] = None,
                              effective_lambda_u: Optional[float] = None,
                              ) -> Dict[str, torch.Tensor]:
        """The objective on already-encoded views (so the caller can own the view encoding).

        ``image_a`` / ``image_b`` are optional and used only for the pixel-distance diagnostic, so
        a caller that has already freed the pixels can still run the objective.
        """
        t_raw, text_hidden = self.clip.encode_text(text, return_full=True)
        m_s, soft, mask_logits = said_mask_from_hidden(self.clip.mask_net, text_hidden,
                                                      soft_mask=self.soft_mask)

        smart = compute_smartclip_terms(v_a, t_raw, m_s, self.rank,
                                        lambda_align=self.lambda_align,
                                        lambda_sparse=self.lambda_sparse)
        mask_u, mask_diag = cvssl.build_arm_mask(
            m_s, self.mask_kind if self.mask_kind != 'none' else 'complement',
            rho=self.rho, generator=random_generator, sample_ids=image_ids,
            mask_seed=0 if mask_seed is None else int(mask_seed),
            override=mask_override)

        # the weight schedule only changes this scalar; which branch runs is decided by the
        # *target*, so a small early effective weight never switches the training path
        weight = self.lambda_u if effective_lambda_u is None else float(effective_lambda_u)
        if self.lambda_u > 0.0:
            u = cvssl.complement_visual_contrastive_loss(
                v_a, v_b, mask_u, image_ids, tau_u=self.tau_u, eps=self.eps, rank=self.rank,
                ddp_gradient_averaging=self.ddp_gradient_averaging,
                duplicate_policy=self.duplicate_policy)
        else:
            # S0: the identical complement is still computed for diagnostics, but there is no
            # graph and no update, so the arm is exactly the reference SmartCLIP objective.
            with torch.no_grad():
                u = cvssl.complement_visual_contrastive_loss(
                    v_a.detach(), v_b.detach(), mask_u.detach(), image_ids, tau_u=self.tau_u,
                    eps=self.eps, rank=self.rank, ddp_gradient_averaging=False,
                    duplicate_policy=self.duplicate_policy)

        loss_total = smart['loss_smart'] + weight * u['loss']
        with torch.no_grad():
            global_similarity = F.cosine_similarity(
                F.normalize(mask_u * v_a.detach(), dim=-1, eps=self.eps),
                F.normalize(v_a.detach(), dim=-1))
            unmasked_crossview_cos = F.cosine_similarity(
                F.normalize(v_a.detach(), dim=-1), F.normalize(v_b.detach(), dim=-1))
            view_pixel_distance = (((image_a.detach().float() - image_b.detach().float()
                                    ).abs().mean()) if image_a is not None and image_b is not None
                                   else v_a.new_zeros(()))
            # retained ENERGY of the image representation (features, not just mask coordinates)
            v_a_detached = v_a.detach().float()
            total_energy = v_a_detached.square().sum(dim=-1).clamp_min(self.eps)
            energy_said = (v_a_detached * m_s.detach().float()).square().sum(dim=-1) / total_energy
            energy_unsaid = (v_a_detached * mask_u.detach().float()).square().sum(
                dim=-1) / total_energy
            smart_norm = smart['loss_smart'].detach()
            u_norm = u['global_mean_loss'].detach()
            weighted_ratio = (weight * u_norm / smart_norm.clamp_min(1e-12))
        out = {
            'loss_total': loss_total,
            'loss_total_for_backward': loss_total,   # what the trainer back-propagates
            'loss': u['loss'],                     # the CVSSL term that back-propagates
            'loss_smart': smart['loss_smart'],
            'loss_smart_unweighted': smart['loss_sidm'] + smart['loss_dism'],
            'loss_sidm': smart['loss_sidm'],
            'loss_dism': smart['loss_dism'],
            'loss_sparsity': smart['loss_sparsity'],
            'loss_vssl_raw': u['global_mean_loss'],
            'loss_vssl_weighted': weight * u['loss'],
            'lambda_U_target': self.lambda_u,
            'lambda_U_effective': weight,
            'vssl_loss_scale': u['loss_scale'],
            'sidm_top1': smart['sidm_top1'],
            'dism_top1': smart['dism_top1'],
            'actual_global_candidate_count': smart['actual_global_candidate_count'],
            'vssl_ab_top1': u['vssl_ab_top1'],
            'vssl_ba_top1': u['vssl_ba_top1'],
            'positive_minus_negative_margin': u['positive_minus_negative_margin'],
            'vssl_valid_anchor_fraction': u['vssl_valid_anchor_fraction'],
            'vssl_valid_ab': u['valid_ab'],
            'vssl_valid_ba': u['valid_ba'],
            'invalid_norm_count': u['invalid_norm_count'],
            'valid_negative_count': u['valid_negative_count'],
            'duplicate_candidates': u['duplicate_candidates'],
            'mask_u_keep_ratio': mask_diag['mask_u_keep_ratio'],
            'mask_s_keep_ratio': smart['mask_s_keep_ratio'],
            'mask_s_count_mean': smart['mask_s_count_mean'],
            'said_retained_energy': energy_said.mean(),
            'said_retained_energy_p10': energy_said.quantile(0.10),
            'unsaid_retained_energy': energy_unsaid.mean(),
            'unsaid_retained_energy_p10': energy_unsaid.quantile(0.10),
            'global_sample_count': u['global_sample_count'],
            'loss_smart_global_mean': smart['loss_smart'].detach(),
            'loss_vssl_global_mean': u['global_mean_loss'].detach(),
            'loss_vssl_for_backward_local': weight * u['loss'],
            'loss_total_for_backward_local': loss_total.detach(),
            'mask_u_histogram_matches_complement':
                mask_diag['mask_value_histogram_identical_to_complement'],
            'cos_u_g': global_similarity.mean(),
            'view_pixel_distance': view_pixel_distance,
            'unmasked_crossview_cos': unmasked_crossview_cos.mean(),
            'weighted_vssl_to_smart_loss_ratio': weighted_ratio,
            'v_a': v_a,
            'v_b': v_b,
            'm_s': m_s,
            'm_u': mask_u,
            'soft_mask_mean': soft.mean(),
            'mask_logits_mean': mask_logits.mean(),
        }
        return out
