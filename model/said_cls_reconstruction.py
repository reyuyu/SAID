"""SAID-C1-TCR v0.1 -- Text-Conditioned visual feature Reconstruction.

One sentence: keep SmartCLIP's verified CLS objective exactly as it is, and add an independent
auxiliary branch that predicts the **complete** image feature produced by a *frozen copy of the
initial* vision tower, from only ``(r_U, t_cond)`` -- the complementary visual input and the text
that has actually been observed.

Frozen maths (this is the specification, not an approximation)::

    v        = E_{I,theta}(I_a)                       student visual output, raw
    g        = Norm(v)                                normalise the FULL vector first
    (t_raw,T)= E_{T,phi}(C_S)
    t        = Norm(t_raw),   t_cond = sg(t)
    p        = sigmoid(M_omega(sg(T)))                the real SmartCLIP mask network
    m_S      = (p >= 0.5).float() - p.detach() + p    hard straight-through
    m_U      = 1 - sg(m_S)                            rho = 0
    r_U      = g . m_U                                <- multiplied AFTER normalisation, r_U is
                                                         deliberately NOT re-normalised
    g0       = Norm(E_I^0(I_a))                       E_I^0 = frozen copy of the initial tower
    g_hat    = D_psi([r_U ; t_cond])
    L_rec    = (1/|V|) sum_{i in V} sum_d (g_hat_i,d - g0_i,d)^2
    L        = 10 (L_SIDM + L_DISM) + 2 L_sparse + lambda_rec L_rec

Gradient boundaries that are intended (and tested):

    L_rec -> decoder           allowed
    L_rec -> student visual    allowed
    L_rec -> text encoder      blocked (t_cond is detached)
    L_rec -> mask_net          blocked (m_U comes from a detached m_S)
    L_rec -> frozen reference  blocked (no_grad, and the reference is not optimised)

NOT claimed: that ``dL_rec/dv`` is exactly zero in the Said coordinates. ``r_U`` normalises the
complete vector *before* masking, so the L2 denominator couples the coordinates; the claim would be
false and is deliberately not asserted anywhere.

The reference visual is a deep copy of the student's *initial* tower and never shares storage with
it. It is not in any optimizer and is forced back to ``eval()`` after any ``train()`` recursion.
"""
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .said_cls_cvssl import (ARM_LAMBDA_U, ARMS, LAMBDA_ALIGN, LAMBDA_SPARSE,  # noqa: F401
                             compute_smartclip_terms, said_mask_from_hidden)

DEFAULT_TAU_REC = 1.0
DECODER_HIDDEN = 512
REFERENCE_EPS = 1e-6
VALID_NORM_EPS = 1e-6
ARM = 'C1_text_conditional_reconstruction'


def visual_embed_dim(clip_model: nn.Module) -> int:
    """The CLS embedding width, read from the real parameter rather than an assumed attribute.

    LongCLIP's ``CLIP`` (``model/model_longclip.py``) exposes ``embed_dim`` only as a constructor
    argument -- it is *not* stored on the module -- so reading ``clip.embed_dim`` works for the test
    stand-in but fails on the production model. The text projection is the authoritative shape.
    """
    projection = getattr(clip_model, 'text_projection', None)
    if projection is not None and hasattr(projection, 'shape') and len(projection.shape) == 2:
        return int(projection.shape[1])
    if hasattr(clip_model, 'embed_dim'):
        return int(clip_model.embed_dim)
    raise AttributeError('cannot determine the CLS embedding width of %r' % type(clip_model))


@contextmanager
def _isolated_rng(seed: int = 0):
    """Snapshot the ambient RNG, seed it, and restore it afterwards.

    Two things consume the ambient RNG while this module is constructed: ``copy.deepcopy`` of the
    visual tower (torch re-draws tensors for it) and the decoder's ``nn.Linear`` initialisation.
    Neither may perturb the data / caption / mask streams, so both happen inside one isolated,
    seed-restorable context. Measured behaviour of this environment, recorded so it stays checkable:
    the deepcopy draws a *variable* number of RNG values (it is not a fixed cost), which is why the
    snapshot/restore is mandatory rather than relying on a fixed draw count.
    """
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(int(seed))
        yield
    finally:
        torch.random.set_rng_state(state)


class TextConditionedReconstructor(nn.Module):
    """The C1 auxiliary head: a two-layer fusion MLP, nothing else.

    Input is exactly ``[r_U ; t_cond]`` with shape ``[B, 2D]``. There is no residual path from the
    student or the teacher feature, no dropout, no normalisation layer and no output L2
    normalisation: the head regresses the unit-length reference vector directly.
    """

    def __init__(self, dim: int, hidden: int = DECODER_HIDDEN, seed: int = 0):
        super().__init__()
        with _isolated_rng(seed):
            self.net = nn.Sequential(
                nn.Linear(2 * dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, dim),
            )

    def forward(self, r_u: torch.Tensor, t_cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([r_u, t_cond], dim=-1))


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor,
                        valid: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """``L_rec`` with the feature dim summed and the valid-sample dim averaged.

    Returning ``sum_d (pred-target)^2`` per sample and dividing by the number of valid samples is
    NOT the same as ``F.mse_loss(pred, target)``, which would additionally divide by D = 512.
    With ``pred = 0`` and unit-norm targets the per-sample value is exactly 1, which the tests pin.
    """
    per_sample = (pred - target.detach()).square().sum(dim=-1)
    valid = valid.to(per_sample.dtype)
    valid_count = valid.sum()
    loss_sum = (per_sample * valid).sum()
    if bool(valid_count > 0):
        loss = loss_sum / valid_count
    else:
        # keep the graph connected (a rank with no valid sample must not stall the reducer) and
        # return an exact zero rather than a NaN
        loss = loss_sum * 0.0
    return loss, {'per_sample': per_sample, 'valid': valid, 'valid_count': valid_count,
                  'loss_sum': loss_sum, 'valid_fraction': valid.mean()}


class C1TrainModule(nn.Module):
    """DDP-visible module for ``objective = said_cls_tcr``.

    Owns the student CLIP once, one separable frozen reference vision tower and one decoder.
    """

    def __init__(self, clip_model, rank: int = 0, lambda_rec: float = DEFAULT_TAU_REC,
                 lambda_align: float = LAMBDA_ALIGN, lambda_sparse: float = LAMBDA_SPARSE,
                 decoder_hidden: int = DECODER_HIDDEN, decoder_seed: int = 0,
                 duplicate_policy: str = 'exclude', eps: float = VALID_NORM_EPS,
                 ddp_gradient_averaging: bool = True, grad_checkpoint_views: bool = False):
        super().__init__()
        self.clip = clip_model
        self.rank = int(rank)
        self.lambda_rec = float(lambda_rec)
        self.lambda_align = float(lambda_align)
        self.lambda_sparse = float(lambda_sparse)
        self.duplicate_policy = duplicate_policy
        self.eps = float(eps)
        self.ddp_gradient_averaging = bool(ddp_gradient_averaging)
        self.grad_checkpoint_views = bool(grad_checkpoint_views)

        # Order matters and is fixed: the reference is deep-copied from the student's *initial*
        # tower first, then the decoder is built. deepcopy consumes a variable number of ambient
        # RNG draws, so both steps share one isolated, seed-restorable context.
        with _isolated_rng(decoder_seed):
            self.reference_visual = copy_visual_for_reference(clip_model)
            self.decoder = TextConditionedReconstructor(visual_embed_dim(clip_model),
                                                        hidden=int(decoder_hidden),
                                                        seed=int(decoder_seed))

        # a real deep copy: no shared storage with the student, no shared parameters
        for parameter in self.reference_visual.parameters():
            parameter.requires_grad_(False)
        self.reference_visual.eval()

    # -- frozen reference invariants ---------------------------------------- #
    def train(self, mode: bool = True):
        """Never let ``.train()`` recursion put the frozen reference back into training mode."""
        result = super().train(mode)
        self.reference_visual.eval()
        return result

    def reference_state(self) -> Dict[str, torch.Tensor]:
        return {key: value.detach().clone()
                for key, value in self.reference_visual.state_dict().items()}

    # -- one training step's forward ---------------------------------------- #
    def forward(self, image_a: torch.Tensor, text: torch.Tensor,
                image_b: Optional[torch.Tensor] = None, image_ids: Optional[torch.Tensor] = None,
                mask_override: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """``image_b`` is accepted only so the existing data pipeline can keep producing it; C1
        never feeds a second view to the student or to the reference.

        ``mask_override`` is a test-only hook (default ``None``) used to exercise the empty
        complementary-support path; it changes no default behaviour.
        """
        del image_b, image_ids
        clip = self.clip
        device = image_a.device

        # ---- student: one image view (with graph), one current text -------------------- #
        if self.grad_checkpoint_views:
            from torch.utils.checkpoint import checkpoint
            v_a = checkpoint(clip.visual, image_a.type(clip.dtype), use_reentrant=False,
                             preserve_rng_state=True)
        else:
            v_a = clip.encode_image(image_a)
        t_raw, text_hidden = clip.encode_text(text, return_full=True)
        m_s, soft, mask_logits = said_mask_from_hidden(clip.mask_net, text_hidden, soft_mask=False)
        if mask_override is not None:
            m_s = torch.as_tensor(mask_override).to(m_s.dtype).to(m_s.device)

        smart = compute_smartclip_terms(v_a, t_raw, m_s, self.rank,
                                        lambda_align=self.lambda_align,
                                        lambda_sparse=self.lambda_sparse)

        # ---- frozen reference target (fp32, no grad, autocast explicitly off) ---------- #
        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=False):
                ref_raw = self.reference_visual(image_a.float())
                target = F.normalize(ref_raw.float(), dim=-1, eps=REFERENCE_EPS)

        # ---- reconstruction core, fp32, autocast off --------------------------------- #
        with torch.autocast(device_type=device.type, enabled=False):
            v32 = v_a.float()
            g = F.normalize(v32, dim=-1, eps=REFERENCE_EPS)
            m_u = 1.0 - m_s.detach().float()
            r_u = g * m_u
            t_cond = F.normalize(t_raw.float(), dim=-1, eps=REFERENCE_EPS).detach()

            finite = torch.isfinite(v32).all(dim=-1) & torch.isfinite(t_cond).all(dim=-1)
            nonempty = m_u.sum(dim=-1) > 0
            norm_r = r_u.norm(dim=-1)
            valid = finite & nonempty & (norm_r > self.eps)

            pred = self.decoder(r_u, t_cond)
            rec, info = reconstruction_loss(pred, target, valid)
            per_sample = info['per_sample']
            valid_count = info['valid_count']

        with torch.no_grad():
            pred_norm = pred.norm(dim=-1)
            ref_norm = target.norm(dim=-1)
            cosine = F.cosine_similarity(pred, target, dim=-1, eps=REFERENCE_EPS)
            g_norm = g.norm(dim=-1)
            g_to_ref = F.cosine_similarity(g, target, dim=-1, eps=REFERENCE_EPS)
            keep = m_s.detach() > 0
            energy = m_s.detach().float()
            total_energy = energy.sum(dim=-1).clamp_min(1e-12)
            v_det = v32.detach()
            total_sq = v_det.square().sum(dim=-1).clamp_min(self.eps)
            energy_said = (v_det * m_s.detach().float()).square().sum(dim=-1) / total_sq
            energy_unsaid = (v_det * m_u.detach()).square().sum(dim=-1) / total_sq
            coordinate_said = keep.float().mean()
            coordinate_unsaid = m_u.detach().float().mean()
            invalid_norm = int((~finite).sum())

        loss_total = smart['loss_smart'] + self.lambda_rec * rec
        out = {
            'loss_total_for_backward': loss_total,
            'loss_total': loss_total,
            'loss_smart': smart['loss_smart'],
            'loss_smart_unweighted': smart['loss_sidm'] + smart['loss_dism'],
            'loss_sidm': smart['loss_sidm'],
            'loss_dism': smart['loss_dism'],
            'loss_sparsity': smart['loss_sparsity'],
            'loss_rec': rec,
            'loss_rec_sum': info['loss_sum'] * (1.0 if bool(valid_count) > 0 else 0.0),
            'loss_rec_backward_local': rec,
            'lambda_rec': rec.new_tensor(self.lambda_rec),
            'weighted_rec': self.lambda_rec * rec,
            # the raw tensors travel with the output so the trainer can build the DDP scaling; the
            # float copies below are what gets logged
            'rec_valid_count_tensor': valid_count,
            'rec_valid_fraction': float(valid_count) / max(int(valid.shape[0]), 1),
            'rec_invalid_finite': float(invalid_norm),
            'prediction_norm': pred_norm.mean() if pred_norm.numel() else rec.new_zeros(()),
            'reference_norm': ref_norm.mean() if ref_norm.numel() else rec.new_zeros(()),
            'cos_pred_reference': cosine[valid].mean() if bool(valid.any()) else cosine.sum() * 0.0,
            'r_u_norm': norm_r.mean(),
            'g_norm': g_norm.mean(),
            'native_global_to_reference_cos': g_to_ref.mean(),
            'coordinate_fraction_said': coordinate_said,
            'coordinate_fraction_unsaid': coordinate_unsaid,
            'retained_energy_said': energy_said.mean(),
            'retained_energy_unsaid': energy_unsaid.mean(),
            'sidm_top1': smart['sidm_top1'],
            'dism_top1': smart['dism_top1'],
            'mask_s_keep_ratio': smart['mask_s_keep_ratio'],
            'mask_u_keep_ratio': m_u.detach().mean(),
            'soft_mask_mean': soft.mean(),
            'mask_logits_mean': mask_logits.mean(),
            'actual_global_candidate_count': smart['actual_global_candidate_count'],
            # raw handles for the input-dependency diagnostics (no_grad, evaluation only)
            'v_a': v_a,
            'g': g.detach(),
            'm_s': m_s.detach(),
            'm_u': m_u.detach(),
            't_cond': t_cond,
            'target': target,
        }
        return out


def reference_fingerprint(reference_visual: nn.Module) -> str:
    """sha256 over the frozen reference state, in a fixed key order."""
    import hashlib
    digest = hashlib.sha256()
    for key, value in sorted(reference_visual.state_dict().items()):
        digest.update(key.encode('utf-8'))
        digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def copy_visual_for_reference(clip_model: nn.Module) -> nn.Module:
    """Deep-copy the student's visual tower into an independent, storage-disjoint module."""
    import copy
    reference = copy.deepcopy(clip_model.visual)
    reference.eval()
    return reference


def arm_lambda_rec(arm: str) -> float:
    """Only C1 exists in this round; kept next to the other arm tables for symmetry."""
    if arm != ARM:
        raise ValueError('arm must be %r, got %r' % (ARM, arm))
    return DEFAULT_TAU_REC
