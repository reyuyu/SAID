"""SAID-CLS-CVSSL v0.1 -- Said-conditioned complementary *visual* self-supervision.

One sentence: keep SmartCLIP's verified CLS-selective image-text objective exactly as it is and
add a single task that only trains vision -- in the coordinates the current caption does *not*
use, distinguish the image from its own cross-view batch, always under the **anchor's** mask.

Math (``v`` is the raw, unnormalised final-CLS embedding ``LN_post(CLS12) W_V``)::

    m_U,i = 1 - (1 - rho) * sg(m_S,i)                      (detached complement, rho = 0 by default)
    Pi_m(v) = (v * m) / max(||v * m||_2, eps)

    Q^ab_ij = < Pi_{m_U,i}(v_i^a), sg[ Pi_{m_U,i}(v_j^b) ] > / tau_U
    L_U^ab  = -(1/N) sum_i log softmax_j(Q^ab)_i,positive
    L_U     = (1/2) (L_U^ab + L_U^ba)                      (b->a recomputed, never transposed)

The one constraint that defines the method: **for a fixed anchor ``i`` every candidate ``j`` --
positive and negative -- is scored under ``m_U,i``**, never under its own ``m_U,j``. Precomputing
``u_j = v_j * m_U,j`` and doing ``u_i @ u_j.T`` is a different objective (and invites a mask-pattern
shortcut) and is explicitly not implemented here.

Efficient exact form (one matmul per direction, no ``[b, N, D]`` tensor)::

    num_ij        = sum_d A_id K_jd M_id^2
    anchor_sq_i   = sum_d A_id^2 M_id^2
    candidate_sq_ij = sum_d K_jd^2 M_id^2        <- depends on the ANCHOR i, never precomputable
    Q_ij = num_ij / (tau_U * sqrt(anchor_sq_i) * sqrt(candidate_sq_ij))

``M.square()`` is kept: with a hard mask and rho = 0 the error would be hidden, but a soft mask or
rho > 0 makes the plain ``M`` form wrong.
"""
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

DEFAULT_TAU_U = 0.1
DEFAULT_RHO = 0.0
DEFAULT_EPS = 1e-6
ARM_MASKS = ('none', 'ones', 'random', 'complement')


@contextmanager
def fp32_context(device: Optional[torch.device] = None):
    """Explicitly run FP32 cores with autocast disabled.

    ``tensor.float()`` alone is *not* enough: a matmul inside an autocast region is re-cast to
    the autocast dtype, which would silently change the masked-cosine numerics.
    """
    device_type = 'cuda' if (device is None or getattr(device, 'type', 'cuda') == 'cuda') \
        else 'cpu'
    if device_type == 'cuda' and not torch.cuda.is_available():
        device_type = 'cpu'
    with torch.autocast(device_type=device_type, enabled=False):
        yield


# --------------------------------------------------------------------------- #
# masks
# --------------------------------------------------------------------------- #
def build_complement_mask(m_s: torch.Tensor, rho: float = DEFAULT_RHO) -> torch.Tensor:
    """``m_U = 1 - (1 - rho) * sg(m_S)`` -- the detached complement of the *actual* Said mask.

    ``m_s`` must be the mask that is really used in the Said forward (the straight-through hard
    mask by default), never a separately recomputed soft mask. rho is a fixed constant: it is not
    learned and not scanned in this round.
    """
    if not 0.0 <= float(rho) <= 1.0:
        raise ValueError('rho must be in [0, 1], got %r' % (rho,))
    return 1.0 - (1.0 - float(rho)) * m_s.detach().float()


def build_random_mask(m_u: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """R0 control: independently permute the coordinates of *this sample's* complementary mask.

    The permutation is per sample, applied identically to both directions (the caller passes the
    same tensor), so every candidate of one anchor row still shares one mask and the shuffled
    mask keeps the sample's own value histogram / hard keep count.
    """
    if m_u.dim() != 2:
        raise ValueError('m_u must be [b, D], got %r' % (tuple(m_u.shape),))
    scores = torch.rand(m_u.shape, generator=generator, device='cpu').to(m_u.device)
    order = scores.argsort(dim=-1)
    shuffled = torch.gather(m_u, 1, order)
    return shuffled


def build_arm_mask(m_s: torch.Tensor, arm: str, rho: float = DEFAULT_RHO,
                   generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, Dict]:
    """``(m_U, diagnostics)`` for one experiment arm (S0 / G0 / R0 / C0).

    * ``ones``       -- G0: no coordinate selection at all.
    * ``random``     -- R0: coordinate permutation of this sample's own complement.
    * ``complement`` -- C0: the detached complement of the Said mask.
    * ``none``       -- S0: the complement is still *computed* for diagnostics but never used.
    """
    if arm not in ARM_MASKS:
        raise ValueError('arm mask must be one of %r, got %r' % (ARM_MASKS, arm))
    complement = build_complement_mask(m_s, rho)
    if arm == 'ones':
        mask = torch.ones_like(complement)
    elif arm == 'random':
        if generator is None:
            raise ValueError("arm 'random' needs an explicit generator (independent RNG)")
        mask = build_random_mask(complement, generator)
    else:
        mask = complement
    diagnostics = {
        'arm_mask': arm,
        'rho': float(rho),
        'mask_u_keep_ratio': mask.mean(),
        'mask_s_keep_ratio': m_s.detach().float().mean(),
        'mask_value_histogram_identical_to_complement':
            bool(torch.allclose(mask.sort(dim=-1).values, complement.sort(dim=-1).values)),
    }
    return mask, diagnostics


# --------------------------------------------------------------------------- #
# anchor-masked cosine
# --------------------------------------------------------------------------- #
def anchor_masked_cosine_logits(anchor_raw: torch.Tensor, candidate_raw: torch.Tensor,
                                anchor_mask: torch.Tensor, tau_u: float = DEFAULT_TAU_U,
                                eps: float = DEFAULT_EPS
                                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """``Q[i, j]`` for local anchors ``i`` and global candidates ``j``, all under ``m_i``.

    Args:
        anchor_raw: ``[b, D]`` raw features of the local anchors (gradient allowed).
        candidate_raw: ``[N, D]`` raw features of the global candidate bank (**must already be
            detached by the caller**; passing a live tensor is a programming error).
        anchor_mask: ``[b, D]`` detached mask, one row per anchor, shared by all its candidates.
        tau_u: temperature (fixed, not learned).

    Returns:
        ``(logits [b, N], diagnostics)`` where ``diagnostics`` carries the FP32 anchor and
        candidate masked norms used for the validity rules.
    """
    if anchor_raw.dim() != 2 or candidate_raw.dim() != 2:
        raise ValueError('anchor_raw and candidate_raw must be [n, D]')
    if anchor_mask.shape != anchor_raw.shape:
        raise ValueError('anchor_mask %r must match anchor_raw %r'
                         % (tuple(anchor_mask.shape), tuple(anchor_raw.shape)))
    if candidate_raw.shape[-1] != anchor_raw.shape[-1]:
        raise ValueError('candidate dim %d != anchor dim %d'
                         % (candidate_raw.shape[-1], anchor_raw.shape[-1]))
    if float(tau_u) <= 0.0:
        raise ValueError('tau_u must be positive, got %r' % (tau_u,))
    if candidate_raw.requires_grad:
        raise ValueError('candidate_raw must be detached (candidates never receive gradient)')

    with fp32_context(anchor_raw.device):
        a = anchor_raw.float()
        k = candidate_raw.float()
        m2 = anchor_mask.detach().float().square()
        numerator = (a * m2) @ k.transpose(0, 1)                       # [b, N]
        anchor_sq = (a.square() * m2).sum(dim=-1, keepdim=True)         # [b, 1]
        candidate_sq = m2 @ k.square().transpose(0, 1)                  # [b, N]
        anchor_norm = anchor_sq.clamp_min(eps * eps).sqrt()
        candidate_norm = candidate_sq.clamp_min(eps * eps).sqrt()
        logits = numerator / (anchor_norm * candidate_norm) / float(tau_u)
        diagnostics = {
            'anchor_masked_norm': anchor_norm[:, 0],
            'candidate_masked_norm': candidate_norm,
            'anchor_masked_sq': anchor_sq[:, 0],
            'mask_nonzero_count': (anchor_mask.detach().float() > 0).sum(dim=-1),
        }
    return logits, diagnostics


def explicit_broadcast_reference(anchor_raw: torch.Tensor, candidate_raw: torch.Tensor,
                                 anchor_mask: torch.Tensor, tau_u: float = DEFAULT_TAU_U,
                                 eps: float = DEFAULT_EPS) -> torch.Tensor:
    """Small-tensor reference: explicit ``[b, N, D]`` broadcasting, same formula.

    Used by the tests to prove the matmul form, including soft masks and rho > 0, where the plain
    ``M`` (instead of ``M^2``) form would silently disagree.
    """
    if candidate_raw.requires_grad:
        raise ValueError('candidate_raw must be detached')
    with fp32_context(anchor_raw.device):
        a = anchor_raw.float()[:, None, :]                # [b, 1, D]
        k = candidate_raw.float()[None, :, :]             # [1, N, D]
        m = anchor_mask.detach().float()[:, None, :]      # [b, 1, D]
        numerator = (a * k * m.square()).sum(dim=-1)
        anchor_norm = (a.square() * m.square()).sum(dim=-1).clamp_min(eps * eps).sqrt()
        candidate_norm = (k.square() * m.square()).sum(dim=-1).clamp_min(eps * eps).sqrt()
        return numerator / (anchor_norm * candidate_norm) / float(tau_u)


# --------------------------------------------------------------------------- #
# cross-rank helpers
# --------------------------------------------------------------------------- #
def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def all_gather_detached(tensor: torch.Tensor) -> torch.Tensor:
    """Plain (non-autograd) all-gather along dim 0, **stop-gradient by construction**.

    This is deliberately *not* ``torch.distributed.nn.all_gather``: the CVSSL candidate bank must
    never carry gradient, so the helper detaches internally and the caller may pass either the live
    view features or an already-detached copy. The SmartCLIP term keeps the reference's
    autograd-aware gather.
    """
    local = tensor.detach().contiguous()
    if not is_distributed():
        return local
    world = dist.get_world_size()
    if world == 1:
        return local
    gathered = [torch.zeros_like(local) for _ in range(world)]
    dist.all_gather(gathered, local)
    return torch.cat(gathered, dim=0)


def all_reduce_sum(value: torch.Tensor) -> torch.Tensor:
    """Sum a scalar tensor over ranks (a no-op when not distributed)."""
    if not is_distributed():
        return value
    total = value.clone()
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return total


# --------------------------------------------------------------------------- #
# one direction
# --------------------------------------------------------------------------- #
def cvssl_direction(anchor_raw: torch.Tensor, candidate_raw_global: torch.Tensor,
                    anchor_mask: torch.Tensor, positive_index: torch.Tensor,
                    anchor_image_ids: torch.Tensor, candidate_image_ids: torch.Tensor,
                    tau_u: float = DEFAULT_TAU_U, eps: float = DEFAULT_EPS,
                    duplicate_policy: str = 'exclude') -> Dict[str, torch.Tensor]:
    """One direction (a->b or b->a): per-anchor losses, validity flags and diagnostics.

    Rules:
      * a candidate that is the same image as the anchor (and is not the positive) is **not a
        negative** for that anchor (``duplicate_policy='exclude'``, the default);
      * an anchor is invalid if its mask has no non-zero coordinate, if its masked anchor norm or
        the masked norm of its own positive is <= eps, or if it has no valid negative left;
      * degenerate *negative* candidates are removed from that row's softmax denominator;
      * an invalid row is filtered out **before** the cross-entropy, so no row is ever fed an
        all ``-inf`` logit vector. With zero valid rows the caller gets a connected zero.
    """
    batch = anchor_raw.shape[0]
    logits, diagnostics = anchor_masked_cosine_logits(anchor_raw, candidate_raw_global,
                                                      anchor_mask, tau_u=tau_u, eps=eps)
    total_candidates = candidate_raw_global.shape[0]

    rows = torch.arange(batch, device=logits.device)
    positive_mask = torch.zeros_like(logits, dtype=torch.bool)
    positive_mask[rows, positive_index] = True

    duplicate = (anchor_image_ids[:, None] == candidate_image_ids[None, :]) & ~positive_mask
    degenerate_negative = (diagnostics['candidate_masked_norm'] <= eps) & ~positive_mask

    keep = torch.ones_like(logits, dtype=torch.bool)
    if duplicate_policy == 'exclude':
        keep = keep & ~duplicate
    elif duplicate_policy != 'none':
        raise ValueError('duplicate_policy must be "exclude" or "none", got %r'
                         % (duplicate_policy,))

    negative_keep = keep & ~positive_mask & ~degenerate_negative
    positive_norm = diagnostics['candidate_masked_norm'][rows, positive_index]
    positive_valid = (positive_norm > eps)
    anchor_valid = (diagnostics['mask_nonzero_count'] > 0) & \
                   (diagnostics['anchor_masked_norm'] > eps) & positive_valid & \
                   (negative_keep.sum(dim=1) >= 1)

    masked_logits = logits.masked_fill(~(negative_keep | positive_mask), float('-inf'))
    per_anchor = F.cross_entropy(masked_logits[anchor_valid],
                                 positive_index[anchor_valid], reduction='sum') \
        if bool(anchor_valid.any()) else (logits.sum() * 0.0)

    with torch.no_grad():
        top1 = (masked_logits.argmax(dim=1) == positive_index).float()
        positive_score = logits[rows, positive_index]
        negative_like = (logits * negative_keep.float()).sum(dim=1) / \
            negative_keep.sum(dim=1).clamp_min(1)
    return {
        'per_anchor_loss': per_anchor,
        'valid': anchor_valid,
        'valid_count': anchor_valid.sum(),
        'top1': top1[anchor_valid].mean() if bool(anchor_valid.any()) else top1.sum(),
        'margin': (positive_score - negative_like)[anchor_valid].mean()
        if bool(anchor_valid.any()) else (positive_score.sum() * 0.0),
        'invalid_norm_count': int((~positive_valid).sum()),
        'duplicate_candidates': int(duplicate.sum()),
        'total_candidates': int(total_candidates),
        'mean_valid_negatives': negative_keep.sum(dim=1)[anchor_valid].float().mean()
        if bool(anchor_valid.any()) else (logits.sum() * 0.0),
        'logits': masked_logits,
        'positive_logit': positive_score,
    }


# --------------------------------------------------------------------------- #
# full two-direction objective
# --------------------------------------------------------------------------- #
def complement_visual_contrastive_loss(v_a: torch.Tensor, v_b: torch.Tensor,
                                       mask_u: torch.Tensor,
                                       image_ids: torch.Tensor,
                                       tau_u: float = DEFAULT_TAU_U,
                                       eps: float = DEFAULT_EPS,
                                       rank: int = 0,
                                       ddp_gradient_averaging: bool = False,
                                       duplicate_policy: str = 'exclude'
                                       ) -> Dict[str, torch.Tensor]:
    """``L_U = (1/2)(L_U^ab + L_U^ba)`` with candidates gathered (detached) across ranks.

    Cross-rank scaling: the loss returned for backward is
    ``(scale/2) * (sum_valid_ab / max(V_ab_global, 1) + sum_valid_ba / max(V_ba_global, 1))``
    with ``scale = world_size`` **only when** the model is wrapped in DDP (whose backward averages
    gradients over ranks). This project's trainer follows the SmartCLIP reference and does *not*
    wrap the model in DDP, because its image-text losses already all-gather with autograd; there
    ``scale = 1`` and the per-rank gradient equals the global-mean gradient by construction. Both
    settings are exercised by ``tests/test_complement_visual_ssl_ddp.py``.

    Every collective runs on every rank unconditionally (an all-zero-valid rank must not return
    early), so a rank with no valid anchor can never deadlock the others.
    """
    if v_a.shape != v_b.shape:
        raise ValueError('v_a %r and v_b %r must match' % (tuple(v_a.shape), tuple(v_b.shape)))
    if mask_u.shape != v_a.shape:
        raise ValueError('mask_u %r must match features %r'
                         % (tuple(mask_u.shape), tuple(v_a.shape)))
    local_batch = v_a.shape[0]

    a_global = all_gather_detached(v_a)
    b_global = all_gather_detached(v_b)
    ids_global = all_gather_detached(image_ids.reshape(-1, 1)).reshape(-1)
    offset = rank * local_batch
    positive_index = offset + torch.arange(local_batch, device=v_a.device)

    ab = cvssl_direction(v_a, b_global, mask_u, positive_index, image_ids, ids_global,
                         tau_u=tau_u, eps=eps, duplicate_policy=duplicate_policy)
    ba = cvssl_direction(v_b, a_global, mask_u, positive_index, image_ids, ids_global,
                         tau_u=tau_u, eps=eps, duplicate_policy=duplicate_policy)

    valid_ab = all_reduce_sum(ab['valid_count'].float())
    valid_ba = all_reduce_sum(ba['valid_count'].float())
    denominator_ab = valid_ab.clamp_min(1.0)
    denominator_ba = valid_ba.clamp_min(1.0)
    scale = float(dist.get_world_size()) if (ddp_gradient_averaging and is_distributed()) else 1.0

    loss_ab = ab['per_anchor_loss'] / denominator_ab
    loss_ba = ba['per_anchor_loss'] / denominator_ba
    loss = 0.5 * scale * (loss_ab + loss_ba)

    with torch.no_grad():
        global_mean = 0.5 * (
            all_reduce_sum(ab['per_anchor_loss'].detach()) / denominator_ab
            + all_reduce_sum(ba['per_anchor_loss'].detach()) / denominator_ba)
    total_valid = valid_ab + valid_ba
    results = {
        'loss': loss,
        'loss_ab': loss_ab.detach(),
        'loss_ba': loss_ba.detach(),
        'global_mean_loss': global_mean,
        'loss_scale': loss.new_tensor(scale),
        'valid_ab': valid_ab,
        'valid_ba': valid_ba,
        'valid_ab_local': ab['valid_count'],
        'valid_ba_local': ba['valid_count'],
        'total_candidates': ab['total_candidates'],
        'vssl_ab_top1': ab['top1'],
        'vssl_ba_top1': ba['top1'],
        'positive_minus_negative_margin': 0.5 * (ab['margin'] + ba['margin']),
        'invalid_norm_count': ab['invalid_norm_count'] + ba['invalid_norm_count'],
        'duplicate_candidates': ab['duplicate_candidates'],
        'valid_negative_count': 0.5 * (ab['mean_valid_negatives'] + ba['mean_valid_negatives']),
        'vssl_valid_anchor_fraction': total_valid / max(2.0 * float(local_batch), 1.0),
    }
    if not torch.isfinite(results['loss']).all():                 # pragma: no cover - guard
        raise RuntimeError('CVSSL loss is not finite')
    return results
