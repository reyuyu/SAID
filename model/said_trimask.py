"""SAID S0-TriMask v0.1 -- tri-path image-text alignment over one shared pair of masks.

    L_total = 10 * L1 + 1 * L2 + 1 * L3 + 2 * L_sparse_I

Path 1 is the reference SmartCLIP Said forward, untouched: the *visual* mask ``mI_j`` comes from
the shared-init ``clip.mask_net`` applied to the current caption hidden state and the score is
``100 * cos(Norm(v_i * mI_j), Norm(t_raw_j))``. Path 2 is its mirror on the text side, using a new
independently parameterised :class:`TextMaskHead`. Path 3 is both masks at once:

    Q1[i, j] = 100 * dot(Norm(v_i * mI_j), t_hat_j)
    Q2[i, j] = 100 * dot(v_hat_i,          rT_j)
    Q3[i, j] = 100 * dot(Norm(v_i * mI_j), rT_j)

where ``v_hat = Norm(v)``, ``t_hat = Norm(t_raw)`` and ``rT_j = Norm(t_raw_j * mT_j)``.

The three paths never rebuild a mask: one forward produces ``mI`` and ``mT`` once and all three
score matrices reuse those exact tensors and their graph. Both masks depend on the text only --
the image never enters a mask generator -- so a fixed caption yields the same ``mI``/``mT`` for
every candidate image of the batch.

Everything below is new code in a new file. ``model/said_cls_cvssl.py`` (the reference objective),
``model/model_longclip.py`` (``MaskNetwork``) and the existing trainers are imported, never edited,
so the default S0/C1 behaviour of this branch is unchanged.

Two implementation choices are recorded here because they are engineering decisions of this
version, not properties of the published SmartCLIP objective:

* the text gate is a *soft suppression* gate ``mT = 0.1 + 0.9 * sigmoid(aT) in [0.1, 1]`` with a
  zeroed output layer and ``bias = log(8)``, so at initialisation ``mT == 0.9`` everywhere and
  ``Norm(t_raw * mT) == Norm(t_raw)``: the text direction is untouched at step 0. There is no
  hard threshold, no top-k and no target keep ratio anywhere in this file;
* the ``100`` scale, the candidate pool, the targets and the two cross-entropies are the
  reference's, and no duplicate-caption filtering is applied -- ``compute_smartclip_terms`` (the
  authority named by the task) contains none. This is stated explicitly in the report.
"""
import math
from typing import Dict, Tuple

import torch
import torch.distributed.nn as nn_dist
import torch.nn as nn
import torch.nn.functional as F

from .model_longclip import MaskNetwork
from .said_cls_cvssl import said_mask_from_hidden

ARM = 'S0_TriMask'
OBJECTIVE = 'smartclip_trimask'
PHASE = 's0-trimask-v0.1'

SMARTCLIP_FIXED_SCALE = 100.0
LAMBDA_1 = 10.0
LAMBDA_2 = 1.0
LAMBDA_3 = 1.0
LAMBDA_SPARSE_I = 2.0
LAMBDA_SPARSE_T = 0.0
LAMBDA_ADV = 0.0

TEXT_GATE_EPS = 0.1
TEXT_GATE_INIT_BIAS = math.log(8.0)
NORM_EPS = 1e-6
ADV_TOLERANCE = 1e-6
SATURATION_LO = 0.01
SATURATION_HI = 0.99
NEAR_ZERO_NORM = 1e-3


def _fp32(tensor: torch.Tensor) -> torch.Tensor:
    """Explicit fp32 for the new scoring / gating / loss core (autocast must not decide this)."""
    return tensor.float()


class TextMaskHead(nn.Module):
    """A text-side mask branch with its own parameters, structurally independent of ``mask_net``.

    ``text_stem`` is a full ``MaskNetwork`` (one text Transformer block + the reference
    ``AttentionPool`` scoring layer) and ``text_gate_projection`` maps its 512-d output to the 512
    coordinates of the text gate logits. The stem is *not* zero-initialised: only the final
    projection is, so the first update flows through the whole branch from step 2 onwards.

    The whole module is built inside an isolated, reproducible RNG window (``seed``), so building it
    consumes neither the ambient torch RNG nor the caption/data streams.
    """

    def __init__(self, width: int = 512, layers: int = 1, heads: int = 8, seed: int = 0,
                 gate_eps: float = TEXT_GATE_EPS):
        super().__init__()
        self.width = int(width)
        self.gate_eps = float(gate_eps)
        self.seed = int(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.text_stem = MaskNetwork(self.width, layers=int(layers), heads=int(heads))
            self.text_gate_projection = nn.Linear(self.width, self.width, bias=True)
        nn.init.zeros_(self.text_gate_projection.weight)
        nn.init.constant_(self.text_gate_projection.bias, TEXT_GATE_INIT_BIAS)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """``mT = eps + (1 - eps) * sigmoid(gate(stem(hidden.detach())))``, shape ``[B, width]``."""
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            features = self.text_stem(hidden.detach().float())
            logits = self.text_gate_projection(features)
            gate = self.gate_eps + (1.0 - self.gate_eps) * torch.sigmoid(logits)
        return gate

    def gate_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """The raw logits (diagnostics only; the training path never calls this)."""
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return self.text_gate_projection(self.text_stem(hidden.detach().float()))


def mask_from_hidden(mask_net, text_hidden: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The reference visual mask, unchanged (hard straight-through, ``soft_mask=False``)."""
    return said_mask_from_hidden(mask_net, text_hidden, soft_mask=False)


def trimask_pairwise_scores(v_raw: torch.Tensor, t_raw: torch.Tensor, mask_i: torch.Tensor,
                            mask_t: torch.Tensor, eps: float = NORM_EPS
                            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The three score matrices against the **global** text candidate set.

    Shapes: ``v_raw`` ``[n_local_images, D]``, ``t_raw``/``mask_i``/``mask_t``
    ``[n_local_texts, D]``; the returned ``Q1``/``Q2``/``Q3`` are ``[n_local_images, N_global]``.

    The masked norm uses the ``sqrt(clamp_min(sum_d v^2 mI^2, eps^2))`` form, i.e. a safe
    denominator that never takes ``sqrt(0)``. ``mask_i.square()`` is kept in the graph on purpose:
    its forward value equals ``mask_i`` for a straight-through mask, but the gradient does not.

    Every text-side tensor is gathered with the reference's autograd-aware
    ``torch.distributed.nn.all_gather``, and nothing is detached, so remote candidates stay live.
    """
    v = _fp32(v_raw)
    t_hat = F.normalize(_fp32(t_raw), dim=-1, eps=eps)
    r_t = F.normalize(_fp32(t_raw) * _fp32(mask_t), dim=-1, eps=eps)
    m_i = _fp32(mask_i)

    t_hat_all = torch.cat(nn_dist.all_gather(t_hat), dim=0)
    r_t_all = torch.cat(nn_dist.all_gather(r_t), dim=0)
    m_i_all = torch.cat(nn_dist.all_gather(m_i), dim=0)

    denominator = (v.square() @ m_i_all.square().t()).clamp_min(eps ** 2).sqrt()
    v_hat = F.normalize(v, dim=-1, eps=eps)

    q1 = SMARTCLIP_FIXED_SCALE * ((v @ (m_i_all * t_hat_all).t()) / denominator)
    q2 = SMARTCLIP_FIXED_SCALE * (v_hat @ r_t_all.t())
    q3 = SMARTCLIP_FIXED_SCALE * ((v @ (m_i_all * r_t_all).t()) / denominator)
    return q1, q2, q3, denominator


def _targets(rank: int, batch: int, device) -> torch.Tensor:
    return torch.linspace(rank * batch, rank * batch + batch - 1, batch,
                          dtype=torch.long, device=device)


def _path_terms(q_local: torch.Tensor, rank: int, batch: int, targets: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(L_I2T, L_T2I, Q_global)`` for one path: local image rows classify over the global texts
    and the global image rows classify over this rank's text columns."""
    i2t = F.cross_entropy(q_local, targets)
    q_global = torch.cat(nn_dist.all_gather(q_local), dim=0)
    columns = q_global[:, rank * batch:(rank + 1) * batch]
    t2i = F.cross_entropy(columns.t(), targets)
    return i2t, t2i, q_global


def _path_diagnostics(q_local: torch.Tensor, q_global: torch.Tensor, rank: int, batch: int,
                      targets: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Per-path positive/negative scores, gap and top-1 for both directions (no gradient)."""
    columns = q_global[:, rank * batch:(rank + 1) * batch]
    diagonal = q_local.gather(1, targets.view(-1, 1)).squeeze(1)
    row_sum = q_local.sum(dim=1)
    negative = (row_sum - diagonal) / max(q_local.shape[1] - 1, 1)
    diagonal_t = columns.t().gather(1, targets.view(-1, 1)).squeeze(1)
    column_sum = columns.sum(dim=0)
    negative_t = (column_sum - diagonal_t) / max(columns.shape[0] - 1, 1)
    return {
        'positive_score_i2t': diagonal.mean(),
        'negative_score_i2t': negative.mean(),
        'score_gap_i2t': (diagonal - negative).mean(),
        'top1_i2t': (q_local.argmax(dim=1) == targets).float().mean(),
        'positive_score_t2i': diagonal_t.mean(),
        'negative_score_t2i': negative_t.mean(),
        'score_gap_t2i': (diagonal_t - negative_t).mean(),
        'top1_t2i': (columns.argmax(dim=0) == targets).float().mean(),
    }


def _quantile(values: torch.Tensor, q: float) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros(())
    return values.quantile(q)


def trimask_terms(v_raw: torch.Tensor, t_raw: torch.Tensor, mask_i: torch.Tensor,
                  mask_t: torch.Tensor, rank: int, lambda_1: float = LAMBDA_1,
                  lambda_2: float = LAMBDA_2, lambda_3: float = LAMBDA_3,
                  lambda_sparse_i: float = LAMBDA_SPARSE_I,
                  soft_i: torch.Tensor = None, eps: float = NORM_EPS
                  ) -> Dict[str, torch.Tensor]:
    """The complete S0-TriMask objective on already-encoded features and masks."""
    batch = int(v_raw.shape[0])
    device = v_raw.device
    targets = _targets(rank, batch, device)

    q1, q2, q3, denominator = trimask_pairwise_scores(v_raw, t_raw, mask_i, mask_t, eps=eps)
    paths = [_path_terms(q, rank, batch, targets) for q in (q1, q2, q3)]
    (l1_i2t, l1_t2i, q1_all), (l2_i2t, l2_t2i, q2_all), (l3_i2t, l3_t2i, q3_all) = paths

    loss_1 = l1_i2t + l1_t2i
    loss_2 = l2_i2t + l2_t2i
    loss_3 = l3_i2t + l3_t2i
    loss_sparse_i = torch.mean(torch.abs(mask_i))
    loss_total = (lambda_1 * loss_1 + lambda_2 * loss_2 + lambda_3 * loss_3
                  + lambda_sparse_i * loss_sparse_i)

    out: Dict[str, torch.Tensor] = {
        'loss_total': loss_total,
        'loss_total_for_backward': loss_total,
        'loss_1': loss_1, 'loss_1_i2t': l1_i2t, 'loss_1_t2i': l1_t2i,
        'loss_2': loss_2, 'loss_2_i2t': l2_i2t, 'loss_2_t2i': l2_t2i,
        'loss_3': loss_3, 'loss_3_i2t': l3_i2t, 'loss_3_t2i': l3_t2i,
        'loss_sparse_i': loss_sparse_i,
        'weighted_loss_1': lambda_1 * loss_1,
        'weighted_loss_2': lambda_2 * loss_2,
        'weighted_loss_3': lambda_3 * loss_3,
        'weighted_loss_sparse_i': lambda_sparse_i * loss_sparse_i,
        'lambda_1': torch.as_tensor(float(lambda_1), device=device),
        'lambda_2': torch.as_tensor(float(lambda_2), device=device),
        'lambda_3': torch.as_tensor(float(lambda_3), device=device),
        'lambda_sparse_i': torch.as_tensor(float(lambda_sparse_i), device=device),
        'lambda_sparse_t': torch.as_tensor(LAMBDA_SPARSE_T, device=device),
        'lambda_adv': torch.as_tensor(LAMBDA_ADV, device=device),
        'global_candidate_count': torch.as_tensor(float(q1.shape[1]), device=device),
        'local_anchor_count': torch.as_tensor(float(batch), device=device),
        'm_i': mask_i, 'm_t': mask_t,
        'q1': q1, 'q2': q2, 'q3': q3,
    }

    with torch.no_grad():
        for index, (q_local, q_global) in enumerate(((q1, q1_all), (q2, q2_all), (q3, q3_all))):
            for key, value in _path_diagnostics(q_local.detach(), q_global.detach(), rank, batch,
                                                targets).items():
                out['path%d_%s' % (index + 1, key)] = value
        # "the third path should be better" is diagnosed only -- lambda_adv stays 0 and these
        # quantities are detached, so they can never enter the backward pass
        per_path = []
        for q_local, q_global in ((q1, q1_all), (q2, q2_all), (q3, q3_all)):
            i2t = F.cross_entropy(q_local.detach(), targets, reduction='none')
            columns = q_global.detach()[:, rank * batch:(rank + 1) * batch]
            t2i = F.cross_entropy(columns.t(), targets, reduction='none')
            per_path.append(torch.cat([i2t, t2i]))
        ell_1, ell_2, ell_3 = per_path
        best_single = torch.minimum(ell_1, ell_2)
        adv_gap = torch.relu(ell_3 - best_single).mean()
        out['adv_gap'] = adv_gap
        out['third_not_worse_fraction'] = (ell_3 <= best_single + ADV_TOLERANCE).float().mean()
        out['third_better_fraction'] = (ell_3 < best_single - ADV_TOLERANCE).float().mean()
        out['ell1_mean'] = ell_1.mean()
        out['ell2_mean'] = ell_2.mean()
        out['ell3_mean'] = ell_3.mean()

        # ---- mask diagnostics (detached; purely descriptive) ----
        mi = mask_i.detach().float()
        mt = mask_t.detach().float()
        v = _fp32(v_raw).detach()
        t = _fp32(t_raw).detach()
        v_energy = v.square().sum(dim=-1).clamp_min(eps)
        t_energy = t.square().sum(dim=-1).clamp_min(eps)
        masked_norm = (v * mi).norm(dim=-1)
        out['mask_i_keep_ratio'] = mi.mean()
        out['mask_i_empty_fraction'] = (mi == 0).all(dim=-1).float().mean()
        out['mask_i_retained_energy_fraction'] = (
            (v * mi).square().sum(dim=-1) / v_energy).mean()
        out['mask_i_masked_norm_min'] = masked_norm.min()
        out['mask_i_near_zero_norm_fraction'] = (masked_norm <= NEAR_ZERO_NORM).float().mean()
        out['mask_i_pairwise_denominator_min'] = denominator.detach().min()
        if soft_i is not None:
            soft = soft_i.detach().float()
            out['mask_i_soft_mean'] = soft.mean()
            out['mask_i_soft_saturation_fraction'] = (
                (soft < SATURATION_LO) | (soft > SATURATION_HI)).float().mean()

        out['mask_t_mean'] = mt.mean()
        out['mask_t_std'] = mt.std(unbiased=False)
        out['mask_t_p10'] = _quantile(mt.flatten(), 0.10)
        out['mask_t_p50'] = _quantile(mt.flatten(), 0.50)
        out['mask_t_p90'] = _quantile(mt.flatten(), 0.90)
        out['mask_t_within_sample_std'] = mt.std(dim=-1, unbiased=False).mean()
        out['mask_t_at_floor_fraction'] = (mt <= TEXT_GATE_EPS + 1e-6).float().mean()
        out['mask_t_at_top_fraction'] = (mt >= 1.0 - 1e-6).float().mean()
        out['mask_t_retained_energy_fraction'] = (
            (t * mt).square().sum(dim=-1) / t_energy).mean()
        cosine = F.cosine_similarity(F.normalize(t * mt, dim=-1, eps=eps),
                                     F.normalize(t, dim=-1, eps=eps), dim=-1, eps=eps)
        out['mask_t_cosine_with_unmasked'] = cosine.mean()
        out['mask_t_cosine_min'] = cosine.min()
        out['text_direction_change_fraction'] = (cosine < 1.0 - 1e-6).float().mean()
    return out


class TriMaskTrainModule(nn.Module):
    """The DDP-visible module: owns the CLIP student (registered once) plus the new text branch.

    ``ddp_gradient_averaging`` is deliberately absent: the task fixes the reference convention
    (per-rank anchor means + autograd-aware gather + standard DDP parameter averaging), under which
    ``mean_r dL_r/dp == dL_global_mean/dp`` and no extra ``world_size`` factor may be applied. The
    two-rank test in ``tests/test_said_trimask.py`` checks exactly that.
    """

    def __init__(self, clip_model, rank: int = 0, lambda_1: float = LAMBDA_1,
                 lambda_2: float = LAMBDA_2, lambda_3: float = LAMBDA_3,
                 lambda_sparse_i: float = LAMBDA_SPARSE_I, text_mask_width: int = 512,
                 text_mask_layers: int = 1, text_mask_heads: int = 8, text_mask_seed: int = 0,
                 grad_checkpoint_views: bool = False):
        super().__init__()
        self.clip = clip_model
        self.text_mask_net = TextMaskHead(width=text_mask_width, layers=text_mask_layers,
                                          heads=text_mask_heads, seed=text_mask_seed)
        self.rank = int(rank)
        self.lambda_1 = float(lambda_1)
        self.lambda_2 = float(lambda_2)
        self.lambda_3 = float(lambda_3)
        self.lambda_sparse_i = float(lambda_sparse_i)
        self.grad_checkpoint_views = bool(grad_checkpoint_views)

    def _encode_view(self, images: torch.Tensor) -> torch.Tensor:
        if not self.grad_checkpoint_views:
            return self.clip.encode_image(images)
        from torch.utils.checkpoint import checkpoint
        return checkpoint(self.clip.visual, images.type(self.clip.dtype),
                          use_reentrant=False, preserve_rng_state=True)

    def forward(self, image_a: torch.Tensor, text: torch.Tensor) -> Dict[str, torch.Tensor]:
        v_a = self._encode_view(image_a)
        t_raw, text_hidden = self.clip.encode_text(text, return_full=True)
        # one visual mask and one text mask per caption, generated once and reused by all three
        # paths; the text head detaches its input internally, the reference mask does the same.
        # The visual mask keeps the reference execution strategy (ambient autocast); the text head
        # is explicit fp32 internally.
        mask_i, soft_i, logits_i = mask_from_hidden(self.clip.mask_net, text_hidden)
        mask_t = self.text_mask_net(text_hidden)
        # the new scoring/gating/loss core is explicit fp32: autocast must not silently turn the
        # masked norms, the similarities or the two cross-entropies into bf16 matmuls
        with torch.autocast(device_type=v_a.device.type, enabled=False):
            out = trimask_terms(v_a, t_raw, mask_i, mask_t, self.rank,
                                lambda_1=self.lambda_1, lambda_2=self.lambda_2,
                                lambda_3=self.lambda_3, lambda_sparse_i=self.lambda_sparse_i,
                                soft_i=soft_i)
        out['v_a'] = v_a
        out['t_raw'] = t_raw
        out['soft_mask_i'] = soft_i
        out['mask_logits_i'] = logits_i
        if not bool(torch.isfinite(out['loss_total'])):
            raise RuntimeError('S0-TriMask produced a non-finite total loss; '
                               'per-term values: 1=%r 2=%r 3=%r sparse=%r'
                               % (float(out['loss_1']), float(out['loss_2']),
                                  float(out['loss_3']), float(out['loss_sparse_i'])))
        return out


def student_state_from_checkpoint(clip_state: Dict[str, torch.Tensor]
                                  ) -> Dict[str, torch.Tensor]:
    """The bare student state: the CLIP keys only, with the new text branch keys rejected."""
    forbidden = [key for key in clip_state
                 if key.startswith('text_mask_net') or key.startswith('text_stem')
                 or key.startswith('text_gate_projection')]
    if forbidden:
        raise ValueError('the student state must not contain the new text-branch keys: %r'
                         % forbidden[:5])
    return dict(clip_state)
