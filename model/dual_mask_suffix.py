"""Clean S0 Dual-Mask Suffix objective.

The module keeps the verified S0 SmartCLIP helpers from ``model.said_cls_cvssl`` and adds one
small suffix task.  ``native`` scores the original image embedding against the separately encoded
suffix.  ``masked`` scores the same suffix after a pairwise, hard straight-through gate whose
input is ``[stop_grad(g), stop_grad(g * mS)]``.  The suffix gate never sees suffix tokens.

The public functions are deliberately small so CPU tests, a two-process DDP worker, and the
production trainer all exercise the same code path.  Cross-rank feature gathers use
``torch.distributed.nn.all_gather`` for the differentiable suffix text bank; only scalar score rows
are gathered after masked pair scoring.  Invalid suffix candidates keep their original global
labels and are filtered before cross entropy, never re-numbered into a compact candidate pool.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
import torch.nn as nn
import torch.nn.functional as F

from .said_cls_cvssl import (LAMBDA_ALIGN, LAMBDA_SPARSE, compute_smartclip_terms,
                             said_mask_from_hidden)


SUFFIX_DIM = 512
SUFFIX_HIDDEN = 512
SUFFIX_LAMBDA = 1.0
# Weight of the valid-positive U-gate sparsity term. 0.0 reproduces the accepted Clean v0.1
# objective bit for bit (the term is then never even accumulated into the total loss).
U_SPARSE_LAMBDA = 0.0
SUFFIX_MODES = ("native", "masked")


def split_caption_suffix(original_caption: str, prefix_k: int, caption_said: str) -> Tuple[str, str]:
    """Return ``(P, R)`` using the original caption and the dataset's already chosen ``K``.

    Newlines are replaced exactly as in the base dataset.  The final non-empty sentence is reserved
    from the suffix; an empty trailing fragment therefore changes neither ``last`` nor ``R``.
    ``caption_said`` is checked character-for-character so a changed prefix stream fails loudly.
    """
    full = str(original_caption).replace("\n", " ")
    parts = full.split(". ")
    k = int(prefix_k)
    if k < 1 or k > len(parts):
        raise ValueError("prefix_k=%d outside caption with %d parts" % (k, len(parts)))
    prefix = ". ".join(parts[:k])
    if prefix != str(caption_said):
        raise AssertionError("caption_said is not the original caption prefix")
    nonempty = [i for i, value in enumerate(parts) if value.strip()]
    last = nonempty[-1] if nonempty else -1
    suffix = ". ".join(value for value in parts[k:last] if value.strip()) if last >= k else ""
    return prefix, suffix


def build_suffix_gate(seed: int = 0, input_dim: int = 1024, hidden_dim: int = SUFFIX_HIDDEN,
                      output_dim: int = SUFFIX_DIM) -> nn.Sequential:
    """Build the masked-only gate while restoring every global RNG stream it touches."""
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(int(seed))
        gate = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                             nn.Linear(hidden_dim, output_dim))
        nn.init.xavier_uniform_(gate[0].weight)
        nn.init.zeros_(gate[0].bias)
        nn.init.zeros_(gate[2].weight)
        nn.init.constant_(gate[2].bias, math.log(8.0))
        return gate
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def hard_st(probability: torch.Tensor) -> torch.Tensor:
    """Binary forward value with the sigmoid derivative in the backward graph."""
    return (probability >= 0.5).float() + (probability - probability.detach())


def differentiable_gather(local: torch.Tensor) -> torch.Tensor:
    """Gather dim-0 with gradients to the local slice in every rank."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return local
    # Gloo's all_gather backward can receive transposed, non-contiguous score
    # gradients. Communicate a flat contiguous tensor and restore row order;
    # the reshape remains differentiable and introduces no gradient scaling.
    world = dist.get_world_size()
    shape = tuple(local.shape)
    flat = local.contiguous().reshape(-1)
    pieces = dist_nn.all_gather(flat)
    return torch.cat(tuple(pieces), dim=0).reshape(world * shape[0], *shape[1:])


def detached_gather(local: torch.Tensor) -> torch.Tensor:
    """Gather a metadata tensor without adding a communication backward edge."""
    value = local.detach().contiguous()
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return value
    gathered = [torch.empty_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, value)
    return torch.cat(gathered, dim=0)


def global_targets(batch_size: int, rank: int, device: torch.device) -> torch.Tensor:
    """Fixed global labels in rank order, never a compressed valid-candidate numbering."""
    return int(rank) * int(batch_size) + torch.arange(batch_size, device=device, dtype=torch.long)


def pairwise_masked_scores(g: torch.Tensor, m_s: torch.Tensor, t_r: torch.Tensor,
                           suffix_gate: nn.Module, image_chunk: int = 16,
                           text_chunk: int = 32, rank: Optional[int] = None,
                           valid_local: Optional[torch.Tensor] = None
                           ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute ``Q_local[B,G]`` in pair blocks using the prescribed detached gate inputs.

    When ``rank`` and ``valid_local`` are given, the tile loop also extracts the gate of the
    *valid positive pair* of this rank.  Local row ``a`` has global candidate column
    ``rank * B + a``, so each valid positive is taken exactly once from the one tile that contains
    that column, and never from the tile diagonal of a foreign rank.  Only the needed
    ``[n, D]`` slices are kept: no ``[B, G, D]`` gate tensor is ever materialised.
    """
    if g.ndim != 2 or m_s.ndim != 2 or t_r.ndim != 2:
        raise ValueError("g, m_s and t_r must be rank-2 tensors")
    if m_s.ndim != 2 or m_s.shape[0] != t_r.shape[0] or m_s.shape[1] != g.shape[1] or g.shape[1] != t_r.shape[1]:
        raise ValueError("feature and mask dimensions do not match")
    rows, candidates, dim = g.shape[0], t_r.shape[0], g.shape[1]
    positive_enabled = rank is not None and valid_local is not None
    if positive_enabled:
        valid_local = valid_local.to(device=g.device, dtype=torch.bool).reshape(-1)
        if valid_local.numel() != rows:
            raise ValueError("valid_local has %d entries, expected %d" % (valid_local.numel(), rows))
        positive_rank = int(rank)
        sparse_positive_sum = g.new_zeros((), dtype=torch.float32)
    else:
        valid_local = None
        positive_rank = 0
        sparse_positive_sum = None
    positive_count_t = g.new_zeros((), dtype=torch.float32)
    positive_keep_sum = g.new_zeros((), dtype=torch.float32)
    positive_probability_sum = g.new_zeros((), dtype=torch.float32)
    positive_cos_sum = g.new_zeros((), dtype=torch.float32)
    positive_all_open_t = g.new_zeros((), dtype=torch.float32)
    positive_all_closed_t = g.new_zeros((), dtype=torch.float32)
    chunks = []
    keep_sum = g.new_zeros((), dtype=torch.float32)
    prob_sum = g.new_zeros((), dtype=torch.float32)
    element_count = 0
    with torch.autocast(device_type="cuda" if g.is_cuda else "cpu", enabled=False):
        g_fp = F.normalize(g.float(), p=2, dim=-1, eps=1e-6)
        m_fp = m_s.detach().float()
        t_fp = F.normalize(t_r.float(), p=2, dim=-1, eps=1e-6)
        for i in range(0, rows, max(1, int(image_chunk))):
            g_block = g_fp[i:i + max(1, int(image_chunk))]
            row_chunks = []
            for j in range(0, candidates, max(1, int(text_chunk))):
                m_block = m_fp[j:j + max(1, int(text_chunk))]
                t_block = t_fp[j:j + max(1, int(text_chunk))]
                g_det = g_block.detach()[:, None, :]
                m_det = m_block.detach()[None, :, :]
                r_s = g_det * m_det
                g_pair = g_det.expand(-1, t_block.shape[0], -1)
                x_u = torch.cat((g_pair, r_s.expand(-1, t_block.shape[0], -1)), dim=-1)
                logits = suffix_gate(x_u.reshape(-1, 2 * dim).float()).reshape(
                    g_block.shape[0], t_block.shape[0], dim)
                probability = torch.sigmoid(logits)
                m_u = hard_st(probability)
                u = F.normalize(g_block[:, None, :] * m_u, p=2, dim=-1, eps=1e-6)
                row_chunks.append(100.0 * (u * t_block[None, :, :]).sum(dim=-1))
                keep_sum = keep_sum + (m_u.detach() >= 0.5).float().sum()
                prob_sum = prob_sum + probability.detach().float().sum()
                element_count += int(m_u.numel())
                if positive_enabled:
                    # Integer tile algebra: local row a's positive column is rank*rows + a, which
                    # lies in exactly one text tile. No GPU synchronisation and no [B, G, D] tensor.
                    block_start = positive_rank * rows + i
                    block_stop = block_start + g_block.shape[0]
                    low = max(block_start, j)
                    high = min(block_stop, j + m_block.shape[0])
                    if high > low:
                        ladder = torch.arange(low, high, device=g.device)
                        row_index = ladder - block_start
                        column_index = ladder - j
                        pair_mask = m_u[row_index, column_index, :]
                        selected = valid_local[i + row_index].float()
                        # live straight-through gate of the positive pair, averaged over D once
                        sparse_positive_sum = sparse_positive_sum + (
                            pair_mask.abs().mean(dim=-1) * selected).sum()
                        positive_count_t = positive_count_t + selected.sum()
                        keep_ratio = (pair_mask.detach() >= 0.5).float().mean(dim=-1)
                        pair_probability = probability[row_index, column_index, :]
                        positive_keep_sum = positive_keep_sum + (keep_ratio * selected).sum()
                        positive_probability_sum = positive_probability_sum + (
                            pair_probability.detach().float().mean(dim=-1) * selected).sum()
                        positive_all_open_t = positive_all_open_t + (
                            (keep_ratio >= 1.0).float() * selected).sum()
                        positive_all_closed_t = positive_all_closed_t + (
                            (keep_ratio <= 0.0).float() * selected).sum()
                        g_rows = g_block[row_index].detach().float()
                        masked_rows = g_rows * pair_mask.detach().float()
                        g_unit = g_rows / g_rows.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                        masked_unit = masked_rows / masked_rows.norm(
                            dim=-1, keepdim=True).clamp_min(1e-6)
                        positive_cos_sum = positive_cos_sum + (
                            (g_unit * masked_unit).sum(dim=-1) * selected).sum()
            chunks.append(torch.cat(row_chunks, dim=1))
    count = g.new_tensor(float(element_count), dtype=torch.float32)
    logs = {
        "m_u_keep_ratio": keep_sum / count.clamp_min(1.0),
        "m_u_probability_mean": prob_sum / count.clamp_min(1.0),
        "m_u_element_count": count.detach(),
        "m_u_keep_ratio_scope": "all B x G x D gate elements of this rank",
    }
    if positive_enabled:
        positive_count = float(positive_count_t.item())
        denominator = max(1.0, positive_count)
        logs.update({
            "u_sparse_positive_sum": sparse_positive_sum,
            "u_sparse_positive_count": positive_count_t.detach(),
            "m_u_positive_keep_ratio": positive_keep_sum / denominator,
            "m_u_positive_probability_mean": positive_probability_sum / denominator,
            "m_u_positive_all_open_fraction": positive_all_open_t / denominator,
            "m_u_positive_all_closed_fraction": positive_all_closed_t / denominator,
            "m_u_positive_cos_to_g": positive_cos_sum / denominator,
        })
    else:
        logs.update({"u_sparse_positive_sum": None,
                     "u_sparse_positive_count": g.new_tensor(0.0),
                     "m_u_positive_keep_ratio": None,
                     "m_u_positive_probability_mean": None,
                     "m_u_positive_all_open_fraction": None,
                     "m_u_positive_all_closed_fraction": None,
                     "m_u_positive_cos_to_g": None})
    return torch.cat(chunks, dim=0), logs


def native_scores(g: torch.Tensor, t_r_global: torch.Tensor) -> torch.Tensor:
    """Native suffix score, with both sides normalized in FP32."""
    with torch.autocast(device_type="cuda" if g.is_cuda else "cpu", enabled=False):
        return 100.0 * F.normalize(g.float(), p=2, dim=-1, eps=1e-6) @ F.normalize(
            t_r_global.float(), p=2, dim=-1, eps=1e-6).transpose(0, 1)


def suffix_loss_from_scores(q_local: torch.Tensor, valid_local: torch.Tensor,
                            rank: int, world_size: int, valid_global: torch.Tensor,
                            q_global: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Compute the two filtered CE sums while retaining original global labels.

    ``q_local`` is finite before candidate masking.  When a rank has no valid local query, the
    connected finite zero keeps all collective and DDP graph participation intact.
    """
    if q_local.ndim != 2 or valid_local.ndim != 1 or q_local.shape[0] != valid_local.numel():
        raise ValueError("q_local and valid_local shape mismatch")
    batch, candidates = q_local.shape
    valid_local = valid_local.to(device=q_local.device, dtype=torch.bool)
    valid_global = valid_global.to(device=q_local.device, dtype=torch.bool).reshape(-1)
    if valid_global.numel() != candidates:
        raise ValueError("valid_global has %d rows, expected %d" % (valid_global.numel(), candidates))
    valid_count = valid_global.sum().to(dtype=q_local.dtype)
    if int(valid_count.item()) < 2:
        zero = q_local.sum() * 0.0
        return {"loss": zero, "global_mean": zero.detach(), "valid_count": valid_count.detach(),
                "i2t_sum": zero.detach(), "t2i_sum": zero.detach()}
    labels = global_targets(batch, rank, q_local.device)
    i2t_logits = q_local.masked_fill(~valid_global[None, :], float("-inf"))
    if bool(valid_local.any()):
        i2t_sum = F.cross_entropy(i2t_logits[valid_local], labels[valid_local], reduction="sum")
    else:
        i2t_sum = q_local.sum() * 0.0
    if q_global is None:
        q_global = differentiable_gather(q_local)
    t2i_logits = q_global.transpose(0, 1).masked_fill(~valid_global[None, :], float("-inf"))
    if bool(valid_local.any()):
        t2i_sum = F.cross_entropy(t2i_logits[labels[valid_local]], labels[valid_local], reduction="sum")
    else:
        t2i_sum = q_global.sum() * 0.0
    a_r = i2t_sum + t2i_sum
    world = max(1, int(world_size))
    loss = (float(world) / valid_count.clamp_min(1.0)) * a_r
    if dist.is_available() and dist.is_initialized() and world > 1:
        global_sum = a_r.detach().clone()
        dist.all_reduce(global_sum, op=dist.ReduceOp.SUM)
    else:
        global_sum = a_r.detach()
    return {"loss": loss, "global_mean": global_sum / valid_count.clamp_min(1.0),
            "valid_count": valid_count.detach(), "i2t_sum": i2t_sum.detach(),
            "t2i_sum": t2i_sum.detach()}


@contextmanager
def fp32_core(device: torch.device):
    """Disable autocast for suffix normalization, MLP, score and CE."""
    with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=False):
        yield


class DualMaskSuffixTrainModule(nn.Module):
    """One DDP-visible forward combining the unchanged S0 objective and the suffix objective."""

    def __init__(self, clip_model: nn.Module, suffix_mode: str = "masked", rank: int = 0,
                 image_chunk: int = 16, text_chunk: int = 32,
                 lambda_suffix: float = SUFFIX_LAMBDA, feature_dim: Optional[int] = None,
                 lambda_u_sparse: float = U_SPARSE_LAMBDA):
        super().__init__()
        if suffix_mode not in SUFFIX_MODES:
            raise ValueError("suffix_mode must be native or masked")
        self.clip = clip_model
        self.suffix_mode = suffix_mode
        self.rank = int(rank)
        self.image_chunk = int(image_chunk)
        self.text_chunk = int(text_chunk)
        self.lambda_suffix = float(lambda_suffix)
        self.lambda_u_sparse = float(lambda_u_sparse)
        if self.lambda_u_sparse != 0.0 and suffix_mode != "masked":
            raise ValueError("the U-gate sparsity term only exists in masked mode")
        feature_dim = int(feature_dim or getattr(clip_model, "embed_dim", SUFFIX_DIM))
        self.feature_dim = feature_dim
        self.suffix_gate = (build_suffix_gate(input_dim=2 * feature_dim, hidden_dim=SUFFIX_HIDDEN,
                                              output_dim=feature_dim)
                            if suffix_mode == "masked" else None)

    def forward(self, image_a: torch.Tensor, prefix_tokens: torch.Tensor,
                suffix_tokens: torch.Tensor, valid_local: torch.Tensor,
                image_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Exactly one image and one prefix encoding per sample.
        g_raw = self.clip.encode_image(image_a)
        t_p_raw, prefix_hidden = self.clip.encode_text(prefix_tokens, return_full=True)
        m_s, _, _ = said_mask_from_hidden(self.clip.mask_net, prefix_hidden)
        s0 = compute_smartclip_terms(g_raw, t_p_raw, m_s, self.rank)

        valid_local = valid_local.to(device=g_raw.device, dtype=torch.bool)
        valid_global = detached_gather(valid_local)
        world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if int(valid_global.sum().item()) < 2:
            zero = g_raw.sum() * 0.0
            return self._pack(s0, zero, zero, zero, zero.detach(), valid_global, valid_local,
                              m_u_keep_ratio=None, m_u_probability_mean=None)

        # Suffix is separately encoded and is never passed to suffix_gate.
        t_r_raw = self.clip.encode_text(suffix_tokens)
        t_r_global = differentiable_gather(t_r_raw)
        u_sparse_fields = {}
        with fp32_core(g_raw.device):
            if self.suffix_mode == "native":
                q_local = native_scores(g_raw, t_r_global)
                gate_logs = {"m_u_keep_ratio": None, "m_u_probability_mean": None,
                             "u_sparse_positive_sum": None}
            else:
                m_s_global = detached_gather(m_s)
                q_local, gate_logs = pairwise_masked_scores(
                    g_raw, m_s_global, t_r_global, self.suffix_gate,
                    image_chunk=self.image_chunk, text_chunk=self.text_chunk,
                    rank=self.rank, valid_local=valid_local)
            q_global = differentiable_gather(q_local)
            suffix = suffix_loss_from_scores(q_local, valid_local, self.rank, world,
                                              valid_global, q_global=q_global)
            sparse_backward, sparse_global, u_sparse_fields = u_sparse_terms(
                gate_logs, suffix["valid_count"], world, g_raw, self.suffix_mode)
        return self._pack(s0, self.lambda_suffix * suffix["loss"], sparse_backward,
                          suffix["global_mean"], sparse_global, valid_global, valid_local,
                          gate_logs["m_u_keep_ratio"], gate_logs["m_u_probability_mean"], suffix,
                          gate_logs, u_sparse_fields)

    def _pack(self, s0: Dict[str, torch.Tensor], suffix_loss: torch.Tensor,
              sparse_backward: torch.Tensor, suffix_global: torch.Tensor,
              sparse_global: torch.Tensor, valid_global: torch.Tensor,
              valid_local: torch.Tensor, m_u_keep_ratio: torch.Tensor,
              m_u_probability_mean: torch.Tensor, suffix: Optional[Dict[str, torch.Tensor]] = None,
              gate_logs: Optional[Dict[str, torch.Tensor]] = None,
              u_sparse_fields: Optional[Dict[str, torch.Tensor]] = None
              ) -> Dict[str, torch.Tensor]:
        # ``sparse_backward`` arrives unweighted: it is (W / V) * local sum, and the coefficient is
        # applied exactly once, here.
        sparse_loss = self.lambda_u_sparse * sparse_backward
        zero = suffix_loss.detach() * 0.0
        total = s0["loss_smart"] + suffix_loss + sparse_loss
        suffix = suffix or {"valid_count": valid_global.sum().detach(),
                            "i2t_sum": zero, "t2i_sum": zero}
        gate_logs = gate_logs or {}
        u_sparse_fields = u_sparse_fields or {}
        unweighted_s0 = s0["loss_sidm"] + s0["loss_dism"]
        # Only total/S0/suffix/sparse losses remain live; every diagnostic is detached.
        packed = {
            "loss_total": total,
            "loss_total_local_backward": total.detach(),
            "loss_s0": s0["loss_smart"],
            "loss_suffix": suffix_loss,
            "loss_u_sparse_backward": sparse_backward.detach(),
            "weighted_u_sparse_backward": sparse_loss.detach(),
            "loss_suffix_global": suffix_global.detach(),
            "loss_u_align_global": suffix_global.detach(),
            "loss_u_sparse_global": sparse_global.detach(),
            "L_S": unweighted_s0.detach(),
            "S_S": s0["loss_sparsity"].detach(),
            "weighted_s0_align": (LAMBDA_ALIGN * unweighted_s0).detach(),
            "weighted_s0_sparse": (LAMBDA_SPARSE * s0["loss_sparsity"]).detach(),
            "weighted_u_align": (self.lambda_suffix * suffix_global).detach(),
            "weighted_u_sparse": (self.lambda_u_sparse * sparse_global).detach(),
            "lambda_suffix": float(self.lambda_suffix),
            "lambda_u_sparse": float(self.lambda_u_sparse),
            "loss_suffix_i2t_sum": suffix["i2t_sum"].detach(),
            "loss_suffix_t2i_sum": suffix["t2i_sum"].detach(),
            "valid_global": valid_global.sum().detach(),
            "valid_local_count": valid_local.sum().detach(),
            "valid_local_fraction": valid_local.float().mean().detach(),
            "m_u_keep_ratio": None if m_u_keep_ratio is None else m_u_keep_ratio.detach(),
            "m_u_probability_mean": None if m_u_probability_mean is None else m_u_probability_mean.detach(),
            "m_u_positive_keep_ratio": gate_logs.get("m_u_positive_keep_ratio"),
            "m_u_positive_probability_mean": gate_logs.get("m_u_positive_probability_mean"),
            "m_u_positive_all_open_fraction": gate_logs.get("m_u_positive_all_open_fraction"),
            "m_u_positive_all_closed_fraction": gate_logs.get("m_u_positive_all_closed_fraction"),
            "m_u_positive_cos_to_g": gate_logs.get("m_u_positive_cos_to_g"),
            "m_u_positive_count_local": gate_logs.get("u_sparse_positive_count"),
            "u_sparse_local_sum": u_sparse_fields.get("local_sum"),
            "u_sparse_w_over_v": u_sparse_fields.get("weight_over_valid"),
            "suffix_mode": self.suffix_mode,
            "s0_sidm": s0["loss_sidm"].detach(),
            "s0_dism": s0["loss_dism"].detach(),
            "s0_sparsity": s0["loss_sparsity"].detach(),
        }
        result: Dict[str, torch.Tensor] = {}
        for key, value in packed.items():
            if value is None or not torch.is_tensor(value):
                result[key] = value
            elif key == "loss_total":
                result[key] = value
            else:
                result[key] = value.detach()
        return result


def u_sparse_terms(gate_logs: Dict[str, torch.Tensor], valid_count: torch.Tensor, world_size: int,
                   reference: torch.Tensor, suffix_mode: str
                   ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Reduce the valid-positive gate sparsity to one backward scalar and one logged scalar.

    ``valid_count`` is the detached global ``V``.  The locally backpropagated value is
    ``(W / V) * sum_local`` so that DDP's gradient averaging over W ranks reproduces exactly
    ``(1 / V) * sum_global``; the logged value is the correctly reduced global mean.  The log
    reduction never participates in training, and every rank takes part in it.
    """
    local_sum = gate_logs.get("u_sparse_positive_sum")
    if suffix_mode != "masked" or local_sum is None:
        zero = reference.sum() * 0.0
        return zero, zero.detach(), {"local_sum": None, "weight_over_valid": None}
    count = valid_count.detach().to(device=local_sum.device, dtype=torch.float32).reshape(())
    if float(count) < 2.0:
        zero = local_sum * 0.0
        return zero, zero.detach(), {"local_sum": local_sum.detach() * 0.0,
                                     "weight_over_valid": count * 0.0}
    weight = float(max(1, int(world_size))) / count.clamp_min(1.0)
    global_sum = local_sum.detach().clone().to(torch.float32)
    if dist.is_available() and dist.is_initialized() and int(world_size) > 1:
        dist.all_reduce(global_sum, op=dist.ReduceOp.SUM)
    global_mean = global_sum / count.clamp_min(1.0)
    return weight * local_sum, global_mean, {"local_sum": local_sum.detach(),
                                             "weight_over_valid": weight.detach()}


def split_suffix_valid(text: str, tokenize_fn, eot_token: Optional[int] = None) -> bool:
    """Whether tokenization contains a real content span between SOT and EOT."""
    tokens = tokenize_fn([text], truncate=True)[0]
    if eot_token is None:
        try:
            eot_token = int(tokenize_fn.__globals__["_tokenizer"].encoder["<|endoftext|>"])
        except (AttributeError, KeyError):
            raise ValueError("pass eot_token when tokenize_fn does not expose the project tokenizer")
    positions = (tokens == int(eot_token)).nonzero(as_tuple=False)
    eot_pos = int(positions[0].item()) if positions.numel() else int(tokens.numel())
    return eot_pos > 1


__all__ = [
    "SUFFIX_DIM", "SUFFIX_HIDDEN", "SUFFIX_LAMBDA", "SUFFIX_MODES", "U_SPARSE_LAMBDA",
    "DualMaskSuffixTrainModule",
    "build_suffix_gate", "detached_gather", "differentiable_gather", "global_targets", "hard_st",
    "native_scores", "pairwise_masked_scores", "split_caption_suffix", "split_suffix_valid",
    "suffix_loss_from_scores", "u_sparse_terms",
]
