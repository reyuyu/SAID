"""High-Dim SmartMask (HDSM) v0.1.

Single-stage 2048-d text-conditioned hard straight-through filtering with one shared decoder.
The implementation keeps the native CLIP representation as the trainable image/text input and
does not introduce a teacher, a second decoder, a sentence loop, or an extra alignment objective.
"""
import contextlib
import hashlib
import math
import os
import random
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ARM = "HDSM_V01"
OBJECTIVE = "clip_highdim_smartmask_fourterm"
PHASE = "hd-smartmask-v0.1"
GATE_MODE = "highdim_hard_st_2048"
LATENT_DIM = 2048
CLIP_DIM = 512
GATE_WIDTH = 512
GATE_HEADS = 8
GATE_LAYERS = 1
GATE_SEED = 0
GATE_BIAS_INIT = math.log(8.0)
SCORE_SCALE = 100.0
NORM_EPS = 1e-6
LAMBDA_ALIGN = 10.0
LAMBDA_REC = 1.0
LAMBDA_CONS = 0.5
LAMBDA_SPARSE = 0.1
IMAGE_CHUNK_DEFAULT = 16
TEXT_CHUNK_DEFAULT = 32
LOSS_WEIGHTS = {"align": LAMBDA_ALIGN, "rec": LAMBDA_REC,
                "cons": LAMBDA_CONS, "sparse": LAMBDA_SPARSE}


@contextlib.contextmanager
def isolated_rng(seed: int, device=None):
    """Initialise new modules without perturbing the CLIP or data RNG streams."""
    py, cpu = random.getstate(), torch.get_rng_state()
    cuda_index = None
    cuda = None
    if torch.cuda.is_available():
        cuda_index = device.index if isinstance(device, torch.device) and device.index is not None else torch.cuda.current_device()
        cuda = torch.cuda.get_rng_state(cuda_index)
    try:
        random.seed(seed); torch.manual_seed(seed)
        if cuda_index is not None:
            torch.cuda.manual_seed(seed)
        yield
    finally:
        random.setstate(py); torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state(cuda, cuda_index)


class HighDimGate(nn.Module):
    """Full hidden sequence -> 2048 hard straight-through mask."""
    def __init__(self, width=GATE_WIDTH, out_dim=LATENT_DIM, layers=GATE_LAYERS,
                 heads=GATE_HEADS, seed=GATE_SEED, device=None):
        super().__init__()
        from model.model_longclip import MaskNetwork
        with isolated_rng(seed, device):
            self.stem = MaskNetwork(width, layers=layers, heads=heads)
            self.projection = nn.Linear(width, out_dim, bias=True)
            nn.init.zeros_(self.projection.weight)
            nn.init.constant_(self.projection.bias, GATE_BIAS_INIT)
        self.width, self.out_dim = width, out_dim
        self.layers, self.heads, self.seed = layers, heads, seed

    def forward(self, hidden):
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            pooled = self.stem(hidden.detach().float())
            p = torch.sigmoid(self.projection(pooled))
            hard = (p >= 0.5).to(p.dtype)
            mask = hard + (p - p.detach())
        return mask, p

    def config(self):
        return {"gate_mode": GATE_MODE, "gate_width": self.width,
                "gate_out": self.out_dim, "gate_layers": self.layers,
                "gate_heads": self.heads, "gate_seed": self.seed,
                "bias_init": GATE_BIAS_INIT,
                "forward": "hard=(p>=0.5); mask=hard+(p-p.detach())",
                "input": "full ln_final token hidden, detached inside gate"}


def native_unit(g_raw):
    return F.normalize(g_raw.float(), dim=-1, eps=NORM_EPS)


def decode_unit(z, decoder):
    return F.normalize(decoder(z.float()), dim=-1, eps=NORM_EPS)


def blocked_masked_units(z, decoder, masks, image_chunk=IMAGE_CHUNK_DEFAULT,
                         text_chunk=TEXT_CHUNK_DEFAULT, use_checkpoint=False):
    """Return ``[images,texts,512]`` for tests/diagnostics; production uses the score variant."""
    rows = []
    for i0 in range(0, z.shape[0], image_chunk):
        blocks = []
        for j0 in range(0, masks.shape[0], text_chunk):
            zi = z[i0:i0 + image_chunk]
            mj = masks[j0:j0 + text_chunk]
            fn = lambda a, b: decode_unit(a.unsqueeze(1) * b.unsqueeze(0), decoder)
            blocks.append(checkpoint(fn, zi, mj, use_reentrant=False) if use_checkpoint else fn(zi, mj))
        rows.append(torch.cat(blocks, dim=1))
    return torch.cat(rows, dim=0)


def conditional_scores(z, decoder, masks, t_unit, image_chunk=IMAGE_CHUNK_DEFAULT,
                       text_chunk=TEXT_CHUNK_DEFAULT, use_checkpoint=False):
    """Exact ``Q[i,j] = 100 dot(Norm(decoder(z_i*m_j)), t_j)`` in differentiable blocks."""
    if any(x.dtype != torch.float32 for x in (z, masks, t_unit, decoder.weight)):
        raise TypeError("HDSM conditional core requires fp32 tensors")
    rows = []
    with torch.autocast(device_type=z.device.type, enabled=False):
        for i0 in range(0, z.shape[0], image_chunk):
            row_blocks = []
            for j0 in range(0, masks.shape[0], text_chunk):
                zi, mj, tj = z[i0:i0 + image_chunk], masks[j0:j0 + text_chunk], t_unit[j0:j0 + text_chunk]
                def fn(a, b, c):
                    return SCORE_SCALE * (decode_unit(a.unsqueeze(1) * b.unsqueeze(0), decoder) * c.unsqueeze(0)).sum(-1)
                row_blocks.append(checkpoint(fn, zi, mj, tj, use_reentrant=False) if use_checkpoint else fn(zi, mj, tj))
            rows.append(torch.cat(row_blocks, dim=1))
    return torch.cat(rows, dim=0)


def positive_masked_units(z, decoder, masks):
    return decode_unit(z * masks, decoder)


def four_losses(q, masked_pos, f, g, masks, targets=None):
    """Return raw four terms and the fixed weighted total."""
    targets = torch.arange(q.shape[0], device=q.device) if targets is None else targets
    align_i2t = F.cross_entropy(q, targets)
    align_t2i = F.cross_entropy(q.t(), targets)
    align = align_i2t + align_t2i
    rec = (f - g.detach()).square().sum(-1).mean()
    # ``masked_pos`` is already the positive diagonal [local_B, 512]; target columns were
    # selected while forming it so this term does not accidentally include mismatched pairs.
    cons = (1.0 - (masked_pos * f.detach()).sum(-1)).mean()
    sparse = masks.mean()
    total = LAMBDA_ALIGN * align + LAMBDA_REC * rec + LAMBDA_CONS * cons + LAMBDA_SPARSE * sparse
    return {"align_i2t": align_i2t, "align_t2i": align_t2i, "align": align,
            "rec": rec, "cons": cons, "sparse": sparse, "total": total,
            "weighted_align": LAMBDA_ALIGN * align, "weighted_rec": rec,
            "weighted_cons": LAMBDA_CONS * cons, "weighted_sparse": LAMBDA_SPARSE * sparse}


def sparse_loss(masks):
    return masks.mean()


def gather_rows(x, group=None):
    if not dist.is_initialized() or dist.get_world_size(group) == 1:
        return x
    return torch.cat(torch.distributed.nn.all_gather(x, group=group), dim=0)


def global_targets(local_size, rank, device):
    return torch.arange(rank * local_size, (rank + 1) * local_size, device=device, dtype=torch.long)


def partition_parameters(clip, latent_encoder, decoder, gate):
    """Three optimizer groups in one AdamW; compatibility mask and logit scale stay frozen."""
    clip_group, latent_group, gate_group = [], list(latent_encoder.parameters()) + list(decoder.parameters()), list(gate.parameters())
    gate_ids = {id(p) for p in gate.parameters()}
    for name, p in clip.named_parameters():
        if name.startswith("mask_net.") or name == "logit_scale":
            p.requires_grad_(False); continue
        if id(p) in gate_ids: raise RuntimeError("module parameter is shared")
        if p.requires_grad: clip_group.append(p)
    all_groups = clip_group + latent_group + gate_group
    if len({id(p) for p in all_groups}) != len(all_groups): raise RuntimeError("duplicate optimizer parameter")
    expected = {id(p) for p in clip.parameters() if p.requires_grad} | {id(p) for p in latent_encoder.parameters()} | {id(p) for p in decoder.parameters()} | {id(p) for p in gate.parameters()}
    if {id(p) for p in all_groups} != expected: raise RuntimeError("optimizer groups do not cover trainable parameters")
    if sum(id(p) == id(clip.visual.proj) for p in clip_group) != 1: raise RuntimeError("visual.proj must be registered once")
    return {"clip": clip_group, "latent": latent_group, "gate": gate_group}


def build_optimizer(clip, latent_encoder, decoder, gate, clip_lr=1e-6, latent_lr=1e-4, gate_lr=1e-3):
    groups = partition_parameters(clip, latent_encoder, decoder, gate)
    optim = torch.optim.AdamW([
        {"params": groups["clip"], "lr": clip_lr, "weight_decay": 1e-2},
        {"params": groups["gate"], "lr": gate_lr, "weight_decay": 0.0},
        {"params": groups["latent"], "lr": latent_lr, "weight_decay": 0.0}], betas=(0.9, 0.999), eps=1e-8)
    return optim, groups


def energy_metrics(z, masks, full_raw, masked_raw):
    with torch.no_grad():
        z2 = z.float().pow(2).sum(-1).clamp_min(1e-12)
        masked_z = z.float() * masks.float()
        return {"latent_retained_energy": float((masked_z.pow(2).sum(-1) / z2).mean()),
                "latent_retained_energy_min": float((masked_z.pow(2).sum(-1) / z2).min()),
                "decoded_energy_ratio": float((masked_raw.float().pow(2).sum(-1) / full_raw.float().pow(2).sum(-1).clamp_min(1e-12)).mean()),
                "decoded_energy_ratio_max": float((masked_raw.float().pow(2).sum(-1) / full_raw.float().pow(2).sum(-1).clamp_min(1e-12)).max()),
                "full_raw_norm": float(full_raw.float().norm(dim=-1).mean()),
                "masked_raw_norm": float(masked_raw.float().norm(dim=-1).mean())}


def mask_stats(mask, probability):
    hard = mask.detach() >= .5
    kept = hard.sum(-1).float()
    p = probability.detach().float()
    return {"mask_kept_mean": float(kept.mean()), "mask_kept_min": float(kept.min()),
            "mask_kept_max": float(kept.max()), "mask_keep_fraction": float(hard.float().mean()),
            "mask_all_off_fraction": float((kept == 0).float().mean()),
            "mask_all_on_fraction": float((kept == mask.shape[-1]).float().mean()),
            "p_mean": float(p.mean()), "p_std": float(p.std()), "p_min": float(p.min()), "p_max": float(p.max()),
            "p_near_threshold_fraction": float(((p - .5).abs() < .05).float().mean()),
            "p_quantiles": [float(x) for x in torch.quantile(p.flatten(), torch.tensor([.05,.25,.5,.75,.95], device=p.device))]}


def config_dict():
    return {"objective": OBJECTIVE, "arm": ARM, "phase": PHASE, "gate_mode": GATE_MODE,
            "latent_dim": LATENT_DIM, "clip_dim": CLIP_DIM, "gate_width": GATE_WIDTH,
            "gate_heads": GATE_HEADS, "gate_layers": GATE_LAYERS, "gate_seed": GATE_SEED,
            "score_scale": SCORE_SCALE, "norm_eps": NORM_EPS, "loss_weights": LOSS_WEIGHTS,
            "loss_formula": "10*align + 1*rec + 0.5*cons + 0.1*sparse",
            "align_formula": "CE(Q,targets)+CE(Q.T,targets)",
            "rec_formula": "mean(sum_d((f-g.detach)^2))",
            "cons_formula": "mean(1-dot(s_pos,f.detach))", "sparse_formula": "mean(mask)"}


def state_digest(state):
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if torch.is_tensor(value): h.update(key.encode()); h.update(value.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def checkpoint_payload(clip, latent_encoder, decoder, gate, optimizer, completed_steps,
                       scheduler_state, config, data_cursor, rng_states, provenance):
    return {"schema_version": "hdsm-v0.1", "clip_state": clip.state_dict(),
            "latent_encoder_state": latent_encoder.state_dict(), "decoder_state": decoder.state_dict(),
            "gate_state": gate.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler_state, "completed_steps": int(completed_steps),
            "data_cursor": data_cursor, "rng_states": rng_states, "config": config,
            "provenance": provenance, "gate_config": gate.config()}


def validate_checkpoint(payload, clip, latent_encoder, decoder, gate):
    required = {"clip_state","latent_encoder_state","decoder_state","gate_state","optimizer_state","scheduler_state","completed_steps","data_cursor","rng_states","config","provenance","gate_config"}
    missing = required - set(payload)
    if missing: raise RuntimeError("HDSM checkpoint missing keys: %s" % sorted(missing))
    if not isinstance(payload["gate_state"], dict) or not payload["gate_state"] or not all(torch.is_tensor(v) for v in payload["gate_state"].values()):
        raise RuntimeError("HDSM gate_state must be a non-empty tensor dictionary")
    for module, key in ((clip,"clip_state"),(latent_encoder,"latent_encoder_state"),(decoder,"decoder_state"),(gate,"gate_state")):
        expected = module.state_dict(); got = payload[key]
        if set(expected) != set(got): raise RuntimeError("%s key mismatch" % key)
        for name in expected:
            if expected[name].shape != got[name].shape: raise RuntimeError("%s shape mismatch: %s" % (key,name))


def atomic_torch_save(payload, path):
    temporary = path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)
