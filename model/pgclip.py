"""PG-CLIP v0.1 -- Pre-Projection Gated CLIP (native alignment + pre-projection text-conditioned
visual selection).

Two paths share one visual projection
-------------------------------------
``h`` is the CLS token **after the 12th transformer block and after the native ``ln_post``**, and
*before* ``clip.visual.proj``. ``W = clip.visual.proj`` is a single ``Parameter`` of the CLIP model:
it is used by both paths and is never copied, cloned or re-initialised here.

    native path          v_global    = Norm(h @ W)
    text-conditioned     v_preproj(i,j) = Norm((h_i * mask_j) @ W)

``mask_j`` is a 768-dimensional hard straight-through gate produced from candidate caption ``j``
(see :class:`PreProjectionGate`). The text side is the native EOT + ``text_projection`` output; no
text mask is added anywhere.

Fixed objective
---------------
``L_total = 5 * L_global + 5 * L_preproj + 1 * L_sparse`` with

    L_global  = CE(QG, y) + CE(QG.T, y)
    L_preproj = CE(QP, y) + CE(QP.T, y)
    L_sparse  = mean(|mask|)                # local captions x 768, computed once

Neither alignment term is halved and no extra factor is applied inside or outside the total.

What this module deliberately does **not** contain: a post-projection 512-d SmartCLIP mask
alignment, a TriMask three-way objective, a text mask, U / reference / teacher / decoder /
reconstruction, cross-attention, extra global distillation, advantage losses, orthogonality or
energy regularisers. The old ``clip.mask_net`` is kept only as a compatibility key for bare student
states: it is frozen, never forwarded, excluded from both optimizers and produces no sparsity term.

Exact conditional scores
------------------------
The conditional vector is ``u = (h * m) @ W`` and its norm is the quadratic form
``(h*m) @ (W @ W.T) @ (h*m).T``. ``W @ W.T`` is **not** diagonal, so the projected norm can never be
replaced by ``sum(h**2 * m**2)`` (the 512-d diagonal shortcut of the post-projection formulation);
:func:`projected_norm_shortcut_error` exists to demonstrate the difference. Scores are computed in
explicit blocks so that no ``[B_global, B_global, 768]`` (or ``[..., 512]``) activation is ever
resident, and every block stays differentiable.
"""
import contextlib
import math
import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

# --------------------------------------------------------------------------- experiment identity
ARM = 'PG_CLIP_V01'
OBJECTIVE = 'clip_native_preproj_mask'
PHASE = 'pgclip-v0.1'
GATE_MODE = 'preproj_hard_st_768'

# fixed coefficients: 5 / 5 / 1, never re-tuned in this round
LAMBDA_GLOBAL = 5.0
LAMBDA_PREPROJ = 5.0
LAMBDA_SPARSE = 1.0
LOSS_WEIGHTS = {'global': LAMBDA_GLOBAL, 'preproj': LAMBDA_PREPROJ, 'sparse': LAMBDA_SPARSE}

FIXED_SCALE = 100.0
NORM_EPS = 1e-6

GATE_WIDTH = 512
GATE_OUT = 768
GATE_HEADS = 8
GATE_LAYERS = 1
GATE_SEED = 0
GATE_BIAS_INIT = math.log(8.0)

IMAGE_CHUNK_DEFAULT = 32
TEXT_CHUNK_DEFAULT = 64

# keys that define the mathematics of a run: a resume may never silently change them
RESUME_CRITICAL_KEYS = ('objective', 'arm', 'gate_mode', 'lambda_global', 'lambda_preproj',
                        'lambda_sparse', 'gate_width', 'gate_out', 'gate_layers', 'gate_heads',
                        'fixed_scale', 'norm_eps', 'gate_seed')

SMARTCLIP_FIXED_SCALE = FIXED_SCALE          # same 100x score convention as the S0 lineage


# --------------------------------------------------------------------------- RNG isolation
def _cuda_device_index(device=None):
    if not torch.cuda.is_available():
        return None
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, torch.device):
        return device.index if device.index is not None else torch.cuda.current_device()
    if isinstance(device, int):
        return device
    return torch.cuda.current_device()


def _seed_cuda_device(seed, index):
    """Seed one device's generator without touching the other devices' streams."""
    generators = getattr(torch.cuda, 'default_generators', None)
    if generators is not None and 0 <= index < len(generators):
        generators[index].manual_seed(int(seed))
    else:                                                        # pragma: no cover
        torch.cuda.manual_seed(int(seed))


@contextlib.contextmanager
def isolated_rng(seed: int, device=None):
    """Run a block under a private Python/CPU/CUDA RNG that is fully restored afterwards.

    The gate is built inside this block, so its random initialisation can never shift the main
    model's or the data loader's random stream (and therefore the caption stream).
    """
    python_state = random.getstate()
    cpu_state = torch.get_rng_state()
    cuda_index = _cuda_device_index(device)
    cuda_state = torch.cuda.get_rng_state(cuda_index) if cuda_index is not None else None
    try:
        random.seed(int(seed))
        torch.manual_seed(int(seed))
        if cuda_index is not None:
            _seed_cuda_device(seed, cuda_index)
        yield
    finally:
        random.setstate(python_state)
        torch.set_rng_state(cpu_state)
        if cuda_index is not None and cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, cuda_index)


# --------------------------------------------------------------------------- the gate
class PreProjectionGate(nn.Module):
    """768-dimensional hard straight-through gate produced by one candidate caption.

    Input is the full ``ln_final`` text hidden state ``[B, 248, 512]`` (it is detached inside: the
    sparse term and the gate must never push gradients back into the text transformer). The stem is
    the repository's own ``MaskNetwork`` (one residual attention block, width 512, 8 heads, then the
    original softmax-over-tokens ``AttentionPool``) so the whole 248-token sequence takes part; the
    output layer is ``Linear(512, 768, bias=True)``.

    Initialisation: the stem is normally initialised, the output weight is exactly zero and the
    output bias is exactly ``log(8)``. Therefore ``p = sigmoid(log(8)) = 8/9 >= 0.5`` and
    ``mask = hard + (p - p.detach()) = 1`` for every candidate text: all candidates start from the
    same fully-open selection. Only the output layer is zeroed -- never the stem as well.

    Forward: ``hard = (p >= 0.5)`` in the forward direction, gradient through ``sigmoid`` in the
    backward direction (straight-through). No soft lower bound, no top-k, no fixed keep count, and
    the mask is never detached.
    """

    def __init__(self, width: int = GATE_WIDTH, out_dim: int = GATE_OUT, heads: int = GATE_HEADS,
                 layers: int = GATE_LAYERS, seed: int = GATE_SEED, device=None):
        super().__init__()
        from model.model_longclip import MaskNetwork
        with isolated_rng(seed, device=device):
            self.stem = MaskNetwork(width, layers=layers, heads=heads)
            self.projection = nn.Linear(width, out_dim, bias=True)
            nn.init.zeros_(self.projection.weight)
            nn.init.constant_(self.projection.bias, GATE_BIAS_INIT)
        self.width = int(width)
        self.out_dim = int(out_dim)
        self.heads = int(heads)
        self.layers = int(layers)
        self.seed = int(seed)

    def forward(self, text_hidden: torch.Tensor):
        """``(mask, p)`` for ``text_hidden`` [B, 248, 512]; the mask is differentiable (hard ST)."""
        with torch.autocast(device_type=text_hidden.device.type, enabled=False):
            hidden = text_hidden.detach().float()
            pooled = self.stem(hidden)
            logits = self.projection(pooled)
            p = torch.sigmoid(logits)
            hard = (p >= 0.5).to(p.dtype)
            mask = hard + (p - p.detach())
        return mask, p

    def config(self) -> dict:
        return {'gate_mode': GATE_MODE, 'gate_width': self.width, 'gate_out': self.out_dim,
                'gate_layers': self.layers, 'gate_heads': self.heads, 'gate_seed': self.seed,
                'gate_bias_init': GATE_BIAS_INIT, 'gate_out_weight_init': 0.0,
                'gate_forward': 'hard straight-through: hard = (p >= 0.5), mask = hard + (p - p.detach())',
                'gate_input': 'ln_final text hidden state, detached inside the gate'}


def gate_init_report(gate: PreProjectionGate, hidden: torch.Tensor) -> dict:
    """Evidence that the initial gate is fully open (used by tests and the step-0 log)."""
    with torch.no_grad():
        mask, p = gate(hidden)
    return {
        'p_mean': float(p.mean()),
        'p_expected': 1.0 / (1.0 + math.exp(-GATE_BIAS_INIT)),
        'mask_min': float(mask.min()),
        'mask_max': float(mask.max()),
        'mask_mean': float(mask.mean()),
        'projection_weight_norm': float(gate.projection.weight.detach().norm()),
        'projection_bias': float(gate.projection.bias.detach().mean()),
    }


# --------------------------------------------------------------------------- scores
def native_scores(h: torch.Tensor, W: torch.Tensor, t_unit: torch.Tensor) -> torch.Tensor:
    """``QG[i, j] = 100 * dot(Norm(h_i @ W), t_j)`` for unnormalised ``h`` and unit ``t``."""
    with torch.autocast(device_type=h.device.type, enabled=False):
        v = F.normalize(h.float() @ W.float(), dim=-1, eps=NORM_EPS)
        return FIXED_SCALE * (v @ t_unit.float().t())


def _conditional_block(h_block, W, t_block, masks_block):
    """One differentiable block of ``QP``: exact ``Norm((h * m) @ W)``, fp32, no norm shortcut."""
    masked = h_block.unsqueeze(1) * masks_block.unsqueeze(0)          # [bi, bj, 768]
    u = F.normalize(masked @ W, dim=-1, eps=NORM_EPS)                 # [bi, bj, 512]
    return FIXED_SCALE * (u * t_block.unsqueeze(0)).sum(-1)            # [bi, bj]


def conditional_scores(h: torch.Tensor, W: torch.Tensor, t_unit: torch.Tensor,
                       masks: torch.Tensor, image_chunk: int = IMAGE_CHUNK_DEFAULT,
                       text_chunk: int = TEXT_CHUNK_DEFAULT,
                       use_checkpoint: bool = False) -> torch.Tensor:
    """``QP[i, j] = 100 * dot(Norm((h_i * mask_j) @ W), t_j)``, computed in exact fp32 blocks.

    ``h`` [Ni, 768], ``masks`` [Nt, 768] (the candidate texts' own masks), ``t_unit`` [Nt, 512] (unit
    text features). Rows are images, columns are candidate captions: a fixed image uses each
    candidate's own mask, exactly like the positive pair does. The result stays differentiable;
    activation checkpointing (non-reentrant) is optional and only saves memory.
    """
    if h.dtype != torch.float32 or W.dtype != torch.float32 or t_unit.dtype != torch.float32 \
            or masks.dtype != torch.float32:
        raise TypeError('the conditional score core must run in fp32, got h=%s W=%s t=%s mask=%s'
                        % (h.dtype, W.dtype, t_unit.dtype, masks.dtype))
    rows = []
    with torch.autocast(device_type=h.device.type, enabled=False):
        for i0 in range(0, h.shape[0], image_chunk):
            h_block = h[i0:i0 + image_chunk]
            blocks = []
            for j0 in range(0, masks.shape[0], text_chunk):
                masks_block = masks[j0:j0 + text_chunk]
                t_block = t_unit[j0:j0 + text_chunk]
                if use_checkpoint:
                    # only local tensor maths inside; there is no collective in this closure
                    blocks.append(activation_checkpoint(
                        _conditional_block, h_block, W, t_block, masks_block,
                        use_reentrant=False, preserve_rng_state=True))
                else:
                    blocks.append(_conditional_block(h_block, W, t_block, masks_block))
            rows.append(torch.cat(blocks, dim=1))
        return torch.cat(rows, dim=0)


def diagonal_norm_shortcut(h: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """The (wrong) diagonal norm ``sum(h**2 * m**2)`` that the 512-d formulation could use."""
    return (h.float() ** 2 * masks.float() ** 2).sum(-1)


def projected_norm_shortcut_error(h: torch.Tensor, W: torch.Tensor, masks: torch.Tensor,
                                  t_unit: torch.Tensor, image_chunk: int = IMAGE_CHUNK_DEFAULT,
                                  text_chunk: int = TEXT_CHUNK_DEFAULT) -> dict:
    """How far the diagonal shortcut is from the real projected norm (per (i, j) pair).

    The real quantity is the quadratic form ``(h*m) @ (W @ W.T) @ (h*m).T``; ``W @ W.T`` carries
    off-diagonal channel cross terms, so a diagonal shortcut is not an approximation of the same
    quantity even though it looks similar.
    """
    with torch.no_grad(), torch.autocast(device_type=h.device.type, enabled=False):
        real_rows, short_rows = [], []
        for i0 in range(0, h.shape[0], image_chunk):
            h_block = h[i0:i0 + image_chunk]
            real_block, short_block = [], []
            for j0 in range(0, masks.shape[0], text_chunk):
                masks_block = masks[j0:j0 + text_chunk]
                masked = h_block.unsqueeze(1) * masks_block.unsqueeze(0)
                real = ((masked @ W) ** 2).sum(-1)
                real_block.append(real)
                short_block.append((masked ** 2).sum(-1))
            real_rows.append(torch.cat(real_block, dim=1))
            short_rows.append(torch.cat(short_block, dim=1))
        real = torch.cat(real_rows, dim=0)
        short = torch.cat(short_rows, dim=0)
        return {
            'max_abs_diff': float((real - short).abs().max()),
            'mean_abs_diff': float((real - short).abs().mean()),
            'max_relative_diff': float(((real - short).abs() / real.clamp_min(1e-12)).max()),
            'W_Wt_offdiagonal_fraction': float(
                ((W @ W.t()) - torch.diag(torch.diagonal(W @ W.t()))).abs().sum()
                / (W @ W.t()).abs().sum()),
        }


# --------------------------------------------------------------------------- losses
def cross_entropy_both_directions(scores: torch.Tensor, targets: torch.Tensor) -> tuple:
    """``CE(scores, y) + CE(scores.T, y)`` -- the two directions are summed, never averaged."""
    return (F.cross_entropy(scores, targets), F.cross_entropy(scores.t(), targets))


def sparse_term(masks: torch.Tensor) -> torch.Tensor:
    """``mean(|mask|)`` over the local captions x 768 coordinates (computed once per step)."""
    return masks.abs().mean()


def total_loss(loss_global: torch.Tensor, loss_preproj: torch.Tensor,
               loss_sparse: torch.Tensor) -> torch.Tensor:
    """``5 * L_global + 5 * L_preproj + 1 * L_sparse``."""
    return LAMBDA_GLOBAL * loss_global + LAMBDA_PREPROJ * loss_preproj + LAMBDA_SPARSE * loss_sparse


# --------------------------------------------------------------------------- distributed helpers
def assert_equal_local_batch(local_size: int, world_size: int = None, group=None,
                             device=None) -> None:
    """Refuse an uneven local batch *before* any large feature gather.

    A ragged batch makes ``all_gather`` abort inside the collective with a length mismatch instead of
    naming the real problem; equal (possibly repeated) sizes are supported.
    """
    if world_size is None:
        world_size = dist.get_world_size(group) if dist.is_initialized() else 1
    if world_size == 1:
        return
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    torch.distributed.all_gather(sizes, torch.tensor([int(local_size)], dtype=torch.long,
                                                      device=device), group=group)
    values = sorted({int(item.item()) for item in sizes})
    if len(values) != 1:
        raise RuntimeError('ragged local batch across ranks: %s (equal sizes are required before '
                           'the first feature gather)' % values)


def gather_rows(tensor: torch.Tensor, group=None) -> torch.Tensor:
    """Autograd-aware row gather in global rank order (identity when not distributed)."""
    if not dist.is_initialized() or dist.get_world_size(group) == 1:
        return tensor
    return torch.cat(torch.distributed.nn.all_gather(tensor, group=group), dim=0)


def global_targets(local_size: int, rank: int, device) -> torch.Tensor:
    return torch.arange(rank * local_size, (rank + 1) * local_size, dtype=torch.long,
                        device=device)


# --------------------------------------------------------------------------- optimizers
def partition_parameters(clip: nn.Module, gate: nn.Module) -> dict:
    """Split trainable parameters into the CLIP group and the new-gate group, by parameter id.

    ``clip.mask_net`` (compatibility-only) and ``clip.logit_scale`` (unused, the score scale is the
    fixed 100) are frozen and must not appear in either group; ``clip.visual.proj`` appears once, in
    the CLIP group, because both paths reach it through the single CLIP parameter.
    """
    frozen = {'mask_net': 0, 'logit_scale': 0}
    clip_parameters, gate_parameters = [], []
    gate_ids = {id(parameter) for parameter in gate.parameters()}
    for name, parameter in clip.named_parameters():
        if name.startswith('mask_net.'):
            parameter.requires_grad_(False)
            frozen['mask_net'] += 1
            continue
        if name == 'logit_scale':
            parameter.requires_grad_(False)
            frozen['logit_scale'] += 1
            continue
        if id(parameter) in gate_ids:
            raise RuntimeError('parameter %s is shared between clip and the gate' % name)
        if parameter.requires_grad:
            clip_parameters.append(parameter)
    for parameter in gate.parameters():
        if parameter.requires_grad:
            gate_parameters.append(parameter)
    ids = [id(p) for p in clip_parameters] + [id(p) for p in gate_parameters]
    if len(set(ids)) != len(ids):
        raise RuntimeError('a parameter was registered in both optimizer groups')
    trainable = {id(p) for p in clip.parameters() if p.requires_grad}
    trainable |= {id(p) for p in gate.parameters() if p.requires_grad}
    if set(ids) != trainable:
        missing = [name for name, p in clip.named_parameters()
                   if p.requires_grad and id(p) not in set(ids)]
        raise RuntimeError('optimizer groups do not cover every trainable parameter: %r' % missing)
    projection = clip.visual.proj
    if [id(p) for p in clip_parameters].count(id(projection)) != 1:
        raise RuntimeError('clip.visual.proj must be in exactly one optimizer group')
    return {'clip': clip_parameters, 'gate': gate_parameters, 'frozen': frozen,
            'clip_count': len(clip_parameters), 'gate_count': len(gate_parameters)}


def build_optimizers(clip: nn.Module, gate: nn.Module, clip_lr: float = 1e-6,
                     gate_lr: float = 1e-3, weight_decay: float = 1e-2,
                     betas=(0.9, 0.999), eps: float = 1e-8) -> dict:
    """The two AdamW optimizers of the specification (identical hyper-parameters to the S0 lineage)."""
    groups = partition_parameters(clip, gate)
    clip_optimizer = torch.optim.AdamW(groups['clip'], lr=clip_lr, weight_decay=weight_decay,
                                       betas=betas, eps=eps)
    gate_optimizer = torch.optim.AdamW(groups['gate'], lr=gate_lr, weight_decay=0.0,
                                       betas=betas, eps=eps)
    return {'clip': clip_optimizer, 'gate': gate_optimizer, 'partition': groups}


# --------------------------------------------------------------------------- diagnostics
def energy_metrics(h: torch.Tensor, W: torch.Tensor, masks: torch.Tensor) -> dict:
    """The two energy ratios of the specification, kept separate and never conflated.

    ``preproj_retained_energy`` = ``||h*m||^2 / ||h||^2`` (in [0, 1] for a hard 0/1 mask),
    ``projected_output_energy_ratio`` = ``||(h*m)@W||^2 / ||h@W||^2`` which **can exceed 1** because
    removing coordinates can reduce cancellation inside the projection; it must never be clamped and
    is not a "semantic retention rate".
    """
    with torch.no_grad(), torch.autocast(device_type=h.device.type, enabled=False):
        h = h.float()
        masked = h * masks.float()
        preproj = (masked.pow(2).sum(-1) / h.pow(2).sum(-1).clamp_min(1e-12))
        native = (h @ W.float())
        conditioned = (masked @ W.float())
        projected = (conditioned.pow(2).sum(-1) / native.pow(2).sum(-1).clamp_min(1e-12))
        conditioned_norm = conditioned.norm(dim=-1)
        cosine = F.cosine_similarity(conditioned, native, dim=-1, eps=1e-12)
    return {
        'preproj_retained_energy_mean': float(preproj.mean()),
        'preproj_retained_energy_min': float(preproj.min()),
        'preproj_retained_energy_max': float(preproj.max()),
        'projected_output_energy_ratio_mean': float(projected.mean()),
        'projected_output_energy_ratio_max': float(projected.max()),
        'projected_output_energy_ratio_above_one_fraction': float((projected > 1.0).float().mean()),
        'conditioned_output_norm_mean': float(conditioned_norm.mean()),
        'conditioned_output_norm_min': float(conditioned_norm.min()),
        'native_output_norm_mean': float(native.norm(dim=-1).mean()),
        'native_vs_conditioned_cosine_mean': float(cosine.mean()),
    }


def mask_statistics(masks: torch.Tensor, probabilities: torch.Tensor) -> dict:
    """Hard keep counts/rates, gate probabilities and their quantiles (rank-local batch)."""
    with torch.no_grad():
        hard = (masks.detach() >= 0.5)
        per_caption = hard.float().sum(dim=1)
        quantiles = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=probabilities.device)
        p = probabilities.detach().float()
        return {
            'mask_kept_mean': float(per_caption.mean()),
            'mask_kept_min': float(per_caption.min()),
            'mask_kept_max': float(per_caption.max()),
            'mask_keep_fraction_mean': float(per_caption.mean() / masks.shape[1]),
            'mask_all_off_fraction': float((per_caption == 0).float().mean()),
            'mask_all_on_fraction': float((per_caption == masks.shape[1]).float().mean()),
            'mask_coordinate_mean': float(hard.float().mean()),
            'gate_probability_mean': float(p.mean()),
            'gate_probability_std': float(p.std()),
            'gate_probability_min': float(p.min()),
            'gate_probability_max': float(p.max()),
            'gate_probability_near_threshold_fraction': float(
                ((p - 0.5).abs() < 0.05).float().mean()),
            'gate_probability_quantiles': [float(value) for value in
                                           torch.quantile(p.flatten().float(), quantiles)],
            'gate_coordinate_variation_across_captions':
                float(p.std(dim=0).mean()) if p.shape[0] > 1 else None,
            'mask_empty_caption_count': int((per_caption == 0).sum()),
        }


# --------------------------------------------------------------------------- configuration
def config_dict() -> dict:
    return {'objective': OBJECTIVE, 'arm': ARM, 'phase': PHASE, 'gate_mode': GATE_MODE,
            'lambda_global': LAMBDA_GLOBAL, 'lambda_preproj': LAMBDA_PREPROJ,
            'lambda_sparse': LAMBDA_SPARSE, 'fixed_scale': FIXED_SCALE, 'norm_eps': NORM_EPS,
            'gate_width': GATE_WIDTH, 'gate_out': GATE_OUT, 'gate_heads': GATE_HEADS,
            'gate_layers': GATE_LAYERS, 'gate_seed': GATE_SEED}


def check_resume_compatible(payload: dict, expected: dict) -> None:
    """Refuse a silent resume across objective / gate type / coefficient changes."""
    config = dict(payload.get('config') or {})
    problems = []
    for key in RESUME_CRITICAL_KEYS:
        want = expected.get(key, config_dict().get(key))
        got = config.get(key, payload.get(key))
        if isinstance(want, float) or isinstance(got, float):
            try:
                same = float(want) == float(got)
            except (TypeError, ValueError):
                same = False
        else:
            same = want == got
        if not same:
            problems.append('%s=%r != %r' % (key, got, want))
    if problems:
        raise SystemExit('refusing to resume: %s' % ', '.join(problems))


def checkpoint_metadata(steps: int, rank: int, world: int, config: dict, digests: dict,
                        batch_size: int, chunking: dict, precision: str,
                        lr_horizon_steps: int = None) -> dict:
    """Self-describing checkpoint header (identity, init file, shapes, weights, cursor).

    Schema v1.1 adds the flat gate identity keys and the LR horizon at the top level so a verifier
    does not have to reach into ``config``; a checkpoint written by the first production run carries
    the same information inside ``config`` and is still accepted by the runner.
    """
    horizon = int(lr_horizon_steps if lr_horizon_steps is not None
                  else (config.get('lr_horizon_steps') or 0))
    return {
        'objective': OBJECTIVE, 'arm': ARM, 'phase': PHASE, 'gate_mode': GATE_MODE,
        'gate_width': GATE_WIDTH, 'gate_out': GATE_OUT, 'gate_layers': GATE_LAYERS,
        'gate_heads': GATE_HEADS, 'gate_seed': GATE_SEED,
        'completed_steps': int(steps), 'rank': int(rank), 'world_size': int(world),
        'batch_size_per_gpu': int(batch_size), 'global_batch': int(batch_size) * int(world),
        'lr_horizon_steps': horizon,
        'loss_weights': dict(LOSS_WEIGHTS), 'fixed_scale': FIXED_SCALE, 'norm_eps': NORM_EPS,
        'gate': {'stem': 'MaskNetwork(width=512, layers=1, heads=8)',
                 'output': 'Linear(512, 768, bias=True)', 'bias_init': GATE_BIAS_INIT,
                 'out_weight_init': 0.0, 'mode': 'hard straight-through 0/1 forward',
                 'seed': GATE_SEED, 'registered_on': 'train module (outside clip)',
                 'shares_W_with_clip': 'clip.visual.proj (single Parameter, never copied)'},
        'precision': precision, 'chunking': dict(chunking), 'config': dict(config),
        'digests': dict(digests),
    }


def model_shapes(clip: nn.Module) -> dict:
    return {'visual_hidden': int(clip.visual.proj.shape[0]),
            'output_dim': int(clip.visual.proj.shape[1]),
            'text_width': int(clip.text_projection.shape[0]),
            'context_length': int(clip.context_length),
            'image_resolution': int(clip.visual.input_resolution),
            'patch_size': int(clip.visual.conv1.kernel_size[0])}


def file_sha256(path: str) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def state_digest(state_dict) -> str:
    """Order-independent digest of every tensor in a state dict (tensor values, not file bytes)."""
    import hashlib
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key]
        if torch.is_tensor(value):
            digest.update(key.encode('utf-8'))
            digest.update(value.detach().to(torch.float32).cpu().numpy().tobytes())
    return digest.hexdigest()


def git_head(repo=None) -> str:
    import subprocess
    repo = repo or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        return subprocess.run(['git', '-C', repo, 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, timeout=20).stdout.strip()
    except Exception:                                            # pragma: no cover
        return 'unknown'
