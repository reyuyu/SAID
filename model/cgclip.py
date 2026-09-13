"""CG-CLIP v0.1 -- Caption-Gated Final-CLS Attention.

Native alignment plus text-gated attention over the visual tokens of the LAST visual block, in front
of the untouched native output head.

    X11 = [CLS11, patch11_1 ... patch11_196]        # tokens ENTERING visual block 12, [B,197,768]

    native path        c = X11[:,0]; U = last.ln_1(X11)
                       q_cls = Q(U[:,0]);  K = K(U);  V = V(U)
                       a[h,p] = softmax_p(q_cls[h]·K[h,p] / sqrt(64))
                       oG = out_proj(concat_h Σ_p a[h,p] V[h,p])
                       cG = c + oG + mlp(ln_2(c + oG));  vG = Norm(ln_post(cG) @ W)

    conditional path   qT_j = A(stop_grad(t_j))               # 512 -> 64, no bias, zero-init
                       kG_i,p = B(stop_grad(U_i,p))           # 768 -> 64, no bias, xavier
                       logits = qT_j·kG_i,p / sqrt(64) + b
                       m = hard(p >= 0.5) + (p - p.detach())  # 196 patch gates, straight-through
                       m_full = [1, m]                        # the CLS slot keeps gate 1
                       a_cond = (a * m_full) / Σ_p (a * m_full)
                       oA = out_proj(concat_h Σ_p a_cond[h,p] V[h,p])
                       cA = c + oA + mlp(ln_2(c + oA));  vA = Norm(ln_post(cA) @ W)

The text therefore only changes **which visual Values the native CLS query reads**. The native CLS
query, the native Keys/Values, the attention output projection, both residuals, `ln_2`/MLP,
`ln_post` and `visual.proj` are all the *same* parameters for both paths: nothing is copied, no
text query replaces the native one, no text vector or text bias is added to a visual output, there is
no second Value source, and no extra post-projection (PG/SmartCLIP style) channel mask is applied.

Fixed objective::

    L_total = 5 * LG + 5 * LA + 1 * LS
    LG = CE(QG, y) + CE(QG.T, y)        QG[i,j] = 100 * <vG_i, t_j>
    LA = CE(QA, y) + CE(QA.T, y)        QA[i,j] = 100 * <vA_i,j, t_j>
    LS = mean over the TRUE positive pairs of the 196 patch gates only

The auxiliary (conditional) path is unused at inference time; that says nothing about whether it
changes the shared trunk while training, and no claim is made that the native ability is preserved.
"""
import contextlib
import math
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

# --------------------------------------------------------------------------- experiment identity
ARM = 'CG_CLIP_V01'
OBJECTIVE = 'clip_native_caption_gated_cls'
PHASE = 'cgclip-v0.1'
GATE_KIND = 'caption_gated_final_cls_attention'

LAMBDA_GLOBAL = 5.0
LAMBDA_ATTENTION = 5.0
LAMBDA_SPARSE = 1.0
LOSS_WEIGHTS = {'global': LAMBDA_GLOBAL, 'attention': LAMBDA_ATTENTION, 'sparse': LAMBDA_SPARSE}

FIXED_SCALE = 100.0
NORM_EPS = 1e-6
GATE_KEY_DIM = 64
GATE_SEED = 0
GATE_BIAS_INIT = math.log(8.0)

IMAGE_CHUNK_DEFAULT = 8
TEXT_CHUNK_DEFAULT = 32

RESUME_CRITICAL_KEYS = ('objective', 'arm', 'gate_kind', 'gate_key_dim', 'gate_seed',
                        'lambda_global', 'lambda_attention', 'lambda_sparse', 'fixed_scale',
                        'norm_eps')

VISUAL_TOKENS = 197
PATCH_TOKENS = 196
VISUAL_WIDTH = 768
OUTPUT_DIM = 512


# --------------------------------------------------------------------------- RNG isolation
@contextlib.contextmanager
def isolated_rng(seed: int, device=None):
    """Private Python / NumPy / CPU / current-CUDA RNG, all restored afterwards.

    The caption gate is built inside this block, so its random initialisation can never shift the
    shared initialisation, the data pipeline or the caption stream. Only the *current* CUDA device is
    seeded and restored, so other devices' streams are left alone.
    """
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    index = None
    cuda_state = None
    if torch.cuda.is_available():
        index = torch.cuda.current_device() if device is None else (
            device.index if isinstance(device, torch.device) and device.index is not None
            else (int(device) if isinstance(device, int) else torch.cuda.current_device()))
        cuda_state = torch.cuda.get_rng_state(index)
    try:
        random.seed(int(seed))
        np.random.seed(int(seed) % (2 ** 32))
        torch.manual_seed(int(seed))
        if index is not None:
            generators = getattr(torch.cuda, 'default_generators', None)
            if generators is not None and 0 <= index < len(generators):
                generators[index].manual_seed(int(seed))
            else:                                                # pragma: no cover
                torch.cuda.manual_seed(int(seed))
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if index is not None and cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, index)


# --------------------------------------------------------------------------- configuration audit
def visual_spec(clip: nn.Module) -> dict:
    """Read the real module and REFUSE anything that is not the expected ViT-B/16 layout."""
    visual = clip.visual
    last = visual.transformer.resblocks[-1]
    spec = {
        'visual_width': int(visual.conv1.out_channels),
        'visual_layers': len(visual.transformer.resblocks),
        'tokens': int(visual.positional_embedding.shape[0]),
        'last_embed_dim': int(last.attn.embed_dim),
        'last_num_heads': int(last.attn.num_heads),
        'last_head_dim': int(last.attn.head_dim),
        'last_in_proj_weight': tuple(last.attn.in_proj_weight.shape),
        'last_in_proj_bias': tuple(last.attn.in_proj_bias.shape),
        'last_out_proj_weight': tuple(last.attn.out_proj.weight.shape),
        'last_out_proj_bias': tuple(last.attn.out_proj.bias.shape),
        'last_attention_dropout': float(last.attn.dropout),
        'last_has_bias_k': last.attn.bias_k is not None,
        'last_has_bias_v': last.attn.bias_v is not None,
        'last_batch_first': bool(last.attn.batch_first),
        'mlp_c_fc': tuple(last.mlp.c_fc.weight.shape),
        'mlp_c_fc_bias': tuple(last.mlp.c_fc.bias.shape),
        'mlp_c_proj': tuple(last.mlp.c_proj.weight.shape),
        'mlp_c_proj_bias': tuple(last.mlp.c_proj.bias.shape),
        'ln_pre': tuple(visual.ln_pre.normalized_shape),
        'ln_post': tuple(visual.ln_post.normalized_shape),
        'visual_proj': tuple(visual.proj.shape),
        'context_length': int(clip.context_length),
        'text_width': int(clip.transformer.width),
        'text_heads': int(clip.transformer.resblocks[0].attn.num_heads),
        'text_projection': tuple(clip.text_projection.shape),
    }
    expected = {
        'visual_width': VISUAL_WIDTH, 'visual_layers': 12, 'tokens': VISUAL_TOKENS,
        'last_embed_dim': VISUAL_WIDTH, 'last_num_heads': 12, 'last_head_dim': 64,
        'last_in_proj_weight': (3 * VISUAL_WIDTH, VISUAL_WIDTH),
        'last_in_proj_bias': (3 * VISUAL_WIDTH,),
        'last_out_proj_weight': (VISUAL_WIDTH, VISUAL_WIDTH),
        'last_out_proj_bias': (VISUAL_WIDTH,),
        'last_attention_dropout': 0.0, 'last_has_bias_k': False, 'last_has_bias_v': False,
        'last_batch_first': False,
        'mlp_c_fc': (4 * VISUAL_WIDTH, VISUAL_WIDTH), 'mlp_c_fc_bias': (4 * VISUAL_WIDTH,),
        'mlp_c_proj': (VISUAL_WIDTH, 4 * VISUAL_WIDTH), 'mlp_c_proj_bias': (VISUAL_WIDTH,),
        'ln_pre': (VISUAL_WIDTH,), 'ln_post': (VISUAL_WIDTH,),
        'visual_proj': (VISUAL_WIDTH, OUTPUT_DIM),
        'context_length': 248, 'text_width': OUTPUT_DIM, 'text_projection': (OUTPUT_DIM, OUTPUT_DIM),
    }
    problems = ['%s=%r (expected %r)' % (key, spec.get(key), value)
                for key, value in expected.items() if spec.get(key) != value]
    if problems:
        raise RuntimeError('the visual/text configuration is not the supported ViT-B/16 layout: %s'
                           % '; '.join(problems))
    # the vision heads are 12 x 64 and the text heads are 8: never mix them up
    if spec['last_num_heads'] == spec['text_heads']:
        raise RuntimeError('vision and text head counts are equal, which is not this model')
    return spec


def gate_config(gate: nn.Module) -> dict:
    return {'gate_kind': GATE_KIND, 'gate_key_dim': gate.key_dim, 'gate_seed': gate.seed,
            'gate_bias_init': GATE_BIAS_INIT, 'gate_query_input': 'Normalize(t_raw) detached',
            'gate_key_input': 'U = last.ln_1(X11) patches, detached',
            'gate_forward': 'hard straight-through: hard = (p >= 0.5), m = hard + (p - p.detach())',
            'gate_patches': PATCH_TOKENS,
            'cls_self_gate': 'fixed 1, not trainable, not in the sparse term',
            'gate_query_init': 'A.weight = 0 (no bias)',
            'gate_key_init': 'B.weight = xavier_uniform (no bias)',
            'gate_bias': 'single trainable scalar initialised to log(8)',
            'shared_over_heads': True, 'top_k': None, 'soft_floor': None}


def config_dict() -> dict:
    return {'objective': OBJECTIVE, 'arm': ARM, 'phase': PHASE, 'gate_kind': GATE_KIND,
            'lambda_global': LAMBDA_GLOBAL, 'lambda_attention': LAMBDA_ATTENTION,
            'lambda_sparse': LAMBDA_SPARSE, 'fixed_scale': FIXED_SCALE, 'norm_eps': NORM_EPS,
            'gate_key_dim': GATE_KEY_DIM, 'gate_seed': GATE_SEED}


# --------------------------------------------------------------------------- the gate
class CaptionGate(nn.Module):
    """Light 64-d image/text interaction gate over the 196 patch tokens of the last block.

    ``A`` (512 -> 64, no bias) is zero-initialised, ``B`` (768 -> 64, no bias) uses Xavier uniform
    and the bias is one trainable scalar initialised to ``log(8)``. With ``A = 0`` the logits are
    exactly ``log(8)``, so ``p = 8/9 >= 0.5`` and every patch gate is exactly 1: both paths start
    identical. Only ``A`` is zeroed -- ``B`` keeps its normal initialisation, so ``B`` receives no
    gradient on the first step and starts receiving one once ``A`` has moved.

    ``stop_grad`` is applied *inside* the gate to the two inputs, so the gate can never push gradient
    into the text trunk or into ``U``; ``t`` still trains the text trunk through the final similarity,
    and the visual main path keeps ``U``, ``Q``, ``K``, ``V`` and ``c`` live.
    """

    def __init__(self, text_dim: int = OUTPUT_DIM, visual_dim: int = VISUAL_WIDTH,
                 key_dim: int = GATE_KEY_DIM, seed: int = GATE_SEED, device=None):
        super().__init__()
        with isolated_rng(seed, device=device):
            self.query = nn.Linear(text_dim, key_dim, bias=False)
            self.key = nn.Linear(visual_dim, key_dim, bias=False)
            nn.init.zeros_(self.query.weight)
            nn.init.xavier_uniform_(self.key.weight)
            self.bias = nn.Parameter(torch.tensor(float(GATE_BIAS_INIT)))
        self.text_dim = int(text_dim)
        self.visual_dim = int(visual_dim)
        self.key_dim = int(key_dim)
        self.seed = int(seed)

    def project_query(self, t_unit: torch.Tensor) -> torch.Tensor:
        """``qT = A(stop_grad(t))`` -- [Nt, key_dim]."""
        with torch.autocast(device_type=t_unit.device.type, enabled=False):
            return self.query(t_unit.detach().float())

    def project_key(self, u_patches: torch.Tensor) -> torch.Tensor:
        """``kG = B(stop_grad(U patches))`` -- [..., 196, key_dim]."""
        with torch.autocast(device_type=u_patches.device.type, enabled=False):
            return self.key(u_patches.detach().float())

    def gate_from_keys(self, qT: torch.Tensor, kG: torch.Tensor):
        """``(m, p, logits)`` for every (text, patch) pair; ``m`` is the straight-through mask."""
        with torch.autocast(device_type=qT.device.type, enabled=False):
            logits = torch.einsum('jd,ipd->jip', qT.float(), kG.float()) / math.sqrt(self.key_dim)
            logits = logits + self.bias.float()
            probability = torch.sigmoid(logits)
            hard = (probability >= 0.5).to(probability.dtype)
            mask = hard + (probability - probability.detach())
        return mask, probability, logits

    def forward(self, t_unit: torch.Tensor, u_patches: torch.Tensor):
        """Convenience path for small cases: ``(mask, p, logits)`` over full tensors."""
        return self.gate_from_keys(self.project_query(t_unit), self.project_key(u_patches))


def gate_init_report(gate: CaptionGate, t_unit: torch.Tensor, u_patches: torch.Tensor) -> dict:
    with torch.no_grad():
        mask, probability, logits = gate(t_unit, u_patches)
    return {
        'query_weight_norm': float(gate.query.weight.detach().norm()),
        'key_weight_norm': float(gate.key.weight.detach().norm()),
        'bias': float(gate.bias.detach()),
        'logits_min': float(logits.min()), 'logits_max': float(logits.max()),
        'probability_mean': float(probability.mean()),
        'probability_expected': 1.0 / (1.0 + math.exp(-GATE_BIAS_INIT)),
        'mask_min': float(mask.min()), 'mask_max': float(mask.max()),
        'mask_all_one': bool((mask.detach() >= 0.5).all()),
    }


# --------------------------------------------------------------------------- native CLS row
def native_cls_row(last: nn.Module, x11: torch.Tensor, ln_post: nn.Module,
                   visual_proj: torch.Tensor, want_attention: bool = True) -> dict:
    """The native last-block CLS row, computed exactly as ``last(x11)[:, 0, :]`` would be.

    Only the CLS row of the last block is produced (the patch rows of that block are never used).
    Every native component is executed: ``ln_1``, the Q/K/V projection with its biases, the per-head
    scaling by ``1/sqrt(64)``, the attention output projection with its bias, the first residual,
    ``ln_2`` and the MLP with both biases, the second residual, ``ln_post`` and ``visual.proj``.
    """
    if x11.dim() != 3 or x11.shape[1] != VISUAL_TOKENS:
        raise ValueError('X11 must be [B, %d, width], got %s' % (VISUAL_TOKENS, tuple(x11.shape)))
    device_type = x11.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        x11 = x11.float()
        c = x11[:, 0, :]
        u = last.ln_1(x11).float()                                  # [B,197,768]
        batch = u.shape[0]
        heads = last.attn.num_heads
        head_dim = last.attn.head_dim
        weight = last.attn.in_proj_weight.float()
        bias = last.attn.in_proj_bias.float()
        qkv = F.linear(u, weight, bias)                              # [B,197,3*768]
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(batch, VISUAL_TOKENS, heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch, VISUAL_TOKENS, heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch, VISUAL_TOKENS, heads, head_dim).permute(0, 2, 1, 3)
        q_cls = q[:, :, 0:1, :]                                      # [B,H,1,64]
        logits = (q_cls @ k.transpose(-2, -1)) / math.sqrt(head_dim)  # [B,H,1,197]
        attention = torch.softmax(logits, dim=-1)                     # p=0 is CLS itself
        read = attention @ v                                          # [B,H,1,64]
        read = read.permute(0, 2, 1, 3).reshape(batch, 1, heads * head_dim)
        out = last.attn.out_proj(read).reshape(batch, VISUAL_WIDTH)
        residual = c + out
        cls = residual + last.mlp(last.ln_2(residual))
        projected = ln_post(cls) @ visual_proj.float()
    result = {'x11': x11, 'c': c, 'u': u, 'q': q, 'k': k, 'v': v, 'attention': attention,
              'logits': logits, 'out': out, 'residual': residual, 'cls': cls,
              'projected': projected}
    if want_attention:
        with torch.no_grad():
            self_mass = attention[..., 0]                             # [B,H]
            result['cls_self_mass'] = self_mass
            result['cls_self_mass_mean'] = float(self_mass.mean())
            result['cls_self_mass_min'] = float(self_mass.min())
            result['cls_self_mass_max'] = float(self_mass.max())
            result['patch_mass_mean'] = float(attention[..., 1:].sum(-1).mean())
            result['attention_output_norm_ratio'] = float(
                (out.norm(dim=-1) / c.norm(dim=-1).clamp_min(1e-12)).mean())
    return result


# --------------------------------------------------------------------------- conditional CLS row
def _conditional_block(x11_block, c_block, u_block, k_block, v_block, attention_block,
                       qT_block, kG_block, bias, out_proj, ln_2, mlp, ln_post, visual_proj,
                       key_dim):
    """One (text block x image block) tile of the conditional CLS, differentiable end to end.

    Everything the closure needs arrives as an argument: no loop variable is captured, which is what
    makes non-reentrant activation checkpointing safe here.
    """
    with torch.autocast(device_type=x11_block.device.type, enabled=False):
        # gate: 196 patch logits per (text j, image i) pair, shared by all attention heads
        logits = torch.einsum('jd,ipd->jip', qT_block.float(), kG_block.float()) / math.sqrt(key_dim)
        logits = logits + bias.float()
        probability = torch.sigmoid(logits)
        hard = (probability >= 0.5).to(probability.dtype)
        mask = hard + (probability - probability.detach())            # [bj,bi,196]
        mask_full = torch.cat([torch.ones_like(mask[..., :1]), mask], dim=-1)   # CLS slot keeps 1
        attention = attention_block.squeeze(2).unsqueeze(0)            # [1,bi,H,197]
        weighted = attention * mask_full.unsqueeze(2)                  # [bj,bi,H,197]
        denominator = weighted.sum(dim=-1, keepdim=True)               # no detach, no clamp
        attention_conditional = weighted / denominator
        images, texts = kG_block.shape[0], qT_block.shape[0]
        heads = attention.shape[2]
        head_dim = v_block.shape[-1]
        flat_attention = attention_conditional.permute(1, 2, 0, 3).reshape(images * heads, texts,
                                                                          VISUAL_TOKENS)
        flat_values = v_block.reshape(images * heads, VISUAL_TOKENS, head_dim)
        head_out = torch.bmm(flat_attention, flat_values)              # [bi*H, bj, head_dim]
        head_out = head_out.reshape(images, heads, texts, head_dim).permute(2, 0, 1, 3)
        out = out_proj(head_out.reshape(texts * images, heads * head_dim))
        out = out.reshape(texts, images, VISUAL_WIDTH)
        residual = c_block.unsqueeze(0) + out
        cls = residual + mlp(ln_2(residual))
        projected = ln_post(cls) @ visual_proj.float()
    return projected, mask, probability, denominator, attention_conditional, cls


def conditional_cls_scores(x11: torch.Tensor, native: dict, last: nn.Module, gate: CaptionGate,
                           t_unit: torch.Tensor, qT: torch.Tensor, kG: torch.Tensor,
                           visual_proj: torch.Tensor, ln_post: nn.Module,
                           image_chunk: int = IMAGE_CHUNK_DEFAULT,
                           text_chunk: int = TEXT_CHUNK_DEFAULT,
                           use_checkpoint: bool = True, positive_columns=None, progress=None,
                           capture: dict = None):
    """``(QA, sparse_positive_mask)`` -- conditional CLS scores for local images x given texts.

    ``t_unit``/``qT`` may be the gathered GLOBAL texts (all candidates), while ``x11``, ``kG`` and the
    native tensors stay local to this rank: only the scalar score rows are produced here. The
    conditional CLS really runs the native ``out_proj``, residuals, ``ln_2``/MLP, ``ln_post`` and
    ``visual.proj`` for every (text, image) pair. ``positive_columns`` (global column index per local
    image, or None) selects the true-positive patch gates for the sparse term -- they are sliced out of
    the tile that contains them, so the gate network is never evaluated a second time for the sparse
    term. ``capture`` optionally receives the last tile's attention and CLS tensors (detached) so the
    heavy log can reuse the forward instead of re-running it.
    """
    images = x11.shape[0]
    texts = t_unit.shape[0]
    device = x11.device
    scores = torch.empty(images, texts, dtype=torch.float32, device=device)
    positive_mask = (torch.empty(images, PATCH_TOKENS, dtype=torch.float32, device=device)
                     if positive_columns is not None else None)
    found = torch.zeros(images, dtype=torch.bool, device=device)
    for i0 in range(0, images, image_chunk):
        i1 = min(i0 + image_chunk, images)
        for j0 in range(0, texts, text_chunk):
            j1 = min(j0 + text_chunk, texts)
            arguments = (x11[i0:i1], native['c'][i0:i1], native['u'][i0:i1], native['k'][i0:i1],
                         native['v'][i0:i1], native['attention'][i0:i1], qT[j0:j1], kG[i0:i1])
            if use_checkpoint:
                outputs = activation_checkpoint(
                    _conditional_block, *arguments, gate.bias, last.attn.out_proj, last.ln_2,
                    last.mlp, ln_post, visual_proj, gate.key_dim,
                    use_reentrant=False, preserve_rng_state=True)
            else:
                outputs = _conditional_block(*arguments, gate.bias, last.attn.out_proj, last.ln_2,
                                             last.mlp, ln_post, visual_proj, gate.key_dim)
            projected, mask, probability, denominator, attention_conditional, cls = outputs
            if not torch.isfinite(denominator).all() or bool((denominator <= 0).any()):
                raise RuntimeError('non-finite or non-positive conditional attention denominator '
                                   '(block images %d:%d, texts %d:%d)' % (i0, i1, j0, j1))
            normalised = F.normalize(projected, dim=-1, eps=NORM_EPS)
            scores[i0:i1, j0:j1] = FIXED_SCALE * torch.einsum('jid,jd->ij', normalised,
                                                              t_unit[j0:j1].float())
            if positive_mask is not None:
                # the positive column of local image i is rank*B + i; take its patch gate from THIS
                # tile when the tile contains it (no second gate evaluation)
                for local in range(i0, i1):
                    column = int(positive_columns[local])
                    if j0 <= column < j1:
                        positive_mask[local] = mask[column - j0, local - i0]
                        found[local] = True
            if capture is not None and i1 == images and j1 == texts:
                capture['attention_conditional'] = attention_conditional.detach()
                capture['cls'] = cls.detach()
                capture['projected'] = projected.detach()
                capture['mask'] = mask.detach()
                capture['probability'] = probability.detach()
                capture['block'] = (i0, i1, j0, j1)
            if progress is not None:
                progress(i1, images, j1, texts)
    if positive_mask is not None and not bool(found.all()):
        raise RuntimeError('the positive patch gates were not collected for %d of %d local images '
                           '(check the chunking and the global positive columns)'
                           % (int((~found).sum()), images))
    return scores, positive_mask


# --------------------------------------------------------------------------- losses and metrics
def lse_margin(rows: torch.Tensor, positive_index: torch.Tensor) -> torch.Tensor:
    """``positive - logsumexp(scores of the negatives only)`` -- the definition used everywhere.

    ``logsumexp(all) - positive`` is the cross entropy and must never be reported as the LSE margin.
    """
    masked = rows.clone()
    masked.scatter_(1, positive_index.view(-1, 1), float('-inf'))
    return rows.gather(1, positive_index.view(-1, 1)).squeeze(1) - torch.logsumexp(masked, dim=1)


def cross_entropy_from_lse_margin(margin: torch.Tensor) -> torch.Tensor:
    """``CE = softplus(-margin)`` for the same rows, used to cross-check the two quantities."""
    return F.softplus(-margin)


def sparse_positive_term(positive_mask: torch.Tensor) -> torch.Tensor:
    """``LS = mean`` of the 196 patch gates of the TRUE positive pairs (CLS slot excluded)."""
    return positive_mask.mean()


def total_loss(loss_global: torch.Tensor, loss_attention: torch.Tensor,
               loss_sparse: torch.Tensor) -> torch.Tensor:
    return (LAMBDA_GLOBAL * loss_global + LAMBDA_ATTENTION * loss_attention
            + LAMBDA_SPARSE * loss_sparse)


def cross_entropy_both_directions(scores: torch.Tensor, targets: torch.Tensor):
    return F.cross_entropy(scores, targets), F.cross_entropy(scores.t(), targets)


def path_statistics(rows: torch.Tensor, targets: torch.Tensor) -> dict:
    """Positive / strongest negative / max margin / LSE margin / CE / top1 for one direction."""
    with torch.no_grad():
        positive = rows.gather(1, targets.view(-1, 1)).squeeze(1)
        margin = lse_margin(rows, targets)
        strongest = rows.clone()
        strongest.scatter_(1, targets.view(-1, 1), float('-inf'))
        strongest = strongest.max(dim=1).values
        return {
            'positive_mean': float(positive.mean()),
            'strongest_negative_mean': float(strongest.mean()),
            'max_margin_mean': float((positive - strongest).mean()),
            'max_margin_min': float((positive - strongest).min()),
            'lse_margin_mean': float(margin.mean()),
            'lse_margin_min': float(margin.min()),
            'ce_mean': float(F.cross_entropy(rows, targets)),
            'ce_from_lse_margin_max_abs_diff': float(
                (F.cross_entropy(rows, targets, reduction='none')
                 - cross_entropy_from_lse_margin(margin)).abs().max()),
            'top1': float((rows.argmax(dim=1) == targets).float().mean()),
            'positive_win_fraction': float(((positive - strongest) > 0).float().mean()),
        }


def gate_statistics(mask: torch.Tensor, probability: torch.Tensor, sample_pairs: int = 16) -> dict:
    """Patch-gate statistics with bounded pair diagnostics (never a full B^2 table)."""
    with torch.no_grad():
        patches = mask[..., 1:] if mask.shape[-1] == VISUAL_TOKENS else mask
        soft = probability[..., 1:] if probability.shape[-1] == VISUAL_TOKENS else probability
        kept = (patches.detach() >= 0.5)
        per_pair = kept.float().sum(dim=-1)
        stats = {
            'gate_kept_mean': float(per_pair.mean()),
            'gate_kept_min': float(per_pair.min()),
            'gate_kept_max': float(per_pair.max()),
            'gate_keep_fraction_mean': float(per_pair.mean() / PATCH_TOKENS),
            'gate_all_off_fraction': float((per_pair == 0).float().mean()),
            'gate_all_on_fraction': float((per_pair == PATCH_TOKENS).float().mean()),
            'gate_coordinate_mean': float(kept.float().mean()),
            'gate_probability_mean': float(soft.mean()),
            'gate_probability_std': float(soft.std()),
            'gate_probability_min': float(soft.min()),
            'gate_probability_max': float(soft.max()),
            'gate_probability_near_threshold_fraction': float(((soft - 0.5).abs() < 0.05).float().mean()),
            'gate_probability_quantiles': [float(v) for v in torch.quantile(
                _bounded(soft), torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=soft.device))],
        }
        if soft.dim() == 3 and soft.shape[0] > 1 and soft.shape[1] > 1:
            sample = min(sample_pairs, soft.shape[0], soft.shape[1])
            row = soft[:sample, :sample, :]
            column = soft[:sample, :sample, :]
            # soft is [texts, images, patches]: dim 1 is the image, dim 0 is the text
            stats['gate_variation_across_images_same_text'] = float(
                row.std(dim=1).mean())
            stats['gate_variation_across_texts_same_image'] = float(
                column.std(dim=0).mean())
            stats['gate_pair_sample'] = int(sample)
        return stats


def _bounded(values: torch.Tensor, limit: int = 1 << 22):
    flat = values.detach().float().flatten()
    if flat.numel() <= limit:
        return flat
    stride = int(math.ceil(flat.numel() / float(limit)))
    return flat[::stride]


def attention_read_diagnostics(native: dict, conditional: dict) -> dict:
    """How much of the CLS read is CLS-self mass, patch mass, and how big the added increment is."""
    with torch.no_grad():
        native_attention = native['attention']                        # [B,H,1,197]
        conditional_attention = conditional['attention_conditional']  # [bj,bi,H,197]
        native_self = native_attention[..., 0]
        conditional_self = conditional_attention[..., 0]
        native_cls = native['cls']                                    # [B,768]
        conditional_cls = conditional['cls']                          # [bj,bi,768]
        native_projected = native['projected']
        conditional_projected = conditional['projected']
        cosine_cls = F.cosine_similarity(conditional_cls, native_cls.unsqueeze(0), dim=-1)
        cosine_projected = F.cosine_similarity(conditional_projected,
                                               native_projected.unsqueeze(0), dim=-1)
        return {
            'native_cls_self_mass_mean': float(native_self.mean()),
            'native_cls_self_mass_per_head_min': float(native_self.mean(dim=0).min()),
            'native_cls_self_mass_per_head_max': float(native_self.mean(dim=0).max()),
            'native_patch_mass_mean': float(native_attention[..., 1:].sum(-1).mean()),
            'conditional_cls_self_mass_mean': float(conditional_self.mean()),
            'conditional_patch_mass_mean': float(conditional_attention[..., 1:].sum(-1).mean()),
            'native_out_norm_ratio_mean': float(
                (native['out'].norm(dim=-1) / native['c'].norm(dim=-1).clamp_min(1e-12)).mean()),
            'conditional_vs_native_cls_cosine_mean': float(cosine_cls.mean()),
            'conditional_vs_native_projected_cosine_mean': float(cosine_projected.mean()),
            'scope': 'rank-local batch; the conditional entries average over the (text, image) tiles',
        }


# --------------------------------------------------------------------------- distributed helpers
def gather_rows(tensor: torch.Tensor, group=None) -> torch.Tensor:
    """Autograd-aware row gather in global rank order (identity when not distributed)."""
    if not dist.is_initialized() or dist.get_world_size(group) == 1:
        return tensor
    return torch.cat(torch.distributed.nn.all_gather(tensor, group=group), dim=0)


def global_targets(local_size: int, rank: int, device) -> torch.Tensor:
    return torch.arange(rank * local_size, (rank + 1) * local_size, dtype=torch.long, device=device)


def positive_global_columns(local_size: int, rank: int, world_size: int = None) -> torch.Tensor:
    """The global candidate column of each local image's true caption (never the local diagonal)."""
    return torch.arange(rank * local_size, (rank + 1) * local_size, dtype=torch.long)


def assert_equal_local_batch(local_size: int, world_size: int = None, group=None,
                             device=None) -> None:
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
        raise RuntimeError('ragged local batch across ranks: %s (equal sizes are required before the '
                           'first feature gather)' % values)


# --------------------------------------------------------------------------- parameters
def partition_parameters(clip: nn.Module, gate: nn.Module) -> dict:
    """The shared CLIP group and the caption-gate group, by parameter id.

    ``clip.mask_net`` (legacy state compatibility only) and ``clip.logit_scale`` (unused, the score
    scale is the fixed 100) are frozen and excluded; every shared parameter, including the last
    block and ``visual.proj``, appears exactly once.
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
        raise RuntimeError('optimizer groups do not cover every trainable parameter')
    last_block_ids = {id(p) for p in clip.visual.transformer.resblocks[-1].parameters()}
    projection_id = id(clip.visual.proj)
    group_ids = set(ids)
    missing = [name for name, parameter in clip.named_parameters()
               if parameter.requires_grad
               and id(parameter) not in group_ids
               and (id(parameter) in last_block_ids or id(parameter) == projection_id)]
    if missing:
        raise RuntimeError('last block / visual.proj parameters are missing from the groups: %r'
                           % missing)
    if [id(p) for p in clip_parameters].count(projection_id) != 1:
        raise RuntimeError('clip.visual.proj must be in exactly one optimizer group')
    return {'clip': clip_parameters, 'gate': gate_parameters, 'frozen': frozen,
            'clip_count': len(clip_parameters), 'gate_count': len(gate_parameters),
            'last_block_tensors': len(last_block_ids)}


def build_optimizers(clip: nn.Module, gate: nn.Module, clip_lr: float = 1e-6,
                     gate_lr: float = 1e-3, weight_decay: float = 1e-2,
                     betas=(0.9, 0.999), eps: float = 1e-8) -> dict:
    groups = partition_parameters(clip, gate)
    clip_optimizer = torch.optim.AdamW(groups['clip'], lr=clip_lr, weight_decay=weight_decay,
                                       betas=betas, eps=eps)
    gate_optimizer = torch.optim.AdamW(groups['gate'], lr=gate_lr, weight_decay=0.0,
                                       betas=betas, eps=eps)
    return {'clip': clip_optimizer, 'gate': gate_optimizer, 'partition': groups}


# --------------------------------------------------------------------------- checkpoint identity
def check_resume_compatible(payload: dict, expected: dict) -> None:
    """Refuse a silent resume across objective / gate kind / coefficient changes."""
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
                        gate_digest: str, clip_digest: str, lr_horizon_steps: int = None,
                        gate_report: dict = None, spec: dict = None) -> dict:
    """Self-describing checkpoint header.

    The gate *tensors* live under ``gate_state`` (written by the trainer); this header owns
    ``gate_config`` and must never collide with them. ``gate_state_digest`` here is the digest of the
    real gate state dict, so a checkpoint whose weights went missing is detectable.
    """
    horizon = int(lr_horizon_steps if lr_horizon_steps is not None
                  else (config.get('lr_horizon_steps') or 0))
    return {
        'objective': OBJECTIVE, 'arm': ARM, 'phase': PHASE, 'gate_kind': GATE_KIND,
        'gate_key_dim': GATE_KEY_DIM, 'gate_seed': GATE_SEED,
        'gate_config': dict(config.get('gate') or {}),
        'gate_init_report': dict(gate_report or {}),
        'gate_state_digest': gate_digest,
        'clip_state_digest': clip_digest,
        'completed_steps': int(steps), 'rank': int(rank), 'world_size': int(world),
        'batch_size_per_gpu': int(batch_size), 'global_batch': int(batch_size) * int(world),
        'lr_horizon_steps': horizon,
        'loss_weights': dict(LOSS_WEIGHTS), 'fixed_scale': FIXED_SCALE, 'norm_eps': NORM_EPS,
        'caption_scope': {'cls_sparse': 'true positive pairs only, 196 patch gates, CLS slot '
                                        'excluded', 'conditional_path': 'training only'},
        'attention_route': {'cls_query': 'native last-block query, never replaced by text',
                            'keys_values': 'native last-block keys/values, shared by both paths',
                            'gate_sharing': 'one 196-d patch gate shared by all 12 heads',
                            'renormalisation': 'a_cond = (a * m_full) / sum_p (a * m_full), no '
                                               'detached denominator, no clamp'},
        'visual_spec': dict(spec or {}),
        'precision': precision, 'chunking': dict(chunking), 'config': dict(config),
        'digests': dict(digests),
    }


def state_digest(state_dict) -> str:
    import hashlib
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key]
        if torch.is_tensor(value):
            digest.update(key.encode('utf-8'))
            digest.update(value.detach().to(torch.float32).cpu().numpy().tobytes())
    return digest.hexdigest()


def file_sha256(path: str) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(repo=None) -> str:
    import subprocess
    repo = repo or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        return subprocess.run(['git', '-C', repo, 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, timeout=20).stdout.strip()
    except Exception:                                            # pragma: no cover
        return 'unknown'


def attention_map_grid(mask_196: torch.Tensor, side: int = 14) -> dict:
    """Reshape one 196-d patch gate into the 14x14 grid the page draws.

    The grid is a *token order* layout of the 14x14 patch sequence, not a semantic map: the tokens
    come from the 11th block and are already contextualised.
    """
    flat = mask_196.detach().float().flatten()
    if flat.numel() != side * side:
        raise ValueError('expected %d patch gates, got %d' % (side * side, flat.numel()))
    rows = [[float(v) for v in row] for row in flat.reshape(side, side).tolist()]
    return {'side': side, 'grid': [v for row in rows for v in row], 'rows': rows,
            'note': 'token order of the 14x14 patch sequence from block 11; not a semantic map'}
