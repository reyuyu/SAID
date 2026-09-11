"""SAID-ExGAP v1.5 -- pre-final evidence routing read out by the ORIGINAL final block.

One sentence: *route caption relevance over the pre-final visual tokens ``H11``, but never
replace CLIP's native global readout -- instead use the original final Transformer block itself
to produce Global, Said and complementary final-CLS representations by softly reweighting only
the CLS-to-patch attention row.*

What this module does
---------------------
CLIP ViT-B/16 has 12 transformer blocks. ``X11 = [c11, H11]`` is the token sequence entering
block 12 (see ``VisionTransformer.forward_prefinal``). Block 12 is the *native* producer of the
image representation::

    g = Norm( ln_post( block12(X11)[CLS] ) @ proj )

This module leaves blocks 1..11 and every parameter of block 12 untouched. It only changes the
**CLS query row** of block 12's self-attention, by adding a per-patch bias before the softmax::

    A^G_p  = softmax_p( a_p )                                 Global   (bias = 0, native)
    A^S_p  = softmax_p( a_p + log r_p )                       Said     (live gate)
    A^U_p  = softmax_p( a_p + log(1 - sg(r_p)) )              Unsaid   (detached complement)

with ``a_p = q_CLS . k_p / sqrt(d_head)`` the *unmodified* native CLS attention logits, and
``bias[CLS key] = 0`` always (the CLS self key is never suppressed).

Everything after the attention -- ``out_proj``, the CLS residual, ``ln_2``, the MLP, the second
residual, ``ln_post`` and the visual projection -- is the original block-12 code path, executed
on the CLS row only (the MLP is token-wise, so the CLS row can be finished on its own).

Exactness
---------
* ``read_global`` reproduces ``clip.encode_image`` (hard gate, tested at 1e-5 in fp32).
* ``r == 1`` gives ``log r == 0`` so ``z_S == g`` (ones-gate identity, tested).
* The optimized CLS-only path is checked against a slow reference that runs the full block with
  a batch-specific additive attention mask (tested for B=2, C=2).

Why CLS-only is exact
---------------------
Only the CLS query row is gated, so ``q_CLS`` and all ``K/V`` are identical to the native block;
the patch query rows are neither recomputed nor masked. Reading out the CLS row of the final
block therefore needs ``LN1``, one ``Q_cls``, ``K_all``, ``V_all`` and the CLS-only MLP -- never
a ``[B, C, 197, 768]`` sequence and never ``B*B`` full block runs.

Terminology (deliberately *not* "semantic removal"): ``c11`` already carries global information
from the previous 11 layers, so this is a **final-stage evidence aggregation** intervention, not
a complete causal erasure of visual semantics.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_GATE_EPS = 1e-6
PARAMETER_FREE_NOTE = ('no new trainable parameter: every tensor used here belongs to the '
                       'original visual encoder')


# --------------------------------------------------------------------------- #
# Said relevance (shared definition with the v1 Said router logit)
# --------------------------------------------------------------------------- #
def said_relevance_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """``r = sigmoid(logits)`` -- an independent per-patch relevance, never a spatial softmax."""
    return torch.sigmoid(logits)


def pairwise_said_relevance(text_features: torch.Tensor, routing_patches: torch.Tensor,
                            w_query: nn.Linear, w_key: nn.Linear,
                            tau_said: float, chunk_size: Optional[int] = None) -> torch.Tensor:
    """``r[i, j, p] = sigmoid( (q_hat(t_j) . k_hat(h_i_p)) / tau_said )`` -> ``[B, C, N]``.

    Batched form of ``exgap.said_relevance_logits`` with the *same* normalisation convention
    (``q_hat = Norm(W_Q t)``, ``k_hat = Norm(W_K h)``); the candidate dimension is chunked so the
    ``[B, C, N]`` relevance is never materialised for the whole batch at once.
    """
    if float(tau_said) <= 0.0:
        raise ValueError('tau_said must be positive, got %r' % (tau_said,))
    n_candidates = int(text_features.shape[0])
    step = n_candidates if not chunk_size else min(int(chunk_size), n_candidates)
    if int(chunk_size or 0) < 0 or step <= 0:
        raise ValueError('chunk_size must be >= 0, got %r' % (chunk_size,))
    k_hat = F.normalize(w_key(routing_patches), dim=-1)
    chunks = []
    for start in range(0, n_candidates, step):
        stop = min(start + step, n_candidates)
        q_hat = F.normalize(w_query(text_features[start:stop]), dim=-1)
        logits = torch.einsum('cd,bpd->bcp', q_hat, k_hat) / float(tau_said)
        chunks.append(torch.sigmoid(logits))
    return torch.cat(chunks, dim=1)


def gate_bias(relevance: torch.Tensor, eps: float = DEFAULT_GATE_EPS,
              complement: bool = False) -> torch.Tensor:
    """``log(clamp(r, eps, 1))`` (or the ``1 - r`` complement) as a ``[B, 197]`` additive bias.

    Position 0 is the CLS key and always gets bias ``0``: the CLS self key is never suppressed.
    """
    if relevance.dim() != 2:
        raise ValueError('relevance must be [B, N], got %r' % (tuple(relevance.shape),))
    values = (1.0 - relevance) if complement else relevance
    patch_bias = torch.log(values.clamp(min=float(eps), max=1.0))
    return F.pad(patch_bias, (1, 0), value=0.0)


# --------------------------------------------------------------------------- #
# CLS-only readout of the original final block
# --------------------------------------------------------------------------- #
class FinalBlockCLSReadout(nn.Module):
    """Read the CLS row of the ORIGINAL final visual block under an additive patch gate.

    The module owns **no parameter**: it holds a reference to the visual encoder and uses
    ``visual.transformer.resblocks[-1]`` (``ln_1`` / ``attn.in_proj_weight`` / ``attn.in_proj_bias``
    / ``attn.out_proj`` / ``ln_2`` / ``mlp``) plus ``visual.ln_post`` and ``visual.proj`` directly,
    so Global / Said / Unsaid always share one final block and one visual projection.

    The reference is stored with ``object.__setattr__`` on purpose: a plain ``self.visual = visual``
    would register the whole vision tower as a *second* submodule of this module, duplicating every
    parameter in ``named_parameters()`` / ``state_dict()`` / the optimizer groups.
    """

    def __init__(self, visual, eps: float = DEFAULT_GATE_EPS):
        super().__init__()
        block = visual.transformer.resblocks[-1]
        attn = block.attn
        if not isinstance(attn, nn.MultiheadAttention):
            raise TypeError('final block attention must be nn.MultiheadAttention, got %r'
                            % (type(attn).__name__,))
        if attn.batch_first:
            raise ValueError('the reference final block must be batch_first=False (L, N, E)')
        self.num_heads = int(attn.num_heads)
        self.width = int(attn.embed_dim)
        if self.width % self.num_heads:
            raise ValueError('width %d is not divisible by heads %d' % (self.width, self.num_heads))
        self.head_dim = self.width // self.num_heads
        self.eps = float(eps)
        self.n_patches = int(visual.positional_embedding.shape[0]) - 1
        object.__setattr__(self, '_visual', visual)

    # -- original modules (shared, never copied) ---------------------------- #
    @property
    def visual(self):
        return object.__getattribute__(self, '_visual')

    @property
    def block(self):
        return self.visual.transformer.resblocks[-1]

    def gate_parameters(self):
        """The original final-block tensors this readout is allowed to use."""
        block = self.block
        return {
            'ln_1': tuple(block.ln_1.parameters()),
            'attn_in_proj_weight': (block.attn.in_proj_weight,),
            'attn_in_proj_bias': (block.attn.in_proj_bias,) if block.attn.in_proj_bias is not None
            else (),
            'attn_out_proj': tuple(block.attn.out_proj.parameters()),
            'ln_2': tuple(block.ln_2.parameters()),
            'mlp': tuple(block.mlp.parameters()),
            'ln_post': tuple(self.visual.ln_post.parameters()),
            'proj': (self.visual.proj,) if self.visual.proj is not None else (),
        }

    def trainable_parameters(self):
        """Every original parameter reachable from this readout (no new ones exist)."""
        parameters = []
        for group in self.gate_parameters().values():
            parameters.extend(group)
        return parameters

    # -- preparation -------------------------------------------------------- #
    def prepare(self, x11_raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Pre-compute the final-block cache from ``X11`` ``[B, 197, width]`` (raw hidden space).

        Computes ``LN1`` once, then ``Q_cls`` (CLS query row only) and ``K_all`` / ``V_all`` and
        the native CLS attention logits. Global, Said, Unsaid and every pairwise candidate reuse
        this single cache, so the expensive projections are computed exactly once per image.
        """
        if x11_raw.dim() != 3:
            raise ValueError('x11_raw must be [B, 197, width], got %r'
                             % (tuple(x11_raw.shape),))
        batch, length, width = x11_raw.shape
        if width != self.width:
            raise ValueError('x11_raw width %d != model width %d' % (width, self.width))
        if length != self.n_patches + 1:
            raise ValueError('x11_raw length %d != 1 + %d patches'
                             % (length, self.n_patches))
        block = self.block
        attn = block.attn
        x = block.ln_1(x11_raw)                                  # [B, L, D]
        qkv = F.linear(x, attn.in_proj_weight, attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        def _heads(tensor):
            # [B, L, D] -> [B, H, L, head_dim] (the layout torch's MHA uses internally)
            return tensor.reshape(batch, length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q = _heads(q)
        k = _heads(k)
        v = _heads(v)
        # torch's need_weights=False path scales q by sqrt(1/E) before the bmm; replicated exactly
        q_cls = q[:, :, :1, :] * (self.head_dim ** -0.5)
        base_logits = torch.matmul(q_cls, k.transpose(-1, -2))   # [B, H, 1, L]
        return {
            'base_logits': base_logits[..., 0, :],               # [B, H, L]
            'values': v,                                         # [B, H, L, head_dim]
            'cls11_raw': x11_raw[:, 0, :],                       # [B, D]
            'batch': batch,
            'length': length,
        }

    # -- readout ------------------------------------------------------------ #
    def _finish_cls(self, attention_output: torch.Tensor, cls11_raw: torch.Tensor):
        """``out_proj -> CLS residual -> ln_2 -> MLP -> residual -> ln_post -> proj -> Norm``.

        ``attention_output`` is ``[..., D]`` with the heads already concatenated; the whole
        tail is token-wise, so it runs on the CLS row alone (any leading batch/candidate dims
        are flattened and restored). No ``[B, C, 197, 768]`` sequence is ever built.
        """
    def _finish_cls(self, attention_output: torch.Tensor, cls11_raw: torch.Tensor,
                    normalize: bool = False):
        """``out_proj -> CLS residual -> ln_2 -> MLP -> residual -> ln_post -> proj``.

        ``attention_output`` is ``[..., D]`` with the heads already concatenated; the whole
        tail is token-wise, so it runs on the CLS row alone (any leading batch/candidate dims
        are flattened and restored). No ``[B, C, 197, 768]`` sequence is ever built.

        The result is the RAW visual feature -- exactly what ``clip.encode_image`` returns -- so
        ``read_global`` can be compared with the native forward bit for bit. Callers that need a
        cosine similarity pass ``normalize=True`` (or normalise themselves).
        """
        block = self.block
        leading = attention_output.shape[:-1]
        # 1. attention output projection (original weights), applied to the attention result
        flat = attention_output.reshape(-1, self.width)
        out = F.linear(flat, block.attn.out_proj.weight, block.attn.out_proj.bias)
        # 2. CLS residual, broadcast over any candidate dims: c_attn = c11 + out_proj(attn)
        cls = cls11_raw.reshape(leading[0], *([1] * (len(leading) - 1)), self.width)
        c_attn = (cls + out.reshape(*leading, self.width)).reshape(-1, self.width)
        # 3. token-wise tail: ln_2 -> MLP -> residual
        c12 = c_attn + block.mlp(block.ln_2(c_attn))
        feature = self.visual.ln_post(c12)
        if self.visual.proj is not None:
            feature = feature @ self.visual.proj
        feature = feature.reshape(*leading, -1)
        if normalize:
            return F.normalize(feature, dim=-1)
        return feature

    def _attend(self, prepared: Dict[str, torch.Tensor], bias: torch.Tensor) -> torch.Tensor:
        """``softmax(a + bias) @ V`` for the CLS row -> ``[B, H * head_dim]``.

        ``bias`` broadcasts against ``base_logits`` ``[B, H, L]`` (a zero CLS column plus a
        patch-only gate, or plain zeros for the native readout).
        """
        logits = prepared['base_logits'] + bias
        attention = F.softmax(logits, dim=-1)
        out = torch.matmul(attention.unsqueeze(-2), prepared['values'])[..., 0, :]  # [B, H, hd]
        return out.reshape(prepared['batch'], self.width)

    def attention_weights(self, prepared: Dict[str, torch.Tensor], relevance: torch.Tensor,
                          complement: bool = False,
                          eps: Optional[float] = None) -> torch.Tensor:
        """``softmax(a + bias)`` -> ``[B, H, L]`` (diagnostics / gate-semantics tests).

        Position 0 is the CLS key and is never biased.
        """
        bias = gate_bias(relevance, eps=self.eps if eps is None else eps,
                         complement=complement)[:, None, :]
        return F.softmax(prepared['base_logits'] + bias, dim=-1)

    def read_global(self, prepared: Dict[str, torch.Tensor],
                    normalize: bool = False) -> torch.Tensor:
        """Native global final CLS: ``softmax(a)`` -- equals ``clip.encode_image``."""
        zero_bias = torch.zeros_like(prepared['base_logits'][:, :1, :])
        return self._finish_cls(self._attend(prepared, zero_bias), prepared['cls11_raw'],
                                normalize=normalize)

    def read_said(self, prepared: Dict[str, torch.Tensor],
                  relevance: torch.Tensor, eps: Optional[float] = None,
                  normalize: bool = False) -> torch.Tensor:
        """Said final CLS with a **live** gate ``A^S = softmax(a + log r)`` -> ``[B, D_out]``."""
        bias = gate_bias(relevance, eps=self.eps if eps is None else eps)[:, None, :]
        return self._finish_cls(self._attend(prepared, bias), prepared['cls11_raw'],
                                normalize=normalize)

    def read_unsaid(self, prepared: Dict[str, torch.Tensor],
                    relevance_detached: torch.Tensor,
                    eps: Optional[float] = None,
                    normalize: bool = False) -> torch.Tensor:
        """Complementary final CLS ``A^U = softmax(a + log(1 - sg(r)))`` -> ``[B, D_out]``.

        The caller must pass an **already detached** relevance: the complement is a constant, so
        the ExGAP gradient can never reach the Said router.
        """
        if relevance_detached.requires_grad:
            raise ValueError('read_unsaid requires a detached relevance (no router gradient)')
        bias = gate_bias(relevance_detached, eps=self.eps if eps is None else eps,
                         complement=True)[:, None, :]
        return self._finish_cls(self._attend(prepared, bias), prepared['cls11_raw'],
                                normalize=normalize)

    def read_pairwise_said(self, prepared: Dict[str, torch.Tensor],
                           pairwise_relevance: torch.Tensor,
                           chunk_size: int = 32,
                           eps: Optional[float] = None,
                           normalize: bool = False) -> torch.Tensor:
        """``[B, C, D_out]`` Said final CLS for ``C`` routing captions per image.

        The candidate dimension is chunked; per chunk the bias is broadcast over heads, so the
        materialised attention is ``[B, c, H, 197]`` at most (never ``[B, B, 197, 768]``).
        """
        if pairwise_relevance.dim() != 3:
            raise ValueError('pairwise_relevance must be [B, C, N], got %r'
                             % (tuple(pairwise_relevance.shape),))
        batch = prepared['batch']
        if pairwise_relevance.shape[0] != batch:
            raise ValueError('pairwise_relevance batch %d != prepared batch %d'
                             % (pairwise_relevance.shape[0], batch))
        candidates = pairwise_relevance.shape[1]
        step = candidates if not chunk_size else min(int(chunk_size), candidates)
        if step <= 0:
            raise ValueError('chunk_size must be positive, got %r' % (chunk_size,))
        eps = self.eps if eps is None else eps
        outputs = []
        for start in range(0, candidates, step):
            stop = min(start + step, candidates)
            chunk = pairwise_relevance[:, start:stop]                      # [B, c, N]
            bias = torch.log(chunk.clamp(min=float(eps), max=1.0))
            bias = F.pad(bias, (1, 0), value=0.0)                          # [B, c, L]
            logits = prepared['base_logits'][:, None, :, :] + bias[:, :, None, :]
            attention = F.softmax(logits, dim=-1)                          # [B, c, H, L]
            out = torch.einsum('bchp,bhpd->bchd', attention, prepared['values'])
            outputs.append(self._finish_cls(out.reshape(batch, stop - start, self.width),
                                            prepared['cls11_raw'], normalize=normalize))
        return torch.cat(outputs, dim=1)


# --------------------------------------------------------------------------- #
# Slow reference path (tests only)
# --------------------------------------------------------------------------- #
def reference_final_cls(visual, x11_raw: torch.Tensor, relevance: torch.Tensor,
                        complement: bool = False, eps: float = DEFAULT_GATE_EPS) -> torch.Tensor:
    """Slow, obviously-correct reference: run the FULL final block once per (image, candidate).

    Used only to prove the optimized CLS-only readout. For every candidate ``j`` of every image
    ``i`` it builds the ``[L, L]`` additive mask that biases the patch *keys* (all query rows are
    computed by the real block and only row 0 is kept) and runs the block end to end, followed by
    the original ``ln_post`` and ``proj``. Returns the raw ``[B, C, output_dim]`` features.

    Deliberately loop-based and O(B*C) full-block runs: it exists to be trusted, not to be fast.
    """
    block = visual.transformer.resblocks[-1]
    if relevance.dim() == 2:
        relevance = relevance[:, None, :]
    batch, candidates, patches = relevance.shape
    if x11_raw.shape[1] != patches + 1:
        raise ValueError('x11_raw has %d tokens for %d patches' % (x11_raw.shape[1], patches))
    outputs = []
    for image in range(batch):
        rows = []
        for candidate in range(candidates):
            tokens = x11_raw[image:image + 1].permute(1, 0, 2)        # [L, 1, D]
            bias = gate_bias(relevance[image, candidate:candidate + 1], eps=eps,
                             complement=complement)[0]               # [L], CLS column = 0
            mask = bias[None, :].expand(tokens.shape[0], tokens.shape[0]).clone()   # [L, L]
            out = block.attention_with_mask(block.ln_1(tokens), mask)              # [L, 1, D]
            # row 0 of the SEQUENCE is the CLS token (L, N, D layout)
            cls = tokens[0:1, 0, :] + out[0:1, 0, :]
            cls = cls + block.mlp(block.ln_2(cls))
            feature = visual.ln_post(cls)
            if visual.proj is not None:
                feature = feature @ visual.proj
            rows.append(feature)
        outputs.append(torch.cat(rows, dim=0))
    return torch.stack(outputs, dim=0)


def attach_reference_helper(visual):
    """Install ``ResidualAttentionBlock.attention_with_mask`` (test/diagnostic helper).

    Kept out of the training path: the reference runs a full ``[B*C, 197, 197]`` attention, which
    is exactly what the optimized readout exists to avoid.
    """
    from model.model_longclip import ResidualAttentionBlock

    if hasattr(ResidualAttentionBlock, 'attention_with_mask'):
        return

    def attention_with_mask(self, x, attn_mask):
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    ResidualAttentionBlock.attention_with_mask = attention_with_mask
