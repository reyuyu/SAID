"""SAID-Token v1 (experiment T1): token-level Said / Unsaid with a frozen-reference reconstruction.

One sentence: aggregate the native last-layer local tokens of CLIP into 32 image slots and 32 text
slots per modality, pick (per image-text candidate pair) 16 visual slots as *Said* by a
text-conditioned router, run one fine-grained best-match ranking objective over
``{native CLS} u {16 Said slots}`` against ``{32 text slots} u {native EOS}``, and let the
complementary 16 visual slots plus the observed text predict the complete feature of a frozen copy of
the initial vision tower.

    L = L_Said + 0.1 * L_rec

There is deliberately NO separate global image-text loss and no global pre-training stage: the native
CLS and EOS stay inside the same fine-grained matching set.

Frozen specification implemented here:

* last-layer interface -- image patches come back already through ``ln_post`` and ``visual.proj``;
  the text hidden state gets the original ``text_projection`` applied exactly once; native CLS/EOS are
  kept as separate candidates. ``d`` is read from ``text_projection.shape[-1]``.
* self-aggregation -- ``A(X) = softmax_L(f(X)^T)``, ``X' = A(X) X`` with
  ``f = LayerNorm(d) -> Linear(d,102) -> GELU -> Linear(102,32)``, logit scale fixed to 1; one
  aggregator per modality, no shared parameters, CLS/EOS never aggregated, text padding/SOT/EOT never
  attended.
* router -- ``a = <Norm(W_V V_p), Norm(W_T g_T)> / tau_r`` over the 32 *aggregated visual* slots ``p``,
  where the query side is the candidate text's **native global EOS feature** ``g_T`` (not the aggregated
  text slots, and with no text-token axis, hence no reduction over ``q``); both projections
  ``Linear(d,128,bias=False)``, ``tau_r = 0.07``; ``hard = 1{p in TopK(a,16)}`` over the trailing visual
  axis only, deterministic on ties, native CLS excluded (it is not even in ``a``). Every candidate pair
  re-evaluates the router, so the mask is a function of *that* candidate text.
* hard-forward / score-layer-backward proxy -- ``s = s_hard + (s_tilde - sg(s_tilde))`` with
  ``s_tilde`` built from a detached cosine matrix, so the forward value is exactly the hard score and
  only the router's ``sigmoid(a)`` receives the proxy gradient. Biased straight-through proxy, not the
  exact derivative of top-k.
* Said set -- ``{g_I} u {selected slots}`` (17) against ``{32 text slots} u {g_T}`` (33), per-token
  normalised, bidirectional max-similarity match, hinge margin 0.2, mean over valid negative pairs.
* U branch -- ``m_U = 1 - sg(h_ii)``, ``u = mean_p m_U,ip V_ip`` over the raw aggregated slots,
  ``g_hat = D([Norm(u); sg(g_T)], L_rec = mean(1 - Norm(g_hat)^T sg(g0))`` with
  ``g0 = Norm(E_I^0(I_a))`` from a frozen deep copy of the *initial* visual tower.
"""
import copy
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# frozen constants (first version: fixed, not tuned)
# --------------------------------------------------------------------------- #
N_SLOTS = 32
K_SAID = 16
AGG_HIDDEN = 102
ROUTER_DIM = 128
ROUTER_TAU = 0.07
SURROGATE_ETA = 0.1
MARGIN = 0.2
LAMBDA_REC = 0.1
EPS = 1e-6
ARM_T1 = 'T1_said_token_reconstruction'
ARM_T0 = 'T0_said_token_only'
ARMS = (ARM_T1, ARM_T0)
ARM_LAMBDA_REC = {ARM_T1: LAMBDA_REC, ARM_T0: 0.0}
N_VISUAL_CANDIDATES = 1 + N_SLOTS          # native CLS + 32 aggregated slots
N_TEXT_CANDIDATES = 1 + N_SLOTS            # native EOS + 32 aggregated slots


@contextmanager
def isolated_rng(seed: int = 0):
    """Snapshot/seed/restore the ambient RNG so new modules never disturb the data/caption stream.

    ``nn.Linear`` initialisation draws from the global generator; without this, adding the
    aggregators/router/decoder would silently shift every later caption and augmentation draw.
    """
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(int(seed))
        yield
    finally:
        torch.random.set_rng_state(state)


def cls_embedding_dim(clip_model: nn.Module) -> int:
    """The shared embedding width, read from the real parameter (CLIP has no ``embed_dim`` attr)."""
    projection = getattr(clip_model, 'text_projection', None)
    if projection is not None and len(getattr(projection, 'shape', ())) == 2:
        return int(projection.shape[1])
    raise AttributeError('cannot determine the shared embedding width of %r' % type(clip_model))


# --------------------------------------------------------------------------- #
# 1. self-aggregation
# --------------------------------------------------------------------------- #
class SelfAggregator(nn.Module):
    """``softmax_L(f(X)^T) X`` over the *local* tokens of one modality."""

    def __init__(self, dim: int, slots: int = N_SLOTS, hidden: int = AGG_HIDDEN, seed: int = 0):
        super().__init__()
        with isolated_rng(seed):
            self.norm = nn.LayerNorm(dim)
            self.score = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, slots),
            )

    def weights(self, tokens: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``[b, slots, L]`` weights, normalised over the input-token dimension L.

        A row with no legal position (an empty caption: nothing strictly between SOT and EOT) is given
        one legal fallback position instead of an all ``-inf`` row, whose softmax would be NaN and whose
        backward would poison every gradient. The caller still learns about the case through
        ``text_content_mask``'s ``empty`` flag; this function only keeps the arithmetic finite.
        """
        logits = self.score(self.norm(tokens)) * 1.0            # fixed logit scale = 1
        logits = logits.transpose(1, 2)
        if valid is not None:
            if tuple(valid.shape) != (tokens.shape[0], tokens.shape[1]):
                raise ValueError('valid mask %r does not match the token axis %r'
                                 % (tuple(valid.shape), (tokens.shape[0], tokens.shape[1])))
            dead = ~valid.any(dim=-1)
            if bool(dead.any()):
                fallback = torch.zeros_like(valid)
                fallback[:, 0] = True
                valid = torch.where(dead[:, None], fallback, valid)
            logits = logits.masked_fill(~valid[:, None, :], float('-inf'))
        return logits.softmax(dim=-1)

    def forward(self, tokens: torch.Tensor, valid: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = self.weights(tokens, valid)
        return torch.bmm(weights, tokens), weights


# --------------------------------------------------------------------------- #
# 2. text validity from the real tokenizer
# --------------------------------------------------------------------------- #
def text_content_mask(token_ids: torch.Tensor, eot_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(valid, empty)`` over the text *content* positions.

    The end-of-text position is ``argmax`` of the ids, exactly what ``encode_text`` uses to pick the
    pooled feature. Valid content lies strictly between SOT and EOT, so padding, SOT and EOT are all
    excluded. A text with no content position is reported through ``empty`` instead of producing an
    all ``-inf`` softmax.
    """
    length = token_ids.shape[1]
    positions = torch.arange(length, device=token_ids.device)[None, :]
    eot_pos = token_ids.argmax(dim=-1, keepdim=True)
    valid = (positions > 0) & (positions < eot_pos)
    empty = ~valid.any(dim=-1)
    return valid, empty


# --------------------------------------------------------------------------- #
# 3. router, hard score and surrogate score
# --------------------------------------------------------------------------- #
class TokenRouter(nn.Module):
    """Text-conditioned selection of the 16 Said visual slots, evaluated per candidate pair."""

    def __init__(self, dim: int, out_dim: int = ROUTER_DIM, seed: int = 0):
        super().__init__()
        with isolated_rng(seed):
            self.visual = nn.Linear(dim, out_dim, bias=False)
            self.text = nn.Linear(dim, out_dim, bias=False)

    def logits(self, visual_slots: torch.Tensor, text_global: torch.Tensor,
               tau: float = ROUTER_TAU) -> torch.Tensor:
        """``[n, m, n_slots]``: the 32 image slots of ``n`` anchors against ``m`` candidate texts.

        ``a[i, j, p] = <Norm(W_V V[i, p]), Norm(W_T g_T[j])> / tau``. The query is the candidate text's
        *native global EOS feature* ``g_T``, a single vector per candidate: there is no text-token axis
        and therefore nothing to reduce over. The top-k is taken by the caller over the trailing
        visual-slot axis only, and the native CLS is not in this tensor at all.

        ``W_T g_T`` is 2-D (``[m, r]``), which is what keeps the n x m pairing unambiguous: with a 3-D
        right operand a batched matmul would broadcast the leading axes against each other and silently
        drop the candidate dimension (measured on this build), so the shapes are asserted as well.
        """
        if visual_slots.dim() != 3:
            raise ValueError('visual slots must be [n, V, d], got %r' % (tuple(visual_slots.shape),))
        if text_global.dim() != 2:
            raise ValueError('router query must be the native global text feature [m, d], got %r'
                             % (tuple(text_global.shape),))
        if visual_slots.shape[-1] != text_global.shape[-1]:
            raise ValueError('router widths differ: %d vs %d'
                             % (visual_slots.shape[-1], text_global.shape[-1]))
        left = F.normalize(self.visual(visual_slots), dim=-1, eps=EPS)        # [n, V, r]
        right = F.normalize(self.text(text_global), dim=-1, eps=EPS)          # [m, r]
        out = torch.matmul(left, right.transpose(-1, -2)) / tau               # [n, V, m]
        out = out.permute(0, 2, 1).contiguous()                               # [n, m, V]
        expected = (visual_slots.shape[0], text_global.shape[0], visual_slots.shape[1])
        if tuple(out.shape) != expected:
            raise RuntimeError('router logits have shape %r, expected %r'
                               % (tuple(out.shape), expected))
        return out


def hard_gate(logits: torch.Tensor, k: int = K_SAID) -> torch.Tensor:
    """Hard 0/1 gate with exactly ``k`` ones per row, deterministic on ties.

    Ties are broken by **ascending slot index** through a stable sort of the negated logits. ``topk``
    alone is not enough for that contract: it is deterministic for a given input, but with equal values
    it is not guaranteed to keep the lowest indices (measured here: on an all-zero row it selected the
    last ``k`` slots). No noise is injected either -- the mask must be a reproducible function of
    (pair, parameters), never of the random stream.
    """
    if logits.shape[-1] < k:
        raise ValueError('cannot select %d slots out of %d' % (k, logits.shape[-1]))
    order = torch.argsort(-logits, dim=-1, stable=True)[..., :k]
    gate = torch.zeros_like(logits).scatter(-1, order, 1.0)
    if not bool((gate.sum(dim=-1) == float(k)).all()):
        raise RuntimeError('the hard gate did not select exactly %d slots' % k)
    return gate


def _candidate_mask(gate: torch.Tensor) -> torch.Tensor:
    """``[..., 1+n_slots]`` boolean candidate keep-mask: CLS/EOS always kept, masked slots dropped.

    Masked slots are *removed* from the candidate set. Setting them to zero and leaving them in the
    max would let them win whenever every legal cosine is negative.
    """
    keep_tail = gate > 0.5
    leading = torch.ones_like(keep_tail[..., :1], dtype=torch.bool)
    return torch.cat([leading, keep_tail], dim=-1)


def _similarity_grid(visual: torch.Tensor, text: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(v2t_grid, t2v_grid)`` as ``[n, W, m, T]`` and ``[m, T, n, W]``.

    Implemented with explicit reshapes and ``bmm`` rather than ``einsum``: on this torch build
    (2.5.1) the einsum subscript form silently summed over a token axis whenever a letter was reused
    between the operands, producing plausible but wrong shapes. The reshape form has no naming
    ambiguity, and the assertions below turn any layout mistake into an immediate failure.

    Callers pass *blocks*, never the whole global batch, so no ``[n, W, m, T]`` tensor for the full
    1024x1024 pair grid is ever materialised.
    """
    if visual.dim() != 3 or text.dim() != 3:
        raise ValueError('expected [n, W, d] and [m, T, d], got %r and %r'
                         % (tuple(visual.shape), tuple(text.shape)))
    n, width, dim = visual.shape
    m, text_width, text_dim = text.shape
    if dim != text_dim:
        raise ValueError('embedding widths differ: %d vs %d' % (dim, text_dim))
    v2t = torch.bmm(visual.reshape(1, n * width, dim),
                    text.reshape(1, m * text_width, dim).transpose(1, 2))
    v2t = v2t.reshape(n, width, m, text_width)
    t2v = torch.bmm(text.reshape(1, m * text_width, dim),
                    visual.reshape(1, n * width, dim).transpose(1, 2))
    t2v = t2v.reshape(m, text_width, n, width)
    if v2t.shape != (n, width, m, text_width):
        raise RuntimeError('visual->text grid layout is wrong: %r' % (tuple(v2t.shape),))
    if t2v.shape != (m, text_width, n, width):
        raise RuntimeError('text->visual grid layout is wrong: %r' % (tuple(t2v.shape),))
    return v2t, t2v


def hard_said_score(visual_all: torch.Tensor, text_all: torch.Tensor, gate: torch.Tensor
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``s_hard`` plus its two directional parts for ``[n, m]`` anchor/candidate pairs.

    ``visual_all`` is ``[n, 1+n_slots, d]`` (index 0 = native CLS), ``text_all`` is
    ``[m, 1+n_slots, d]`` (index 0 = native EOS), ``gate`` is ``[n, m, n_slots]``.

    Masked visual slots are dropped from the candidate set *before* the max: zero-filling them and
    leaving them in the max would let them win whenever every legal cosine is negative. In the
    visual->text direction the dropped slots are excluded by the mean's denominator instead, which
    is algebraically the same thing and avoids a second ``[n, W, m, T]`` tensor.
    """
    n, width = visual_all.shape[0], visual_all.shape[1]
    m, text_width = text_all.shape[0], text_all.shape[1]
    visual = F.normalize(visual_all, dim=-1, eps=EPS)
    text = F.normalize(text_all, dim=-1, eps=EPS)
    keep = _candidate_mask(gate)                                          # [n, m, W]
    v2t_grid, t2v_grid = _similarity_grid(visual, text)
    # visual -> text: best text token per (visual token, candidate), then mean over kept tokens.
    # ``best_text`` and ``kept`` are both [n, W, m]: the sum *and* the denominator run over the visual
    # token axis W (axis 1), never over the candidate axis.
    best_text = v2t_grid.max(dim=3).values                                # [n, W, m]
    kept = keep.permute(0, 2, 1)                                          # [n, W, m]
    v2t = (best_text * kept).sum(dim=1) / kept.sum(dim=1)                 # [n, m]
    # text -> visual: best KEPT visual token per (text token, candidate), then mean over text tokens.
    # ``keep`` is [n, m, W] and ``t2v_grid`` is [m, T, n, W], so the mask has to broadcast as
    # [m, 1, n, W]: permuting to [m, W, n] (or to [1, m, n, W]) would line the candidate axis up
    # against the text-token axis and mask the wrong entries.
    mask = keep.permute(1, 0, 2).unsqueeze(1)                             # [m, 1, n, W]
    masked = t2v_grid.masked_fill(~mask, float('-inf'))                   # [m, T, n, W]
    t2v = masked.max(dim=3).values.mean(dim=1).permute(1, 0)              # [n, m]
    score = v2t + t2v
    assert score.shape == (n, m), score.shape
    return score, v2t, t2v


def surrogate_said_score(visual_all: torch.Tensor, text_all: torch.Tensor,
                         soft_gate: torch.Tensor, eta: float = SURROGATE_ETA) -> torch.Tensor:
    """Mask-learning-only proxy from ``C_bar = C.detach()``.

        s_v2t = sum_p w_p max_q C_bar_pq / sum_p w_p
        s_t2v = (1/|T|) sum_q eta [logsumexp_p(C_bar_pq/eta + log w_p) - logsumexp_p(log w_p)]

    ``w = [1; sigmoid(a)]`` has ``1 + n_slots`` entries per pair; the extra leading weight belongs to
    the native CLS and its log-weight is exactly ``log 1 = 0`` (not special-cased). The similarity grid
    is detached and the whole function is fp32, so this path reaches only ``sigmoid(a)``; the features
    take their gradient from ``s_hard`` alone.
    """
    n, width = visual_all.shape[0], visual_all.shape[1]
    m, text_width = text_all.shape[0], text_all.shape[1]
    visual = F.normalize(visual_all.float(), dim=-1, eps=EPS)
    text = F.normalize(text_all.float(), dim=-1, eps=EPS)
    w = torch.cat([torch.ones_like(soft_gate[:, :, :1]), soft_gate], dim=-1)  # [n, m, W]
    log_w = torch.log(w.clamp_min(EPS))
    weight_sum = w.sum(dim=-1).clamp_min(EPS)                                 # [n, m]
    with torch.no_grad():
        v2t_grid, _ = _similarity_grid(visual.detach(), text.detach())
        row_max = v2t_grid.max(dim=3).values                                 # [n, W, m]
        scaled = v2t_grid.permute(0, 2, 1, 3) / eta                          # [n, m, W, T]
        scaled = scaled + log_w[:, :, :, None]
        lse = torch.logsumexp(scaled, dim=2)                                 # [n, m, T]
    v2t = (w * row_max.permute(0, 2, 1)).sum(dim=-1) / weight_sum
    t2v = eta * (lse - log_w.sum(dim=-1)[:, :, None]).mean(dim=-1)
    out = (v2t + t2v).float()
    assert out.shape == (n, m), out.shape
    return out


class PairwiseScorer:
    """Holds the student-side pieces needed to score one image-anchor/text-candidate block."""

    def __init__(self, module, tau: float = ROUTER_TAU, eta: float = SURROGATE_ETA,
                 k_said: int = K_SAID):
        self.module = module
        self.tau = tau
        self.eta = eta
        self.k_said = k_said

    def score(self, visual_all: torch.Tensor, text_all: torch.Tensor) -> Dict[str, torch.Tensor]:
        """``visual_all`` ``[n, 1+V, d]`` anchors, ``text_all`` ``[m, 1+T, d]`` candidates.

        Index 0 of the visual side is the native CLS: never routed, never a top-k candidate, but still a
        member of the Said matching set. Index 0 of the text side is the candidate's native global EOS
        feature, and that -- not the aggregated text slots -- is the router's query.

        The returned score is the plain pair score of the given image and text, so both ranking
        directions use one and the same evaluation: callers run the block once and read the result,
        transposed, for the text-anchor role.
        """
        logits = self.module.router.logits(visual_all[:, 1:], text_all[:, 0], tau=self.tau)
        gate = hard_gate(logits, self.k_said)                        # [n, m, V]
        hard, v2t, t2v = hard_said_score(visual_all, text_all, gate)
        soft_gate = torch.sigmoid(logits)
        soft = surrogate_said_score(visual_all, text_all, soft_gate, self.eta)
        score = hard + (soft - soft.detach())                        # forward == hard
        return {'score': score, 'hard': hard, 'hard_v2t': v2t, 'hard_t2v': t2v,
                'logits': logits, 'gate': gate, 'soft': soft}


def _pairwise_cos(tokens: torch.Tensor) -> torch.Tensor:
    """Mean off-diagonal pairwise cosine between slots (diagnostic only, no anti-collapse loss)."""
    if tokens.shape[0] < 2 or tokens.shape[1] < 2:
        return tokens.new_zeros(())
    flat = F.normalize(tokens.reshape(-1, tokens.shape[-1]).float(), dim=-1, eps=EPS)
    similarity = flat @ flat.t()
    mask = ~torch.eye(similarity.shape[0], dtype=torch.bool, device=similarity.device)
    return similarity[mask].mean()


# --------------------------------------------------------------------------- #
# 4. decoder + frozen reference
# --------------------------------------------------------------------------- #
class ReconstructionDecoder(nn.Module):
    """``Linear(2d,d) -> GELU -> Linear(d,d)``: no dropout, no norm, no residual shortcut."""

    def __init__(self, dim: int, seed: int = 0):
        super().__init__()
        with isolated_rng(seed):
            self.net = nn.Sequential(
                nn.Linear(2 * dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )

    def forward(self, u_norm: torch.Tensor, text_global: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([u_norm, text_global], dim=-1))


def reference_fingerprint(reference_visual: nn.Module) -> str:
    import hashlib
    digest = hashlib.sha256()
    for key, value in sorted(reference_visual.state_dict().items()):
        digest.update(key.encode('utf-8'))
        digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# 5. the training module
# --------------------------------------------------------------------------- #
class SaidTokenV1Module(nn.Module):
    """Student CLIP + two aggregators + router + decoder + one frozen reference vision tower."""

    def __init__(self, clip_model, rank: int = 0, arm: str = ARM_T1, lambda_rec: float = LAMBDA_REC,
                 margin: float = MARGIN, n_slots: int = N_SLOTS, k_said: int = K_SAID,
                 router_dim: int = ROUTER_DIM, router_tau: float = ROUTER_TAU,
                 surrogate_eta: float = SURROGATE_ETA, seed: int = 0,
                 grad_checkpoint_views: bool = False, chunk_image: int = 8,
                 chunk_text: int = 32):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('arm must be one of %r' % (ARMS,))
        self.clip = clip_model
        self.rank = int(rank)
        self.arm = arm
        self.lambda_rec = float(lambda_rec)
        self.margin = float(margin)
        self.n_slots = int(n_slots)
        self.k_said = int(k_said)
        self.router_tau = float(router_tau)
        self.surrogate_eta = float(surrogate_eta)
        self.grad_checkpoint_views = bool(grad_checkpoint_views)
        self.chunk_image = max(1, int(chunk_image))
        self.chunk_text = max(1, int(chunk_text))
        self.dim = cls_embedding_dim(clip_model)

        with isolated_rng(seed):
            self.image_aggregator = SelfAggregator(self.dim, slots=self.n_slots, seed=seed)
            self.text_aggregator = SelfAggregator(self.dim, slots=self.n_slots, seed=seed + 1)
            self.router = TokenRouter(self.dim, out_dim=router_dim, seed=seed + 2)
            self.decoder = ReconstructionDecoder(self.dim, seed=seed + 3)
            # deep copy of the student's *initial* tower: taken before any update, never re-copied
            self.reference_visual = copy.deepcopy(clip_model.visual)
        for parameter in self.reference_visual.parameters():
            parameter.requires_grad_(False)
        self.reference_visual.eval()

        # the historic 512-d mask_net and the unused logit_scale stay in the checkpoint so the
        # exported student still loads strictly, but they are frozen and never optimised
        if hasattr(clip_model, 'mask_net'):
            for parameter in clip_model.mask_net.parameters():
                parameter.requires_grad_(False)
        if hasattr(clip_model, 'logit_scale'):
            clip_model.logit_scale.requires_grad_(False)
        self.scorer = PairwiseScorer(self, tau=self.router_tau, eta=self.surrogate_eta,
                                     k_said=self.k_said)

    # -- invariants ---------------------------------------------------------- #
    def train(self, mode: bool = True):
        result = super().train(mode)
        self.reference_visual.eval()
        return result

    # -- encoders ------------------------------------------------------------ #
    def encode_image_tokens(self, images: torch.Tensor):
        """``(g_I_raw, patch_raw)`` from one forward; both already in the shared space."""
        if self.grad_checkpoint_views:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self.clip.visual, images.type(self.clip.dtype), False, True,
                              use_reentrant=False, preserve_rng_state=True)
        return self.clip.encode_image_with_patches(images, use_checkpoint=False)

    def encode_text_tokens(self, text: torch.Tensor):
        """``(g_T_raw, text_local_raw)``: the hidden state projected exactly once by text_projection."""
        pooled, hidden = self.clip.encode_text(text, return_full=True)
        return pooled, hidden @ self.clip.text_projection

    def reference_global(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            with torch.autocast(device_type=images.device.type, enabled=False):
                raw = self.reference_visual(images.float())
                return F.normalize(raw.float(), dim=-1, eps=EPS)

    # -- one training step's forward ----------------------------------------- #
    def forward(self, images: torch.Tensor, text: torch.Tensor, image_ids: torch.Tensor,
                eot_id: int, all_gather=None) -> Dict[str, torch.Tensor]:
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        gather = all_gather
        if gather is None:
            if world_size > 1:
                def gather(value):
                    """Differentiable all_gather (``torch.distributed.nn``).

                    The Said term couples *every* anchor with *every* candidate, so the gathered features
                    cannot be treated as constants: the functional collective's backward all-reduces the
                    incoming gradients and gives each rank the part belonging to its own shard, which is
                    what makes DDP's average over ranks reproduce the single-process gradient. Two
                    alternatives were measured and rejected: a plain ``all_gather`` silently detached the
                    local slice as well, leaving the text tower and the image projection with no gradient
                    at all, and an "own anchors only" loss lost every candidate-side partial gradient
                    (the two-rank equivalence test caught both).
                    """
                    import torch.distributed.nn.functional as distributed_nn
                    result = distributed_nn.all_gather(value.contiguous())
                    # this build returns a tuple of per-rank tensors; older/newer ones may return the
                    # concatenation directly, so accept both rather than assume one
                    return result if isinstance(result, tuple) else (result,)
            else:
                def gather(value):
                    return (value,)

        g_i_raw, patch_raw = self.encode_image_tokens(images)
        g_t_raw, text_local = self.encode_text_tokens(text)
        if patch_raw.shape[-1] != self.dim or g_i_raw.shape[-1] != self.dim:
            raise RuntimeError('embedding width mismatch: patches %r, global %r, expected %d'
                               % (tuple(patch_raw.shape), tuple(g_i_raw.shape), self.dim))

        valid_text, empty_text = text_content_mask(text, eot_id)
        v_slots, _ = self.image_aggregator(patch_raw)
        t_slots, _ = self.text_aggregator(text_local, valid=valid_text)
        g_i = F.normalize(g_i_raw, dim=-1, eps=EPS)
        g_t = F.normalize(g_t_raw, dim=-1, eps=EPS)

        visual_local = torch.cat([g_i[:, None, :], v_slots], dim=1)      # [b, 1+V, d]
        text_local_all = torch.cat([g_t[:, None, :], t_slots], dim=1)    # [b, 1+T, d]

        visual_global = torch.cat(gather(visual_local), dim=0)
        text_global = torch.cat(gather(text_local_all), dim=0)
        ids_global = torch.cat(gather(image_ids.reshape(-1, 1)), dim=0).reshape(-1)
        local = visual_local.shape[0]
        if visual_global.shape[0] != ids_global.shape[0] or text_global.shape[0] != ids_global.shape[0]:
            raise RuntimeError('the gathered candidate pools disagree: visual %r text %r ids %r'
                               % (tuple(visual_global.shape), tuple(text_global.shape),
                                  tuple(ids_global.shape)))
        # The per-anchor positive scores and the anchors' own windows are indexed by absolute position,
        # so the gathered order has to be exactly the rank order, every rank included. That convention is
        # checked -- with one tiny collective -- rather than assumed: a gather that silently dropped a
        # rank would otherwise mis-rank candidates instead of failing.
        global_index = self.rank * local + torch.arange(local, device=images.device)
        order = torch.cat(gather(global_index.reshape(-1)), dim=0)
        if not bool((order == torch.arange(ids_global.shape[0], device=images.device)).all()):
            raise RuntimeError('gathered candidate order does not match rank order: rank %d, local %d'
                               % (self.rank, local))

        # ---- the true pairs, row-aligned: mask h_ii and score s_ii ------------- #
        # One router evaluation of image i against text i. This is the only place the positive gate
        # comes from, and the U branch uses exactly this mask (m_U = 1 - sg(h_ii)). The diagonal is the
        # true pair, and the per-anchor positive score is gathered so that anchors owned by another rank
        # can be ranked too. Advanced indexing is used rather than ``Tensor.diagonal``: the latter moves
        # the diagonal to the *last* axis ([local, local, V] would come back as [V, local]).
        local_rows = torch.arange(local, device=images.device)
        positive_rows = self.scorer.score(visual_local, text_local_all)
        positive_gate = positive_rows['gate'][local_rows, local_rows].detach()        # [local, V]
        positive_score = positive_rows['score'][local_rows, local_rows].detach()      # [local]
        positive_logits = positive_rows['logits'][local_rows, local_rows].detach()    # [local, V]
        positive_global = torch.cat(gather(positive_score.reshape(-1)), dim=0).detach().reshape(-1)

        # ---- chunked pair scoring over the global candidate grid ------------- #
        # Every rank walks the *same* grid: this rank's anchors against the whole gathered candidate
        # pool, so each (image, text) pair carries its own hard mask h[i, j, p] -- nothing reuses the
        # positive pair's mask and nothing is transposed into the other role. The scores, the counts and
        # therefore the loss are already global on every rank, so this term carries NO ``world_size``
        # factor: the differentiable gather in ``gather`` is what makes the gradient correct under
        # DDP's averaging. Only small score/gate blocks live in the graph; the full [B, B, 33, 33]
        # tensor never exists.
        sum_i2t = visual_local.new_zeros(())
        sum_t2i = visual_local.new_zeros(())
        count_i2t = visual_local.new_zeros(())
        count_t2i = visual_local.new_zeros(())
        sum_hinge_i2t = visual_local.new_zeros(())
        sum_hinge_t2i = visual_local.new_zeros(())
        margin_active = visual_local.new_zeros(())

        for image_start in range(0, visual_global.shape[0], self.chunk_image):
            image_stop = min(image_start + self.chunk_image, visual_global.shape[0])
            image_block = visual_global[image_start:image_stop]
            image_ids_block = ids_global[image_start:image_stop]
            for text_start in range(0, text_global.shape[0], self.chunk_text):
                text_stop = min(text_start + self.chunk_text, text_global.shape[0])
                text_block = text_global[text_start:text_stop]
                text_ids_block = ids_global[text_start:text_stop]
                # A pair is a valid negative unless both sides belong to the same original image. That
                # one test drops the true positive as well as any second caption of the same image, and
                # it is the same matrix in both directions (the score of a pair does not depend on which
                # side is called the anchor).
                valid_pair = image_ids_block[:, None] != text_ids_block[None, :]   # [i_block, t_block]
                rows = self.scorer.score(image_block, text_block)
                # ---- image anchors x text candidates --------------------- #
                block_positive = positive_global[image_start:image_stop, None]
                hinge = F.relu(self.margin + rows['score'] - block_positive) * valid_pair
                sum_hinge_i2t = sum_hinge_i2t + hinge.sum()
                margin_active = margin_active + (hinge > 0).sum()
                sum_i2t = sum_i2t + (rows['score'] * valid_pair).sum()
                count_i2t = count_i2t + valid_pair.sum().float()
                # ---- text anchors x image candidates --------------------- #
                score_t = rows['score'].transpose(0, 1)                            # [t_block, i_block]
                valid_t = valid_pair.transpose(0, 1)
                block_positive_t = positive_global[text_start:text_stop, None]
                hinge_t = F.relu(self.margin + score_t - block_positive_t) * valid_t
                sum_hinge_t2i = sum_hinge_t2i + hinge_t.sum()
                sum_t2i = sum_t2i + (score_t * valid_t).sum()
                count_t2i = count_t2i + valid_t.sum().float()

        # The counts and the hinge sums are already global on every rank (the whole grid was scored
        # above), so they are used directly: all-reducing them would multiply the denominator by
        # ``world_size``. No ``world_size`` factor appears here either -- the differentiable gather is
        # what carries the cross-rank gradient.
        count_i2t_global, count_t2i_global = count_i2t.detach(), count_t2i.detach()
        loss_i2t = sum_hinge_i2t / count_i2t_global.clamp_min(1.0)
        loss_t2i = sum_hinge_t2i / count_t2i_global.clamp_min(1.0)
        loss_said = 0.5 * (loss_i2t + loss_t2i)

        # ---- U branch: the true pairs only, mask from that same positive pair - #
        # m_U = 1 - sg(h_ii); ``u`` averages the raw aggregated slots of the complementary set.
        m_u = 1.0 - positive_gate.float()
        u = (m_u[:, :, None] * v_slots).sum(dim=1) / self.k_said
        u_norm = F.normalize(u, dim=-1, eps=EPS)
        reference = self.reference_global(images)
        prediction = self.decoder(u_norm, g_t.detach())
        prediction_norm = F.normalize(prediction, dim=-1, eps=EPS)
        per_sample = 1.0 - (prediction_norm * reference).sum(dim=-1)
        valid_rec = u_norm.norm(dim=-1) > EPS
        rec_sum = (per_sample * valid_rec).sum()
        rec_count = valid_rec.sum().float()
        if world_size > 1 and torch.distributed.is_initialized():
            total = rec_count.detach().clone()
            torch.distributed.all_reduce(total)
            rec_count_global = total
        else:
            rec_count_global = rec_count.detach()
        # The U branch is per sample: each sample's term is computed exactly once, by the rank that owns
        # it, and never depends on another rank's features (the decoder input uses the same sample's
        # detached g_T). That is why this term -- unlike the Said term -- carries an explicit
        # ``world_size`` factor so DDP's average turns the summed per-rank contributions back into the
        # global mean.
        loss_rec = float(world_size) * rec_sum / rec_count_global.clamp_min(1.0)

        loss_total = loss_said + self.lambda_rec * loss_rec
        if not bool(torch.isfinite(loss_total) and torch.isfinite(prediction).all()):
            raise RuntimeError('non-finite loss or prediction: a real error, not a degenerate case')

        with torch.no_grad():
            logits_pos = positive_logits
            return {
                'loss_total_for_backward': loss_total,
                'loss_said': loss_said,
                'loss_said_i2t': loss_i2t.detach(),
                'loss_said_t2i': loss_t2i.detach(),
                'loss_said_i2t_score_sum': sum_i2t.detach(),
                'loss_said_t2i_score_sum': sum_t2i.detach(),
                'loss_said_i2t_hinge_sum': sum_hinge_i2t.detach(),
                'loss_said_t2i_hinge_sum': sum_hinge_t2i.detach(),
                'loss_said_pairs_i2t': count_i2t_global,
                'loss_said_pairs_t2i': count_t2i_global,
                'loss_said_pairs_total': (count_i2t_global + count_t2i_global),
                'loss_rec': loss_rec,
                'loss_rec_sum_local': rec_sum.detach(),
                'loss_rec_valid_local': rec_count.detach(),
                'loss_rec_valid_global': rec_count_global,
                'weighted_rec': (self.lambda_rec * loss_rec).detach(),
                'lambda_rec': loss_total.new_tensor(self.lambda_rec),
                'positive_said_score': positive_score.detach().mean(),
                'negative_said_score': sum_i2t.detach() / count_i2t.clamp_min(1.0),
                'margin_active_fraction': margin_active / count_i2t.clamp_min(1.0),
                'router_logit_std': logits_pos.std(),
                'router_logit_mean': logits_pos.mean(),
                'soft_gate_mean': torch.sigmoid(logits_pos).mean(),
                'said_token_count': positive_gate.detach().sum(dim=-1).mean(),
                'unsaid_token_count': m_u.sum(dim=-1).mean(),
                'selected_router_score': (logits_pos[positive_gate > 0.5].mean()
                                          if bool((positive_gate > 0.5).any())
                                          else logits_pos.sum() * 0.0),
                'excluded_router_score': (logits_pos[positive_gate < 0.5].mean()
                                          if bool((positive_gate < 0.5).any())
                                          else logits_pos.sum() * 0.0),
                'u_norm': u_norm.norm(dim=-1).mean(),
                'u_raw_norm': u.norm(dim=-1).mean(),
                'prediction_norm': prediction.norm(dim=-1).mean(),
                'cos_prediction_reference': (prediction_norm * reference).sum(dim=-1).mean(),
                'reference_norm': reference.norm(dim=-1).mean(),
                'rec_valid_fraction': rec_count / max(local, 1),
                'empty_text_count': empty_text.sum(),
                'native_cls_pairwise_cos': _pairwise_cos(g_i.detach()),
                'aggregation_token_pairwise_cos': _pairwise_cos(v_slots.detach()),
                # handles for the fixed-cohort diagnostics (no_grad consumers only)
                'v_slots': v_slots.detach(),
                't_slots': t_slots.detach(),
                'g_i': g_i.detach(),
                'g_t': g_t.detach(),
                'gate_positive': positive_gate.detach(),
                'm_u': m_u,
                'u_norm_detached': u_norm.detach(),
                'reference': reference,
            }
