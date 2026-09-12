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
  the proxy follows router logits, including their feature inputs, but never the cosine grid. Biased straight-through proxy, not the
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
def core_dtype(x):
    # FP64 is retained only for the continuous-formula finite-difference tests.
    return x if x.dtype == torch.float64 else x.float()


class TokenRouter(nn.Module):
    def __init__(self, dim, out_dim=ROUTER_DIM, seed=0):
        super().__init__()
        with isolated_rng(seed):
            self.visual = nn.Linear(dim, out_dim, bias=False)
            self.text = nn.Linear(dim, out_dim, bias=False)

    def project_visual(self, slots):
        with torch.autocast(device_type=slots.device.type, enabled=False):
            return F.normalize(self.visual(core_dtype(slots)), dim=-1, eps=EPS)

    def project_text(self, query):
        if query.ndim != 2:
            raise ValueError('router query must be native global EOS [m,d]')
        with torch.autocast(device_type=query.device.type, enabled=False):
            return F.normalize(self.text(core_dtype(query)), dim=-1, eps=EPS)

    def logits(self, visual_slots, text_global, tau=ROUTER_TAU):
        with torch.autocast(device_type=visual_slots.device.type, enabled=False):
            left = self.project_visual(visual_slots)
            right = self.project_text(text_global)
            return (left @ right.T).permute(0, 2, 1) / tau


def hard_gate(logits, k=K_SAID):
    if not 0 <= k <= logits.shape[-1]:
        raise ValueError('invalid selected slot count')
    order = torch.argsort(-logits, dim=-1, stable=True)[..., :k]
    return torch.zeros_like(logits).scatter(-1, order, 1.0)


def _candidate_mask(gate):
    return torch.cat([torch.ones_like(gate[..., :1], dtype=torch.bool), gate > .5], -1)


def _similarity_grid(visual, text):
    """One matmul, canonical layout [n,m,W,T]; inputs already normalized."""
    n, w, d = visual.shape
    m, t, _ = text.shape
    return (visual.reshape(n*w, d) @ text.reshape(m*t, d).T).reshape(n, w, m, t).permute(0, 2, 1, 3)


def hard_from_grid(cosine, gate, text_valid):
    keep = _candidate_mask(gate)
    best_text = cosine.masked_fill(~text_valid[..., None, :], -torch.inf).max(-1).values
    v2t = (best_text * keep).sum(-1) / keep.sum(-1)
    best_visual = cosine.masked_fill(~keep[..., :, None], -torch.inf).max(-2).values
    t2v = (best_visual * text_valid).sum(-1) / text_valid.sum(-1)
    return v2t + t2v, v2t, t2v


def soft_from_grid(cosine, router_logits, text_valid=None, eta=SURROGATE_ETA):
    """Continuous approved proxy: only C is detached, never log-weight arithmetic."""
    with torch.autocast(device_type=cosine.device.type, enabled=False):
        c = core_dtype(cosine).detach()
        a = core_dtype(router_logits)
        if text_valid is None:
            text_valid = torch.ones_like(c[..., 0, :], dtype=torch.bool)
        log_w = torch.cat([torch.zeros_like(a[..., :1]), F.logsigmoid(a)], -1)
        log_den = torch.logsumexp(log_w, -1)
        maxima = c.masked_fill(~text_valid[..., None, :], -torch.inf).max(-1).values
        v2t = (torch.softmax(log_w, -1) * maxima).sum(-1)
        numerator = torch.logsumexp(c / eta + log_w[..., :, None], -2)
        t2v = eta * ((numerator - log_den[..., None]) * text_valid).sum(-1) / text_valid.sum(-1)
        return v2t + t2v


def hard_said_score(visual_all, text_all, gate, text_valid=None):
    with torch.autocast(device_type=visual_all.device.type, enabled=False):
        v = F.normalize(core_dtype(visual_all), dim=-1, eps=EPS)
        t = F.normalize(core_dtype(text_all), dim=-1, eps=EPS)
        c = _similarity_grid(v, t)
        if text_valid is None:
            text_valid = torch.ones(text_all.shape[:2], device=t.device, dtype=torch.bool)
        return hard_from_grid(c, gate, text_valid[None])


def surrogate_said_score(visual_all, text_all, router_logits, eta=SURROGATE_ETA, text_valid=None):
    with torch.autocast(device_type=visual_all.device.type, enabled=False):
        v = F.normalize(core_dtype(visual_all), dim=-1, eps=EPS)
        t = F.normalize(core_dtype(text_all), dim=-1, eps=EPS)
        c = _similarity_grid(v, t)
        return soft_from_grid(c, router_logits, None if text_valid is None else text_valid[None], eta)


class PairwiseScorer:
    def __init__(self, module, tau=ROUTER_TAU, eta=SURROGATE_ETA, k_said=K_SAID):
        self.module, self.tau, self.eta, self.k_said = module, tau, eta, k_said

    def prepare(self, visual_all, text_all):
        with torch.autocast(device_type=visual_all.device.type, enabled=False):
            return (F.normalize(core_dtype(visual_all), dim=-1, eps=EPS),
                    F.normalize(core_dtype(text_all), dim=-1, eps=EPS),
                    self.module.router.project_visual(visual_all[:, 1:]),
                    self.module.router.project_text(text_all[:, 0]))

    def from_prepared(self, visual, text, projected_visual, projected_text, text_valid,
                      matched=False):
        with torch.autocast(device_type=visual.device.type, enabled=False):
            if matched:
                c = torch.bmm(visual, text.transpose(1, 2))
                logits = (projected_visual * projected_text[:, None]).sum(-1) / self.tau
                valid = text_valid
            else:
                c = _similarity_grid(visual, text)
                logits = (projected_visual @ projected_text.T).permute(0, 2, 1) / self.tau
                valid = text_valid[None]
            gate = hard_gate(logits, self.k_said)
            hard, v2t, t2v = hard_from_grid(c, gate, valid)
            soft = soft_from_grid(c, logits, valid, self.eta)
            return {'score': hard + (soft - soft.detach()), 'hard': hard,
                    'hard_v2t': v2t, 'hard_t2v': t2v, 'logits': logits, 'gate': gate, 'soft': soft}

    def block_score(self, visual, text, projected_visual, projected_text, text_valid):
        return self.from_prepared(visual, text, projected_visual, projected_text, text_valid)['score']

    def score(self, visual_all, text_all, text_valid=None, matched=False):
        prepared = self.prepare(visual_all, text_all)
        if text_valid is None:
            text_valid = torch.ones(text_all.shape[:2], device=text_all.device, dtype=torch.bool)
        return self.from_prepared(*prepared, text_valid, matched=matched)

    def score_matched_pairs(self, visual_all, text_all, text_valid=None):
        return self.score(visual_all, text_all, text_valid, matched=True)


def ranking_sums(scores, positive_rows, positive_columns, valid_pairs, margin=MARGIN):
    """Symmetric legal ordered pairs: two directions SUM, no half factor."""
    with torch.autocast(device_type=scores.device.type, enabled=False):
        scores = core_dtype(scores)
        first = F.relu(margin + scores - positive_rows[:, None]) * valid_pairs
        second = F.relu(margin + scores - positive_columns[None, :]) * valid_pairs
        return first.sum(), second.sum(), (first.detach() > 0).sum()


def _pairwise_cos(tokens):
    flat = F.normalize(tokens.reshape(-1, tokens.shape[-1]).float(), dim=-1, eps=EPS)
    n = flat.shape[0]
    return (flat.sum(0).square().sum() - flat.square().sum()) / max(n*(n-1), 1)


def intra_image_slot_cos(tokens):
    x = F.normalize(tokens.float(), dim=-1, eps=EPS)
    k = x.shape[1]
    return ((x.sum(1).square().sum(-1) - x.square().sum((1, 2))) / max(k*(k-1), 1)).mean()


def complement_pool(slots, gate):
    m = 1.0 - gate.detach().float()
    count = m.sum(-1)
    raw = (slots.float() * m[..., None]).sum(1) / count.clamp_min(1)[..., None]
    valid = (count > 0) & (raw.norm(dim=-1) > EPS)
    return raw, F.normalize(raw, dim=-1, eps=EPS), valid, m


def gather_metadata(value, world):
    if world == 1:
        return value
    result = [torch.empty_like(value) for _ in range(world)]
    torch.distributed.all_gather(result, value.contiguous())
    return torch.cat(result, 0)


def caption_identity(token_ids, world):
    """Exact effective token sequences (through EOS); padding never affects identity."""
    length = torch.tensor([token_ids.shape[1]], device=token_ids.device, dtype=torch.long)
    if world > 1:
        torch.distributed.all_reduce(length, op=torch.distributed.ReduceOp.MAX)
    pos = torch.arange(token_ids.shape[1], device=token_ids.device)[None]
    canonical = token_ids.masked_fill(pos > token_ids.argmax(-1)[:, None], 0)
    canonical = F.pad(canonical, (0, int(length.item()) - token_ids.shape[1]))
    all_tokens = gather_metadata(canonical, world)
    _, identity = torch.unique(all_tokens, dim=0, return_inverse=True)
    return identity



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
                 chunk_text: int = 32, checkpoint_pairwise: bool = True):
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
        self.checkpoint_pairwise = bool(checkpoint_pairwise)
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
    def forward(self, images, text, image_ids, eot_id, all_gather=None, diagnostics=False):
        import torch.distributed as dist
        from torch.utils.checkpoint import checkpoint
        from torch.profiler import record_function
        world = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else self.rank
        local = images.shape[0]
        sizes = gather_metadata(torch.tensor([local], device=images.device), world)
        if not bool((sizes == local).all()):
            raise RuntimeError('unequal per-rank batch sizes: %s' % sizes.tolist())
        def gather_live(value):
            if all_gather is not None:
                return torch.cat(all_gather(value), 0)
            if world == 1:
                return value
            from torch.distributed.nn.functional import all_gather as differentiable_gather
            return torch.cat(differentiable_gather(value.contiguous()), 0)

        with record_function('t1_student_encoders_aggregators'):
            g_i_raw, patches = self.encode_image_tokens(images)
            g_t_raw, text_local = self.encode_text_tokens(text)
            valid, empty = text_content_mask(text, eot_id)
            v_slots, _ = self.image_aggregator(patches)
            t_slots, _ = self.text_aggregator(text_local, valid)
        with torch.autocast(device_type=images.device.type, enabled=False):
            g_i = F.normalize(g_i_raw.float(), dim=-1, eps=EPS)
            g_t = F.normalize(g_t_raw.float(), dim=-1, eps=EPS)
            visual = torch.cat([g_i[:, None], v_slots.float()], 1)
            text_all = torch.cat([g_t[:, None], t_slots.float()], 1)
            text_valid = torch.cat([torch.ones_like(empty[:, None]),
                                    (~empty[:, None]).expand(-1, self.n_slots)], 1)
            with record_function('t1_prepare_and_collectives'):
                vn, tn, vp, tp = self.scorer.prepare(visual, text_all)
                global_tn, global_tp = gather_live(tn), gather_live(tp)
                global_valid = gather_metadata(text_valid, world)
                global_ids = gather_metadata(image_ids, world)
                caption_ids = caption_identity(text, world)
                offset = rank * local
                valid_pairs = ((image_ids[:, None] != global_ids[None]) &
                    (caption_ids[offset:offset+local, None] != caption_ids[None]))
            with record_function('t1_matched_positives'):
                positive = self.scorer.from_prepared(vn, tn, vp, tp, text_valid, matched=True)
                positive_score_live = positive['score']
                if torch.is_grad_enabled() and not positive_score_live.requires_grad:
                    raise RuntimeError('positive ranking score lost its gradient')
                if diagnostics and positive_score_live.requires_grad:
                    positive_score_live.retain_grad()
                    self._positive_live_for_check = positive_score_live
                else:
                    self._positive_live_for_check = None
                positive_global_live = gather_live(positive_score_live)
                # U is intentionally isolated from the gate; ranking positives remain live.
                positive_gate = positive['gate'].detach()
            # Connected zeros preserve all gather backward paths on a rank with no legal negatives.
            first = (vn.sum() + global_tn.sum() + global_tp.sum() + positive_global_live.sum()) * 0
            second = first * 0
            negative_sum = first.detach().clone()
            active = first.detach().clone()
            with record_function('t1_pairwise_hard_proxy'):
                for i in range(0, local, self.chunk_image):
                    si = slice(i, min(i+self.chunk_image, local))
                    for j in range(0, global_tn.shape[0], self.chunk_text):
                        sj = slice(j, min(j+self.chunk_text, global_tn.shape[0]))
                        args = (vn[si], global_tn[sj], vp[si], global_tp[sj], global_valid[sj])
                        if self.checkpoint_pairwise and torch.is_grad_enabled():
                            score = checkpoint(self.scorer.block_score, *args,
                                               use_reentrant=False, preserve_rng_state=True)
                        else:
                            score = self.scorer.block_score(*args)
                        legal = valid_pairs[si, sj]
                        a, b, nactive = ranking_sums(score, positive_score_live[si],
                                                    positive_global_live[sj], legal, self.margin)
                        first, second = first+a, second+b
                        negative_sum = negative_sum + (score.detach()*legal).sum()
                        active = active + nactive
            count_local = valid_pairs.sum().float()
            count_global = count_local.clone()
            if world > 1:
                dist.all_reduce(count_global)
            loss_i2t = world * first / count_global.clamp_min(1)
            loss_t2i = world * second / count_global.clamp_min(1)
            loss_said = loss_i2t + loss_t2i
            with record_function('t1_frozen_reference_decoder'):
                u_raw, u_norm, valid_rec, m_u = complement_pool(v_slots, positive_gate)
                reference = self.reference_global(images)
                prediction = self.decoder(u_norm, g_t.detach())
                predicted = F.normalize(prediction, dim=-1, eps=EPS)
                rec_each = 1 - (predicted * reference).sum(-1)
                rec_sum = (rec_each * valid_rec).sum()
                rec_count = valid_rec.sum().float()
                rec_global = rec_count.clone()
                if world > 1:
                    dist.all_reduce(rec_global)
                loss_rec = world * rec_sum / rec_global.clamp_min(1)
            loss_total = loss_said + self.lambda_rec * loss_rec
            finite = torch.stack([torch.isfinite(x).all() for x in
                (g_i_raw, g_t_raw, v_slots, t_slots, u_raw, reference, prediction, loss_total)]).all().int()
            if world > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not bool(finite):
                raise RuntimeError('non-finite input/features/target/prediction/loss')
            logits = positive['logits'].detach()
            result = {
                'loss_total_for_backward': loss_total, 'loss_said': loss_said,
                'loss_said_i2t': loss_i2t.detach(), 'loss_said_t2i': loss_t2i.detach(),
                'loss_said_i2t_hinge_sum': first.detach(), 'loss_said_t2i_hinge_sum': second.detach(),
                'loss_said_pairs_local': count_local, 'loss_said_pairs_total': count_global,
                'loss_said_pairs_i2t': count_global, 'loss_said_pairs_t2i': count_global,
                'loss_rec': loss_rec, 'loss_rec_sum_local': rec_sum.detach(),
                'loss_rec_valid_local': rec_count, 'loss_rec_valid_global': rec_global,
                'weighted_rec': (self.lambda_rec*loss_rec).detach(),
                'lambda_rec': loss_total.new_tensor(self.lambda_rec),
                'positive_score_sum_local': positive_score_live.detach().sum(),
                'negative_score_sum_local': negative_sum, 'margin_active_local': active,
                'positive_said_score': positive_score_live.detach().mean(),
                'negative_said_score': negative_sum / count_local.clamp_min(1),
                'margin_active_fraction': active / count_local.clamp_min(1),
                'router_logit_std': logits.std(), 'router_logit_mean': logits.mean(),
                'router_within_pair_logit_std': logits.std(-1).mean(),
                'soft_gate_mean': logits.sigmoid().mean(),
                'soft_gate_saturated_fraction': (logits.sigmoid() > .999).float().mean(),
                'said_token_count': positive_gate.sum(-1).mean(),
                'unsaid_token_count': m_u.sum(-1).mean(),
                'selected_router_score': (logits*positive_gate).sum()/positive_gate.sum().clamp_min(1),
                'excluded_router_score': (logits*m_u).sum()/m_u.sum().clamp_min(1),
                'u_raw_norm': u_raw.norm(dim=-1).mean(), 'u_norm': u_norm.norm(dim=-1).mean(),
                'prediction_norm': prediction.norm(dim=-1).mean(),
                'cos_prediction_reference': (predicted*reference).sum(-1).mean(),
                'rec_valid_fraction': rec_count/max(local, 1), 'empty_text_count': empty.sum(),
                'actual_global_candidates': float(global_tn.shape[0]),
                'actual_local_image_rows': float(local),
                'v_slots': v_slots.detach(), 't_slots': t_slots.detach(), 'g_i': g_i.detach(),
                'g_t': g_t.detach(), 'gate_positive': positive_gate, 'm_u': m_u,
                'u_norm_detached': u_norm.detach(), 'reference': reference,
                'core_dtypes': {'student_image': str(g_i_raw.dtype), 'student_text': str(g_t_raw.dtype),
                    'router_projection': str(vp.dtype), 'router_logits': str(positive['logits'].dtype),
                    'matched_hard': str(positive['hard'].dtype), 'matched_proxy': str(positive['soft'].dtype),
                    'ranking': str(loss_said.dtype), 'reference': str(reference.dtype),
                    'decoder': str(prediction.dtype), 'reconstruction': str(loss_rec.dtype)},
            }
            if diagnostics:
                with torch.no_grad():
                    result.update(native_cls_pairwise_cos=_pairwise_cos(g_i),
                                  intra_image_slot_cos=intra_image_slot_cos(v_slots),
                                  all_image_slot_cos=_pairwise_cos(v_slots))
            return result
