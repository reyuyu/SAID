"""Tests for SAID-Token v1 (arms ``T1_said_token_reconstruction`` / ``T0_said_token_only``).

The suites here are the ones the specification asks for, grouped as: last-layer interface, text
validity, self-aggregation, router/gate, hard score, the hard-forward/score-layer-backward proxy, the
gradient boundary of the two loss terms, chunking invariance, the T0 switch, strict export, RNG
isolation, the 2-rank DDP equality, and the CLI entry point.
"""
import json
import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'train'))

from model import longclip  # noqa: E402
from model.model_longclip import CLIP  # noqa: E402
from model.said_token_v1 import (ARM_T0, ARM_T1, EPS, K_SAID, N_SLOTS,  # noqa: E402
                                 SaidTokenV1Module, SelfAggregator, hard_gate, hard_said_score,
                                 surrogate_said_score, text_content_mask)

EOT_ID = 49407
SOT_ID = 49406
BATCH = 3
CAPTIONS = ['a cat on a chair', 'a dog on the grass', 'a red bus downtown']
DIM = 16


def tiny_clip(seed=1234):
    torch.manual_seed(seed)
    clip = CLIP(embed_dim=DIM, image_resolution=16, vision_layers=1, vision_width=128,
                vision_patch_size=8, context_length=8, vocab_size=50000, transformer_width=DIM,
                transformer_heads=4, transformer_layers=1, load_from_clip=False)
    clip = clip.float()
    with torch.no_grad():
        # Measured here: ``initialize_parameters()`` initialises ``positional_embedding`` but NOT
        # ``positional_embedding_res``, which the ``load_from_clip=False`` path creates on
        # ``torch.empty`` memory. The text encoder adds it for positions >= 20, so without this the
        # fixture would silently run on garbage -- and NaN, non-deterministically (one earlier run of
        # this very file produced NaN text features). The real training path is unaffected: it loads
        # the pretrained checkpoint, which carries the rescaled positional embedding.
        clip.positional_embedding_res.normal_(0.0, 0.01)
    for name, parameter in clip.named_parameters():
        assert bool(torch.isfinite(parameter).all()), name
    return clip


def build_env(arm=ARM_T1, lambda_rec=None, seed=1234, chunk=(8, 32), batch=BATCH,
              captions=CAPTIONS, image_ids=None):
    clip = tiny_clip(seed)
    torch.manual_seed(seed + 5)
    images = torch.randn(batch, 3, 16, 16) * 0.4
    text = longclip.tokenize(list(captions), truncate=True)
    kwargs = {'lambda_rec': lambda_rec} if lambda_rec is not None else {}
    module = SaidTokenV1Module(clip, rank=0, arm=arm, seed=0, chunk_image=chunk[0],
                               chunk_text=chunk[1], **kwargs)
    ids = torch.arange(batch) if image_ids is None else image_ids
    return clip, images, text, module, ids


@pytest.fixture
def env():
    return build_env()


def manual_hard_score(visual_all, text_all, gate):
    """The specification, written out literally in float64 with no tensor algebra shortcuts."""
    visual = F.normalize(visual_all.double(), dim=-1, eps=EPS)
    text = F.normalize(text_all.double(), dim=-1, eps=EPS)
    keep = torch.cat([torch.ones_like(gate[..., :1], dtype=torch.bool), gate > 0.5], dim=-1)
    n, width = visual.shape[0], visual.shape[1]
    m, text_width = text.shape[0], text.shape[1]
    out = torch.zeros(n, m, dtype=torch.float64)
    for i in range(n):
        for j in range(m):
            similarity = visual[i] @ text[j].t()                 # [width, text_width]
            legal = keep[i, j]                                   # [width]
            v2t = similarity.max(dim=1).values[legal].mean()
            t2v = torch.stack([similarity[:, q][legal].max()
                               for q in range(text_width)]).mean()
            out[i, j] = v2t + t2v
    return out


def slot_tensors(module, env, batch=BATCH):
    """``(visual_all, text_all)`` built the way the module builds them, detached."""
    clip, images, text, _, _ = env
    with torch.no_grad():
        g_i_raw, patch_raw = module.encode_image_tokens(images[:batch])
        g_t_raw, hidden = module.encode_text_tokens(text[:batch])
        valid, _ = text_content_mask(text[:batch], EOT_ID)
        v_slots = module.image_aggregator(patch_raw)[0]
        t_slots = module.text_aggregator(hidden @ clip.text_projection, valid=valid)[0]
        g_i = F.normalize(g_i_raw, dim=-1, eps=EPS)
        g_t = F.normalize(g_t_raw, dim=-1, eps=EPS)
        visual_all = torch.cat([g_i[:, None, :], v_slots], dim=1)
        text_all = torch.cat([g_t[:, None, :], t_slots], dim=1)
    return visual_all, text_all


# --------------------------------------------------------------------------- #
# 1. last-layer interface
# --------------------------------------------------------------------------- #
def test_last_layer_interface_matches_the_native_paths(env):
    clip, images, text, module, _ = env
    g_i_raw, patch_raw = module.encode_image_tokens(images)
    assert torch.equal(g_i_raw, clip.encode_image(images)), \
        'the image global must be the native CLS of the same forward'
    grid = (16 // 8) ** 2
    assert patch_raw.shape == (BATCH, grid, DIM)
    assert patch_raw.shape[-1] == module.dim == clip.text_projection.shape[1]
    g_t_raw, text_local = module.encode_text_tokens(text)
    assert g_t_raw.shape == (BATCH, DIM)
    assert text_local.shape == (BATCH, text.shape[1], DIM)


def test_text_hidden_state_is_projected_exactly_once(env):
    clip, _, text, module, _ = env
    pooled, hidden = clip.encode_text(text, return_full=True)
    g_t_raw, text_local = module.encode_text_tokens(text)
    assert torch.allclose(text_local, hidden @ clip.text_projection, atol=1e-6)
    assert torch.allclose(g_t_raw, pooled, atol=1e-6), \
        'the native EOS feature keeps the original pooled-text path'
    double = hidden @ clip.text_projection @ clip.text_projection
    assert not torch.allclose(text_local, double, atol=1e-4)


def test_text_content_mask_is_strictly_inside_sot_and_eot(env):
    _, _, _, _, _ = env
    ids = torch.zeros(3, 10, dtype=torch.long)
    ids[0, :6] = torch.tensor([SOT_ID, 11, 12, 13, EOT_ID, 0])
    ids[1, :4] = torch.tensor([SOT_ID, 21, EOT_ID, 0])
    ids[2, :3] = torch.tensor([SOT_ID, EOT_ID, 0])
    valid, empty = text_content_mask(ids, EOT_ID)
    assert valid[0].tolist() == [False, True, True, True, False, False, False, False, False, False]
    assert valid[1].tolist()[:4] == [False, True, False, False]
    assert valid[2].sum() == 0 and bool(empty[2])
    assert empty.tolist() == [False, False, True]
    # the real tokenizer prepends SOT and appends EOT, so an empty caption has no content position
    real = longclip.tokenize(['', 'cat'], truncate=True)
    valid_real, empty_real = text_content_mask(real, EOT_ID)
    assert bool(empty_real[0]) and valid_real[0].sum() == 0
    assert not bool(empty_real[1]) and valid_real[1].sum() == 1


def test_empty_caption_is_reported_and_never_produces_nan(env):
    clip, images, text, module, ids = env
    captions = [CAPTIONS[0], '', CAPTIONS[2]]
    text = longclip.tokenize(captions, truncate=True)
    valid, empty = text_content_mask(text, EOT_ID)
    assert bool(empty[1]) and not bool(empty[0])
    tokens = torch.randn(3, text.shape[1], DIM)
    weights = module.text_aggregator.weights(tokens, valid=valid)
    assert torch.isfinite(weights).all()
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3, N_SLOTS), atol=1e-5)
    assert (weights[1] >= 0).all() and float(weights[1].sum()) > 0
    out = module(images, text, ids, EOT_ID)
    for key in ('loss_said', 'loss_rec', 'loss_total_for_backward', 'positive_said_score'):
        assert torch.isfinite(out[key]), key


# --------------------------------------------------------------------------- #
# 2. self-aggregation
# --------------------------------------------------------------------------- #
def test_aggregation_weights_normalise_over_the_legal_tokens_only(env):
    _, _, _, module, _ = env
    tokens = torch.randn(2, 9, DIM)
    valid = torch.zeros(2, 9, dtype=torch.bool)
    valid[0, 2:6] = True
    valid[1, :] = True
    aggregator = module.image_aggregator
    weights = aggregator.weights(tokens, valid=valid)
    assert weights.shape == (2, N_SLOTS, 9)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, N_SLOTS), atol=1e-5)
    assert float(weights[0][:, ~valid[0]].abs().max()) == 0.0
    assert float(weights[0][:, valid[0]].min()) > 0.0
    # a row with no legal position falls back to one position instead of an all-(-inf) softmax
    dead = torch.zeros(1, 9, dtype=torch.bool)
    dead_weights = aggregator.weights(tokens[:1], valid=dead)
    assert torch.isfinite(dead_weights).all()
    assert torch.allclose(dead_weights.sum(dim=-1), torch.ones(1, N_SLOTS), atol=1e-5)
    assert float(dead_weights[:, :, 0].min()) == 1.0


def test_aggregators_are_separate_and_have_the_frozen_shape(env):
    _, _, _, module, _ = env
    image_params = {id(p) for p in module.image_aggregator.parameters()}
    text_params = {id(p) for p in module.text_aggregator.parameters()}
    assert not (image_params & text_params), 'the two modalities must not share parameters'
    assert len(image_params) == len(text_params) == 6      # norm weight/bias + two Linear layers
    slots, weights = module.image_aggregator(torch.randn(2, 7, DIM))
    assert slots.shape == (2, N_SLOTS, DIM)
    assert weights.shape == (2, N_SLOTS, 7)
    fresh = SelfAggregator(DIM, slots=N_SLOTS, seed=3)
    assert not ({id(p) for p in fresh.parameters()} & image_params)


# --------------------------------------------------------------------------- #
# 3. router and hard gate
# --------------------------------------------------------------------------- #
def test_gate_selects_exactly_k_visual_slots_and_never_the_native_cls(env):
    _, _, _, module, _ = env
    visual_all, text_all = slot_tensors(module, env)
    logits = module.router.logits(visual_all[:, 1:], text_all[:, 0])
    assert logits.shape == (BATCH, BATCH, N_SLOTS), \
        'the router looks at the 32 aggregated visual slots, not at the native CLS'
    gate = hard_gate(logits, K_SAID)
    assert gate.shape == (BATCH, BATCH, N_SLOTS)
    assert gate.sum(dim=-1).tolist() == [[float(K_SAID)] * BATCH] * BATCH
    assert set(gate.reshape(-1).tolist()) <= {0.0, 1.0}
    # deterministic on ties: topk breaks by index, no noise
    tied = torch.zeros(2, 2, N_SLOTS)
    tied_gate = hard_gate(tied, K_SAID)
    assert tied_gate[0, 0, :K_SAID].sum() == K_SAID
    assert tied_gate[0, 0, K_SAID:].sum() == 0.0
    assert torch.equal(tied_gate, hard_gate(tied.clone(), K_SAID))


def test_router_query_is_the_candidate_native_global_not_the_text_slots(env):
    _, _, _, module, _ = env
    visual_all, text_all = slot_tensors(module, env)
    with torch.no_grad():
        scored = module.scorer.score(visual_all, text_all)
        assert torch.equal(scored['gate'],
                           hard_gate(module.router.logits(visual_all[:, 1:], text_all[:, 0])))
        # the aggregated text slots must not influence the gate at all
        garbage = text_all.clone()
        garbage[:, 1:] = torch.randn_like(garbage[:, 1:]) * 3.0
        other = module.scorer.score(visual_all, garbage)
        assert torch.equal(other['gate'], scored['gate']), \
            'the first-version router must not reduce over the text slots'
        # the native global EOS *is* the query: replacing it must move the gate
        changed = text_all.clone()
        changed[:, 0] = torch.randn_like(changed[:, 0])
        moved = module.scorer.score(visual_all, changed)
        assert not torch.equal(moved['gate'], scored['gate'])


# --------------------------------------------------------------------------- #
# 4. hard score
# --------------------------------------------------------------------------- #
def test_hard_score_matches_the_manual_definition(env):
    _, _, _, module, _ = env
    visual_all, text_all = slot_tensors(module, env)
    with torch.no_grad():
        gate = module.scorer.score(visual_all, text_all)['gate']
        score, v2t, t2v = hard_said_score(visual_all, text_all, gate)
        expected = manual_hard_score(visual_all, text_all, gate)
    assert torch.allclose(score.double(), expected, atol=1e-5)
    assert torch.allclose((v2t + t2v).double(), score.double(), atol=1e-6)


def test_masked_slots_are_removed_before_the_max(env):
    _, _, _, module, _ = env
    direction = torch.randn(DIM)
    direction = direction / direction.norm()
    text_all = direction[None, :].repeat(33, 1)                   # every text token points at +x
    visual_all = -direction[None, :].repeat(33, 1)                # every visual token points at -x
    visual_all[1] = 0.0                                          # the masked slot is exactly zero
    gate = torch.zeros(1, 1, N_SLOTS)
    gate[0, 0, :K_SAID] = 1.0
    gate[0, 0, 0] = 0.0                                          # slot 1 is NOT selected
    score, v2t, t2v = hard_said_score(visual_all[None], text_all[None], gate)
    assert float(v2t) < 0 and float(t2v) < 0, (float(v2t), float(t2v))
    assert float(score) == pytest.approx(-2.0, abs=1e-5), \
        'a zeroed masked slot would win the max as soon as every legal cosine is negative'
    expected = manual_hard_score(visual_all[None], text_all[None], gate)
    assert float(score) == pytest.approx(float(expected), abs=1e-5)
    # if the same slot is selected, the +1 cosine is legal and must count
    selected = gate.clone()
    selected[0, 0, 0] = 1.0
    visual_all[1] = direction
    legal = hard_said_score(visual_all[None], text_all[None], selected)[0]
    assert float(legal) > float(score)
    assert float(legal) == pytest.approx(
        float(manual_hard_score(visual_all[None], text_all[None], selected)), abs=1e-5)


def test_all_negative_cosines_and_single_legal_token(env):
    _, _, _, module, _ = env
    a = torch.randn(DIM)
    b = torch.randn(DIM)
    visual_all = torch.stack([a, b, -b, b * 0.5] + [b * 0.25] * (N_SLOTS - 3))
    text_all = torch.stack([b] + [b] * N_SLOTS)
    gate = torch.zeros(1, 1, N_SLOTS, dtype=torch.float32)       # nothing selected: CLS only
    score, v2t, t2v = hard_said_score(visual_all[None], text_all[None], gate)
    assert torch.isfinite(score).all()
    expected = manual_hard_score(visual_all[None], text_all[None], gate)
    assert float(score) == pytest.approx(float(expected), abs=1e-5)
    cosine = float(F.cosine_similarity(a[None], b[None]))
    assert float(v2t) == pytest.approx(cosine, abs=1e-5), \
        'with nothing selected the visual side is the native CLS'
    assert float(score) == pytest.approx(2.0 * cosine, abs=1e-5)


# --------------------------------------------------------------------------- #
# 5. proxy: hard forward, score-layer backward
# --------------------------------------------------------------------------- #
def test_proxy_forward_is_exactly_the_hard_value(env):
    _, _, _, module, _ = env
    visual_all, text_all = slot_tensors(module, env)
    with torch.no_grad():
        scored = module.scorer.score(visual_all, text_all)
        assert torch.equal(scored['score'], scored['hard'])
        assert torch.allclose(scored['score'], scored['hard'] + 0.0)


def test_proxy_gradient_reaches_the_router_and_never_the_features_through_the_grid(env):
    _, _, _, module, _ = env
    visual_all, text_all = slot_tensors(module, env)
    visual = visual_all.clone().requires_grad_(True)
    text = text_all.clone().requires_grad_(True)
    with torch.no_grad():
        frozen_gate = torch.sigmoid(module.router.logits(visual_all[:, 1:], text_all[:, 0]))
    # (a) with the gate held constant the proxy is a pure function of the *detached* cosine grid: it
    #     does not even require grad, so no gradient at all can reach the features or the router.
    surrogate = surrogate_said_score(visual, text, frozen_gate)
    assert not surrogate.requires_grad, \
        'the proxy must be built from a detached cosine grid'
    assert visual.grad is None and text.grad is None
    assert module.router.visual.weight.grad is None
    assert module.router.text.weight.grad is None
    # (b) with the real gate, the router parameters must receive a finite non-zero gradient, and the
    #     only feature path is through sigmoid(a) -- never through the similarity grid.
    module.zero_grad(set_to_none=True)
    visual.grad = None
    text.grad = None
    soft_gate = torch.sigmoid(module.router.logits(visual[:, 1:], text[:, 0]))
    surrogate = surrogate_said_score(visual, text, soft_gate)
    assert surrogate.requires_grad
    surrogate.sum().backward()
    for name, parameter in module.router.named_parameters():
        grad = parameter.grad
        assert grad is not None and torch.isfinite(grad).all(), name
        assert float(grad.abs().max()) > 0.0, name
    assert visual.grad is not None and float(visual.grad[:, 1:].abs().max()) > 0.0
    # the text *slots* appear only inside the detached grid; the only text path is the native query
    # inside sigmoid(a), which is a legitimate part of the gate
    assert text.grad is not None
    assert float(text.grad[:, 1:].abs().max()) == 0.0
    assert float(text.grad[:, 0].abs().max()) > 0.0


def test_proxy_leaves_the_hard_path_differentiable_for_the_features(env):
    _, _, _, module, _ = env
    visual_all, text_all = slot_tensors(module, env)
    visual = visual_all.clone().requires_grad_(True)
    text = text_all.clone().requires_grad_(True)
    with torch.no_grad():
        gate = module.scorer.score(visual_all, text_all)['gate']
    score = hard_said_score(visual, text, gate)[0]
    score.sum().backward()
    assert visual.grad is not None and float(visual.grad.abs().max()) > 0.0
    assert text.grad is not None and float(text.grad.abs().max()) > 0.0


def test_scored_block_exposes_the_documented_keys(env):
    _, _, _, module, _ = env
    visual_all, text_all = slot_tensors(module, env)
    scored = module.scorer.score(visual_all, text_all)
    assert set(scored) == {'score', 'hard', 'hard_v2t', 'hard_t2v', 'logits', 'gate', 'soft'}
    assert scored['soft'].shape == (BATCH, BATCH)
    assert scored['gate'].shape == (BATCH, BATCH, N_SLOTS)


# --------------------------------------------------------------------------- #
# 6. gradient boundary of the two loss terms
# --------------------------------------------------------------------------- #
def _max_grad(module, prefix):
    values = []
    for name, parameter in module.named_parameters():
        if name.startswith(prefix) and parameter.grad is not None:
            values.append(float(parameter.grad.abs().max()))
    return max(values) if values else 0.0


def test_reconstruction_gradients_stay_on_the_image_side(env):
    clip, images, text, module, ids = env
    out = module(images, text, ids, EOT_ID)
    module.zero_grad(set_to_none=True)
    out['loss_rec'].backward(retain_graph=True)
    assert _max_grad(module, 'clip.transformer') == 0.0, \
        'L_rec must not send a direct gradient into the text tower'
    assert clip.text_projection.grad is None or float(clip.text_projection.grad.abs().max()) == 0.0
    assert _max_grad(module, 'text_aggregator') == 0.0
    assert _max_grad(module, 'router') == 0.0
    assert _max_grad(module, 'reference_visual') == 0.0
    assert _max_grad(module, 'clip.visual') > 0.0
    assert _max_grad(module, 'image_aggregator') > 0.0
    assert _max_grad(module, 'decoder') > 0.0

    module.zero_grad(set_to_none=True)
    out['loss_said'].backward(retain_graph=True)
    assert _max_grad(module, 'clip.transformer') > 0.0
    assert float(clip.text_projection.grad.abs().max()) > 0.0
    assert _max_grad(module, 'text_aggregator') > 0.0
    assert _max_grad(module, 'router') > 0.0
    assert _max_grad(module, 'image_aggregator') > 0.0
    assert _max_grad(module, 'clip.visual') > 0.0
    assert _max_grad(module, 'decoder') == 0.0, 'the Said term must not touch the decoder'
    assert _max_grad(module, 'reference_visual') == 0.0

    module.zero_grad(set_to_none=True)
    out['loss_total_for_backward'].backward()
    assert _max_grad(module, 'clip.visual') > 0.0
    assert _max_grad(module, 'decoder') > 0.0
    assert out['m_u'].requires_grad is False
    assert clip.logit_scale.grad is None
    assert all(p.grad is None for p in clip.mask_net.parameters())


def test_unsaid_mask_is_the_detached_complement_of_the_said_mask(env):
    _, images, text, module, ids = env
    out = module(images, text, ids, EOT_ID)
    assert out['m_u'].requires_grad is False
    assert torch.allclose(out['m_u'].sum(dim=-1), torch.full((BATCH,), float(K_SAID)))
    assert torch.allclose(out['m_u'] + out['gate_positive'], torch.ones_like(out['m_u']))


def test_one_optimizer_step_leaves_the_frozen_reference_untouched(env):
    clip, images, text, module, ids = env
    before = {name: parameter.detach().clone()
              for name, parameter in module.reference_visual.named_parameters()}
    optimizer = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad], lr=1e-3)
    out = module(images, text, ids, EOT_ID)
    out['loss_total_for_backward'].backward()
    optimizer.step()
    for name, parameter in module.reference_visual.named_parameters():
        assert torch.equal(before[name], parameter.detach()), name
    assert module.reference_visual.training is False
    module.train()
    assert module.reference_visual.training is False


# --------------------------------------------------------------------------- #
# 7. chunking invariance
# --------------------------------------------------------------------------- #
def test_chunking_does_not_change_the_objective_or_the_gradients():
    batch = 20
    captions = ['a photo of number %d on a table' % i for i in range(batch)]
    image_ids = torch.arange(batch) // 2            # two captions per image: duplicates are exercised
    clip, images, text, module, ids = build_env(chunk=(4, 5), batch=batch, captions=captions,
                                                image_ids=image_ids)
    out_a = module(images, text, ids, EOT_ID)
    out_a['loss_total_for_backward'].backward()
    grads_a = {name: (None if p.grad is None else p.grad.detach().clone())
               for name, p in module.named_parameters()}
    module.zero_grad(set_to_none=True)
    module.chunk_image = batch
    module.chunk_text = batch
    out_b = module(images, text, ids, EOT_ID)
    out_b['loss_total_for_backward'].backward()
    for key in ('loss_said', 'loss_rec', 'positive_said_score', 'loss_said_pairs_total'):
        left, right = float(out_a[key]), float(out_b[key])
        assert left == pytest.approx(right, rel=1e-5, abs=1e-6), key
    assert float(out_b['loss_said_pairs_total']) == 2 * (batch * (batch - 2))
    for name, parameter in module.named_parameters():
        other = parameter.grad
        first = grads_a[name]
        if first is None or other is None:
            assert first is None and other is None, name
            continue
        difference = float((first - other.detach()).abs().max())
        assert difference <= 1e-6 + 1e-5 * float(first.abs().max()), (name, difference)


def test_candidate_mask_always_keeps_the_native_cls():
    from model.said_token_v1 import _candidate_mask
    gate = torch.zeros(2, 2, N_SLOTS)
    keep = _candidate_mask(gate)
    assert keep.shape == (2, 2, 1 + N_SLOTS)
    assert keep[..., 0].all()
    assert not keep[..., 1:].any()


# --------------------------------------------------------------------------- #
# 8. the T0 switch
# --------------------------------------------------------------------------- #
def test_t0_drops_the_reconstruction_term_and_trains_nothing_else():
    clip, images, text, module, ids = build_env(arm=ARM_T0, lambda_rec=0.0)
    assert module.lambda_rec == 0.0
    out = module(images, text, ids, EOT_ID)
    assert float(out['lambda_rec']) == 0.0
    assert float(out['weighted_rec']) == 0.0
    assert float(out['loss_total_for_backward']) == pytest.approx(float(out['loss_said']), abs=1e-7)
    out['loss_total_for_backward'].backward()
    assert _max_grad(module, 'decoder') == 0.0
    assert _max_grad(module, 'clip.visual') > 0.0
    assert _max_grad(module, 'clip.transformer') > 0.0
    objective_keys = [key for key in out if 'loss' in key or 'smart' in key or 'mask_sparsity' in key]
    assert not [key for key in objective_keys
                if 'smart' in key or 'nce' in key or 'cvssl' in key or 'sparsity' in key], \
        'T0 is the same token objective with lambda_rec = 0, never a fallback to SmartCLIP'


def test_lambda_rec_weight_is_a_plain_constant():
    _, images, text, module, ids = build_env(arm=ARM_T1, lambda_rec=0.1)
    out = module(images, text, ids, EOT_ID)
    assert float(out['lambda_rec']) == pytest.approx(0.1)
    assert float(out['weighted_rec']) == pytest.approx(0.1 * float(out['loss_rec']), rel=1e-6)
    assert float(out['loss_total_for_backward']) == pytest.approx(
        float(out['loss_said']) + 0.1 * float(out['loss_rec']), rel=1e-6)


# --------------------------------------------------------------------------- #
# 9. export, RNG isolation, CLI
# --------------------------------------------------------------------------- #
def test_bare_student_state_loads_strictly(env):
    clip, _, _, module, _ = env
    state = module.clip.state_dict()
    assert not [key for key in state if key.startswith(('image_aggregator', 'text_aggregator',
                                                        'router', 'decoder', 'reference_visual'))]
    fresh = tiny_clip(seed=99)
    missing, unexpected = fresh.load_state_dict(state, strict=True)
    assert not missing and not unexpected, (missing, unexpected)


def test_module_construction_does_not_disturb_the_global_rng():
    clip = tiny_clip(seed=11)
    state = torch.random.get_rng_state()
    expected = torch.randn(4)
    torch.random.set_rng_state(state)
    SaidTokenV1Module(clip, rank=0, seed=0)
    assert torch.equal(torch.randn(4), expected), \
        'the aggregators/router/decoder must not shift the caption and augmentation stream'


def test_cli_entry_point_is_documented_and_rejects_t0_with_a_weight():
    script = os.path.join(REPO_ROOT, 'train', 'train_said_token_v1.py')
    result = subprocess.run([sys.executable, script, '--help'], capture_output=True, text=True,
                            cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr[-2000:]
    for flag in ('--arm', '--lambda_rec', '--n_slots', '--k_said', '--router_tau', '--chunk_image',
                 '--chunk_text', '--save_completed_steps', '--online_check_steps', '--diag_every',
                 '--init_state', '--output_dir'):
        assert flag in result.stdout, flag
    rejected = subprocess.run([sys.executable, script, '--arm', ARM_T0, '--lambda_rec', '0.1',
                               '--init_state', '/nonexistent.pt', '--output_dir', '/tmp/t1'],
                              capture_output=True, text=True, cwd=REPO_ROOT)
    assert rejected.returncode != 0
    assert 'lambda_rec 0' in (rejected.stdout + rejected.stderr)


# --------------------------------------------------------------------------- #
# 10. 2-rank DDP equality
# --------------------------------------------------------------------------- #
WORKER = os.path.join(REPO_ROOT, 'tests', '_t1_ddp_worker.py')
ATOL = 0.0
RTOL = 1e-5


def _run_worker(mode, rows, out_path, port):
    command = [sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node=2',
               '--master_port', str(port), WORKER, '--mode', mode, '--rows', str(rows),
               '--out', out_path]
    env = {**os.environ, 'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port),
           'GLOO_SOCKET_IFNAME': 'lo'}
    return subprocess.run(command, capture_output=True, text=True, env=env, cwd=REPO_ROOT)


def test_two_rank_t1_step_matches_the_single_process_reference(tmp_path):
    out = str(tmp_path / 't1_{rank}.json')
    result = _run_worker('single', 2, out, 29721)
    assert result.returncode == 0, result.stderr[-3000:]
    single = json.load(open(out.format(rank=0)))
    assert single['rows'] == 4, 'the single-process run must hold the whole batch'
    result = _run_worker('rank', 2, out, 29722)
    assert result.returncode == 0, result.stderr[-3000:]
    ranks = [json.load(open(out.format(rank=rank))) for rank in range(2)]
    assert [payload['rows'] for payload in ranks] == [2, 2]
    reference = single['grads']
    for rank, payload in enumerate(ranks):
        assert payload['grads'].keys() == reference.keys()
        for name, value in reference.items():
            other = payload['grads'][name]
            if value is None or other is None:
                assert value is None and other is None, (rank, name)
                continue
            left = torch.tensor(value, dtype=torch.float64)
            right = torch.tensor(other, dtype=torch.float64)
            difference = float((left - right).abs().max())
            allowed = ATOL + RTOL * float(left.abs().max())
            assert difference <= allowed, (rank, name, difference, allowed)
    for name in reference:
        if reference[name] is None:
            continue
        assert ranks[0]['grads'][name] == ranks[1]['grads'][name], name
    assert ranks[0]['param_digest'] == ranks[1]['param_digest']
    assert ranks[0]['module_digest'] == ranks[1]['module_digest']
    assert ranks[0]['decoder_digest'] == ranks[1]['decoder_digest']
    # every rank reports the same global means, and they match the single-process numbers
    for key in ('loss_said_global_mean', 'loss_rec_global_mean', 'loss_total_global_mean'):
        assert ranks[0][key] == pytest.approx(ranks[1][key], rel=1e-6, abs=1e-8)
        assert ranks[0][key] == pytest.approx(single[key], rel=1e-5, abs=1e-7), key
    # the reported global mean is exactly what the amplified backward value averages to
    assert max(ranks[0]['said_scaling_abs_diff'], ranks[0]['rec_scaling_abs_diff']) < 1e-6


def test_ragged_batches_are_refused_before_any_gather(tmp_path):
    out = str(tmp_path / 'ragged_{rank}.json')
    result = _run_worker('ragged', 2, out, 29723)
    assert result.returncode != 0
    assert 'unequal per-rank batch sizes' in result.stderr, result.stderr[-2000:]
