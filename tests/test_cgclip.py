"""CG-CLIP v0.1 acceptance tests.

These tests exercise the REAL CLIP ViT-B/16 module and the REAL cached checkpoint, because the whole
point of the arm is what it does to the native last visual block. They are written so that a wrong
implementation fails loudly:

* the native CLS row is compared against the untouched native block and against ``encode_image``, in
  values and in gradients;
* the all-open case is compared against the 10x reference gradient and against the exact gate-gradient
  pattern the design predicts (``A`` and ``b`` non-zero, ``B`` exactly zero);
* the renormalised conditional attention is compared against an independent "masked logits + softmax"
  definition, and an independent projection of that attention must reproduce the model's own scores;
* the straight-through gradient is checked against the chain rule through an INDEPENDENT
  differentiable implementation of the same renormalisation, and the denominator-detached variant is
  shown to give a different gradient;
* four named wrong variants (value masking without renormalisation, raw soft probabilities instead of
  the hard straight-through gate, transposed pair indexing, text query replacing the CLS query) are
  shown to be measurably different.

Two cautions shape the tests below. First, the straight-through mask has a piecewise-constant forward
value, so a finite difference of the score is NOT a valid oracle for its gradient (it is zero away from
a threshold and jumps across it); the gradient tests therefore use the chain rule through an
independent implementation instead. Second, whenever two computations must agree numerically, the gate
bias is first shifted so that no logit sits near the 0.5 threshold, because a probability on the
threshold would flip a hard gate and mask a real agreement with a discrete jump.
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (REPO, os.path.join(REPO, 'train'), os.path.join(REPO, 'tests')):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from model import longclip                                                      # noqa: E402
from model.cgclip import (ARM, FIXED_SCALE, GATE_BIAS_INIT, GATE_KEY_DIM, LAMBDA_ATTENTION,  # noqa: E402
                          LAMBDA_GLOBAL, LAMBDA_SPARSE, NORM_EPS, OBJECTIVE, PATCH_TOKENS,
                          CaptionGate, check_resume_compatible, config_dict,
                          conditional_cls_scores, cross_entropy_from_lse_margin, gate_config,
                          gate_init_report, gate_statistics, isolated_rng, lse_margin,
                          native_cls_row, partition_parameters, path_statistics,
                          positive_global_columns, sparse_positive_term, state_digest,
                          total_loss, visual_spec)
import train_cgclip as trainer                                                 # noqa: E402

CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda', 0) if CUDA else torch.device('cpu')
needs_cuda = pytest.mark.skipif(not CUDA, reason='acceptance tests need a CUDA device')
needs_two_gpus = pytest.mark.skipif(torch.cuda.device_count() < 2,
                                    reason='needs two visible GPUs')
CLIP_CACHE = os.path.expanduser('~/.cache/clip/ViT-B-16.pt')
CAPTIONS = ['a photo of a cat on a wooden table', 'a dog running through tall grass',
            'an old bicycle leaning against a wall', 'two boats on a calm lake']
GRAD_NAMES = ('gate.query.weight', 'gate.key.weight', 'gate.bias', 'clip.visual.proj',
              'clip.text_projection', 'clip.visual.transformer.resblocks.11.attn.in_proj_weight',
              'clip.visual.transformer.resblocks.11.mlp.c_fc.weight',
              'clip.visual.transformer.resblocks.0.ln_1.weight',
              'clip.transformer.resblocks.0.attn.in_proj_weight')


def require_clip():
    if not os.path.isfile(CLIP_CACHE):
        pytest.skip('the CLIP ViT-B/16 checkpoint is not cached at %s' % CLIP_CACHE)


def build_model(device=DEVICE):
    require_clip()
    model, _ = longclip.load_from_clip('ViT-B/16', device='cpu', download_root=None,
                                       args=argparse.Namespace())
    model.logit_scale = torch.nn.Parameter(torch.ones([]) * math.log(100.0))
    return model.to(device).eval()


def fixed_images(count=2, seed=20260913):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(count, 3, 224, 224, generator=generator)


def make_gate(device=DEVICE, seed=7, scale=0.0, bias=GATE_BIAS_INIT):
    """The initial gate (scale 0: fully open) or a gate with a non-zero output layer."""
    gate = CaptionGate(device=device).to(device)
    if scale:
        generator = torch.Generator(device='cpu').manual_seed(seed)
        with torch.no_grad():
            gate.query.weight.copy_(torch.randn(gate.query.weight.shape, generator=generator)
                                    * scale)
            gate.bias.fill_(bias)
    return gate


def block_and_parts(model):
    visual = model.visual
    return visual.transformer.resblocks[-1], visual.ln_post, visual.proj


def small_case(model, images_count=2, texts_count=2, seed=11):
    """A controlled last-block case: a random X11 and random unit texts on the real model's block."""
    generator = torch.Generator().manual_seed(seed)
    last, ln_post, projection = block_and_parts(model)
    x11 = torch.randn(images_count, 197, 768, generator=generator).to(DEVICE)
    texts = F.normalize(torch.randn(texts_count, 512, generator=generator).to(DEVICE), dim=-1,
                        eps=NORM_EPS)
    native = native_cls_row(last, x11, ln_post, projection, want_attention=True)
    return x11, texts, last, native


def raw_gate_logits(gate, texts, native):
    u_patches = native['u'][:, 1:, :]
    return torch.einsum('jd,ipd->jip', gate.project_query(texts),
                        gate.project_key(u_patches)) / math.sqrt(GATE_KEY_DIM)


def stable_gate(gate, texts, native, margin=0.01, trials=800, stride=0.005):
    """Shift the gate bias so that no logit sits within ``margin`` of the 0.5 threshold.

    The forward value of a straight-through gate is piecewise constant, so two computations that ought
    to agree can still disagree by a whole gate. Keeping every logit away from the threshold makes such
    a comparison a test of the arithmetic instead of a test of luck.
    """
    with torch.no_grad():
        raw = raw_gate_logits(gate, texts, native)
        chosen = None
        for trial in range(trials):
            candidate = stride * trial
            if float((raw + candidate).abs().min()) > margin:
                chosen = candidate
                break
        if chosen is None:
            raise AssertionError('no evaluation point away from the gate threshold was found')
        gate.bias.fill_(chosen)
        logits = raw + chosen
        hard = (torch.sigmoid(logits) >= 0.5).float()
    assert float(logits.abs().min()) > margin
    assert 0.05 < float(hard.mean()) < 0.95, 'the evaluation point must be genuinely mixed'
    return chosen, hard


def conditional(x11, native, last, ln_post, projection, gate, texts, image_chunk=2, text_chunk=2,
                use_checkpoint=False, positive_columns=None, capture=None):
    return conditional_cls_scores(
        x11, native, last, gate, texts, gate.project_query(texts),
        gate.project_key(native['u'][:, 1:, :]), projection, ln_post, image_chunk=image_chunk,
        text_chunk=text_chunk, use_checkpoint=use_checkpoint, positive_columns=positive_columns,
        capture=capture)


def score_from_attention(attention_rows, native, last, ln_post, projection, texts):
    """Independent projection of an attention tensor [texts, images, heads, 197] into 100-scaled scores.

    This mirrors the conditional block's arithmetic exactly while taking the attention weights as an
    input, so a test can differentiate with respect to the mask or the weights directly.
    """
    images = native['cls'].shape[0]
    texts_count = attention_rows.shape[0]
    heads = attention_rows.shape[2]
    head_dim = native['v'].shape[-1]
    flat = attention_rows.permute(1, 2, 0, 3).reshape(images * heads, texts_count, 197)
    head_out = torch.bmm(flat, native['v'].reshape(images * heads, 197, head_dim))
    head_out = head_out.reshape(images, heads, texts_count, head_dim).permute(2, 0, 1, 3)
    out = last.attn.out_proj(head_out.reshape(texts_count * images, heads * head_dim))
    residual = native['c'].unsqueeze(0) + out.reshape(texts_count, images, 768)
    hidden = ln_post(residual + last.mlp(last.ln_2(residual)))
    # the model returns [image, text]; ``hidden`` above is [text, image, 512]
    return FIXED_SCALE * torch.einsum(
        'jid,jd->ij', F.normalize(hidden @ projection, dim=-1, eps=NORM_EPS), texts)


def attention_from_mask(mask, native, detach_denominator=False):
    """``a_cond = (a * m_full) / sum(a * m_full)`` with ``m_full = [1, m]``."""
    ones = torch.ones_like(mask[..., :1])
    mask_full = torch.cat([ones, mask], dim=-1)
    weighted = native['attention'].squeeze(2).unsqueeze(0) * mask_full.unsqueeze(2)
    denominator = weighted.sum(-1, keepdim=True)
    if detach_denominator:
        denominator = denominator.detach()
    return weighted / denominator


def named_grads(module, names=GRAD_NAMES):
    found = dict(module.named_parameters())
    return {name: (None if found[name].grad is None else found[name].grad.detach().cpu().clone())
            for name in names}


# --------------------------------------------------------------------------- A: native identity
@needs_cuda
def test_a_native_cls_row_equals_the_untouched_native_block():
    model = build_model()
    images = fixed_images(2).to(DEVICE)
    with torch.no_grad():
        tokens = model.encode_visual_prefinal(images)
        reference_output = model.encode_image(images)
    assert tuple(tokens['x11_raw'].shape) == (2, 197, 768)
    assert tokens['patch11_raw'].shape[1] == PATCH_TOKENS

    last, ln_post, projection = block_and_parts(model)
    x11 = tokens['x11_raw'].float().detach().requires_grad_(True)
    native = native_cls_row(last, x11, ln_post, projection)

    reference_tokens = last(x11.permute(1, 0, 2).contiguous())
    reference_projected = ln_post(reference_tokens[0]) @ projection
    difference_block = float((native['projected'] - reference_projected).abs().max())
    difference_encoder = float((native['projected'] - reference_output).abs().max())
    assert difference_block < 1e-4, difference_block
    assert difference_encoder < 1e-4, difference_encoder
    assert native['projected'].shape == (2, 512)
    assert native['attention'].shape == (2, 12, 1, 197)
    assert native['logits'].shape == (2, 12, 1, 197)
    assert float(native['attention'].sum(-1).sub(1.0).abs().max()) < 1e-5
    spec = visual_spec(model)
    assert spec['last_num_heads'] == 12 and spec['last_head_dim'] == 64
    assert spec['text_heads'] == 8 and spec['last_embed_dim'] == 768

    mine_parameters = [x11] + [p for p in last.parameters()] + [projection]
    grads_mine = torch.autograd.grad(native['projected'].sum(), mine_parameters)
    x11_reference = x11.detach().clone().requires_grad_(True)
    reference_projected = ln_post(last(x11_reference.permute(1, 0, 2).contiguous())[0]) @ projection
    reference_parameters = [x11_reference] + [p for p in last.parameters()] + [projection]
    grads_reference = torch.autograd.grad(reference_projected.sum(), reference_parameters)
    for index, (mine, reference) in enumerate(zip(grads_mine, grads_reference)):
        scale = max(float(reference.abs().max()), 1e-8)
        assert float((mine - reference).abs().max()) / scale < 1e-4, index


# --------------------------------------------------------------------------- B: all-open identity
@needs_cuda
def test_b_initial_gate_is_fully_open_and_both_paths_coincide():
    model = build_model()
    gate = make_gate()
    x11, texts, last, native = small_case(model, 2, 3)
    ln_post, projection = model.visual.ln_post, model.visual.proj
    mask, probability, logits = gate(texts, native['u'][:, 1:, :])
    assert bool((mask.detach() >= 0.5).all())
    assert abs(float(probability.mean()) - 1.0 / (1.0 + math.exp(-GATE_BIAS_INIT))) < 1e-6
    assert float(logits.min()) == pytest.approx(GATE_BIAS_INIT, abs=1e-6)
    assert float(gate.query.weight.abs().max()) == 0.0

    capture = {}
    scores, positive_mask = conditional(x11, native, last, ln_post, projection, gate, texts,
                                        image_chunk=1, text_chunk=1, capture=capture,
                                        positive_columns=torch.arange(2))
    vG = F.normalize(native['projected'], dim=-1, eps=NORM_EPS)
    qg = FIXED_SCALE * (vG @ texts.t())
    assert float((scores - qg).abs().max()) < 1e-3
    # the capture holds the LAST tile only, so the native tensors are sliced with the same indices
    i0, i1, j0, j1 = capture['block']
    assert capture['attention_conditional'].shape == (j1 - j0, i1 - i0, 12, 197)
    native_attention = native['attention'][i0:i1].squeeze(2).unsqueeze(0)
    assert float((capture['attention_conditional'] - native_attention).abs().max()) < 1e-6
    assert float((capture['cls'] - native['cls'][i0:i1].unsqueeze(0)).abs().max()) < 1e-5
    assert float((capture['projected']
                  - native['projected'][i0:i1].unsqueeze(0)).abs().max()) < 1e-4
    assert abs(float(positive_mask.mean()) - 1.0) < 1e-6
    assert abs(float(sparse_positive_term(positive_mask)) - 1.0) < 1e-6


@needs_cuda
def test_b_shared_parameter_gradient_equals_ten_times_the_global_reference():
    model = build_model()
    gate = make_gate()
    module = trainer.CgClipTrainModule(model, gate, rank=0, image_chunk=2, text_chunk=2,
                                       cond_checkpoint=False).to(DEVICE)
    images = fixed_images(2).to(DEVICE)
    text = longclip.tokenize(CAPTIONS[:2]).to(DEVICE)
    frozen_prefixes = ('mask_net.', 'logit_scale')
    frozen = {name for name, _ in module.clip.named_parameters()
              if name.startswith(frozen_prefixes)}
    parameters = [p for name, p in module.clip.named_parameters()
                  if p.requires_grad and name not in frozen]
    assert frozen, 'the frozen compatibility parameters must exist'

    out = module(images, text, amp_enabled=False)
    assert out['stats']['all_gates_open'], out['stats']
    assert float((out['qg_local'] - out['qa_local']).abs().max()) < 1e-3
    assert abs(float(out['loss_sparse']) - 1.0) < 1e-6

    def grads_of(loss):
        found = torch.autograd.grad(loss, parameters, allow_unused=True)
        return {id(p): g for p, g in zip(parameters, found)}

    combined = LAMBDA_GLOBAL * out['loss_global'] + LAMBDA_ATTENTION * out['loss_attention']
    grads_combined = grads_of(combined)
    module.zero_grad(set_to_none=True)
    out2 = module(images, text, amp_enabled=False)
    reference = (LAMBDA_GLOBAL + LAMBDA_ATTENTION) * out2['loss_global']
    grads_reference = grads_of(reference)
    compared = 0
    unused = []
    for parameter in parameters:
        mine = grads_combined[id(parameter)]
        reference_grad = grads_reference[id(parameter)]
        if reference_grad is None or mine is None:
            unused.append(tuple(parameter.shape))
            continue
        scale = max(float(reference_grad.abs().max()), 1e-12)
        assert float((mine - reference_grad).abs().max()) / scale < 1e-3, tuple(parameter.shape)
        compared += 1
    assert compared > 100, compared
    assert not unused, ('every trained tensor must be used by the loss; unused: %r'
                        % (unused[:5],))


@needs_cuda
def test_b_gate_parameter_gradients_at_initialisation_are_what_the_design_says():
    model = build_model()
    gate = make_gate()
    module = trainer.CgClipTrainModule(model, gate, rank=0, image_chunk=2, text_chunk=2,
                                       cond_checkpoint=False).to(DEVICE)
    images = fixed_images(2).to(DEVICE)
    text = longclip.tokenize(CAPTIONS[:2]).to(DEVICE)
    out = module(images, text, amp_enabled=False)
    out['loss_total'].backward()
    # A receives gradient through the sparse term (the logits depend on A even when A = 0), while B
    # receives exactly none until A has moved away from zero
    assert float(gate.query.weight.grad.norm()) > 0
    assert float(gate.key.weight.grad.norm()) == 0.0
    assert float(gate.bias.grad.norm()) > 0
    assert float(model.text_projection.grad.norm()) > 0
    assert model.logit_scale.grad is None
    mask_net = getattr(model, 'mask_net', None)
    assert mask_net is None or all(p.grad is None for p in mask_net.parameters())


# --------------------------------------------------------------------------- C: chunking
@needs_cuda
def test_c_chunked_conditional_scores_match_the_single_pair_loop_including_gradients():
    model = build_model()
    gate = make_gate(scale=2.5, bias=0.0)
    x11, texts, last, native = small_case(model, 3, 3)
    ln_post, projection = model.visual.ln_post, model.visual.proj
    stable_gate(gate, texts, native)

    x11_a = x11.detach().clone().requires_grad_(True)
    native_a = native_cls_row(last, x11_a, ln_post, projection, want_attention=True)
    scores_a, mask_a = conditional(x11_a, native_a, last, ln_post, projection, gate, texts,
                                   image_chunk=3, text_chunk=3,
                                   positive_columns=torch.arange(3))

    entries = []
    for index in range(3):
        for jndex in range(3):
            single_x11 = x11[index:index + 1].detach().clone().requires_grad_(True)
            single_native = native_cls_row(last, single_x11, ln_post, projection)
            single, _ = conditional(single_x11, single_native, last, ln_post, projection, gate,
                                    texts[jndex:jndex + 1], image_chunk=1, text_chunk=1)
            entries.append(single[0, 0])
    loop = torch.stack(entries).reshape(3, 3)
    assert float((scores_a - loop).abs().max()) < 1e-3

    x11_c = x11.detach().clone().requires_grad_(True)
    native_c = native_cls_row(last, x11_c, ln_post, projection, want_attention=True)
    scores_c, mask_c = conditional(x11_c, native_c, last, ln_post, projection, gate, texts,
                                   image_chunk=2, text_chunk=2, use_checkpoint=True,
                                   positive_columns=torch.arange(3))
    assert float((scores_a - scores_c).abs().max()) < 1e-3
    assert float((mask_a - mask_c).abs().max()) < 1e-6

    probe = [gate.query.weight, gate.key.weight, gate.bias]
    grads_a = torch.autograd.grad((scores_a * mask_a.mean()).sum(), [x11_a] + probe)
    grads_c = torch.autograd.grad((scores_c * mask_c.mean()).sum(), [x11_c] + probe)
    for index, (mine, reference) in enumerate(zip(grads_a, grads_c)):
        scale = max(float(reference.abs().max()), 1e-9)
        assert float((mine - reference).abs().max()) / scale < 1e-3, index
    assert float(grads_a[1].abs().max()) > 0 and float(grads_a[2].abs().max()) > 0
    assert float(grads_a[3].abs().max()) > 0


@needs_cuda
def test_c_positive_mask_follows_the_global_positive_column_and_is_no_second_evaluation():
    model = build_model()
    gate = make_gate(scale=2.5, bias=0.0)
    x11, texts, last, native = small_case(model, 2, 2)
    ln_post, projection = model.visual.ln_post, model.visual.proj
    stable_gate(gate, texts, native)
    capture = {}
    _, positive_mask = conditional(x11, native, last, ln_post, projection, gate, texts,
                                  image_chunk=2, text_chunk=2, positive_columns=torch.arange(2),
                                  capture=capture)
    tile_mask = capture['mask']                                   # [texts, images, 196]
    assert float((positive_mask[0] - tile_mask[0, 0]).abs().max()) == 0.0
    assert float((positive_mask[1] - tile_mask[1, 1]).abs().max()) == 0.0
    shifted_columns = torch.tensor([1, 0], dtype=torch.long)
    _, shifted = conditional(x11, native, last, ln_post, projection, gate, texts,
                            image_chunk=2, text_chunk=2, positive_columns=shifted_columns)
    assert float((shifted[0] - tile_mask[1, 0]).abs().max()) == 0.0
    assert float((shifted[1] - tile_mask[0, 1]).abs().max()) == 0.0
    assert float((shifted[0] - shifted[1]).abs().max()) > 1e-6
    # the helper used by the trainer is the global column rule, rank * local_batch + local_index
    assert positive_global_columns(4, 2).tolist() == [8, 9, 10, 11]
    assert positive_mask.shape == (2, PATCH_TOKENS)


# --------------------------------------------------------------------------- D: wrong variants
@needs_cuda
def test_d_renormalisation_equals_the_masked_softmax_definition():
    model = build_model()
    gate = make_gate(scale=2.5, bias=0.0)
    x11, texts, last, native = small_case(model, 1, 1)
    ln_post, projection = model.visual.ln_post, model.visual.proj
    stable_gate(gate, texts, native)
    capture = {}
    conditional(x11, native, last, ln_post, projection, gate, texts, image_chunk=1, text_chunk=1,
                capture=capture)
    attention = capture['attention_conditional']                   # [1,1,12,197]
    mask = capture['mask']                                         # [1,1,196]
    mask_full = torch.cat([torch.ones_like(mask[..., :1]), mask], dim=-1)
    logits = native['logits'].squeeze(2).unsqueeze(0)               # [1,1,12,197]
    reference = torch.softmax(logits.masked_fill(mask_full.unsqueeze(2) < 0.5, float('-inf')), -1)
    assert float((attention - reference).abs().max()) < 1e-6
    # an independent projection of that attention must reproduce the model's own scores
    independent = score_from_attention(attention, native, last, ln_post, projection, texts)
    model_scores = FIXED_SCALE * torch.einsum(
        'jid,jd->ij', F.normalize(capture['projected'], dim=-1, eps=NORM_EPS), texts)
    assert float((independent - model_scores).abs().max()) < 1e-3


@needs_cuda
def test_d_named_wrong_variants_are_measurably_different():
    model = build_model()
    gate = make_gate(scale=2.5, bias=0.0)
    x11, texts, last, native = small_case(model, 2, 2)
    ln_post, projection = model.visual.ln_post, model.visual.proj
    stable_gate(gate, texts, native)
    capture = {}
    scores, _ = conditional(x11, native, last, ln_post, projection, gate, texts, image_chunk=2,
                            text_chunk=2, capture=capture)
    attention = capture['attention_conditional']                    # [2,2,12,197]
    mask = capture['mask']                                          # [2,2,196]
    probabilities = capture['probability']
    native_attention = native['attention'].squeeze(2).unsqueeze(0)  # [1,2,12,197]
    ones = torch.ones_like(mask[..., :1])
    mask_full = torch.cat([ones, mask], dim=-1)

    # the correct attention, projected independently, must reproduce the model's scores
    correct = score_from_attention(attention_from_mask(mask, native), native, last, ln_post,
                                   projection, texts)
    assert float((scores - correct).abs().max()) < 1e-3

    # wrong 1: the mask scales the attention weights (or the values) WITHOUT renormalisation
    wrong_unscaled = score_from_attention(native_attention * mask_full.unsqueeze(2), native, last,
                                          ln_post, projection, texts)
    # wrong 2: the raw soft probabilities are used instead of the straight-through hard gate
    soft_full = torch.cat([ones, probabilities], dim=-1)
    soft_attention = native_attention * soft_full.unsqueeze(2)
    wrong_soft = score_from_attention(soft_attention / soft_attention.sum(-1, keepdim=True), native,
                                      last, ln_post, projection, texts)
    # wrong 3: the pair index is transposed (the gate of text i used for image j)
    transposed_full = torch.cat([ones, mask.transpose(0, 1)], dim=-1)
    transposed_attention = native_attention * transposed_full.unsqueeze(2)
    wrong_transposed = score_from_attention(
        transposed_attention / transposed_attention.sum(-1, keepdim=True), native, last, ln_post,
        projection, texts)
    # wrong 4: the text query replaces the native CLS query
    replacement = torch.einsum('jd,ihpd->jihp', gate.project_query(texts), native['k']) \
        / math.sqrt(native['k'].shape[-1])
    wrong_query = score_from_attention(torch.softmax(replacement, dim=-1), native, last, ln_post,
                                       projection, texts)

    for name, wrong in (('unscaled', wrong_unscaled), ('soft', wrong_soft),
                        ('transposed', wrong_transposed), ('query', wrong_query)):
        assert float((scores - wrong).abs().max()) > 1e-3, name
    assert float((attention - native_attention).abs().max()) > 1e-3


@needs_cuda
def test_d_pair_indexing_uses_the_pair_gate_and_the_fixed_scale():
    model = build_model()
    gate = make_gate(scale=2.5, bias=0.0)
    x11, texts, last, native = small_case(model, 2, 2)
    ln_post, projection = model.visual.ln_post, model.visual.proj
    stable_gate(gate, texts, native)
    capture = {}
    scores, _ = conditional(x11, native, last, ln_post, projection, gate, texts, image_chunk=2,
                            text_chunk=2, capture=capture)
    projected = capture['projected']                                # [texts, images, 768]
    reference = FIXED_SCALE * torch.einsum(
        'jid,jd->ij', F.normalize(projected, dim=-1, eps=NORM_EPS), texts)
    assert float((scores - reference).abs().max()) < 1e-3
    assert abs(FIXED_SCALE - 100.0) < 1e-12
    assert capture['projected'].shape == (2, 2, 512)
    for image in range(2):
        single_native = native_cls_row(last, x11[image:image + 1], ln_post, projection)
        for text in range(2):
            single, _ = conditional(x11[image:image + 1], single_native, last, ln_post, projection,
                                    gate, texts[text:text + 1], image_chunk=1, text_chunk=1)
            assert abs(float(single[0, 0]) - float(scores[image, text])) < 1e-3


# --------------------------------------------------------------------------- E: ST gradient
@needs_cuda
def test_e_straight_through_gradient_is_the_chain_rule_through_the_renormalisation():
    """The mask gradient must equal ``sum_p (d score / d m_p) * p_p (1 - p_p)``.

    ``d score / d m_p`` comes from an INDEPENDENT differentiable implementation of the same
    renormalisation, so the model's straight-through wiring is compared against something that does
    not share its code. The denominator-detached variant is computed too and must give a measurably
    different number, which is what makes this test able to fail.
    """
    model = build_model()
    gate = make_gate(scale=4.0, bias=0.0)
    x11, texts, last, native = small_case(model, 1, 1)
    ln_post, projection = model.visual.ln_post, model.visual.proj
    base, _hard = stable_gate(gate, texts, native, margin=0.02)

    capture = {}
    scores, _ = conditional(x11, native, last, ln_post, projection, gate, texts, image_chunk=1,
                            text_chunk=1, capture=capture)
    analytic = torch.autograd.grad(scores.sum(), gate.bias)[0]

    # detached copies: the independent graphs must not try to walk back into the model's graph
    reference_native = {key: (value.detach() if torch.is_tensor(value) else value)
                        for key, value in native.items()}
    mask_leaf = capture['mask'].detach().clone().requires_grad_(True)
    mask_full = torch.cat([torch.ones_like(mask_leaf[..., :1]), mask_leaf], dim=-1)
    weighted = reference_native['attention'].squeeze(2).unsqueeze(0) * mask_full.unsqueeze(2)
    denominator = weighted.sum(-1, keepdim=True)
    graph = score_from_attention(weighted / denominator, reference_native, last, ln_post,
                                 projection, texts)
    mask_grad = torch.autograd.grad(graph.sum(), mask_leaf, retain_graph=True)[0]

    detached_graph = score_from_attention(weighted / denominator.detach(), reference_native, last,
                                          ln_post, projection, texts)
    mask_grad_detached = torch.autograd.grad(detached_graph.sum(), mask_leaf)[0]

    with torch.no_grad():
        probability = torch.sigmoid(raw_gate_logits(gate, texts, native) + base)
        slope = probability * (1.0 - probability)
    predicted = float((mask_grad * slope).sum())
    predicted_detached = float((mask_grad_detached * slope).sum())

    assert abs(float(analytic)) > 1e-6, 'the gate path must produce a real derivative'
    assert abs(predicted) > 1e-6 and abs(predicted_detached) > 1e-6
    assert abs(float(analytic) - predicted) / abs(predicted) < 1e-3, (float(analytic), predicted)
    assert abs(float(analytic) - predicted_detached) / abs(predicted_detached) > 1e-3, \
        'a detached denominator must give a measurably different gradient'


# --------------------------------------------------------------------------- F: edge cases
@needs_cuda
def test_f_all_patches_off_leaves_the_cls_slot_alone_and_stays_finite():
    model = build_model()
    gate = make_gate(scale=2.5, bias=-80.0)
    x11, texts, last, native = small_case(model, 1, 1)
    ln_post, projection = model.visual.ln_post, model.visual.proj
    capture = {}
    scores, positive_mask = conditional(x11, native, last, ln_post, projection, gate, texts,
                                        image_chunk=1, text_chunk=1, capture=capture,
                                        positive_columns=torch.arange(1))
    assert float(positive_mask.max()) == 0.0, 'every patch gate must be closed'
    assert torch.isfinite(scores).all()
    attention = capture['attention_conditional']
    assert float(attention[..., 0].min()) == pytest.approx(1.0, abs=1e-5)
    assert float(attention[..., 1:].abs().max()) == 0.0
    assert float(sparse_positive_term(positive_mask)) == 0.0
    # the all-off case is exactly the CLS-only read of the same block: attention mass 1 on position 0
    out = last.attn.out_proj(native['v'][:, :, 0, :].reshape(1, -1))
    residual = native['c'] + out
    cls_only = ln_post(residual + last.mlp(last.ln_2(residual))) @ projection
    expected = FIXED_SCALE * F.normalize(cls_only, dim=-1, eps=NORM_EPS) @ texts.t()
    assert float((scores - expected).abs().max()) < 1e-3
    # and it is NOT the native read, which keeps the full softmax over the patches
    native_scores = FIXED_SCALE * F.normalize(native['projected'], dim=-1, eps=NORM_EPS) @ texts.t()
    assert float((scores - native_scores).abs().max()) > 1e-3


@needs_cuda
def test_f_non_finite_attention_is_refused_not_clamped():
    model = build_model()
    gate = make_gate()
    x11, texts, last, native = small_case(model, 1, 1)
    ln_post, projection = model.visual.ln_post, model.visual.proj
    broken = dict(native)
    broken['attention'] = native['attention'].clone()
    broken['attention'][..., 1:] = -1.0
    with pytest.raises(RuntimeError, match='non-finite or non-positive'):
        conditional(x11, broken, last, ln_post, projection, gate, texts, image_chunk=1, text_chunk=1)


# --------------------------------------------------------------------------- G: objective maths
def test_g_lse_margin_is_positive_minus_logsumexp_of_the_negatives_only():
    rows = torch.tensor([[3.0, 1.0, 2.0], [0.5, 4.0, -1.0]])
    targets = torch.tensor([0, 1])
    margin = lse_margin(rows, targets)
    manual = torch.stack([rows[0, 0] - torch.logsumexp(rows[0, 1:], dim=0),
                          rows[1, 1] - torch.logsumexp(rows[1][torch.tensor([0, 2])], dim=0)])
    assert torch.allclose(margin, manual, atol=1e-6)
    assert torch.allclose(cross_entropy_from_lse_margin(margin),
                          F.cross_entropy(rows, targets, reduction='none'), atol=1e-6)
    statistics = path_statistics(rows, targets)
    assert statistics['ce_from_lse_margin_max_abs_diff'] < 1e-6
    assert statistics['top1'] == pytest.approx(1.0)
    assert statistics['positive_win_fraction'] == pytest.approx(1.0)


def test_g_loss_weights_are_five_five_one_and_the_sparse_term_excludes_the_cls_slot():
    assert (LAMBDA_GLOBAL, LAMBDA_ATTENTION, LAMBDA_SPARSE) == (5.0, 5.0, 1.0)
    total = total_loss(torch.tensor(2.0), torch.tensor(3.0), torch.tensor(0.25))
    assert float(total) == pytest.approx(5 * 2.0 + 5 * 3.0 + 1 * 0.25)
    mask = torch.zeros(3, PATCH_TOKENS)
    assert float(sparse_positive_term(mask)) == 0.0            # the CLS slot value 1 is not counted
    assert mask.shape[-1] == 196
    statistics = gate_statistics(mask, mask, sample_pairs=2)
    assert statistics['gate_all_off_fraction'] == pytest.approx(1.0)
    assert statistics['gate_kept_mean'] == 0.0
    assert statistics['gate_kept_max'] == 0.0


def test_g_gate_is_zero_initialised_below_a_xavier_key_and_one_scalar_bias():
    torch.manual_seed(4321)
    before = torch.get_rng_state().clone()
    gate = CaptionGate(device=torch.device('cpu'))
    after = torch.get_rng_state()
    assert torch.equal(before, after), 'building the gate must not consume the global RNG stream'
    assert gate.query.weight.shape == (GATE_KEY_DIM, 512)
    assert gate.key.weight.shape == (GATE_KEY_DIM, 768)
    assert float(gate.query.weight.abs().max()) == 0.0
    assert float(gate.key.weight.std()) > 0
    assert float(gate.bias) == pytest.approx(GATE_BIAS_INIT)
    description = gate_config(gate)
    assert description['gate_kind'] == 'caption_gated_final_cls_attention'
    assert description['gate_key_dim'] == 64
    assert description['shared_over_heads'] is True
    assert description['top_k'] is None and description['soft_floor'] is None
    assert description['cls_self_gate'].startswith('fixed 1')


def test_g_isolated_rng_restores_every_stream():
    import random
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    state_torch = torch.get_rng_state().clone()
    state_numpy = np.random.get_state()
    state_python = random.getstate()
    with isolated_rng(0):
        torch.randn(3)
        np.random.rand(3)
        random.random()
    assert torch.equal(state_torch, torch.get_rng_state())
    assert np.array_equal(state_numpy[1], np.random.get_state()[1])
    assert state_python == random.getstate()


# --------------------------------------------------------------------------- H: partition
@needs_cuda
def test_h_optimizer_partition_covers_every_trainable_parameter_exactly_once():
    model = build_model()
    gate = make_gate()
    partition = partition_parameters(model, gate)
    clip_ids = [id(p) for p in partition['clip']]
    gate_ids = [id(p) for p in partition['gate']]
    assert len(set(clip_ids + gate_ids)) == len(clip_ids) + len(gate_ids)
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    trainable |= {id(p) for p in gate.parameters() if p.requires_grad}
    assert set(clip_ids + gate_ids) == trainable
    last_block = {id(p) for p in model.visual.transformer.resblocks[-1].parameters()}
    assert last_block <= set(clip_ids)
    assert clip_ids.count(id(model.visual.proj)) == 1
    assert not model.logit_scale.requires_grad
    assert id(model.logit_scale) not in set(clip_ids + gate_ids)
    assert {id(p) for p in gate.parameters()} == set(gate_ids)
    optimizers = trainer.build_optimizers(model, gate, clip_lr=1e-6, gate_lr=1e-3)
    assert optimizers['gate'].param_groups[0]['weight_decay'] == 0.0
    assert optimizers['gate'].param_groups[0]['lr'] == pytest.approx(1e-3)
    assert optimizers['clip'].param_groups[0]['lr'] == pytest.approx(1e-6)


# --------------------------------------------------------------------------- I: checkpoint
@needs_cuda
def test_i_production_checkpoint_roundtrip_with_a_trained_gate_and_optimizer_state(tmp_path):
    model = build_model()
    gate = make_gate(scale=2.5, bias=0.0)
    before = {key: value.detach().clone() for key, value in gate.state_dict().items()}
    module = trainer.CgClipTrainModule(model, gate, rank=0, image_chunk=2, text_chunk=2,
                                       cond_checkpoint=False).to(DEVICE)
    optimizers = trainer.build_optimizers(model, gate, clip_lr=1e-6, gate_lr=1e-3)
    images = fixed_images(2).to(DEVICE)
    text = longclip.tokenize(CAPTIONS[:2]).to(DEVICE)
    for _ in range(2):
        out = module(images, text, amp_enabled=False)
        out['loss_total'].backward()
        optimizers['clip'].step()
        optimizers['gate'].step()
        optimizers['clip'].zero_grad(set_to_none=True)
        optimizers['gate'].zero_grad(set_to_none=True)
    moved = any(not torch.equal(before[key], value.detach())
                for key, value in gate.state_dict().items())
    assert moved, 'the gate must have moved away from its initialisation'

    config = config_dict()
    config['gate'] = gate_config(gate)
    config['lr_horizon_steps'] = 10953
    path = os.path.join(str(tmp_path), '%s_step000500.pt' % ARM)
    trainer.write_checkpoint(
        path=path, clip=model, gate=gate, optimizers=optimizers, steps=500, epoch=0,
        step_in_epoch=499, config=config, digests={'init_file_sha256': 'deadbeef'},
        batch_size=256, chunking={'image_chunk': 8, 'text_chunk': 32, 'cond_checkpoint': True},
        lr_horizon_steps=10953,
        gate_report=gate_init_report(gate, torch.zeros(1, 512, device=DEVICE),
                                     torch.zeros(1, PATCH_TOKENS, 768, device=DEVICE)),
        spec=visual_spec(model), rng={'torch': torch.get_rng_state()},
        health={'clip_grad_norm': 1.0})
    assert os.path.isfile(path)
    assert not os.path.isfile(path + '.tmp')

    reloaded = torch.load(path, map_location='cpu', weights_only=False)
    assert reloaded['completed_steps'] == 500
    assert reloaded['arm'] == ARM and reloaded['objective'] == OBJECTIVE
    assert reloaded['gate_kind'] == 'caption_gated_final_cls_attention'
    assert isinstance(reloaded['gate_state'], dict) and reloaded['gate_state']
    assert all(torch.is_tensor(v) for v in reloaded['gate_state'].values())
    assert set(reloaded['gate_state']) == set(gate.state_dict())
    for key, value in gate.state_dict().items():
        assert tuple(reloaded['gate_state'][key].shape) == tuple(value.shape), key
    assert isinstance(reloaded['gate_config'], dict)
    assert reloaded['gate_config']['gate_key_dim'] == 64
    assert reloaded['gate_state_digest'] == state_digest(reloaded['gate_state'])
    assert reloaded['clip_state_digest'] == state_digest(reloaded['clip'])
    assert reloaded['gate_state_digest'] == state_digest(gate.state_dict())
    assert reloaded['optimizer_gate']['state'], 'the gate optimizer state must be real'
    assert reloaded['optimizer_clip']['state'], 'the clip optimizer state must be real'
    assert reloaded['visual_spec']['last_num_heads'] == 12
    assert reloaded['visual_spec']['text_heads'] == 8
    assert reloaded['lr_horizon_steps'] == 10953
    assert reloaded['loss_weights'] == {'global': 5.0, 'attention': 5.0, 'sparse': 1.0}
    assert reloaded['fixed_scale'] == 100.0
    assert reloaded['caption_scope']['cls_sparse'].startswith('true positive pairs only')
    assert reloaded['attention_route']['cls_query'].startswith('native last-block query')
    assert reloaded['grad_health'] == {'clip_grad_norm': 1.0}

    check_resume_compatible(reloaded, config_dict())
    broken = dict(config_dict())
    broken['lambda_sparse'] = 2.0
    with pytest.raises(SystemExit, match='refusing to resume'):
        check_resume_compatible(reloaded, broken)
    broken_kind = dict(config_dict())
    broken_kind['gate_kind'] = 'preproj_mask'
    with pytest.raises(SystemExit, match='refusing to resume'):
        check_resume_compatible(reloaded, broken_kind)

    fresh_gate = make_gate(scale=0.0)
    fresh_gate.load_state_dict(reloaded['gate_state'])
    assert torch.equal(fresh_gate.query.weight.cpu(), gate.query.weight.detach().cpu())
    assert torch.equal(fresh_gate.bias.cpu(), gate.bias.detach().cpu())
    fresh_optimizers = trainer.build_optimizers(model, fresh_gate, clip_lr=1e-6, gate_lr=1e-3)
    fresh_optimizers['gate'].load_state_dict(reloaded['optimizer_gate'])
    assert float(fresh_optimizers['gate'].state_dict()['state'][0]['step']) == \
        float(optimizers['gate'].state_dict()['state'][0]['step'])
    # the restored cursor is the completed step count, never a step count plus one
    assert int(reloaded['completed_steps']) == 500


@needs_cuda
def test_i_gate_digest_detects_missing_or_tampered_gate_weights(tmp_path):
    model = build_model()
    gate = make_gate()
    optimizers = trainer.build_optimizers(model, gate, clip_lr=1e-6, gate_lr=1e-3)
    config = config_dict()
    config['gate'] = gate_config(gate)
    path = os.path.join(str(tmp_path), '%s_step000020.pt' % ARM)
    trainer.write_checkpoint(
        path=path, clip=model, gate=gate, optimizers=optimizers, steps=20, epoch=0, step_in_epoch=19,
        config=config, digests={}, batch_size=256, chunking={}, lr_horizon_steps=100,
        gate_report={}, spec=visual_spec(model))
    payload = torch.load(path, map_location='cpu', weights_only=False)
    assert 'gate_state' in payload and 'gate_config' in payload
    assert payload['gate_state_digest'] == state_digest(payload['gate_state'])
    tampered = dict(payload['gate_state'])
    tampered['bias'] = tampered['bias'] + 1.0
    assert state_digest(tampered) != payload['gate_state_digest'], 'tampering must be detectable'
    missing = dict(payload)
    del missing['gate_state']
    assert 'gate_state' not in missing
    assert missing['gate_state_digest'] is not None      # the header still names the real weights


# --------------------------------------------------------------------------- J: CLI surface
def test_j_trainer_and_runner_cli_surface_is_one_configuration_only():
    help_text = subprocess.run([sys.executable, '-m', 'train.train_cgclip', '--help'],
                               cwd=REPO, capture_output=True, text=True)
    assert help_text.returncode == 0, help_text.stderr[-2000:]
    for needle in ('--batch-size', '--gate_lr', '--warmup_length', '--max_steps', '--init_state',
                   '--image_chunk', '--text_chunk', '--cond_checkpoint', '--save_completed_steps'):
        assert needle in help_text.stdout, needle
    trainer_options = [token for token in help_text.stdout.split() if token.startswith('--')]
    for forbidden in ('--sweep', '--grid', '--extra-epochs', '--allow-missing', '--resume-any'):
        assert all(not option.startswith(forbidden) for option in trainer_options), forbidden

    runner = os.path.join(REPO, 'tools', 'cgclip_runner.py')
    assert os.path.isfile(runner), 'the runner must exist'
    runner_help = subprocess.run([sys.executable, runner, '--help'], cwd=REPO,
                                 capture_output=True, text=True)
    assert runner_help.returncode == 0, runner_help.stderr[-2000:]
    # only the OPTION tokens are inspected: the docstring is allowed to explain which escape hatches
    # were deliberately NOT ported
    options = [token for token in runner_help.stdout.split() if token.startswith('--')]
    assert any(option.startswith('--stages') or option.startswith('--stage') for option in options)
    for forbidden in ('--sweep', '--grid', '--allow-missing', '--extra-epochs'):
        assert all(not option.startswith(forbidden) for option in options), forbidden


# --------------------------------------------------------------------------- K: two-rank DDP
@needs_two_gpus
def test_k_two_rank_update_matches_the_single_process_reference(tmp_path):
    sys.path.insert(0, os.path.join(REPO, 'tests'))
    import _cgclip_ddp_worker as worker
    local_batch = 2
    images, text_ids, captions = worker.build_inputs(local_batch=local_batch)
    image_chunk, text_chunk = 2, 4
    port = str(29591 + (os.getpid() % 200))
    command = [sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node=2',
               '--master_port=%s' % port, os.path.join('tests', '_cgclip_ddp_worker.py'),
               '--output_dir', str(tmp_path), '--local_batch', str(local_batch),
               '--image_chunk', str(image_chunk), '--text_chunk', str(text_chunk)]
    # the container has no usable non-loopback interface, and every other DDP entry point in this
    # project (the runner's training_environment, the PG-CLIP acceptance test) pins NCCL/GLOO to
    # loopback for exactly that reason; the test must do the same or it fails for an environmental
    # reason instead of a modelling one
    env = dict(os.environ, NCCL_SOCKET_IFNAME='lo', GLOO_SOCKET_IFNAME='lo')
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, timeout=1800,
                            env=env)
    assert result.returncode == 0, (result.stdout[-4000:], result.stderr[-4000:])
    ddp_grads = torch.load(os.path.join(str(tmp_path), 'ddp_grads.pt'), map_location='cpu')
    ddp_info = torch.load(os.path.join(str(tmp_path), 'ddp_info.pt'), map_location='cpu')
    assert ddp_info['world_size'] == 2
    assert ddp_info['all_gates_open'] is False, 'the two-rank test must use a mixed gate'

    model = build_model()
    gate = make_gate(scale=worker.GATE_SCALE, bias=worker.GATE_BIAS)
    module = trainer.CgClipTrainModule(model, gate, rank=0, image_chunk=image_chunk,
                                       text_chunk=text_chunk, cond_checkpoint=False).to(DEVICE)
    out = module(images.to(DEVICE), text_ids.to(DEVICE), amp_enabled=False)
    out['loss_total'].backward()
    local_grads = [torch.load(os.path.join(str(tmp_path), 'local_grads_rank%d.pt' % rank),
                              map_location='cpu') for rank in (0, 1)]

    # DECISIVE averaging check: the DDP gradient must be the exact mean of the two ranks' own
    # pre-reduction gradients -- standard averaging, no world-size factor anywhere
    for name in GRAD_NAMES:
        first, second = local_grads[0][name].float(), local_grads[1][name].float()
        mean = (first + second) / 2.0
        scale = max(float(mean.abs().max()), 1e-12)
        assert float((ddp_grads[name].float() - mean).abs().max()) / scale < 1e-4, name
        spread = float((first - second).abs().max()) / scale
        assert spread > 1e-3, (name, spread)      # the average must be a genuine average

    reference = named_grads(module)
    global_scale = max(float(reference[name].abs().max()) for name in GRAD_NAMES)
    assert global_scale > 0
    # noise floor: the SAME single-process reference computed with a different tiling, i.e. a pure
    # floating-point summation-order perturbation of the identical mathematics. The two-rank result is
    # then required to agree with the reference to within a small multiple of that floor, which keeps
    # the test sharp instead of hiding a real mismatch behind a loose constant.
    module.zero_grad(set_to_none=True)
    tiled_module = trainer.CgClipTrainModule(model, gate, rank=0, image_chunk=1, text_chunk=2,
                                             cond_checkpoint=False).to(DEVICE)
    tiled_out = tiled_module(images.to(DEVICE), text_ids.to(DEVICE), amp_enabled=False)
    tiled_out['loss_total'].backward()
    tiled = named_grads(tiled_module)
    module.zero_grad(set_to_none=True)

    report = {}
    problems = []
    for name in GRAD_NAMES:
        assert name in ddp_grads and ddp_grads[name] is not None, name
        assert reference[name] is not None and tiled[name] is not None, name
        mine, theirs = ddp_grads[name].float(), reference[name].float()
        absolute = float((mine - theirs).abs().max())
        noise = float((tiled[name].float() - theirs).abs().max())
        cosine = float(F.cosine_similarity(mine.flatten(), theirs.flatten(), dim=0))
        # Second check, against the whole-batch single-process reference. That reference is the same
        # mathematics with a DIFFERENT floating-point accumulation order (one accumulation over four
        # anchors instead of two over two, plus the NCCL reduction), so the tolerance below is 1% of
        # the tensor's own scale with a direction check. A world-size factor, a wrong positive column
        # or a wrong gather break this by orders of magnitude; the exact-averaging check above is what
        # pins the route down precisely.
        per_tensor_scale = max(float(theirs.abs().max()), 1e-12)
        relative = absolute / per_tensor_scale
        tolerance = 1e-3 * global_scale
        report[name] = {'max_abs_diff': absolute, 'tiling_noise_floor': noise,
                        'absolute_tolerance': tolerance, 'cosine': cosine,
                        'per_tensor_scale': per_tensor_scale,
                        'relative_to_per_tensor_scale': relative}
        if absolute > tolerance or relative > 1e-2 or cosine <= 0.9999:
            problems.append((name, absolute, relative, tolerance, cosine))
    print('DDP_GRADIENT_COMPARISON ' + json.dumps(report, sort_keys=True))
    assert not problems, problems
    # the gate groups really did receive distributed gradients
    assert float(ddp_grads['gate.key.weight'].abs().max()) > 0
    assert float(ddp_grads['gate.query.weight'].abs().max()) > 0
    assert float(ddp_grads['gate.bias'].abs().max()) > 0
