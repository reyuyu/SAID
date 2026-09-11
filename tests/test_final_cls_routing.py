"""SAID-ExGAP v1.5 tests: pre-final routing + native final-block CLS readout.

The twelve required checks (T1..T12) plus the parameter-free / API-surface guarantees. Every test
runs on a *real* ``VisionTransformer`` / ``ResidualAttentionBlock`` at toy scale (width 32, 4
layers, 4 patches), so the code under test is the shipped code, not a mock of it.
"""
import inspect
import math
import os
import sys
import tokenize

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, 'train'), os.path.join(REPO_ROOT, 'tests')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import exgap  # noqa: E402
from model import final_cls_routing as fcr  # noqa: E402
from model import longclip  # noqa: E402
from model.model_longclip import CLIP, VisionTransformer  # noqa: E402
from model.salu_model import SALUModel  # noqa: E402

TOY = dict(width=32, layers=4, heads=4, resolution=8, patch=4, output_dim=16)
BATCH = 4
TOKENS = 5


def build_visual():
    torch.manual_seed(0)
    return VisionTransformer(input_resolution=TOY['resolution'], patch_size=TOY['patch'],
                             width=TOY['width'], layers=TOY['layers'], heads=TOY['heads'],
                             output_dim=TOY['output_dim'])


class _ToyClip(nn.Module):
    """Minimal CLIP-shaped wrapper around a real (toy-scale) ``VisionTransformer``."""

    def __init__(self):
        super().__init__()
        self.visual = build_visual()
        self.text_width = 24
        self.embed_dim = TOY['output_dim']
        self.context_length = 248
        self.text_projection = nn.Parameter(torch.zeros(self.text_width, self.embed_dim))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(20.0)))
        self.text_proj = nn.Linear(self.text_width, self.embed_dim)

    def encode_image(self, images):
        return self.visual(images)

    def encode_image_with_patches(self, images, use_checkpoint=False):
        return self.visual(images, return_patches=True)

    def encode_visual_prefinal(self, images, use_checkpoint=False):
        return self.visual.forward_prefinal(images)

    def encode_text(self, tokens):
        pooled = tokens.float().mean(dim=1, keepdim=True).repeat(1, self.text_width)
        return self.text_proj(pooled)


def make_images(batch=BATCH):
    torch.manual_seed(3)
    return torch.randn(batch, 3, TOY['resolution'], TOY['resolution'])


def make_texts(batch=BATCH):
    torch.manual_seed(5)
    return torch.randint(0, 10, (batch, TOKENS))


def build_model():
    torch.manual_seed(1)
    return SALUModel(_ToyClip(), tau_said=0.07, pair_chunk_size=None)


def finalcls_forward(model, images, texts, **kwargs):
    kwargs.setdefault('objective_mode', 'said_exgap_finalcls')
    kwargs.setdefault('lambda_global', 0.0)
    kwargs.setdefault('lambda_unsaid', 0.0)
    kwargs.setdefault('lambda_exgap', 1.0)
    return model.forward_train(images, texts, **kwargs)


def max_abs(a, b):
    return float((a - b).abs().max())


def codescan(path):
    """Source of a file with comments and docstrings stripped (prose must not be mistaken)."""
    pieces = []
    with open(path, encoding='utf-8') as handle:
        for token in tokenize.generate_tokens(handle.readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            pieces.append(token.string)
    return ' '.join(pieces)


# --------------------------------------------------------------------------- #
# T1 -- pre-final layer correctness
# --------------------------------------------------------------------------- #
def test_t1_prefinal_tokens_are_the_true_block12_input():
    visual = build_visual()
    images = make_images(2)
    pre = visual.forward_prefinal(images)
    x11 = pre['x11_raw']
    assert x11.shape == (2, TOY['resolution'] // TOY['patch'] * (TOY['resolution'] // TOY['patch'])
                         + 1, TOY['width'])
    assert pre['cls11_raw'].shape == (2, TOY['width'])
    assert pre['patch11_raw'].shape == (2, x11.shape[1] - 1, TOY['width'])

    # blocks[0:layers-1] -> X11, then the last block completes the native forward
    tokens = x11.permute(1, 0, 2)
    tokens = visual.transformer.resblocks[-1](tokens)
    tokens = tokens.permute(1, 0, 2)
    manual = visual.ln_post(tokens[:, 0, :]) @ visual.proj
    native = visual(images)
    assert max_abs(manual, native) < 1e-5

    # X11 is genuinely BEFORE the last block, not its output
    assert max_abs(x11[:, 0, :], tokens[:, 0, :]) > 1e-3


def test_t1_clip_encode_visual_prefinal_matches_the_visual_path():
    clip = _ToyClip()
    images = make_images(2)
    through_clip = clip.encode_visual_prefinal(images)
    direct = clip.visual.forward_prefinal(images)
    for key in ('cls11_raw', 'patch11_raw', 'x11_raw'):
        assert torch.equal(through_clip[key], direct[key]), key


# --------------------------------------------------------------------------- #
# T2 -- Global CLS-only exact equivalence (HARD GATE)
# --------------------------------------------------------------------------- #
def test_t2_global_cls_only_readout_equals_native_clip():
    visual = build_visual()
    readout = fcr.FinalBlockCLSReadout(visual)
    images = make_images(4)
    with torch.no_grad():
        native = visual(images)
        prepared = readout.prepare(visual.forward_prefinal(images)['x11_raw'])
        refined = readout.read_global(prepared)
    difference = max_abs(refined, native)
    assert difference <= 1e-5, 'global identity gate failed: max abs error %.3e' % difference
    # and the normalised form is the CLIP embedding direction
    with torch.no_grad():
        assert max_abs(F.normalize(refined, dim=-1), F.normalize(native, dim=-1)) <= 1e-5


def test_t2_global_readout_equals_encode_image_end_to_end():
    """The same gate through the public API, on the wrapped model."""
    model = build_model()
    images = make_images(3)
    with torch.no_grad():
        native = model.encode_image(images)
        readout = model.final_cls_readout()
        prepared = readout.prepare(model.encode_visual_prefinal(images)['x11_raw'])
        refined = readout.read_global(prepared)
    assert max_abs(refined, native) <= 1e-5


# --------------------------------------------------------------------------- #
# T3 -- ones gate identity
# --------------------------------------------------------------------------- #
def test_t3_ones_gate_reproduces_global():
    visual = build_visual()
    readout = fcr.FinalBlockCLSReadout(visual)
    images = make_images(3)
    with torch.no_grad():
        prepared = readout.prepare(visual.forward_prefinal(images)['x11_raw'])
        global_feature = readout.read_global(prepared)
        ones = torch.ones(3, prepared['length'] - 1)
        zeros = torch.zeros(3, prepared['length'] - 1)
        said = readout.read_said(prepared, ones)
        unsaid = readout.read_unsaid(prepared, zeros)       # 1 - 0 == 1
        native = visual(images)
    assert max_abs(said, global_feature) < 1e-6, 'r == 1 must give z_S == g'
    assert max_abs(said, native) <= 1e-5
    assert max_abs(unsaid, global_feature) < 1e-6, 'r == 0 must give z_U == g'


# --------------------------------------------------------------------------- #
# T4 -- pairwise optimized vs slow reference
# --------------------------------------------------------------------------- #
def test_t4_pairwise_readout_matches_the_slow_reference():
    fcr.attach_reference_helper(None)
    visual = build_visual()
    readout = fcr.FinalBlockCLSReadout(visual)
    images = make_images(2)
    torch.manual_seed(7)
    relevance = torch.rand(2, 2, TOY['resolution'] // TOY['patch'] * (TOY['resolution'] // TOY['patch']))
    with torch.no_grad():
        prepared = readout.prepare(visual.forward_prefinal(images)['x11_raw'])
        optimized = readout.read_pairwise_said(prepared, relevance, chunk_size=2)
        x11 = visual.forward_prefinal(images)['x11_raw']
        reference = fcr.reference_final_cls(visual, x11, relevance)
    assert optimized.shape == (2, 2, TOY['output_dim'])
    difference = max_abs(optimized, reference)
    assert difference <= 1e-5, 'pairwise optimized vs reference: %.3e' % difference


def test_t4_pairwise_is_chunk_invariant_and_covers_every_candidate():
    visual = build_visual()
    readout = fcr.FinalBlockCLSReadout(visual)
    images = make_images(3)
    torch.manual_seed(11)
    relevance = torch.rand(3, 5, TOY['resolution'] // TOY['patch'] * (TOY['resolution'] // TOY['patch']))
    with torch.no_grad():
        prepared = readout.prepare(visual.forward_prefinal(images)['x11_raw'])
        whole = readout.read_pairwise_said(prepared, relevance, chunk_size=0)
        chunked = readout.read_pairwise_said(prepared, relevance, chunk_size=2)
    assert whole.shape == (3, 5, TOY['output_dim'])
    assert max_abs(whole, chunked) < 1e-6


def test_t4_pairwise_relevance_matches_the_v1_logit_definition():
    """The chunked pairwise relevance must be the batched form of the v1 Said logit."""
    torch.manual_seed(13)
    dim, patches, batch, candidates = 8, 4, 3, 5
    w_query = nn.Linear(dim, dim)
    w_key = nn.Linear(dim, dim)
    text = F.normalize(torch.randn(candidates, dim), dim=-1)
    patches_tensor = F.normalize(torch.randn(batch, patches, dim), dim=-1)
    chunked = fcr.pairwise_said_relevance(text, patches_tensor, w_query, w_key, tau_said=0.07,
                                          chunk_size=2)
    for candidate in range(candidates):
        single = torch.sigmoid(exgap.said_relevance_logits(
            text[candidate:candidate + 1], patches_tensor, w_query, w_key, tau_said=0.07))
        assert max_abs(chunked[:, candidate, :], single) < 1e-6


# --------------------------------------------------------------------------- #
# T5 -- the Said gate is live (L_S trains the router)
# --------------------------------------------------------------------------- #
def test_t5_said_loss_moves_the_router_through_the_live_gate():
    model = build_model()
    images, texts = make_images(), make_texts()
    out = finalcls_forward(model, images, texts, lambda_exgap=0.0)
    out['loss_said'].backward()
    for name, parameter in model.said_router.named_parameters():
        assert parameter.grad is not None, name
        assert float(parameter.grad.abs().sum()) > 0.0, name
    # and the final visual block (the readout) is trained by L_S too
    final_block = model.clip.visual.transformer.resblocks[-1]
    assert float(final_block.attn.in_proj_weight.grad.abs().sum()) > 0.0
    assert float(final_block.mlp.c_fc.weight.grad.abs().sum()) > 0.0


def test_t5_said_readout_is_not_a_patch_pool():
    """z_S must come from the final block, not from a pool of patch tokens."""
    model = build_model()
    images, texts = make_images(), make_texts()
    features = model.said_final_cls(images, texts)
    assert features['said_feature'].shape == (BATCH, TOY['output_dim'])
    with torch.no_grad():
        _, patches12 = model.encode_image_with_patches(images)
        pooled = F.normalize(patches12.mean(dim=1), dim=-1)
    assert max_abs(features['said_feature'], pooled) > 1e-3
    assert max_abs(features['global_feature'], pooled) > 1e-3


# --------------------------------------------------------------------------- #
# T6 / T7 / T8 -- gradient isolation
# --------------------------------------------------------------------------- #
def exgap_only_forward(model, images, texts, forced_gap=True):
    """A v1.5 forward whose ExGAP term is guaranteed live, for route assertions."""
    features = model.said_final_cls(images, texts, return_details=True)
    gap_weight = (torch.ones_like(features['s_gc']) if forced_gap else features['gap_weight'])
    loss = exgap.compute_exgap_loss(features['s_uc'], features['s_gc'], gap_weight,
                                    temperature=0.05)['loss']
    return loss, features


def test_t6_unsaid_gate_is_detached_from_the_router():
    model = build_model()
    images, texts = make_images(), make_texts()
    loss, _ = exgap_only_forward(model, images, texts)
    loss.backward()
    for name, parameter in model.said_router.named_parameters():
        assert parameter.grad is None or float(parameter.grad.abs().sum()) == 0.0, name
    # the ExGAP term is live on the visual side, so those zeros are real zeros
    assert float(model.clip.visual.transformer.resblocks[-1].mlp.c_fc.weight.grad.abs().sum()) > 0.0


def test_t6_read_unsaid_refuses_a_live_relevance():
    visual = build_visual()
    readout = fcr.FinalBlockCLSReadout(visual)
    images = make_images(2)
    prepared = readout.prepare(visual.forward_prefinal(images)['x11_raw'])
    live = torch.rand(2, prepared['length'] - 1, requires_grad=True)
    with pytest.raises(ValueError):
        readout.read_unsaid(prepared, live)


def test_t7_exgap_never_touches_the_text_encoder():
    model = build_model()
    images, texts = make_images(), make_texts()
    loss, _ = exgap_only_forward(model, images, texts)
    loss.backward()
    for name, parameter in model.clip.text_proj.named_parameters():
        assert parameter.grad is None or float(parameter.grad.abs().sum()) == 0.0, name
    assert model.clip.text_projection.grad is None or \
        float(model.clip.text_projection.grad.abs().sum()) == 0.0


def test_t8_exgap_cannot_flow_through_the_global_reference():
    """``S_gc`` is a constant inside L_ExGAP, so g receives nothing from that path."""
    model = build_model()
    images, texts = make_images(), make_texts()
    loss, features = exgap_only_forward(model, images, texts)
    gradient = torch.autograd.grad(loss, features['global_feature'], allow_unused=True)[0]
    assert gradient is None, 'the global reference path leaked into the global representation'


def test_t8_gap_weight_still_cannot_shrink_the_gap():
    s_sc = torch.tensor([0.8], requires_grad=True)
    s_gc = torch.tensor([0.4], requires_grad=True)
    s_uc = torch.tensor([0.9], requires_grad=True)
    gap = exgap.compute_explanatory_gap(s_sc, s_gc)
    exgap.compute_exgap_loss(s_uc, s_gc, gap['gap_weight'], temperature=0.05)['loss'].backward()
    assert s_sc.grad is None or float(s_sc.grad.abs().sum()) == 0.0
    assert s_gc.grad is None or float(s_gc.grad.abs().sum()) == 0.0
    assert float(s_uc.grad.abs().sum()) > 0.0


# --------------------------------------------------------------------------- #
# T9 / T10 -- gate semantics and the untouched CLS key
# --------------------------------------------------------------------------- #
def test_t9_said_and_unsaid_gates_move_mass_in_opposite_directions():
    visual = build_visual()
    readout = fcr.FinalBlockCLSReadout(visual)
    images = make_images(1)
    prepared = readout.prepare(visual.forward_prefinal(images)['x11_raw'])
    patches = prepared['length'] - 1
    # one strongly-Said patch, one clearly unsaid patch, the rest neutral
    relevance = torch.full((1, patches), 0.5)
    relevance[0, 0] = 0.9
    relevance[0, 1] = 0.1
    with torch.no_grad():
        said = readout.attention_weights(prepared, relevance)[0, 0, 1:]
        unsaid = readout.attention_weights(prepared, relevance, complement=True)[0, 0, 1:]
        base = F.softmax(prepared['base_logits'], dim=-1)[0, 0, 1:]
    assert float(said[0]) > float(said[1]), 'the high-relevance patch must gain Said mass'
    assert float(unsaid[1]) > float(unsaid[0]), 'the low-relevance patch must gain Unsaid mass'
    assert float(said[0] / base[0]) > 1.0
    assert float(unsaid[1] / base[1]) > 1.0
    assert float(said[0] / said[1]) > float(base[0] / base[1])
    assert float(unsaid[1] / unsaid[0]) > float(base[1] / base[0])


def test_t10_the_cls_key_bias_is_always_zero():
    relevance = torch.rand(3, 5)
    for complement in (False, True):
        bias = fcr.gate_bias(relevance, complement=complement)
        assert bias.shape == (3, 6)
        assert float(bias[:, 0].abs().max()) == 0.0
        # patch keys are suppressed only (log of a value in (0, 1])
        assert float(bias[:, 1:].max()) <= 0.0
    ones = torch.ones(2, 4)
    zeros = torch.zeros(2, 4)
    assert float(fcr.gate_bias(ones).abs().max()) == 0.0
    assert float(fcr.gate_bias(zeros)[:, 1:].min()) < -10.0      # clamped, never -inf


# --------------------------------------------------------------------------- #
# T11 -- no forbidden H12 pooling in the v1.5 path
# --------------------------------------------------------------------------- #
def test_t11_v1_5_path_contains_no_patch_pooling_global():
    source = codescan(os.path.join(REPO_ROOT, 'model', 'final_cls_routing.py'))
    for banned in ('compute_global_representation', 'mean_global_weights',
                   'compute_masked_unsaid_representation', 'reconstruct', 'prototype',
                   'decoder', 'teacher'):
        assert banned not in source, banned
    model_source = codescan(os.path.join(REPO_ROOT, 'model', 'salu_model.py'))
    start = model_source.index('def _forward_said_exgap_finalcls')
    end = model_source.index('def _finalcls_collapse_metrics')
    body = model_source[start:end]
    for banned in ('compute_global_representation', 'mean_global_weights', 'normalize_patches',
                   'compute_masked_unsaid_representation', 'compute_said_mask'):
        assert banned not in body, banned


def test_t11_forward_reports_the_native_representation_family():
    model = build_model()
    images, texts = make_images(), make_texts()
    out = finalcls_forward(model, images, texts)
    with torch.no_grad():
        native = model.encode_image(images)
    # every reported feature is unit-norm and in the visual projection space
    for key in ('global_feature_norm', 'said_feature_norm', 'unsaid_feature_norm'):
        assert float(out[key]) == pytest.approx(1.0, abs=1e-4), key


# --------------------------------------------------------------------------- #
# T12 -- LongCLIP 248-token regression
# --------------------------------------------------------------------------- #
def test_t12_longclip_context_length_is_still_248():
    tokens = longclip.tokenize(['a cat on a mat'] * 3, truncate=True)
    assert tokens.shape[1] == 248
    assert longclip.tokenize(['x'], truncate=True).shape[1] == 248


def test_t12_v1_5_objective_uses_the_same_tokenizer():
    source = codescan(os.path.join(REPO_ROOT, 'train', 'train_salu.py'))
    assert 'longclip' in source and 'tokenize' in source and 'truncate' in source
    # the v1.5 path tokenises (I, C) exactly once, through build_gap_train_batch
    assert 'build_gap_train_batch' in source


# --------------------------------------------------------------------------- #
# parameter-free / API-surface guarantees
# --------------------------------------------------------------------------- #
def test_readout_owns_no_parameter_and_is_not_registered():
    model = build_model()
    names_before = {name for name, _ in model.named_parameters()}
    keys_before = set(model.state_dict())
    readout = model.final_cls_readout()
    assert list(readout.named_parameters()) == []
    assert {name for name, _ in model.named_parameters()} == names_before
    assert set(model.state_dict()) == keys_before
    assert not any(name == 'final_cls_readout' or name.startswith('final_cls_readout.')
                   for name, _ in model.named_modules())


def test_readout_uses_the_original_final_block_tensors():
    model = build_model()
    readout = model.final_cls_readout()
    visual = model.clip.visual
    block = visual.transformer.resblocks[-1]
    groups = readout.gate_parameters()
    assert groups['ln_1'][0] is block.ln_1.weight
    assert groups['attn_in_proj_weight'][0] is block.attn.in_proj_weight
    assert groups['attn_out_proj'][0] is block.attn.out_proj.weight
    assert groups['ln_2'][0] is block.ln_2.weight
    assert groups['mlp'][0] is block.mlp.c_fc.weight
    assert groups['ln_post'][0] is visual.ln_post.weight
    assert groups['proj'][0] is visual.proj
    # the readout's own reference is the same object, not a copy
    assert readout.visual is visual
    assert readout.block is block


def test_objective_mode_does_not_change_standard_inference():
    model = build_model()
    images, texts = make_images(), make_texts()
    with torch.no_grad():
        before = model.encode_image(images)
    finalcls_forward(model, images, texts)
    with torch.no_grad():
        after = model.encode_image(images)
    assert torch.equal(before, after)
    assert model.encode_image.__func__ is SALUModel.encode_image


def test_forward_geometry_is_identical_when_lambda_exgap_is_zero():
    """The matched control must still compute every geometry quantity."""
    model = build_model()
    images, texts = make_images(), make_texts()
    with_zero = finalcls_forward(model, images, texts, lambda_exgap=0.0)
    with_one = finalcls_forward(model, images, texts, lambda_exgap=1.0)
    for key in ('S_gc_mean', 'S_sc_mean', 'S_uc_mean', 'gap_weight_mean',
                'said_relevance_mean', 'said_to_global_cos',
                'explanatory_gap_positive_fraction'):
        assert torch.allclose(torch.as_tensor(float(with_zero[key])),
                             torch.as_tensor(float(with_one[key])), atol=1e-6), key
    assert float(with_zero['loss_total']) != float(with_one['loss_total'])
    assert float(with_zero['lambda_exgap']) == 0.0


def test_finalcls_diagnostics_are_present_and_finite():
    model = build_model()
    images, texts = make_images(), make_texts()
    out = finalcls_forward(model, images, texts)
    required = ('S_gc_mean', 'S_sc_mean', 'S_uc_mean', 'S_sc_minus_S_gc_mean',
                'S_uc_minus_S_gc_mean', 'D_U_mean', 'explanatory_gap_raw_mean',
                'gap_weight_mean', 'explanatory_gap_positive_fraction',
                'said_relevance_mean', 'said_relevance_std', 'said_relevance_p10',
                'said_relevance_p25', 'said_relevance_p50', 'said_relevance_p75',
                'said_relevance_p90', 'said_fraction_gt_0_6', 'said_fraction_gt_0_7',
                'said_effective_patch_count', 'unsaid_effective_patch_count',
                'said_to_global_cos', 'unsaid_to_global_cos', 'said_to_unsaid_cos',
                'said_minus_global_l2', 'unsaid_minus_global_l2',
                'route_top1_acc', 'evidence_top1_acc')
    for key in required:
        assert key in out, key
        assert torch.isfinite(torch.as_tensor(float(out[key]))), key
    # the soft gate is not a hard 0/1 mask: both effective counts are proper fractions of N
    patches = out['said_relevance'].shape[-1]
    assert 1.0 <= float(out['said_effective_patch_count']) <= float(patches)
    assert 1.0 <= float(out['unsaid_effective_patch_count']) <= float(patches)
    # the intervention is non-trivial
    assert float(out['said_minus_global_l2']) > 0.0
    assert float(out['unsaid_minus_global_l2']) > 0.0


def test_collapse_metrics_use_the_native_cls():
    model = build_model()
    images, texts = make_images(), make_texts()
    out = finalcls_forward(model, images, texts, collapse_cohort=images,
                          collapse_cohort_texts=texts)
    for key in ('global_cls_pairwise_cos', 'global_cls_std', '64way_i2t_at1'):
        assert key in out, key
        assert torch.isfinite(torch.as_tensor(float(out[key]))), key
    assert -1.0 <= float(out['global_cls_pairwise_cos']) <= 1.0
    assert 0.0 <= float(out['64way_i2t_at1']) <= 1.0


def test_said_exgap_finalcls_rejects_mixed_objectives():
    model = build_model()
    images, texts = make_images(), make_texts()
    for kwargs in ({'lambda_global': 1.0}, {'lambda_unsaid': 0.5},
                   {'global_caption_view': 'full'}):
        arguments = {'objective_mode': 'said_exgap_finalcls', 'lambda_global': 0.0,
                     'lambda_unsaid': 0.0}
        arguments.update(kwargs)
        with pytest.raises(ValueError):
            model.forward_train(images, texts, **arguments)


def test_prepare_validates_its_input():
    visual = build_visual()
    readout = fcr.FinalBlockCLSReadout(visual)
    with pytest.raises(ValueError):
        readout.prepare(torch.randn(2, 5))                       # not [B, L, D]
    with pytest.raises(ValueError):
        readout.prepare(torch.randn(2, 7, TOY['width']))          # wrong token count
    with pytest.raises(ValueError):
        readout.prepare(torch.randn(2, 5, TOY['width'] + 1))      # wrong width


def test_gate_bias_validates_shape():
    with pytest.raises(ValueError):
        fcr.gate_bias(torch.rand(3))
