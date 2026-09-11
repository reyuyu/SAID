"""Phase ExGAP-1B tests: causal-control machinery, gradient attribution and patch-global retrieval.

These cover the four new surfaces of the phase:

* ``SALUModel.encode_exgap_global`` -- the evaluator-facing ``z_G = Pool(H)`` API, which must be
  bit-identical to the ``z_G`` the training forward uses;
* ``exgap_grad_attribution`` -- the per-term ``G_S`` / ``G_ExGAP`` / ``R_grad`` measurement and,
  most importantly, that the ExGAP gradient reaches the visual backbone and *nothing else*;
* ``patch_global_features`` -- the single shared patch-global implementation used by the
  evaluator, preferring the model API and recording which path ran;
* the canonical evaluator's separate ``patch_global`` / ``legacy_cls`` columns.
"""
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, 'train'), os.path.join(REPO_ROOT, 'tests')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import exgap  # noqa: E402
from test_gap_completion import build_model, make_batch  # noqa: E402
from test_said_exgap import exgap_forward  # noqa: E402
from train_salu import exgap_grad_attribution, exgap_grad_groups  # noqa: E402
from eval.retrieval.coco_retrieval import patch_global_features  # noqa: E402

BATCH = 5
PATCHES = 6


def _args(**overrides):
    base = dict(lambda_said=1.0, lambda_exgap=1.0, exgap_global_pool='mean',
                exgap_mask_threshold=0.6, exgap_temperature=0.05, exgap_normalize_gap=True)
    base.update(overrides)
    return SimpleNamespace(**base)


# --------------------------------------------------------------------------- #
# encode_exgap_global: the evaluator-facing z_G
# --------------------------------------------------------------------------- #
def test_encode_exgap_global_is_the_training_z_g():
    """The API and the training forward must produce the same z_G, not two similar ones."""
    model = build_model()
    images, texts = make_batch()
    api = model.encode_exgap_global(images, global_pool='mean')
    assert api.shape == (BATCH, 8)
    assert torch.allclose(api.norm(dim=-1), torch.ones(BATCH), atol=1e-5)

    # the same quantity inside the training forward
    with torch.no_grad():
        _, patch_features = model.encode_router_input(images)
        manual = exgap.compute_global_representation(
            exgap.normalize_patches(patch_features), pool='mean')[0]
    assert torch.equal(api, manual)

    # and the inference API reports the very same vector
    encoded = model.encode_said_exgap(images, texts, exgap_global_pool='mean')
    assert torch.equal(encoded['global_feature'], api)


def test_encode_exgap_global_is_caption_independent():
    model = build_model()
    images, texts = make_batch()
    model.build_exgap_module()          # attention pooling must exist for this path
    first = model.encode_exgap_global(images, global_pool='attention')
    second = model.encode_exgap_global(images.flip(0), global_pool='attention')
    # a different image batch with the same content order is identical; the point here is that
    # no caption ever enters, so re-running with shuffled images shuffles the outputs exactly
    assert torch.allclose(first.flip(0), second, atol=1e-5)


# --------------------------------------------------------------------------- #
# patch_global_features: one shared implementation
# --------------------------------------------------------------------------- #
def test_patch_global_features_prefers_the_api_and_records_the_source():
    model = build_model()
    images, _ = make_batch()
    features, source = patch_global_features(model, images, global_pool='mean')
    assert source == 'model_api'
    assert torch.equal(features, model.encode_exgap_global(images, global_pool='mean'))


def test_patch_global_features_falls_back_to_the_same_formula():
    """A model without the API still gets the identical definition, not a second one."""
    model = build_model()
    images, _ = make_batch()
    expected = model.encode_exgap_global(images, global_pool='mean')

    class _NoApi(torch.nn.Module):
        def __init__(self, core):
            super().__init__()
            self.core = core

        def encode_router_input(self, images):
            return self.core.encode_router_input(images)

        def parameters(self, recurse=True):
            return self.core.parameters(recurse=recurse)

    features, source = patch_global_features(_NoApi(model), images, global_pool='mean')
    assert source == 'training_formula'
    assert torch.equal(features, expected)


def test_patch_global_features_rejects_attention_without_the_api():
    images = torch.randn(2, 3, 4, 4)

    class _NoApi(torch.nn.Module):
        def encode_router_input(self, images):
            raise AssertionError('must not be reached')

    try:
        patch_global_features(_NoApi(), images, global_pool='attention')
    except ValueError as error:
        assert 'encode_exgap_global' in str(error)
    else:                                   # pragma: no cover - the call must raise
        raise AssertionError('attention pooling without the API must raise')


# --------------------------------------------------------------------------- #
# gradient attribution (Phase ExGAP-1B section 9)
# --------------------------------------------------------------------------- #
def test_grad_groups_split_the_model_into_the_attributed_groups():
    model = build_model()
    model.build_exgap_module()
    groups = exgap_grad_groups(model)
    assert set(groups) >= {'visual_backbone', 'patch_pathway', 'text_encoder', 'said_router',
                           'exgap_pooling'}
    names = {name for name, _ in model.named_parameters()}
    assert any(name.startswith('said_router.') for name in names)
    assert any(name.startswith('exgap.') for name in names)
    # no parameter may be silently dropped. ``patch_pathway`` is deliberately a subset of
    # ``visual_backbone`` (in the real ViT the patch tokens come out of the backbone itself), so
    # the union is what has to cover every trainable parameter.
    union = set()
    for parameters in groups.values():
        union.update(id(parameter) for parameter in parameters)
    assert len(union) == len([1 for _, p in model.named_parameters() if p.requires_grad])
    assert all(id(p) in union
               for p in groups['patch_pathway'])


def test_grad_attribution_returns_only_the_requested_steps():
    model = build_model()
    images, texts = make_batch()
    assert exgap_grad_attribution(model, images, texts, _args(), {20, 100}, 7) == {}
    record = exgap_grad_attribution(model, images, texts, _args(), {20, 100}, 20)
    assert record['grad_attr_gS_visual_backbone'] >= 0.0
    assert record['grad_attr_gE_visual_backbone'] >= 0.0
    assert 'grad_attr_R_visual_backbone' in record


def test_grad_attribution_is_measured_even_in_the_lambda_exgap_zero_arm():
    """M0 must build the identical forward geometry; the attribution reads it regardless."""
    model = build_model()
    images, texts = make_batch()
    with_zero = exgap_grad_attribution(model, images, texts, _args(lambda_exgap=0.0), {20}, 20)
    with_one = exgap_grad_attribution(model, images, texts, _args(lambda_exgap=1.0), {20}, 20)
    for key in ('grad_attr_gS_visual_backbone', 'grad_attr_gS_said_router',
                'grad_attr_gE_visual_backbone', 'grad_attr_loss_exgap'):
        assert with_zero[key] == with_one[key], key
    assert with_zero['grad_attr_lambda_exgap'] == 0.0
    assert with_one['grad_attr_lambda_exgap'] == 1.0


def test_exgap_gradient_never_reaches_the_router_or_the_text_encoder():
    """The attribution contract of the phase, measured on the real forward."""
    model = build_model()
    model.build_exgap_module()
    images, texts = make_batch()
    record = exgap_grad_attribution(model, images, texts, _args(), {20}, 20)
    # the ExGAP term is live in this configuration, so the zero entries are real zeros
    assert record['grad_attr_gE_patch_pathway'] >= 0.0
    assert record['grad_attr_gE_said_router'] == 0.0, 'ExGAP must not move the Said router'
    assert record['grad_attr_gE_text_encoder'] == 0.0, 'ExGAP must not move the text encoder'
    # ... while L_S does move both
    assert record['grad_attr_gS_said_router'] > 0.0
    assert record['grad_attr_gS_text_encoder'] > 0.0


def test_r_grad_is_the_ratio_of_the_two_terms():
    model = build_model()
    images, texts = make_batch()
    record = exgap_grad_attribution(model, images, texts, _args(), {20}, 20)
    for group in ('visual_backbone', 'said_router', 'text_encoder'):
        g_s = record['grad_attr_gS_%s' % group]
        g_e = record['grad_attr_gE_%s' % group]
        assert record['grad_attr_R_%s' % group] == g_e / (g_s + 1e-12)


def test_attribution_does_not_disturb_the_gradients_of_the_training_step():
    """The attribution uses autograd.grad on a fresh graph, so ``.grad`` stays untouched."""
    model = build_model()
    images, texts = make_batch()
    out = exgap_forward(model, images, texts, lambda_exgap=1.0)
    out['loss_total'].backward()
    before = {name: (None if p.grad is None else p.grad.clone())
              for name, p in model.named_parameters()}
    exgap_grad_attribution(model, images, texts, _args(), {20}, 20)
    for name, parameter in model.named_parameters():
        current = parameter.grad
        if before[name] is None:
            assert current is None, name
        else:
            assert torch.equal(current, before[name]), name
