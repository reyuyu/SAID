"""Phase 3.0A.1b tests: base gap completion training wiring.

Covers the training-side acceptance points of the phase brief:

A. ``objective_mode=gap_completion`` selects ``caption_views=False``.
B. the real batch is exactly ``(I, C_S)`` (2-tuple), with no ``caption_full`` /
   ``caption_unsaid`` / ``has_unsaid`` / ``caption_uss`` anywhere in it.
C. the gap path never uses ``tri_caption_collate``.
D. only ``C_S`` is tokenized (exactly one ``tokenize`` call per batch).
E. the logger turns ``loss_global=None`` / ``loss_unsaid=None`` into JSON ``null``.
F. the legacy dataloader / logger behaviour is unchanged.
G. ``encode_said_unsaid`` honours its ``gap_anti_temperature`` argument.

The pure math of ``model/gap_completion.py`` is covered by ``test_gap_completion.py``.
"""
import json
import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, 'train'), os.path.join(REPO_ROOT, 'tests')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from test_gap_completion import build_model, gap_forward, make_batch  # noqa: E402

import train_salu  # noqa: E402
from train_salu import build_gap_log_fields, build_gap_train_batch  # noqa: E402

FORBIDDEN_CAPTION_KEYS = ('caption_full', 'caption_unsaid', 'caption_uss', 'has_unsaid')


def parse(**overrides):
    """Parse the real CLI with gap-mode-safe defaults, then apply overrides."""
    argv = ['--lambda_global', '0', '--lambda_unsaid', '0']
    for key, value in overrides.items():
        argv.extend(['--' + key, str(value)])
    return train_salu.parse_args(argv)


# --------------------------------------------------------------------------- #
# A. objective_mode chooses the dataset shape
# --------------------------------------------------------------------------- #
def test_a_gap_mode_selects_caption_views_false():
    gap_mode, tri_mode, caption_views, collate = train_salu.resolve_gap_mode(
        parse(objective_mode='gap_completion'))
    assert gap_mode is True
    assert tri_mode is False                 # not the 2.9A suffix branch
    assert caption_views is False            # (I, C_S) tuple, no caption-view dict
    assert collate is None                   # never tri_caption_collate


def test_a_gap_mode_overrides_a_debiased_suffix_setting():
    """Even if --unsaid_mode asks for tri views, the gap run must not build C_F / C_U."""
    args = parse(objective_mode='gap_completion', unsaid_mode='debiased_suffix')
    gap_mode, tri_mode, caption_views, collate = train_salu.resolve_gap_mode(args)
    assert gap_mode is True and tri_mode is True
    assert caption_views is False and collate is None


def test_a_legacy_modes_keep_the_original_dataset_shapes():
    legacy = train_salu.resolve_gap_mode(parse(objective_mode='legacy'))
    assert legacy == (False, False, False, None)

    args = train_salu.parse_args(['--unsaid_mode', 'debiased_suffix'])
    tri_resolved = train_salu.resolve_gap_mode(args)
    assert tri_resolved[0] is False and tri_resolved[1] is True
    assert tri_resolved[2] is True
    assert tri_resolved[3] is train_salu.tri_caption_collate


# --------------------------------------------------------------------------- #
# B. the batch is exactly (I, C_S)
# --------------------------------------------------------------------------- #
def test_b_real_batch_is_an_image_and_prefix_caption_pair():
    torch.manual_seed(0)
    images = torch.randn(3, 3, 8, 8)
    captions = ['a cat on a mat', 'a dog in a park', 'a bird on a wire']
    seen = []

    def tokenize(texts):
        seen.append(list(texts))
        return torch.zeros(len(texts), 4)

    built = build_gap_train_batch((images, captions), tokenize)
    assert set(built) == {'images', 'texts_said'}
    assert built['images'] is images
    assert seen == [captions]                      # sequence and content preserved
    # there is no key, and no place, for C_F / C_U in this batch
    for key in FORBIDDEN_CAPTION_KEYS:
        assert key not in built


def test_b_caption_view_dicts_are_rejected():
    batch = {'image': torch.randn(2, 3, 8, 8),
             'caption_full': ['a', 'b'],
             'caption_said': ['a', 'b'],
             'caption_unsaid': ['c', 'd'],
             'has_unsaid': torch.tensor([True, False])}
    with pytest.raises(ValueError):
        build_gap_train_batch(batch, lambda texts: texts)


def test_b_non_pair_batches_are_rejected():
    images = torch.randn(2, 3, 8, 8)
    with pytest.raises(ValueError):
        build_gap_train_batch((images, ['a', 'b'], ['c', 'd']), lambda texts: texts)
    with pytest.raises(ValueError):
        build_gap_train_batch(images, lambda texts: texts)


# --------------------------------------------------------------------------- #
# C. tri_caption_collate is never used by the gap path
# --------------------------------------------------------------------------- #
def test_c_gap_mode_never_uses_tri_caption_collate():
    calls = []
    original = train_salu.tri_caption_collate

    def counting(samples):
        calls.append(1)
        return original(samples)

    train_salu.tri_caption_collate = counting
    try:
        args = parse(objective_mode='gap_completion', unsaid_mode='debiased_suffix')
        _, _, caption_views, collate = train_salu.resolve_gap_mode(args)
        assert collate is None
        built = build_gap_train_batch(
            (torch.randn(2, 3, 8, 8), ['a cat.', 'a dog.']), lambda texts: torch.zeros(2, 4))
    finally:
        train_salu.tri_caption_collate = original
    assert calls == []
    assert set(built) == {'images', 'texts_said'}


# --------------------------------------------------------------------------- #
# D. only C_S is tokenized
# --------------------------------------------------------------------------- #
def test_d_only_the_prefix_caption_is_tokenized():
    calls = []

    def tokenize(texts, truncate=True):
        calls.append((list(texts), truncate))
        return torch.zeros(len(texts), 4)

    captions = ['first sentence. second sentence. third sentence.', 'another one.']
    build_gap_train_batch((torch.randn(2, 3, 8, 8), captions), tokenize)
    assert len(calls) == 1                       # exactly one tokenize call per batch
    assert calls[0][0] == captions               # and it is C_S
    assert calls[0][1] is True


# --------------------------------------------------------------------------- #
# E. logger: None -> JSON null
# --------------------------------------------------------------------------- #
def test_e_gap_log_fields_map_missing_terms_to_json_null():
    model = build_model()
    images, texts = make_batch()
    out = gap_forward(model, images, texts)
    fields = build_gap_log_fields(out)

    # loss_global / loss_unsaid do not exist in this objective: they must be None, and
    # nothing in the logger may call .detach() on them.
    assert out['loss_global'] is None and out['loss_unsaid'] is None
    assert fields['loss_global'] is None and fields['loss_unsaid'] is None

    for key in ('loss_said', 'loss_gap_discover', 'loss_global_absorb', 'loss_total',
                'loss_route', 'loss_evidence', 'route_top1_acc', 'evidence_top1_acc',
                'route_margin', 'evidence_margin', 'gap_before_mean', 'gap_after_mean',
                'gap_reduction_mean', 'gap_closure_ratio_mean',
                'gap_closure_positive_fraction', 'said_unsaid_feature_cosine',
                'unsaid_novel_component_norm', 'said_attention_entropy',
                'unsaid_attention_entropy', 'said_unsaid_attention_overlap',
                'said_unsaid_attention_jsd', 'gap_anti_temperature',
                'global_feature_norm', 'said_feature_norm', 'unsaid_feature_norm'):
        value = fields[key]
        assert value is not None, key
        assert isinstance(value, float), key
        assert value == value and abs(value) != float('inf'), key

    # objective identity is explicit, and the flags are not overloaded
    assert fields['objective_mode'] == 'gap_completion'
    assert fields['said_loss_mode'] == 'identifiable'
    assert fields['visual_complement_enabled'] is True
    assert fields['legacy_unsaid_enabled'] is False
    assert fields['global_text_alignment_enabled'] is False

    text = json.dumps(fields, sort_keys=True)
    assert '"loss_global": null' in text
    assert '"loss_unsaid": null' in text
    assert json.loads(text)['loss_global'] is None
    assert json.loads(text)['loss_unsaid'] is None


def test_e_log_record_is_writable_and_holds_no_non_finite_numbers():
    model = build_model()
    images, texts = make_batch()
    record = build_gap_log_fields(gap_forward(model, images, texts))
    record.update({'step': 0, 'epoch': 0})
    line = json.dumps(record, sort_keys=True)
    reloaded = json.loads(line)
    assert reloaded['loss_global'] is None and reloaded['loss_unsaid'] is None
    for key, value in reloaded.items():
        if isinstance(value, float):
            assert value == value and abs(value) != float('inf'), key


# --------------------------------------------------------------------------- #
# F. legacy behaviour is unchanged
# --------------------------------------------------------------------------- #
class _FakeTrainSet:
    """Stand-in for ``share4v_train_dataset(caption_views=False)``."""

    def __init__(self, batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


def test_f_legacy_batch_path_and_logger_are_unchanged():
    images = torch.randn(2, 3, 8, 8)
    captions = ['a cat.', 'a dog.']
    args = parse(objective_mode='legacy')

    # the legacy dataset request and collate are exactly what they were before Phase 3.0A
    gap_mode, tri_mode, caption_views, collate = train_salu.resolve_gap_mode(args)
    assert (gap_mode, tri_mode, caption_views, collate) == (False, False, False, None)

    # the legacy batch shape is still a plain 2-tuple and still unpacks the same way
    batch = (images, captions)
    batch_images, batch_texts = batch
    assert batch_images is images and batch_texts == captions

    # the legacy logger still reads real tensors straight out of the output
    model = build_model()
    _, tokens = make_batch(batch=2)              # the legacy path gets tokenized text
    out = model.forward_train(images, tokens, 1.0, 1.0, lambda_unsaid=0.0,
                              objective_mode='legacy')
    assert out['loss_global'] is not None and out['loss_unsaid'] is not None
    legacy_record = {
        'loss_global': float(out['loss_global'].detach()),
        'loss_said': float(out['loss_said'].detach()),
        'loss_unsaid': float(out['loss_unsaid'].detach()),
        'unsaid_enabled': bool(out['unsaid_enabled']),
        'loss_total': float(out['loss_total'].detach()),
    }
    assert all(value == value for value in legacy_record.values())
    assert legacy_record['unsaid_enabled'] is False
    reloaded = json.loads(json.dumps(legacy_record, sort_keys=True))
    assert reloaded['loss_global'] is not None
    assert reloaded['loss_unsaid'] == 0.0


def test_f_legacy_cli_defaults_are_unchanged():
    args = train_salu.parse_args([])
    assert args.objective_mode == 'legacy'
    assert args.lambda_global == 1.0
    assert args.lambda_unsaid == 0.0
    assert args.lambda_said == 1.0
    assert args.lambda_gap_discover == 1.0
    assert args.lambda_global_absorb == 1.0
    assert args.gap_anti_temperature == 1.0
    # no USS switch was introduced
    assert not hasattr(args, 'lambda_uss')
    assert not hasattr(args, 'unsaid_uss')


# --------------------------------------------------------------------------- #
# gap-mode configuration validation (fail early, never silently mixed)
# --------------------------------------------------------------------------- #
def test_gap_mode_rejects_a_global_or_unsaid_term():
    with pytest.raises(ValueError):
        train_salu.validate_objective_args(
            train_salu.parse_args(['--objective_mode', 'gap_completion',
                                   '--lambda_global', '1.0']))
    with pytest.raises(ValueError):
        train_salu.validate_objective_args(
            train_salu.parse_args(['--objective_mode', 'gap_completion',
                                   '--lambda_global', '0', '--lambda_unsaid', '0.5']))
    with pytest.raises(ValueError):
        train_salu.validate_objective_args(
            train_salu.parse_args(['--objective_mode', 'gap_completion',
                                   '--lambda_global', '0', '--lambda_unsaid', '0',
                                   '--global_caption_view', 'full']))


def test_gap_mode_accepts_zero_gap_weights_as_matched_controls():
    """A zero gap / absorb weight is a legal matched control, not a configuration error.

    (0, 0) = Said-only control, (1, 0) = discovery only, (1, 1) = the full base objective.
    """
    for gap_weight, absorb_weight in (('0', '0'), ('1', '0'), ('0', '1'), ('1', '1')):
        args = train_salu.validate_objective_args(train_salu.parse_args([
            '--objective_mode', 'gap_completion', '--lambda_global', '0', '--lambda_unsaid', '0',
            '--lambda_gap_discover', gap_weight, '--lambda_global_absorb', absorb_weight]))
        assert args.lambda_gap_discover == float(gap_weight)
        assert args.lambda_global_absorb == float(absorb_weight)


def test_gap_mode_rejects_negative_or_non_finite_gap_weights():
    for flag, value in (('lambda_gap_discover', '-1'), ('lambda_global_absorb', '-0.5'),
                        ('lambda_gap_discover', 'inf'), ('lambda_global_absorb', 'nan'),
                        ('gap_anti_temperature', '0'), ('gap_anti_temperature', '-1'),
                        ('lambda_said', '0'), ('lambda_said', '-1')):
        argv = ['--objective_mode', 'gap_completion', '--lambda_global', '0',
                '--lambda_unsaid', '0', '--' + flag, value]
        with pytest.raises(ValueError):
            train_salu.validate_objective_args(train_salu.parse_args(argv))


def test_said_only_control_still_computes_the_gap_diagnostics():
    """With (0, 0) the representations are still built, but L_total is exactly L_S."""
    model = build_model()
    images, texts = make_batch()
    out = gap_forward(model, images, texts, lambda_gap_discover=0.0, lambda_global_absorb=0.0)
    assert torch.allclose(out['loss_total'], out['loss_said'], atol=1e-6)
    for key in ('loss_gap_discover', 'loss_global_absorb', 'gap_before_mean', 'gap_after_mean',
                'patch_pair_cosine_mean', 'raw_pool_cosine', 'cos_said_unsaid'):
        assert out[key] is not None and torch.isfinite(out[key]).all(), key
    assert out['lambda_gap_discover'] == 0.0 and out['lambda_global_absorb'] == 0.0


def test_gap_mode_accepts_the_phase30a_configuration():
    args = train_salu.validate_objective_args(train_salu.parse_args([
        '--objective_mode', 'gap_completion', '--lambda_global', '0', '--lambda_unsaid', '0',
        '--lambda_said', '1', '--lambda_gap_discover', '1', '--lambda_global_absorb', '1',
        '--gap_anti_temperature', '1.0']))
    assert args.objective_mode == 'gap_completion'


def test_legacy_configuration_is_never_touched_by_the_gap_validation():
    args = train_salu.parse_args(['--lambda_global', '0.7', '--lambda_unsaid', '0.25'])
    assert train_salu.validate_objective_args(args) is args


# --------------------------------------------------------------------------- #
# G. encode_said_unsaid temperature argument
# --------------------------------------------------------------------------- #
def test_g_gap_anti_temperature_changes_only_the_anti_said_attention():
    torch.manual_seed(0)
    model = build_model()
    images, texts = make_batch()

    cold = model.encode_said_unsaid(images, texts, gap_anti_temperature=0.25,
                                    return_details=True)
    reference = model.encode_said_unsaid(images, texts, gap_anti_temperature=1.0,
                                         return_details=True)

    # deterministic for one temperature ...
    assert torch.equal(cold['unsaid_attention'],
                       model.encode_said_unsaid(images, texts, gap_anti_temperature=0.25,
                                                return_details=True)['unsaid_attention'])
    assert torch.equal(reference['unsaid_attention'],
                       model.encode_said_unsaid(images, texts, gap_anti_temperature=1.0,
                                                return_details=True)['unsaid_attention'])
    # ... and a genuinely different A_U (hence z_U) for another one
    assert not torch.allclose(cold['unsaid_attention'], reference['unsaid_attention'])
    assert not torch.allclose(cold['unsaid_feature'], reference['unsaid_feature'])
    # the Said side does not depend on the anti-Said temperature at all
    assert torch.equal(cold['said_attention'], reference['said_attention'])
    assert torch.equal(cold['said_feature'], reference['said_feature'])
    for key in ('unsaid_attention', 'unsaid_feature', 'gap_after', 'u_new'):
        assert torch.isfinite(cold[key]).all(), key

    # training and inference agree at the training default
    out = model.forward_train(images, texts, 0.0, 1.0, lambda_unsaid=0.0,
                              objective_mode='gap_completion', gap_anti_temperature=1.0)
    assert torch.allclose(torch.tensor(float(out['gap_after_mean'])),
                          reference['gap_after'].mean(), atol=1e-6)


def test_g_temperature_is_a_keyword_argument_not_a_text_input():
    import inspect

    parameters = inspect.signature(train_salu.SALUModel.encode_said_unsaid).parameters
    assert list(parameters) == ['self', 'images', 'said_texts', 'gap_anti_temperature',
                               'return_details']
    assert parameters['gap_anti_temperature'].default == 1.0
    for banned in ('unsaid_text', 'candidate', 'uss', 'hidden_text'):
        assert not any(banned in name for name in parameters), banned
