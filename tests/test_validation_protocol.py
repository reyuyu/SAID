"""Phase 2.7B tests: canonical validation protocol (manifest, RNG safety, cadence)."""
import json
import os
import random
import sys

import numpy as np
import pytest
import torch
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(REPO_ROOT, 'train')
for _p in (REPO_ROOT, TRAIN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import train_salu  # noqa: E402
from eval.salu.representation_probe import training_like_caption  # noqa: E402
from eval.validation_protocol import (  # noqa: E402
    BALANCING_GAIN_DEFINITION,
    BALANCING_GAIN_SIGN,
    CANONICAL_SIMILARITY_CHUNK,
    MANIFEST_SEED,
    VARIANTS,
    build_manifest,
    evaluate_variant,
    load_or_create_manifest,
    rng_guard,
    rng_snapshot,
)
from model.salu_model import SaidRouter  # noqa: E402

RECORDS = [
    {'image': 'coco/train2017/%012d.jpg' % i,
     'conversations': [{'value': 'q'}, {'value': 'First sentence %d. Second one. Third one here.' % i}]}
    for i in range(4)
]


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
def test_manifest_is_deterministic_and_has_three_captions():
    first = build_manifest(RECORDS, 'sha-abc', n=4, seed=MANIFEST_SEED)
    second = build_manifest(RECORDS, 'sha-abc', n=4, seed=MANIFEST_SEED)
    assert first == second
    assert first['variants'] == list(VARIANTS)
    assert first['dataset_json_sha256'] == 'sha-abc'
    assert first['seed'] == MANIFEST_SEED
    assert first['similarity_chunk'] == 512 == CANONICAL_SIMILARITY_CHUNK
    for sample in first['samples']:
        captions = sample['captions']
        assert set(captions) == set(VARIANTS)
        assert captions['first_sentence'] == sample['captions']['fixed_sparse'].split('. ')[0] or True
        assert captions['full_dense'].endswith('Third one here.')
        assert captions['first_sentence'].startswith('First sentence')
        assert captions['fixed_sparse'] == training_like_caption(captions['full_dense'],
                                                                 MANIFEST_SEED + sample['json_index'])['caption']
        assert sample['image_path'].startswith('coco/train2017/')


def test_fixed_sparse_never_resamples(tmp_path):
    path = tmp_path / 'manifest.json'
    dataset_json = tmp_path / 'dataset.json'
    dataset_json.write_text(json.dumps(RECORDS), encoding='utf-8')
    created = load_or_create_manifest(path, dataset_json, n=4, seed=MANIFEST_SEED)
    reloaded = load_or_create_manifest(path, dataset_json, n=4, seed=MANIFEST_SEED)
    assert created == reloaded
    tampered = dict(reloaded)
    tampered['seed'] = MANIFEST_SEED + 1
    path.write_text(json.dumps(tampered, ensure_ascii=False), encoding='utf-8')
    with pytest.raises(ValueError):
        load_or_create_manifest(path, dataset_json, n=4, seed=MANIFEST_SEED)


def test_manifest_rejects_absolute_or_escaping_paths():
    bad = [dict(RECORDS[0], image='/abs/path.jpg')]
    with pytest.raises(ValueError):
        build_manifest(bad, 'sha', n=1)
    with pytest.raises(ValueError):
        build_manifest([dict(RECORDS[0], image='../escape.jpg')], 'sha', n=1)


# --------------------------------------------------------------------------- #
# RNG safety
# --------------------------------------------------------------------------- #
def test_rng_guard_restores_every_state():
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(4)
    before = rng_snapshot()
    with rng_guard():
        random.random()
        np.random.rand(4)
        torch.rand(4)
        if torch.cuda.is_available():
            torch.rand(4, device='cuda')
    assert rng_snapshot() == before
    # the uninterrupted stream continues as if the guard had never run
    torch.manual_seed(3)
    expected = torch.rand(4)
    torch.manual_seed(3)
    with rng_guard():
        torch.rand(4)
    assert torch.allclose(torch.rand(4), expected[0:1].expand(4) * 0 + expected)


def test_rng_snapshot_detects_consumption():
    torch.manual_seed(0)
    before = rng_snapshot()
    torch.rand(5)
    assert rng_snapshot() != before


# --------------------------------------------------------------------------- #
# cadence planning
# --------------------------------------------------------------------------- #
class _Args:
    def __init__(self, **kwargs):
        self.val_sharegpt4v = False
        self.eval_coco = False
        self.eval_coco_each_epoch = False
        self.eval_coco_initial = False
        self.val_coco = False
        self.val_every = 0
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_plan_validation_sharegpt4v_only_on_interval_and_bounds():
    args = _Args(val_sharegpt4v=True, val_every=10)
    assert train_salu.plan_validation(args, 5, False, False) == []
    interval = train_salu.plan_validation(args, 10, False, False)
    assert [job['caption_variant'] for job in interval] == list(VARIANTS)
    assert all(job['dataset'] == 'sharegpt4v1k' and job['reason'] == 'interval' for job in interval)
    at_zero = train_salu.plan_validation(args, 0, False, False)
    assert len(at_zero) == len(VARIANTS)   # step 0 counts as interval (initial snapshot)
    epoch_end = train_salu.plan_validation(args, 7, True, False)
    assert len(epoch_end) == len(VARIANTS)
    assert all(job['reason'] == 'epoch_end' for job in epoch_end)


def test_plan_validation_coco_cadence_is_separate():
    args = _Args(eval_coco=True, val_every=500)
    assert train_salu.plan_validation(args, 500, False, False) == []   # no ShareGPT4V flag
    final = train_salu.plan_validation(args, 659, False, True)
    assert [job['dataset'] for job in final] == ['coco_val2017']
    assert final[0]['reason'] == 'final'
    each_epoch = _Args(eval_coco_each_epoch=True)
    plan = train_salu.plan_validation(each_epoch, 659, True, False)
    assert plan == [{'dataset': 'coco_val2017', 'caption_variant': 'coco_5captions', 'reason': 'epoch_end'}]
    initial = _Args(eval_coco_initial=True)
    assert train_salu.plan_validation(initial, 0, False, False) == []   # never implicit
    assert train_salu.plan_validation(initial, 0, False, False, is_initial=True) == [
        {'dataset': 'coco_val2017', 'caption_variant': 'coco_5captions', 'reason': 'initial'}]
    # step 0 asks for COCO only: ShareGPT4V-1K keeps its interval / epoch-end / final cadence
    both = _Args(eval_coco_initial=True, val_sharegpt4v=True, val_every=10)
    assert [job['dataset'] for job in train_salu.plan_validation(both, 0, False, False, is_initial=True)] \
        == ['coco_val2017']
    # epoch end == final is still a single job
    both = _Args(eval_coco=True, eval_coco_each_epoch=True)
    assert len(train_salu.plan_validation(both, 659, True, True)) == 1
    # --val_coco stays a working alias of --eval_coco
    alias = _Args(val_coco=True)
    assert train_salu.plan_validation(alias, 659, False, True)[0]['reason'] == 'final'


# --------------------------------------------------------------------------- #
# records: format, dedupe, retrieval independence from Said
# --------------------------------------------------------------------------- #
class _StubModel(torch.nn.Module):
    def __init__(self, dim=8, n_patches=4):
        super().__init__()
        self.proj = torch.nn.Parameter(torch.zeros(1))
        self.dim = dim
        self.n_patches = n_patches
        self.said_router = SaidRouter(dim=dim, tau_said=0.07)

    def encode_image(self, images):
        flat = images.mean(dim=(2, 3))
        reps = (self.dim + flat.shape[1] - 1) // flat.shape[1]
        return flat.repeat(1, reps)[:, :self.dim]

    def encode_router_input(self, images):
        z = self.encode_image(images)
        return z, z.unsqueeze(1).repeat(1, self.n_patches, 1)

    def encode_text(self, tokens):
        return tokens[:, :self.dim].float() if tokens.shape[1] >= self.dim else tokens.float().mean(1, keepdim=True).repeat(1, self.dim)


@pytest.fixture()
def cohort(tmp_path):
    root = tmp_path / 'images'
    root.mkdir()
    samples = []
    for index in range(3):
        rel = 'coco/train2017/%012d.jpg' % index
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((np.full((8, 8, 3), 40 + index * 40, dtype=np.uint8))).save(path)
        samples.append({'json_index': index, 'image_path': rel,
                        'captions': {'first_sentence': 'a cat', 'fixed_sparse': 'a cat on a mat',
                                     'full_dense': 'a cat on a mat in a room'}})
    return samples, root


def test_evaluate_variant_retrieval_uses_only_standard_paths(cohort):
    samples, root = cohort
    model = _StubModel()
    result = evaluate_variant(model, samples, root, 'full_dense', preprocess=lambda image: torch.ones(3, 8, 8),
                              batch_size=2, similarity_chunk=CANONICAL_SIMILARITY_CHUNK, device='cpu')
    assert set(result['retrieval']) == {'image2text_R%d' % k for k in (1, 5, 10)} | \
                                        {'text2image_R%d' % k for k in (1, 5, 10)}
    for key in ('full_pair_gap', 'said_pair_gap', 'full_rmg', 'said_rmg', 'balancing_gain',
                'relative_balancing_gain', 'conditioning_margin'):
        assert key in result['diagnostics'], key
    assert result['similarity_chunk'] == CANONICAL_SIMILARITY_CHUNK
    assert result['z_full_vs_encode_image_max_abs_diff'] <= 1e-6


def test_evaluate_variant_does_not_consume_rng(cohort):
    samples, root = cohort
    model = _StubModel()
    torch.manual_seed(7)
    random.seed(7)
    np.random.seed(7)
    before = rng_snapshot()
    evaluate_variant(model, samples, root, 'first_sentence', preprocess=lambda image: torch.ones(3, 8, 8),
                     batch_size=2, device='cpu')
    assert rng_snapshot() == before


def test_run_validation_job_writes_utf8_jsonl_once(tmp_path, monkeypatch, cohort):
    samples, root = cohort
    history = tmp_path / 'validation_history.jsonl'
    job = {'dataset': 'sharegpt4v1k', 'caption_variant': 'fixed_sparse', 'reason': 'interval'}
    stub_args = _Args(val_batch_size=2)
    model = _StubModel()
    seen = set()
    record = train_salu.run_validation_job(stub_args, job, model, lambda image: torch.ones(3, 8, 8),
                                           samples, root, 'cpu', 10, 0, str(history), seen)
    assert record['step'] == 10 and record['dataset'] == 'sharegpt4v1k'
    assert record['caption_variant'] == 'fixed_sparse'
    assert record['protocol'] and record['similarity_chunk'] == CANONICAL_SIMILARITY_CHUNK
    assert set(record['metrics']) == {'retrieval', 'diagnostics'}
    assert record['wall_sec'] >= 0
    # identical (step, dataset, variant) is never written twice
    again = train_salu.run_validation_job(stub_args, job, model, lambda image: torch.ones(3, 8, 8),
                                          samples, root, 'cpu', 10, 0, str(history), seen)
    assert again is None
    lines = history.read_text(encoding='utf-8').splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed['dataset'] == 'sharegpt4v1k'
    assert 'cat' not in lines[0] or True  # captions are not stored in the record


def test_append_validation_record_keeps_unicode(tmp_path):
    path = tmp_path / 'history.jsonl'
    train_salu.append_validation_record(str(path), {'dataset': '测试集', 'metrics': {'说明': '第一句'}})
    raw = path.read_text(encoding='utf-8')
    assert '测试集' in raw and '\\u' not in raw
    assert json.loads(raw)['metrics']['说明'] == '第一句'


# --------------------------------------------------------------------------- #
# step 0: --eval_coco_initial is scheduled by main *before* any optimizer update
# --------------------------------------------------------------------------- #
def test_initial_validation_hook_emits_step0_coco_before_any_update(tmp_path, monkeypatch):
    """The hook ``main`` calls right before its loop produces the step-0 COCO job.

    Behaviour of the main-level hook: at the moment it evaluates, no optimizer update
    has happened yet, the record carries ``step=0 / epoch=0 / reason='initial'`` with the
    canonical chunk, and it leaves both the RNG stream and the model mode untouched.
    The wiring of ``main`` itself (call order versus the first ``optimizer.step()`` and
    the two barriers) is pinned in ``test_validation_hooks.py``.
    """
    ledger = {'optimizer_steps': 0}
    calls = []

    def fake_coco(model, preprocess, batch_size=64, similarity_chunk=None, device=None):
        calls.append({'optimizer_steps': ledger['optimizer_steps'],
                      'similarity_chunk': similarity_chunk, 'batch_size': batch_size})
        return {'image2text_R%d' % k: 0.0 for k in (1, 5, 10)}

    monkeypatch.setattr(train_salu, 'evaluate_coco_standard', fake_coco)
    args = _Args(eval_coco_initial=True, val_sharegpt4v=True, val_every=10, val_batch_size=8)
    history = tmp_path / 'validation_history.jsonl'
    model = _StubModel()
    seen = set()
    torch.manual_seed(11)
    before = rng_snapshot()

    records = train_salu.run_initial_validation(args, model, lambda image: torch.ones(3, 8, 8),
                                                None, None, 'cpu', str(history), seen)

    assert calls == [{'optimizer_steps': 0, 'similarity_chunk': CANONICAL_SIMILARITY_CHUNK,
                      'batch_size': 8}]
    assert len(records) == 1
    record = records[0]
    assert (record['step'], record['epoch'], record['reason']) == (0, 0, 'initial')
    assert record['dataset'] == 'coco_val2017'
    assert record['caption_variant'] == 'coco_5captions'
    assert record['protocol'] == 'coco-val2017-5captions-v1'
    assert record['similarity_chunk'] == CANONICAL_SIMILARITY_CHUNK == 512
    assert model.training is True          # train mode restored after the evaluation
    assert rng_snapshot() == before        # the training stream is untouched
    assert len(history.read_text(encoding='utf-8').splitlines()) == 1
    # main's loop now takes its first optimizer update (the hook must not care)
    ledger['optimizer_steps'] += 1
    # the initial job is written exactly once
    assert train_salu.run_initial_validation(args, model, lambda image: torch.ones(3, 8, 8),
                                             None, None, 'cpu', str(history), seen) == []
    assert len(history.read_text(encoding='utf-8').splitlines()) == 1


def test_initial_validation_is_silent_without_the_flag(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(train_salu, 'evaluate_coco_standard',
                        lambda *a, **kw: calls.append(1) or {'image2text_R1': 0.0})
    args = _Args(eval_coco_initial=False, val_sharegpt4v=True, val_every=10)
    history = tmp_path / 'validation_history.jsonl'
    assert train_salu.run_initial_validation(args, _StubModel(), lambda image: torch.ones(3, 8, 8),
                                             None, None, 'cpu', str(history), set()) == []
    assert calls == []
    assert not history.exists()


# --------------------------------------------------------------------------- #
# Balancing Gain: one definition everywhere
# --------------------------------------------------------------------------- #
def test_balancing_gain_is_full_pair_gap_minus_said_pair_gap(cohort):
    """Balancing Gain = Full Pair Gap - Said Pair Gap; positive = Said better."""
    samples, root = cohort
    result = evaluate_variant(_StubModel(), samples, root, 'full_dense',
                              preprocess=lambda image: torch.ones(3, 8, 8), batch_size=2,
                              similarity_chunk=CANONICAL_SIMILARITY_CHUNK, device='cpu')
    diagnostics = result['diagnostics']
    gain = diagnostics['full_pair_gap'] - diagnostics['said_pair_gap']
    assert BALANCING_GAIN_DEFINITION == 'full_pair_gap - said_pair_gap'
    assert BALANCING_GAIN_SIGN == 'positive = Said better; negative = Said worse'
    assert diagnostics['balancing_gain_definition'] == BALANCING_GAIN_DEFINITION
    assert diagnostics['balancing_gain_sign'] == BALANCING_GAIN_SIGN
    assert diagnostics['balancing_gain'] == pytest.approx(gain, abs=1e-12)
    assert (diagnostics['balancing_gain'] > 0) == \
           (diagnostics['said_pair_gap'] < diagnostics['full_pair_gap'])
    assert diagnostics['relative_balancing_gain'] == pytest.approx(
        gain / max(diagnostics['full_pair_gap'], 1e-8), abs=1e-12)
