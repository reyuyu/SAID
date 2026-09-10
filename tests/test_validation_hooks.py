"""Phase 2.7b tests: validation hooks (cohort evaluator, cadence flags, records)."""
import inspect
import json
import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(REPO_ROOT, 'train')
for _p in (REPO_ROOT, TRAIN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval.retrieval.sharegpt4v_retrieval import evaluate_sharegpt4v  # noqa: E402
import train_salu  # noqa: E402
from train_salu import append_validation_record, parse_args, validation_key  # noqa: E402

METRIC_KEYS = {'%s_R%d' % (d, k) for d in ('image2text', 'text2image') for k in (1, 5, 10)}


class _DummyDataset:
    """Returns a constant image and a caption; only batch handling is exercised."""

    def __init__(self, size=5):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return torch.full((3, 8, 8), index * 0.1), 'caption %d' % index


class _RecordingModel(torch.nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.proj = torch.nn.Parameter(torch.zeros(dim))
        self.images_seen = 0

    def encode_image(self, images):
        self.images_seen += int(images.shape[0])
        return images.flatten(1)[:, :self.proj.numel()]

    def encode_text(self, tokens):
        return torch.zeros(tokens.shape[0], self.proj.numel(), device=tokens.device)


def test_evaluate_sharegpt4v_covers_dataset_and_is_deterministic():
    dataset = _DummyDataset(size=5)
    model = _RecordingModel()
    first = evaluate_sharegpt4v(model, dataset, batch_size=2, device='cpu')
    assert model.images_seen == len(dataset)
    assert set(first) == METRIC_KEYS
    assert all(0.0 <= value <= 1.0 for value in first.values())
    second = evaluate_sharegpt4v(model, dataset, batch_size=4, device='cpu')
    assert first == second


def test_evaluate_sharegpt4v_rejects_empty_dataset():
    with pytest.raises(ValueError):
        evaluate_sharegpt4v(_RecordingModel(), _DummyDataset(size=0), device='cpu')


def test_validation_key_identity():
    assert validation_key(10, 'sharegpt4v1k', 'fixed_sparse') == (10, 'sharegpt4v1k', 'fixed_sparse')
    assert validation_key(10, 'sharegpt4v1k', 'fixed_sparse') != validation_key(10, 'sharegpt4v1k', 'full_dense')
    assert validation_key(10, 'coco_val2017', 'coco_5captions') != validation_key(11, 'coco_val2017', 'coco_5captions')


def test_append_validation_record_writes_jsonl(tmp_path):
    path = str(tmp_path / 'validation_history.jsonl')
    append_validation_record(path, {'step': 1, 'metrics': {'image2text_R1': 0.5}})
    append_validation_record(path, {'step': 2, 'metrics': {'image2text_R1': 0.6}})
    with open(path, encoding='utf-8') as fp:
        rows = [json.loads(line) for line in fp if line.strip()]
    assert [row['step'] for row in rows] == [1, 2]
    assert rows[1]['metrics']['image2text_R1'] == 0.6


def test_validation_flags_are_exposed():
    args = parse_args(['--val_every', '500', '--val_sharegpt4v', '--eval_coco',
                       '--eval_coco_each_epoch', '--eval_coco_initial', '--val_batch_size', '8'])
    assert args.val_every == 500
    assert args.val_sharegpt4v is True
    assert args.eval_coco is True
    assert args.eval_coco_each_epoch is True
    assert args.eval_coco_initial is True
    assert args.val_batch_size == 8
    assert args.legacy_eval_coco is False
    assert args.validation_manifest.endswith(os.path.join('validation', 'sharegpt4v1k_manifest.json'))
    defaults = parse_args([])
    assert defaults.val_every == 0
    assert defaults.val_sharegpt4v is False and defaults.eval_coco is False
    assert defaults.eval_coco_each_epoch is False and defaults.eval_coco_initial is False
    assert defaults.val_batch_size == 64


def test_main_schedules_initial_validation_before_the_first_optimizer_step():
    """``main()`` itself must produce the step-0 COCO job before any optimizer update.

    This is a wiring check on ``main``'s own source, not on ``plan_validation``: the
    ``--eval_coco_initial`` block has to sit before the training loop, call
    ``run_initial_validation``, and be enclosed by the two ``dist.barrier()`` calls
    that keep every rank in lockstep around the rank-0-only evaluation.
    """
    source = inspect.getsource(train_salu.main)
    loop_at = source.index('for epoch in range(start_epoch, args.epochs):')
    block_at = source.index('if args.eval_coco_initial and start_step == 0:')
    block = source[block_at:loop_at]

    assert 'run_initial_validation(' in block
    assert block.count('dist.barrier()') == 2      # every rank waits around the rank-0 job
    assert 'if rank == 0:' in block                # only rank 0 evaluates and writes
    # the block sits before the training loop, therefore before every optimizer update
    assert block_at < loop_at
    assert block_at < source.index('loss.backward(')
    assert block_at < source.index('optimizer.step()')
    # a resumed run has already passed step 0, so the initial job is skipped there
    assert 'start_step == 0' in block.splitlines()[0]
    assert callable(train_salu.run_initial_validation)
