"""Phase 2.7b tests: standard retrieval validation hooks (validation monitor)."""
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

from eval.retrieval.coco_retrieval import retrieval_metrics  # noqa: E402
from eval.retrieval.sharegpt4v_retrieval import evaluate_sharegpt4v  # noqa: E402
from train_salu import append_val_record, parse_args, run_standard_validation  # noqa: E402

METRIC_KEYS = {'%s_R%d' % (d, k) for d in ('image2text', 'text2image') for k in (1, 5, 10)}


def test_retrieval_metrics_perfect_alignment_five_captions():
    n, c, d = 8, 5, 8  # d >= n keeps every image feature unique
    images = torch.zeros(n, d)
    texts = torch.zeros(n * c, d)
    for i in range(n):
        images[i, i] = 1.0
        for j in range(c):
            texts[i * c + j, i] = 1.0
    metrics = retrieval_metrics(images, texts, captions_per_image=5)
    assert set(metrics) == METRIC_KEYS
    for key, value in metrics.items():
        assert value == pytest.approx(1.0), key


def test_retrieval_metrics_single_caption_is_deterministic():
    torch.manual_seed(0)
    images = torch.randn(16, 8)
    texts = torch.randn(16, 8)
    first = retrieval_metrics(images, texts, captions_per_image=1)
    second = retrieval_metrics(images, texts, captions_per_image=1)
    assert first == second
    assert set(first) == METRIC_KEYS
    assert all(0.0 <= v <= 1.0 for v in first.values())


def test_retrieval_metrics_scale_invariance():
    torch.manual_seed(1)
    images = torch.randn(12, 8) * 7.5
    texts = torch.randn(12, 8) * 0.25
    plain = retrieval_metrics(images, texts, captions_per_image=1)
    normalized = retrieval_metrics(
        torch.nn.functional.normalize(images, dim=-1),
        torch.nn.functional.normalize(texts, dim=-1),
        captions_per_image=1,
    )
    assert plain == normalized


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
    assert all(0.0 <= v <= 1.0 for v in first.values())
    second = evaluate_sharegpt4v(model, dataset, batch_size=4, device='cpu')
    assert first == second


def test_evaluate_sharegpt4v_rejects_empty_dataset():
    with pytest.raises(ValueError):
        evaluate_sharegpt4v(_RecordingModel(), _DummyDataset(size=0), device='cpu')


def test_run_standard_validation_record_shape():
    class _Args:
        val_sharegpt4v = True
        val_coco = False
        val_batch_size = 2

    record = run_standard_validation(_Args(), _RecordingModel(), None, _DummyDataset(4), 'cpu', 7, 4)
    assert record['step'] == 7
    assert record['world_size'] == 4
    assert record['kind'] == 'standard_retrieval'
    assert set(record['sharegpt4v']) == METRIC_KEYS
    assert 'coco' not in record


def test_append_val_record_writes_jsonl(tmp_path):
    path = str(tmp_path / 'val_metrics.jsonl')
    append_val_record(path, {'step': 1, 'coco': {'image2text_R1': 0.5}})
    append_val_record(path, {'step': 2, 'coco': {'image2text_R1': 0.6}})
    with open(path) as fp:
        rows = [json.loads(line) for line in fp if line.strip()]
    assert [row['step'] for row in rows] == [1, 2]
    assert rows[1]['coco']['image2text_R1'] == 0.6


def test_validation_flags_are_exposed():
    args = parse_args(['--val_every', '25', '--val_sharegpt4v', '--val_coco', '--val_batch_size', '8'])
    assert args.val_every == 25
    assert args.val_sharegpt4v is True
    assert args.val_coco is True
    assert args.val_batch_size == 8
    defaults = parse_args([])
    assert defaults.val_every == 0
    assert defaults.val_sharegpt4v is False and defaults.val_coco is False
    assert defaults.val_batch_size == 64
