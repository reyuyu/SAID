"""Phase 2.8B tests: Unsaid mechanism & reachability tooling.

Synthetic coverage: span reachability (A/B), best-patch cosine (C), the approximate
softmax-mixture oracle (D), determinism (E), no model mutation (F) and invalid /
degenerate residuals (G).
"""
import math
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, 'train')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval.unsaid_mechanism_probe import (  # noqa: E402
    best_patch_cosine,
    distribution,
    parse_checkpoints,
    probe_variant,
    softmax_mixture_oracle,
    span_reachability,
)
from eval.validation_protocol import VARIANTS  # noqa: E402
from model.salu_modules import SaidRouter  # noqa: E402

DIM = 8
PATCHES = 4


def _unit(vector):
    return F.normalize(vector, dim=-1)


# --------------------------------------------------------------------------- #
# A. span fully reachable
# --------------------------------------------------------------------------- #
def test_target_inside_patch_span_has_full_reachability():
    torch.manual_seed(0)
    patches = torch.randn(3, 6, 16)
    target = torch.stack([
        _unit(0.7 * patches[i, 1] + 0.3 * patches[i, 4]) for i in range(3)
    ])
    c_span, rank = span_reachability(patches, target)
    assert c_span.min().item() > 0.999
    assert (rank == 6).all()          # 6 independent rows in a 16-dim space


# --------------------------------------------------------------------------- #
# B. span unreachable
# --------------------------------------------------------------------------- #
def test_target_orthogonal_to_patch_span_has_zero_reachability():
    torch.manual_seed(1)
    patches = torch.randn(3, 5, 16)
    patches[:, :, 8:] = 0.0                       # row space lives in the first 8 dims
    target = torch.zeros(3, 16)
    target[:, 8:] = torch.tensor([[1.0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 1.0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 1.0, 0, 0, 0, 0, 0]])
    c_span, rank = span_reachability(patches, target)
    assert c_span.max().item() < 1e-6
    assert (rank == 5).all()


def test_zero_patch_matrix_is_handled():
    patches = torch.zeros(2, 4, 8)
    target = _unit(torch.randn(2, 8))
    c_span, rank = span_reachability(patches, target)
    assert torch.isfinite(c_span).all()
    assert c_span.abs().max().item() == 0.0
    assert (rank == 0).all()


# --------------------------------------------------------------------------- #
# C. best single patch
# --------------------------------------------------------------------------- #
def test_best_patch_cosine_finds_an_aligned_patch():
    torch.manual_seed(2)
    patches = torch.randn(2, 5, 8)
    target = torch.randn(2, 8)
    patches[0, 3] = target[0]
    patches[1, 1] = target[1] * 3.0                # scale invariant
    c_patch = best_patch_cosine(patches, target)
    assert c_patch[0].item() == pytest.approx(1.0, abs=1e-6)
    assert c_patch[1].item() == pytest.approx(1.0, abs=1e-6)
    # a target that no patch matches stays below 1
    other = best_patch_cosine(patches, _unit(torch.randn(2, 8)))
    assert (other <= 1.0 + 1e-6).all() and other.max().item() < 1.0


# --------------------------------------------------------------------------- #
# D. approximate softmax-mixture oracle
# --------------------------------------------------------------------------- #
def test_softmax_oracle_beats_the_uniform_mixture_on_a_positive_target():
    patches = torch.zeros(1, 4, 8)
    patches[0, 0, 0] = 1.0
    patches[0, 1, 1] = 1.0
    patches[0, 2, 2] = 1.0
    patches[0, 3, 3] = 1.0
    target = _unit(torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))

    result = softmax_mixture_oracle(patches, target, init_logits=torch.zeros(1, 4),
                                    steps=100, lr=0.1)
    # uniform mixture: normalize(0.25, 0.25, 0.25, 0.25) -> cos = 1/sqrt(2)
    assert result['cos_init'].item() == pytest.approx(1.0 / math.sqrt(2.0), abs=1e-6)
    assert result['cos_best'].item() > 0.99
    assert result['cos_best'].item() > result['cos_init'].item() + 0.2
    assert torch.isfinite(result['cos_best']).all()
    # the oracle may only use positive weights summing to one
    assert result['steps'] == 100 and result['lr'] == 0.1


def test_softmax_oracle_cannot_exceed_span_reachability():
    torch.manual_seed(3)
    patches = torch.randn(2, 5, 8)
    patches[:, :, 4:] = 0.0
    target = torch.zeros(2, 8)
    target[:, 4:] = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    c_span, _ = span_reachability(patches, target)
    result = softmax_mixture_oracle(patches, target, steps=50, lr=0.1)
    assert c_span.max().item() < 1e-6
    assert result['cos_best'].abs().max().item() < 1e-3     # positive mixture cannot escape either


# --------------------------------------------------------------------------- #
# E. determinism
# --------------------------------------------------------------------------- #
def test_closed_form_metrics_and_oracle_are_deterministic():
    torch.manual_seed(4)
    patches = torch.randn(3, 6, 8)
    target = _unit(torch.randn(3, 8))

    first_span, first_rank = span_reachability(patches, target)
    second_span, second_rank = span_reachability(patches, target)
    assert torch.equal(first_span, second_span) and torch.equal(first_rank, second_rank)
    assert torch.equal(best_patch_cosine(patches, target), best_patch_cosine(patches, target))

    first = softmax_mixture_oracle(patches, target, steps=40, lr=0.1)
    second = softmax_mixture_oracle(patches, target, steps=40, lr=0.1)
    # rule: the Adam-based oracle repeats within 1e-6 (closed-form metrics repeat exactly)
    assert torch.allclose(first['cos_best'], second['cos_best'], atol=1e-6)
    assert torch.allclose(first['cos_final'], second['cos_final'], atol=1e-6)


# --------------------------------------------------------------------------- #
# helpers: distribution / checkpoint spec
# --------------------------------------------------------------------------- #
def test_distribution_reports_mean_median_p10_p90():
    stats = distribution([1.0, 2.0, 3.0, 4.0, 5.0])
    assert stats['n'] == 5
    assert stats['mean'] == pytest.approx(3.0)
    assert stats['median'] == pytest.approx(3.0)
    assert stats['p10'] == pytest.approx(1.4)
    assert stats['p90'] == pytest.approx(4.6)
    empty = distribution([])
    assert empty == {'n': 0, 'mean': None, 'median': None, 'p10': None, 'p90': None}
    assert distribution([float('nan')])['n'] == 0


def test_parse_checkpoints_validates_the_spec():
    assert parse_checkpoints('initial:a.pt, step50:b.pt ,') == [('initial', 'a.pt'), ('step50', 'b.pt')]
    with pytest.raises(ValueError):
        parse_checkpoints('no-colon')
    with pytest.raises(ValueError):
        parse_checkpoints(' , ')


# --------------------------------------------------------------------------- #
# model-level probe: cohort fixture + stub model
# --------------------------------------------------------------------------- #
class _ProbeStub(torch.nn.Module):
    """Minimal stand-in: global feature, patch features, Said router."""

    def __init__(self, dim=DIM, patches=PATCHES, degenerate=False):
        super().__init__()
        self.dim = dim
        self.n_patches = patches
        self.degenerate = degenerate
        self.proj = torch.nn.Linear(dim, dim)
        self.offset = torch.nn.Parameter(torch.randn(patches, dim) * 0.1)
        self.said_router = SaidRouter(dim=dim, tau_said=0.07)

    def _flat(self, images):
        return images.flatten(1)[:, :self.dim]

    def encode_router_input(self, images):
        base = self.proj(self._flat(images))
        if self.degenerate:
            repeated = base.unsqueeze(1).repeat(1, self.n_patches, 1)
            return repeated[:, 0], repeated
        return base, base.unsqueeze(1) + self.offset.unsqueeze(0)

    def encode_text(self, tokens):
        pooled = tokens.float().mean(dim=1, keepdim=True).repeat(1, self.dim)
        return pooled


@pytest.fixture()
def cohort(tmp_path):
    root = tmp_path / 'images'
    root.mkdir()
    samples = []
    for index in range(4):
        rel = 'coco/train2017/%012d.jpg' % index
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((8, 8, 3), 30 + index * 40, dtype=np.uint8)).save(path)
        samples.append({'json_index': index, 'image_path': rel,
                        'captions': {name: 'caption %d' % index for name in VARIANTS}})
    return samples, root


def run_probe(model, samples, root, **kwargs):
    kwargs.setdefault('batch_size', 2)
    kwargs.setdefault('oracle_samples', 2)
    kwargs.setdefault('oracle_steps', 20)
    kwargs.setdefault('oracle_lr', 0.1)
    return probe_variant(model, samples, root, 'first_sentence',
                         lambda image: torch.ones(3, 8, 8), device='cpu', **kwargs)


def _all_numbers(record):
    """Every float the record reports (distributions collapse to their stats)."""
    numbers = []

    def walk(node):
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, float):
            numbers.append(node)

    walk(record['metrics'])
    walk(record['oracle'])
    return numbers


# --------------------------------------------------------------------------- #
# F. no model mutation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('start_training', [True, False])
def test_probe_does_not_mutate_the_model(cohort, start_training):
    samples, root = cohort
    model = _ProbeStub()
    if start_training:
        model.train()
    else:
        model.eval()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    record = run_probe(model, samples, root)

    assert model.training is start_training                 # mode restored
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name]), name      # weights and buffers unchanged
    assert all(parameter.grad is None for parameter in model.parameters())
    assert record['n_valid'] >= 1
    assert all(math.isfinite(value) for value in _all_numbers(record))


def test_probe_is_deterministic_for_the_same_checkpoint(cohort):
    samples, root = cohort
    model = _ProbeStub()
    first = run_probe(model, samples, root)
    second = run_probe(model, samples, root)

    assert first['samples_detail']['c_current'] == second['samples_detail']['c_current']
    assert first['samples_detail']['c_span'] == second['samples_detail']['c_span']
    assert first['samples_detail']['span_rank'] == second['samples_detail']['span_rank']
    assert first['metrics']['unsaid_target_cosine'] == second['metrics']['unsaid_target_cosine']
    oracle_first = first['metrics']['reachability']['c_softmax_oracle']['mean']
    oracle_second = second['metrics']['reachability']['c_softmax_oracle']['mean']
    assert oracle_first is not None and abs(oracle_first - oracle_second) < 1e-6


def test_probe_reports_reachability_layers_and_gaps(cohort):
    samples, root = cohort
    record = run_probe(_ProbeStub(), samples, root)
    reach = record['metrics']['reachability']
    for key in ('c_current', 'c_patch_max', 'c_span', 'c_softmax_oracle',
                'c_softmax_oracle_init', 'patch_span_rank'):
        assert key in reach, key
    assert reach['optimization_gap'] is not None
    assert reach['positive_mixture_gap'] is not None
    # c_span is a (loose) bound of any positive mixture, so the oracle stays below it
    assert reach['c_softmax_oracle']['mean'] <= reach['c_span']['mean'] + 1e-6
    assert record['oracle']['samples'] == 2 and record['oracle']['steps'] == 20
    assert set(record['samples_detail']) >= {'valid', 'c_current', 'c_patch_max', 'c_span',
                                             'residual_norm', 'said_unsaid_cosine', 'span_rank'}
    assert all(0.0 <= value <= 1.0 + 1e-6 for value in record['samples_detail']['c_span'])


# --------------------------------------------------------------------------- #
# G. invalid / degenerate residual
# --------------------------------------------------------------------------- #
def test_degenerate_residual_yields_no_valid_samples_and_no_nan(cohort):
    samples, root = cohort
    record = run_probe(_ProbeStub(degenerate=True), samples, root)

    assert record['n_valid'] == 0
    assert record['metrics']['unsaid_valid_ratio'] == 0.0
    assert record['metrics']['loss_unsaid_eval'] == 0.0
    for key in ('unsaid_target_cosine', 'said_unsaid_cosine', 'said_target_cosine',
                'unsaid_residual_norm_mean'):
        assert record['metrics'][key]['n'] == 0, key
        assert record['metrics'][key]['mean'] is None, key
    reach = record['metrics']['reachability']
    assert reach['c_current']['n'] == 0 and reach['c_span']['n'] == 0
    assert reach['optimization_gap'] is None and reach['positive_mixture_gap'] is None
    # the oracle is simply not run (no crash, no fabricated number)
    assert record['oracle']['samples'] == 0
    assert record['oracle']['cos_best'] == {'n': 0, 'mean': None, 'median': None,
                                            'p10': None, 'p90': None}
    assert all(math.isfinite(value) for value in _all_numbers(record))
    # attention diagnostics still exist (they do not depend on the target)
    assert record['metrics']['said_attention_entropy']['n'] == len(samples)
