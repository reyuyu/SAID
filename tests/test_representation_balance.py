"""Analytical and integration checks for diagnostic-only representation balance."""
import numpy as np
import pytest
import torch

from model.representation_metrics import batch_representation_gaps


def test_detached_batch_gaps_and_gain():
    full = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
    text = full.flip(0)
    result = batch_representation_gaps(full, text, text)
    assert result['pair_gap_full'].item() == pytest.approx(1.)
    assert result['pair_gap_said'].item() == pytest.approx(0.)
    assert result['balancing_gain'].item() == pytest.approx(1.)
    assert result['relative_balancing_gain'].item() == pytest.approx(1.)
    assert all(not x.requires_grad and x.grad_fn is None for x in result.values())
    assert all(torch.isfinite(x) for x in batch_representation_gaps(full, full, full).values())
