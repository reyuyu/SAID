"""Phase 2.1 test: throughput logging utility (compute vs wall time)."""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(REPO_ROOT, 'train')
for _p in (REPO_ROOT, TRAIN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train_salu import summarize_throughput  # noqa: E402


def test_summarize_throughput_compute_vs_wall():
    compute_times = [0.4, 0.6]
    wall_times = [0.8, 1.2]
    metrics = summarize_throughput(global_batch=1024, compute_times=compute_times, wall_times=wall_times)

    assert abs(metrics['compute_sec_per_step'] - 0.5) < 1e-9
    assert abs(metrics['wall_sec_per_step'] - 1.0) < 1e-9
    assert abs(metrics['compute_samples_per_sec'] - 2048.0) < 1e-6
    assert abs(metrics['wall_samples_per_sec'] - 1024.0) < 1e-6
    # wall time can never be smaller than compute time here
    assert metrics['wall_sec_per_step'] >= metrics['compute_sec_per_step']


def test_summarize_throughput_empty_is_zero():
    metrics = summarize_throughput(global_batch=1024, compute_times=[], wall_times=[])
    assert metrics['compute_sec_per_step'] == 0.0
    assert metrics['wall_sec_per_step'] == 0.0
    assert metrics['compute_samples_per_sec'] == 0.0
    assert metrics['wall_samples_per_sec'] == 0.0
