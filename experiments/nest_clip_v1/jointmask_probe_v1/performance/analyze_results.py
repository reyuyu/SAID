"""Condense profiler-off JointMask benchmark artifacts into reviewable JSON."""
import json
from pathlib import Path


ROOT = Path('/root/lk_projects/SAID-nest-clip-v1/jointmask_perf_diagnosis')
HERE = Path(__file__).resolve().parent
RUNS = [
    'baseline-T-32x64-cp1-r2', 'optimized-T-32x64-cp1-r2',
    'optimized-T-64x128-cp1-r2', 'optimized-T-128x128-cp1-r2',
    'optimized-T-128x128-cp0', 'matrix-T-r2',
    'baseline-TI-32x64-cp1-r2', 'optimized-TI-32x64-cp1-r2',
    'optimized-TI-64x128-cp1-r2', 'optimized-TI-128x128-cp1-r2',
    'optimized-TI-128x128-cp0',
]


def load(name):
    with (ROOT / name / 'result.json').open() as handle:
        return json.load(handle)


def condensed(name, data):
    summary = data['max_rank_summary']
    metrics = {}
    for key in ('step_ms', 'data_h2d_ms', 'forward_ms', 'encoder_ms',
                'pair_forward_ms', 'forward_communication_ms',
                'backward_recompute_ms', 'optimizer_ms', 'logging_ms'):
        metrics[key] = {
            'median_ms': summary[key]['median_ms'],
            'p90_ms': summary[key]['p90_ms'],
        }
    return {
        'name': name,
        'args': data['args'],
        'timing': metrics,
        'peak_allocated_gib_by_rank': [rank['peak_allocated_gib'] for rank in data['ranks']],
        'peak_reserved_gib_by_rank': [rank['peak_reserved_gib'] for rank in data['ranks']],
        'block_calls_per_measured_step': sorted(set(data['block_calls_per_rank'])),
        'losses': data['losses'],
        'rank_identities': [rank['identity'] for rank in data['ranks']],
    }


def loss_delta(candidate, reference):
    assert len(candidate) == len(reference)
    diffs = [abs(a - b) for a, b in zip(candidate, reference)]
    return {'max_abs': max(diffs), 'values': diffs}


def main():
    raw = {name: load(name) for name in RUNS}
    runs = {name: condensed(name, value) for name, value in raw.items()}
    comparisons = {}
    for mode in ('T', 'TI'):
        baseline_name = f'baseline-{mode}-32x64-cp1-r2'
        baseline = runs[baseline_name]
        base_ms = baseline['timing']['step_ms']['median_ms']
        for name, run in runs.items():
            if run['args']['mode'] != mode or name == baseline_name:
                continue
            candidate_ms = run['timing']['step_ms']['median_ms']
            comparisons[name] = {
                'versus': baseline_name,
                'median_step_reduction_percent': 100 * (base_ms - candidate_ms) / base_ms,
                'throughput_multiplier': base_ms / candidate_ms,
                'loss_delta': loss_delta(run['losses'], baseline['losses']),
            }
    worker_probe = load('harness-check')
    output = {
        'baseline_commit': '82e06b2e5b0a76e51a924928a036be447e8fce02',
        'benchmark_root': str(ROOT),
        'protocol': {
            'world_size': 4,
            'per_rank_batch': 256,
            'global_candidates': 1024,
            'warmup_steps': 2,
            'measured_steps': 8,
            'profiler_enabled': False,
            'fixed_input_and_initial_state': True,
            'precision': 'existing mixed precision (BF16 encoders, FP32 remaining paths)',
            'encoder_checkpoint': True,
        },
        'runs': runs,
        'comparisons': comparisons,
        'trajectory_note': (
            'Loss deltas compare repeated-update trajectories and are not the numerical '
            'acceptance criterion across chunk shapes. Fixed-state loss, gradient, and '
            'one-step update tolerances are recorded in validation_summary.json.'),
        'production_loader_probe': worker_probe['data_wait'],
        'production_loader_probe_rank_waits_ms': [rank['data_wait_ms'] for rank in worker_probe['ranks']],
    }
    with (HERE / 'performance_results.json').open('w') as handle:
        json.dump(output, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
