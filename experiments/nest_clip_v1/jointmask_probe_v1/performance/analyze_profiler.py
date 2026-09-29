"""Summarize PyTorch profiler traces without retaining them in Git."""
import json
from pathlib import Path
from statistics import median


ROOT = Path('/root/lk_projects/SAID-nest-clip-v1/jointmask_perf_diagnosis')
HERE = Path(__file__).resolve().parent
PROFILES = ('profile-baseline-T', 'profile-baseline-TI',
            'profile-final-T', 'profile-final-TI')
KEYS = ('aten::nonzero', 'aten::index', 'aten::index_select',
        'cudaStreamSynchronize', 'cudaDeviceSynchronize')


def merged_idle(intervals):
    if not intervals:
        return {}
    intervals.sort()
    merged = []
    start, stop = intervals[0]
    for next_start, next_stop in intervals[1:]:
        if next_start <= stop:
            stop = max(stop, next_stop)
        else:
            merged.append((start, stop))
            start, stop = next_start, next_stop
    merged.append((start, stop))
    span = merged[-1][1] - merged[0][0]
    busy = sum(stop - start for start, stop in merged)
    gaps = [merged[index + 1][0] - merged[index][1]
            for index in range(len(merged) - 1)]
    return {
        'profile_gpu_span_ms': span / 1000,
        'kernel_busy_ms': busy / 1000,
        'idle_ms': (span - busy) / 1000,
        'idle_fraction': (span - busy) / span if span else 0,
        'gaps_over_10us': sum(gap > 10 for gap in gaps),
        'gap_ms_over_10us': sum(gap for gap in gaps if gap > 10) / 1000,
        'gaps_over_50us': sum(gap > 50 for gap in gaps),
        'gap_ms_over_50us': sum(gap for gap in gaps if gap > 50) / 1000,
        'largest_gap_ms': max(gaps, default=0) / 1000,
    }


def summarize_rank(directory, rank):
    with (directory / f'profiler-rank{rank}.json').open() as handle:
        averages = {row['key']: row for row in json.load(handle)}
    with (directory / f'trace-rank{rank}.json').open() as handle:
        trace = json.load(handle)['traceEvents']
    kernels = [event for event in trace
               if event.get('cat') == 'kernel' and event.get('ph') == 'X']
    durations = [float(event.get('dur', 0)) for event in kernels]
    intervals = [(float(event['ts']), float(event['ts']) + float(event.get('dur', 0)))
                 for event in kernels]
    nccl = [event for event in kernels if event.get('name', '').startswith('nccl')]
    return {
        'rank': rank,
        'trace_bytes': (directory / f'trace-rank{rank}.json').stat().st_size,
        'operator_counts': {key: int(averages.get(key, {}).get('count', 0)) for key in KEYS},
        'kernel_count': len(kernels),
        'kernels_under_10us': sum(value < 10 for value in durations),
        'kernels_under_25us': sum(value < 25 for value in durations),
        'kernel_duration_ms': sum(durations) / 1000,
        'nccl_kernel_count': len(nccl),
        'nccl_kernel_duration_ms': sum(float(event.get('dur', 0)) for event in nccl) / 1000,
        'gpu_timeline': merged_idle(intervals),
    }


def aggregate(ranks):
    numeric = ('kernel_count', 'kernels_under_10us', 'kernels_under_25us',
               'kernel_duration_ms', 'nccl_kernel_count', 'nccl_kernel_duration_ms')
    result = {}
    for key in numeric:
        values = [rank[key] for rank in ranks]
        result[key] = {'min': min(values), 'median': median(values), 'max': max(values)}
    for key in KEYS:
        values = [rank['operator_counts'][key] for rank in ranks]
        result[key] = {'min': min(values), 'median': median(values), 'max': max(values)}
    for key in ranks[0]['gpu_timeline']:
        values = [rank['gpu_timeline'][key] for rank in ranks]
        result['gpu_timeline_' + key] = {
            'min': min(values), 'median': median(values), 'max': max(values)}
    return result


def main():
    profiles = {}
    for name in PROFILES:
        directory = ROOT / name
        ranks = [summarize_rank(directory, rank) for rank in range(4)]
        with (directory / 'result.json').open() as handle:
            benchmark = json.load(handle)
        profiles[name] = {
            'args': benchmark['args'],
            'pair_block_calls_per_rank': benchmark['block_calls_per_rank'],
            'ranks': ranks,
            'four_rank_summary': aggregate(ranks),
        }
    output = {
        'baseline_commit': '82e06b2e5b0a76e51a924928a036be447e8fce02',
        'note': ('Profiler covers one active step after warmup. Timings are reported only '
                 'as diagnostic context; profiler-off runs are authoritative.'),
        'raw_trace_root': str(ROOT),
        'profiles': profiles,
    }
    with (HERE / 'profiler_summary.json').open('w') as handle:
        json.dump(output, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
