"""Observe the existing run; never start, stop, or modify training/evaluation."""
import datetime
import hashlib
import json
import math
import pathlib
import statistics
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
REFERENCE = pathlib.Path('/root/lk_projects/SAID-reproduction/runs/full_replica01_3651/salu_log.jsonl')


def finite(value):
    if isinstance(value, float):
        assert math.isfinite(value), 'Non-finite recorded value'
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def observe():
    raw = (ROOT / 'run/salu_log.jsonl').read_bytes()
    # A live writer may be between the last JSON byte and its terminating newline.
    lines = raw[:raw.rfind(b'\n') + 1].splitlines()
    rows = [json.loads(line) for line in lines]
    reference = [json.loads(line) for line in REFERENCE.read_text().splitlines() if line.strip()]
    assert len(reference) == 3151
    assert [row['completed_steps'] for row in rows] == list(range(501, 501 + len(rows)))
    assert len(rows) <= 3151
    for actual, expected in zip(rows, reference):
        finite(actual)
        assert actual['batch_stream_sha256'] == expected['batch_stream_sha256']
        assert actual['completed_steps'] == expected['completed_steps']
        assert actual['lambda_suffix'] == 3 and actual['lambda_u_sparse'] == 0
        assert actual['debug_optimizer_updates'] == 0
        assert len(actual['rank_health']) == len(expected['rank_health']) == 4
        assert [h['rank'] for h in actual['rank_health']] == list(range(4))
        for left, right in zip(actual['rank_health'], expected['rank_health']):
            assert all(left[key] == right[key] for key in ('rank', 'consumed_samples', 'stream_sha256'))
    state = json.loads((ROOT / 'status.json').read_text())
    last = rows[-1]
    record = {
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'phase': state['phase'],
        'completed_steps': last['completed_steps'],
        'verified_new_rows': len(rows),
        'all_four_rank_streams_match': True,
        'all_recorded_numbers_finite': True,
        'loss_total': last['loss_total'],
        'epoch': last['epoch'],
        'mean_recent_step_seconds': statistics.mean(r['synchronized_step_seconds'] for r in rows[-100:]),
        'log_age_seconds': round(time.time() - (ROOT / 'run/salu_log.jsonl').stat().st_mtime, 2),
        'train_launcher_exists': pathlib.Path('/proc', (ROOT / 'train_launcher.pid').read_text().strip()).exists(),
        'postprocess_exists': pathlib.Path('/proc', (ROOT / 'postprocess.pid').read_text().strip()).exists(),
        'exit_codes': {str(p.relative_to(ROOT)): int(p.read_text()) for pattern in ('run/exitcode', 'logs/*.exitcode', 'evaluation/*/exitcode') for p in ROOT.glob(pattern)},
        'checkpoints': [p.name for p in sorted((ROOT / 'run').glob('*.pt'))],
    }
    progress = ROOT / 'reports/evaluation_progress.json'
    if progress.exists():
        record['evaluated_protocols'] = [x['protocol'] for x in json.loads(progress.read_text())['completed']]
    if state.get('error'):
        record['pipeline_error'] = state['error']
    if record['phase'] == 'TRAINING' and record['log_age_seconds'] > 180:
        record['attention'] = 'Training log has not advanced for over 180 seconds; inspect without restarting.'
    with (ROOT / 'logs/supervision.jsonl').open('a') as handle:
        handle.write(json.dumps(record) + '\n')
    temporary = ROOT / 'validation/supervision_latest.tmp'
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(ROOT / 'validation/supervision_latest.json')
    print(json.dumps(record), flush=True)
    return record


if __name__ == '__main__':
    while True:
        snapshot = observe()
        if snapshot['phase'] in ('COMPLETE', 'FAILED'):
            raise SystemExit(0 if snapshot['phase'] == 'COMPLETE' else 1)
        time.sleep(50)
