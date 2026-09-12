"""Tests for the read-only training dashboard (task section 9.3).

Every case starts the real HTTP service on an ephemeral loopback port, drives it with ``urllib`` and
shuts it down again. The suite also asserts the two invariants that matter for training: the service
never imports torch (so it cannot touch a GPU) and it opens only whitelisted files inside a
registered run directory.
"""
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(REPO_ROOT, 'tools')
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, 'train'))

from dashboard_data import (DashboardData, MetricsReader, RunNotFound,  # noqa: E402
                            RunRegistry, read_json)
from serve_training_dashboard import build_server  # noqa: E402

WEB_DIR = os.path.join(REPO_ROOT, 'web', 'training_dashboard')


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


class Server:
    def __init__(self, registry, port):
        self.server, self.data, self.port = build_server(registry, '127.0.0.1', port, WEB_DIR)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def get(self, path, expect_json=True):
        url = 'http://127.0.0.1:%d%s' % (self.port, path)
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                body = response.read()
                status = response.status
        except urllib.error.HTTPError as error:
            body, status = error.read(), error.code
        if expect_json:
            try:
                return status, json.loads(body.decode('utf-8'))
            except ValueError:
                return status, {'raw': body.decode('utf-8', 'replace')}
        return status, body.decode('utf-8', 'replace')


def write_log(path, records, trailing_partial=None):
    with open(path, 'a', encoding='utf-8') as handle:
        for record in records:
            handle.write('LOG ' + json.dumps(record, sort_keys=True) + '\n')
        if trailing_partial is not None:
            handle.write(trailing_partial)


def sample_record(step, **extra):
    record = {'completed_steps': step, 'loss_total': 10.0 - step * 0.01, 'loss_1': 0.5,
              'loss_1_i2t': 0.25, 'loss_1_t2i': 0.25, 'loss_2': 0.4, 'loss_2_i2t': 0.2,
              'loss_2_t2i': 0.2, 'loss_3': 0.45, 'loss_3_i2t': 0.22, 'loss_3_t2i': 0.23,
              'loss_sparse_i': 0.8, 'loss_sparse_t': 0.9, 'weighted_loss_1': 5.0,
              'weighted_loss_2': 0.4, 'weighted_loss_3': 0.45, 'weighted_loss_sparse_i': 1.6,
              'weighted_loss_sparse_t': 0.18, 'mask_i_keep_ratio': 0.82, 'mask_t_mean': 0.6,
              'sec_per_step': 1.0, 'peak_memory_gb': 24.0, 'grads_finite': 1.0}
    record.update(extra)
    return record


def build_run(tmp_path, run_id='hs_test', records=3, status=None, snapshot=True,
              evaluation=False):
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_log(str(run_dir / 'salu_log.jsonl'), [sample_record(i + 1) for i in range(records)])
    (run_dir / 'config.json').write_text(json.dumps({
        'objective': 'smartclip_trimask_hs', 'arm': 'S0_TriMask_HS', 'git_head': 'deadbeef',
        'text_gate_mode': 'hard_st', 'lambda_sparse_t': 0.2, 'world_size': 4,
        'batch_size_per_gpu': 256, 'lr_horizon_steps': 3651, 'max_steps': 500,
    }), encoding='utf-8')
    if status is not None:
        (run_dir / 'run_status.json').write_text(json.dumps(status), encoding='utf-8')
    if snapshot:
        (run_dir / 'mask_snapshot.json').write_text(json.dumps({
            'completed_steps': records, 'source': 'rank0/local_batch',
            'mask_i_coordinate_mean': [0.8] * 512, 'mask_t_coordinate_mean': [0.6] * 512,
            'intersection_coordinate_mean': [0.5] * 512,
            'text_gate_probability_sample0': [0.9] * 512,
            'text_gate_keep_ratio': 0.6, 'visual_mask_keep_ratio': 0.8,
            'text_gate_zero_fraction': 0.4,
            'sample0': {'caption_said': 'a photo of a cat .', 'sha256': 'abc'},
        }), encoding='utf-8')
    if evaluation:
        evaluation_dir = run_dir / 'evaluation'
        evaluation_dir.mkdir(exist_ok=True)
        (evaluation_dir / ('%s_canonical.json' % run_id)).write_text(json.dumps({
            'canonical': {'S0_TriMask_HS@500': {
                'coco_val2017': {'image2text_R1': 0.61, 'image2text_R5': 0.83,
                                 'image2text_R10': 0.9, 'text2image_R1': 0.42,
                                 'text2image_R5': 0.68, 'text2image_R10': 0.77},
                'checkpoint_sha256': 'cafe'}}}), encoding='utf-8')
        (evaluation_dir / ('%s_urban1k.json' % run_id)).write_text(json.dumps({
            'urban1k': {'image2text': {'R1': 0.88, 'R5': 0.97, 'R10': 0.99},
                        'text2image': {'R1': 0.85, 'R5': 0.96, 'R10': 0.98}},
            'checkpoint_sha256': 'cafe', 'label': 'S0_TriMask_HS@500'}), encoding='utf-8')
    return run_dir


def registry_for(run_dir, run_id='hs_test', **kwargs):
    registry = RunRegistry()
    registry.register(run_id, str(run_dir), evaluation_prefix=run_id, **kwargs)
    return registry


# ---------------------------------------------------------------- health / runs
def test_health_and_run_listing(tmp_path):
    run_dir = build_run(tmp_path)
    with Server(registry_for(run_dir), free_port()) as server:
        status, payload = server.get('/health')
        assert status == 200 and payload['status'] == 'ok'
        assert payload['read_only'] is True and payload['gpu_used'] is False
        assert payload['checkpoints_loaded'] is False
        status, payload = server.get('/api/runs')
        assert status == 200
        assert [run['run_id'] for run in payload['runs']] == ['hs_test']
        assert payload['runs'][0]['demo'] is False


def test_demo_entries_are_flagged_and_never_mixed_with_real_runs(tmp_path):
    real = build_run(tmp_path, run_id='real_run')
    demo = build_run(tmp_path, run_id='demo_run')
    registry = RunRegistry()
    registry.register('real_run', str(real))
    registry.register('demo_run', str(demo), label='DEMO 演示数据', demo=True)
    with Server(registry, free_port()) as server:
        status, payload = server.get('/api/runs')
        entries = {run['run_id']: run for run in payload['runs']}
        assert entries['demo_run']['demo'] is True
        assert entries['real_run']['demo'] is False
        assert entries['demo_run']['label'].startswith('DEMO')


# ---------------------------------------------------------------- empty / half-written
def test_empty_run_directory_reports_nulls(tmp_path):
    run_dir = tmp_path / 'empty'
    run_dir.mkdir()
    registry = RunRegistry()
    registry.register('empty', str(run_dir))
    with Server(registry, free_port()) as server:
        status, payload = server.get('/api/run/empty/status')
        assert status == 200
        assert payload['phase'] == 'not_started'
        assert payload['completed_steps'] == 0
        assert payload['progress'] == 0.0
        assert payload['objective'] is None
        status, metrics = server.get('/api/run/empty/metrics?after=0')
        assert status == 200 and metrics['records'] == [] and metrics['cursor'] == 0
        status, masks = server.get('/api/run/empty/masks')
        assert masks['snapshot'] is None and masks['snapshot_error'] is None
        status, evaluation = server.get('/api/run/empty/evaluation')
        assert evaluation['datasets']['coco']['available'] is False
        assert evaluation['datasets']['coco']['metrics'] is None
        assert evaluation['verdict'] is None


def test_half_written_jsonl_line_is_ignored_until_complete(tmp_path):
    run_dir = tmp_path / 'partial'
    run_dir.mkdir()
    log = str(run_dir / 'salu_log.jsonl')
    write_log(log, [sample_record(1), sample_record(2)])
    with open(log, 'a', encoding='utf-8') as handle:
        handle.write('LOG {"completed_steps": 3, "loss_to')      # unfinished
    registry = RunRegistry()
    registry.register('partial', str(run_dir))
    with Server(registry, free_port()) as server:
        status, payload = server.get('/api/run/partial/metrics?after=0')
        assert [record['completed_steps'] for record in payload['records']] == [1, 2]
        assert payload['cursor'] == 2
        assert payload['warnings'] == []
    with open(log, 'a', encoding='utf-8') as handle:
        handle.write('tal": 9.5}\n')
    with Server(registry, free_port()) as server:
        status, payload = server.get('/api/run/partial/metrics?after=2')
        assert [record['completed_steps'] for record in payload['records']] == [3]
        assert payload['records'][0]['loss_total'] == 9.5
        assert payload['cursor'] == 3


def test_corrupt_complete_line_warns_and_never_crashes(tmp_path):
    run_dir = tmp_path / 'corrupt'
    run_dir.mkdir()
    log = str(run_dir / 'salu_log.jsonl')
    write_log(log, [sample_record(1)])
    with open(log, 'a', encoding='utf-8') as handle:
        handle.write('LOG {"completed_steps": 2, "loss_total": }\n')     # invalid JSON
        handle.write('this is not a log line at all\n')
    write_log(log, [sample_record(3)])
    registry = RunRegistry()
    registry.register('corrupt', str(run_dir))
    with Server(registry, free_port()) as server:
        status, payload = server.get('/api/run/corrupt/metrics?after=0')
        assert status == 200
        assert [record['completed_steps'] for record in payload['records']] == [1, 3]
        assert len(payload['warnings']) == 2


def test_truncated_or_rotated_log_resets_its_cursor(tmp_path):
    run_dir = tmp_path / 'rotate'
    run_dir.mkdir()
    log = str(run_dir / 'salu_log.jsonl')
    write_log(log, [sample_record(i + 1) for i in range(5)])
    reader = MetricsReader()
    first = reader.read(log, after=0)
    assert first['available'] == 5
    with open(log, 'w', encoding='utf-8') as handle:        # rotation: shorter file, new content
        handle.write('LOG ' + json.dumps(sample_record(99)) + '\n')
    second = reader.read(log, after=0)
    assert second['available'] == 1
    assert second['records'][0]['completed_steps'] == 99
    assert second['warnings'], 'a truncation must be reported, not silently absorbed'


def test_historical_read_and_incremental_cursor(tmp_path):
    run_dir = build_run(tmp_path, records=6)
    with Server(registry_for(run_dir), free_port()) as server:
        status, first = server.get('/api/run/hs_test/metrics?after=0&limit=4')
        assert [r['completed_steps'] for r in first['records']] == [1, 2, 3, 4]
        assert first['cursor'] == 4 and first['available'] == 6
        status, second = server.get('/api/run/hs_test/metrics?after=%d' % first['cursor'])
        assert [r['completed_steps'] for r in second['records']] == [5, 6]
        assert second['cursor'] == 6
        # appending new records only returns the new ones
        write_log(str(run_dir / 'salu_log.jsonl'), [sample_record(7)])
        status, third = server.get('/api/run/hs_test/metrics?after=%d' % second['cursor'])
        assert [r['completed_steps'] for r in third['records']] == [7]


# ---------------------------------------------------------------- status semantics
def test_phase_and_eta_semantics(tmp_path):
    run_dir = build_run(tmp_path, records=30, status={
        'phase': 'training', 'completed_steps': 30, 'max_steps': 500,
        'started_at': time.time() - 120, 'train_exit_code': None})
    with Server(registry_for(run_dir), free_port()) as server:
        status, payload = server.get('/api/run/hs_test/status')
        assert payload['phase'] == 'training' and payload['phase_label'] == '训练中'
        assert payload['progress'] == pytest.approx(0.06, abs=1e-6)
        assert payload['eta_is_estimate'] is True and payload['eta_seconds'] > 0
        assert payload['mean_sec_per_step'] == pytest.approx(1.0, rel=1e-6)
        assert payload['elapsed_seconds'] >= 0

    # 500/500 updates with the evaluations still pending must NOT read as "complete"
    run_dir = build_run(tmp_path, run_id='hs_done', records=5, status={
        'phase': 'evaluating_coco', 'completed_steps': 500, 'max_steps': 500,
        'coco_exit_code': None, 'urban_exit_code': None})
    with Server(registry_for(run_dir, run_id='hs_done'), free_port()) as server:
        status, payload = server.get('/api/run/hs_done/status')
        assert payload['completed_steps'] == 500 and payload['max_steps'] == 500
        assert payload['phase'] == 'evaluating_coco'
        assert payload['phase_label'] == 'COCO评估中'


def test_eta_is_withheld_while_the_sample_is_too_small(tmp_path):
    run_dir = build_run(tmp_path, records=3, status={'phase': 'training', 'completed_steps': 3})
    with Server(registry_for(run_dir), free_port()) as server:
        status, payload = server.get('/api/run/hs_test/status')
        assert payload['eta_is_estimate'] is False and payload['eta_seconds'] is None


def test_failed_run_reports_failure_not_completion(tmp_path):
    run_dir = build_run(tmp_path, status={
        'phase': 'failed', 'completed_steps': 500, 'max_steps': 500, 'train_exit_code': 1,
        'failure_reason': 'training exited 1'})
    with Server(registry_for(run_dir), free_port()) as server:
        status, payload = server.get('/api/run/hs_test/status')
        assert payload['phase'] == 'failed' and payload['phase_label'] == '失败'
        status, logs = server.get('/api/run/hs_test/logs')
        assert any(issue['level'] == 'error' for issue in logs['issues'])
        assert any('退出码' in issue['text'] or 'exit' in issue['text'] for issue in logs['issues'])


# ---------------------------------------------------------------- masks and evaluation
def test_mask_snapshot_and_intersection_payload(tmp_path):
    run_dir = build_run(tmp_path, records=25)
    record = sample_record(25, heavy_diagnostics=True, mask_i_empty_fraction=0.0,
                           mask_t_empty_fraction=0.0,
                           mask_intersection_intersection_count_mean=410.0,
                           mask_intersection_jaccard_mean=None,
                           mask_intersection_intersection_empty_fraction=0.0,
                           mask_t_cross_caption_std_mean=0.1, adv_gap=0.05,
                           third_not_worse_fraction=0.4, text_gate_pT_mean=0.55)
    write_log(str(run_dir / 'salu_log.jsonl'), [record])
    with Server(registry_for(run_dir), free_port()) as server:
        status, payload = server.get('/api/run/hs_test/masks')
        assert status == 200
        assert len(payload['snapshot']['mask_t_coordinate_mean']) == 512
        assert len(payload['snapshot']['mask_i_coordinate_mean']) == 512
        assert len(payload['snapshot']['intersection_coordinate_mean']) == 512
        heavy = payload['latest_heavy_scalars']
        assert heavy['completed_steps'] == 25
        assert heavy['mask_intersection_jaccard_mean'] is None      # explicit null, not 0


def test_evaluation_appears_only_when_the_file_exists(tmp_path):
    run_dir = build_run(tmp_path, run_id='evalrun', records=2)
    with Server(registry_for(run_dir, run_id='evalrun'), free_port()) as server:
        status, payload = server.get('/api/run/evalrun/evaluation')
        assert payload['datasets']['coco']['available'] is False
        assert payload['verdict'] is None
        assert payload['baselines']['S0@500']['coco']['i2t_r1'] == 0.6058

    build_run(tmp_path, run_id='evalrun', records=2, evaluation=True)
    with Server(registry_for(run_dir, run_id='evalrun'), free_port()) as server:
        status, payload = server.get('/api/run/evalrun/evaluation')
        assert payload['datasets']['coco']['available'] is True
        assert payload['datasets']['coco']['metrics']['i2t_r1'] == 0.61
        assert payload['datasets']['urban1k']['metrics']['t2i_r1'] == 0.85
        # 0.61 >= 0.6058 and 0.42 >= 0.41236 with one strictly higher -> the gate is evaluated on
        # raw precision, per direction
        assert payload['verdict']['verdict'] == 'PROMISING_AT_500'
        assert payload['verdict']['i2t_pass'] is True and payload['verdict']['t2i_pass'] is True


# ---------------------------------------------------------------- security
@pytest.mark.parametrize('path', [
    '/api/run/..%2f..%2fetc/status',
    '/api/run/../../etc/passwd/status',
    '/api/run/%2e%2e/status',
    '/api/run/unknown_run/status',
    '/api/run/../../etc/status',
])
def test_unknown_and_traversal_run_ids_are_rejected(tmp_path, path):
    run_dir = build_run(tmp_path)
    with Server(registry_for(run_dir), free_port()) as server:
        status, payload = server.get(path)
        assert status == 404
        assert payload.get('error') in ('unknown run', 'unknown endpoint')
        # nothing outside the registry can be read, whatever the shape of the attempt
        assert 'root:' not in json.dumps(payload)


@pytest.mark.parametrize('path', [
    '/static/../../../../etc/passwd',
    '/static/..%2f..%2f..%2ftmp/x.css',
    '/static/%2e%2e/%2e%2e/train/train_said_trimask.py',
])
def test_static_traversal_is_rejected(tmp_path, path):
    run_dir = build_run(tmp_path)
    with Server(registry_for(run_dir), free_port()) as server:
        status, _ = server.get(path, expect_json=False)
        assert status in (403, 404)


def test_unknown_endpoint_and_no_mutating_methods(tmp_path):
    run_dir = build_run(tmp_path)
    with Server(registry_for(run_dir), free_port()) as server:
        status, _ = server.get('/api/run/hs_test/execute', expect_json=False)
        assert status == 404
        request = urllib.request.Request('http://127.0.0.1:%d/api/runs' % server.port,
                                         data=b'{}', method='POST')
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=10)
        assert error.value.code == 501        # no POST handler exists at all


def test_service_never_imports_torch_or_touches_a_checkpoint(tmp_path):
    """Run the service in a fresh interpreter and assert its own invariants."""
    run_dir = build_run(tmp_path)
    script = (
        'import json, os, sys, threading, urllib.request\n'
        'sys.path.insert(0, %r)\n'
        'from serve_training_dashboard import build_server\n'
        'from dashboard_data import RunRegistry\n'
        'registry = RunRegistry()\n'
        'registry.register("hs_test", %r)\n'
        'server, data, port = build_server(registry, "127.0.0.1", 0, %r)\n'
        'thread = threading.Thread(target=server.serve_forever, daemon=True)\n'
        'thread.start()\n'
        'for path in ("/health", "/api/runs", "/api/run/hs_test/status",\n'
        '             "/api/run/hs_test/metrics?after=0", "/api/run/hs_test/masks",\n'
        '             "/api/run/hs_test/evaluation", "/api/run/hs_test/logs"):\n'
        '    with urllib.request.urlopen("http://127.0.0.1:%%d%%s" %% (port, path)) as r:\n'
        '        r.read()\n'
        'server.shutdown(); server.server_close()\n'
        'print(json.dumps({"torch_imported": "torch" in sys.modules,\n'
        '                  "cuda_initialised": getattr(__import__("torch").cuda, "is_initialized",\n'
        '                                              lambda: False)() if "torch" in '
        'sys.modules else False}))\n'
    ) % (TOOLS_DIR, str(run_dir), WEB_DIR)
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                            cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr[-2000:]
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {'torch_imported': False, 'cuda_initialised': False}


def test_only_whitelisted_files_inside_the_run_directory_are_opened(tmp_path, monkeypatch):
    run_dir = build_run(tmp_path)
    opened = []
    real_open = open

    def spy(path, *args, **kwargs):
        opened.append(str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr('builtins.open', spy)
    registry = registry_for(run_dir)
    data = DashboardData(registry)
    data.list_runs()
    data.status('hs_test')
    data.metrics('hs_test')
    data.masks('hs_test')
    data.evaluation('hs_test')
    data.logs('hs_test')
    for path in opened:
        if 'training_dashboard' in path:
            continue
        assert str(run_dir) in path or path.endswith('.py') or '/proc/' in path, path
    allowed = {'salu_log.jsonl', 'config.json', 'run_status.json', 'run_summary.json',
               'mask_snapshot.json'}
    for path in opened:
        if str(run_dir) not in path:
            continue
        relative = os.path.relpath(path, str(run_dir))
        if relative.startswith('evaluation' + os.sep):
            assert relative.endswith(('_canonical.json', '_urban1k.json')), relative
            continue
        assert os.path.basename(path) in allowed, path
    with pytest.raises(RunNotFound):
        registry.path('hs_test', '../../etc/passwd')


def test_evaluation_is_found_by_its_fixed_suffix_without_an_explicit_prefix(tmp_path):
    """The runner names files '<arm>_step500_canonical.json'; the panel must still find them."""
    run_dir = build_run(tmp_path, run_id='hs_dirname', records=2)
    evaluation_dir = run_dir / 'evaluation'
    evaluation_dir.mkdir(exist_ok=True)
    (evaluation_dir / 'S0_TriMask_HS_step000500_canonical.json').write_text(json.dumps({
        'canonical': {'S0_TriMask_HS@500': {
            'coco_val2017': {'image2text_R1': 0.6006, 'image2text_R5': 0.824,
                             'image2text_R10': 0.887, 'text2image_R1': 0.41208,
                             'text2image_R5': 0.671, 'text2image_R10': 0.76604},
            'checkpoint_sha256': 'abc'}}}), encoding='utf-8')
    (evaluation_dir / 'S0_TriMask_HS_step000500_urban1k.json').write_text(json.dumps({
        'urban1k': {'image2text': {'R1': 0.874, 'R5': 0.975, 'R10': 0.989},
                    'text2image': {'R1': 0.837, 'R5': 0.963, 'R10': 0.982}},
        'checkpoint_sha256': 'abc'}), encoding='utf-8')
    registry = RunRegistry()
    registry.register('hs_dirname', str(run_dir), evaluation_prefix='unrelated_name')
    with Server(registry, free_port()) as server:
        status, payload = server.get('/api/run/hs_dirname/evaluation')
        assert payload['datasets']['coco']['available'] is True
        assert payload['datasets']['coco']['file'] == 'S0_TriMask_HS_step000500_canonical.json'
        assert payload['datasets']['coco']['metrics']['i2t_r1'] == 0.6006
        assert payload['datasets']['urban1k']['available'] is True
        # both COCO directions are below the frozen gate, so the verdict is FAIL on raw precision
        assert payload['verdict']['i2t_pass'] is False
        assert payload['verdict']['t2i_pass'] is False
        assert payload['verdict']['verdict'] == 'FAIL'


def test_run_option_strings_are_parsed(tmp_path):
    from serve_training_dashboard import build_registry
    run_dir = build_run(tmp_path, run_id='opt')
    registry = build_registry(['opt=%s|prefix=FOO_step000500|label=%s' % (run_dir, 'HS 演示'),
                               'opt2=%s|demo=true' % run_dir])
    record = registry.resolve('opt')
    assert record['evaluation_prefix'] == 'FOO_step000500'
    assert record['label'] == 'HS 演示'
    assert registry.resolve('opt2')['demo'] is True
    with pytest.raises(SystemExit):
        build_registry(['opt=%s|bogus=1' % run_dir])


def test_metrics_csv_and_active_run_agreement(tmp_path):
    run_dir = build_run(tmp_path, records=4)
    with Server(registry_for(run_dir), free_port()) as server:
        status, csv_text = server.get('/api/run/hs_test/metrics.csv', expect_json=False)
        assert status == 200
        header = csv_text.splitlines()[0]
        assert 'completed_steps' in header and 'loss_total' in header
        assert len(csv_text.splitlines()) == 5          # header + 4 records
        assert 'caption' not in header
