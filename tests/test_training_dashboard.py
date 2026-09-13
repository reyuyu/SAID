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


def _probe_payload():
    """A small stand-in for the geometry probe output, with the fields the page displays."""
    return {
        'probe': 'trimask_hs_geometry_probe', 'read_only': True, 'new_optimizer_updates': 0,
        'not_a_canonical_evaluation': True, 'scope': 'S0-TriMask-HS @500, 256 x 256 forward-only',
        'checkpoint': {'path': '/tmp/ckpt.pt', 'sha256_before': 'a' * 64, 'sha256_after': 'a' * 64,
                       'sha256_unchanged': True,
                       'identity': {'objective': 'smartclip_trimask_hs', 'arm': 'S0_TriMask_HS',
                                    'completed_steps': 500, 'text_gate_mode': 'hard_st',
                                    'lambda_sparse_t': 0.2},
                       'clip_tensors': 317, 'text_mask_net_tensors': 16},
        'parameter_state': {'before': {'clip': 'x'}, 'after': {'clip': 'x'}, 'unchanged': True},
        'run_status': {'unchanged': True, 'sha256_before': 'b' * 64, 'sha256_after': 'b' * 64},
        'manifest': {'count': 256, 'sha256': 'c' * 64,
                     'selection': {'seed': 0, 'n_images': 256},
                     'source': {'annotations_sha256': 'd' * 64},
                     'image_ids': [1, 2], 'annotation_ids': [10, 11], 'skipped_captions': []},
        'timing': {'wall_seconds': 12.5},
        'dtypes_and_precision': {'image_features': 'torch.float32'},
        'mask_statistics': {'mT_keep_ratio_mean': 0.95},
        'reconciliation_with_trained_scoring_form': {'matmul_vs_broadcast_cosine_q1_max_abs_diff': 1e-5},
        'diagnostic_A_variance': {
            'v_total': 0.05, 'v_level': 0.001, 'v_profile': 0.013, 'v_interaction': 0.036,
            'v_caption_dependent': 0.037,
            'share_of_total': {'level': 0.02, 'profile': 0.26, 'interaction': 0.72},
            'share_of_caption_dependent': {'level': 0.027, 'interaction': 0.973},
            'identity_v_total_minus_parts': 0.0,
            'identity_v_caption_dependent_minus_parts': 0.0,
            'historical_field_check': {'value': 0.001, 'equals_v_level': True}},
        'diagnostic_B_replacements': {
            'variants': {
                'NORMAL': {'q1_equals_normal_max_abs_diff': 0.0,
                           'L3_I2T': {'ce': 1.1, 'R@1': 0.7, 'mrr': 0.8, 'per_query_rank': [1, 2]},
                           'L2_I2T': {'ce': 0.77, 'R@1': 0.79, 'mrr': 0.86, 'per_query_rank': [1, 1]},
                           'L3_T2I': {'ce': 1.02, 'R@1': 0.75, 'mrr': 0.83}},
                'SHUFFLED_seed0': {'q1_equals_normal_max_abs_diff': 0.0,
                                   'L3_I2T': {'ce': 1.22, 'R@1': 0.7, 'mrr': 0.79},
                                   'paired_vs_normal': {'L3_I2T': {'delta_ce_mean': 0.11,
                                                                  'changed_rank_queries': 47}}}},
            'identity_checks': {'ones_q3_equals_normal_q1_max_abs_diff': 0.0,
                                'ones_q2_equals_raw_global_cosine_max_abs_diff': 0.0},
            'shuffled_aggregate': {'L3_I2T': {'ce_mean': 1.2, 'ce_min': 1.18, 'ce_max': 1.23,
                                              'R@1_mean': 0.705}},
            'shuffle_permutations': {'seeds': [0], 'report': [{'seed': 0}]},
            'not_used': 'L1 and loss_total'},
        'diagnostic_C_geometry': {
            'stats': {'valid_pairs': 65536, 'total_pairs': 65536, 'valid_fraction': 1.0,
                      'max_abs_error_on_valid_pairs': 2e-5,
                      'factor_positive_mean': 0.9166, 'factor_negative_mean': 0.9165},
            'q3_metrics': {'I2T': {'ce': 1.12, 'R@1': 0.707, 'mrr': 0.805, 'per_query_rank': [1]},
                           'T2I': {'ce': 1.02, 'R@1': 0.746, 'mrr': 0.834}},
            'qcap_metrics': {'I2T': {'ce': 1.16, 'R@1': 0.703, 'mrr': 0.807, 'per_query_rank': [2]},
                             'T2I': {'ce': 1.06, 'R@1': 0.746, 'mrr': 0.833}},
            'qcap_vs_q3_paired': {'I2T': {'delta_ce_mean': 0.043, 'delta_R@1': -0.004,
                                          'changed_rank_queries': 38, 'rank_worsened': 19,
                                          'rank_improved': 19, 'delta_s_pos_mean': 2.67,
                                          'delta_s_max_negative_mean': 2.30},
                                  'T2I': {'delta_ce_mean': 0.046, 'delta_R@1': 0.0,
                                          'changed_rank_queries': 6, 'rank_worsened': 3,
                                          'rank_improved': 3, 'delta_s_pos_mean': 2.67,
                                          'delta_s_max_negative_mean': 2.32}}},
        'hard_queries': [{'path': 'L3', 'direction': 'I2T', 'query_index': 5,
                          'query_label': '100/200', 'rank': 14, 'ce': 13.4, 'm_max': -13.2,
                          'm_lse': -13.4, 'strongest_negative_index': 9,
                          'strongest_negative_label': '300/400', 'score_of_worst_negative': 29.8,
                          'qcap_rank': 6, 'qcap_score_of_worst_negative': 33.4,
                          'factor_of_positive': 0.75, 'factor_of_worst_negative': 0.89,
                          'query_caption': 'a cat', 'strongest_negative_caption': 'a dog',
                          'visual_verification': 'NOT RUN'}],
        'tie_rule': 'rank = 1 + #{j != pos : score[j] > score[pos]}',
        'not_run': ['image-level verification of the hard-query negatives'],
    }


def write_probe(run_dir, payload=None):
    directory = run_dir / 'diagnostics'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'hs_mask_geometry_probe.json').write_text(
        json.dumps(payload if payload is not None else _probe_payload()), encoding='utf-8')
    (directory / 'manifest.json').write_text(json.dumps({'count': 256}), encoding='utf-8')
    return directory


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


# ---------------------------------------------------------------- F: offline diagnostics
def test_frontend_assets_parse_and_are_served(tmp_path):
    """A syntax error in app.js breaks every dynamic part of the page while the HTML still loads.

    That is exactly what happened once: ``obj.R@1_mean`` is not valid JavaScript (``@`` cannot appear
    in an identifier), the whole script failed to parse, and the page silently showed no run list and
    no curves even though every endpoint returned 200. Hence two checks here -- a node-free scan for
    the invalid property pattern, and a real ``node --check`` whenever node is available.
    """
    import re
    import shutil

    node = shutil.which('node')
    checked = []
    for name in ('app.js', 'index.html', 'style.css'):
        path = os.path.join(WEB_DIR, name)
        assert os.path.isfile(path), path
        source = open(path, encoding='utf-8').read()
        if name.endswith('.js'):
            # '@' is legal inside a string but never inside an identifier; this pattern is precisely
            # the bug that shipped: (x).R@1_mean / paired.delta_R@1
            bad = re.findall(r'\.[A-Za-z_][A-Za-z0-9_]*@[A-Za-z0-9_]', source)
            assert not bad, 'invalid JS property access in %s: %r' % (name, bad[:5])
            assert source.count('{') == source.count('}'), 'unbalanced braces in %s' % name
            assert source.count('(') == source.count(')'), 'unbalanced parentheses in %s' % name
        if node and name.endswith('.js'):
            result = subprocess.run([node, '--check', path], capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
            checked.append(name)
    with Server(registry_for(build_run(tmp_path)), free_port()) as server:
        for name in ('app.js', 'index.html', 'style.css'):
            status, body = server.get('/static/' + name, expect_json=False)
            assert status == 200 and len(body) > 200, name
        status, index = server.get('/', expect_json=False)
        assert status == 200 and '离线诊断' in index


def test_diagnostics_endpoint_reports_未诊断_without_fabricating_zeros(tmp_path):
    run_dir = build_run(tmp_path)
    with Server(registry_for(run_dir), free_port()) as server:
        status, payload = server.get('/api/run/hs_test/diagnostics')
    assert status == 200
    assert payload['available'] is False
    assert payload['status'] == '未诊断'
    assert payload['file'] == 'hs_mask_geometry_probe.json'
    assert payload['error'] is None                     # a missing file is not an error
    # every numeric block is explicitly absent rather than 0
    for key in ('variance', 'geometry', 'replacements', 'hard_queries', 'checkpoint',
                'new_optimizer_updates', 'scope'):
        assert payload[key] is None, key
    assert payload['files']['probe']['available'] is False
    assert payload['reminders']                        # the fixed reminders travel with the payload


def test_diagnostics_endpoint_serves_the_probe_and_prunes_query_ranks(tmp_path):
    run_dir = build_run(tmp_path)
    write_probe(run_dir)
    with Server(registry_for(run_dir), free_port()) as server:
        status, payload = server.get('/api/run/hs_test/diagnostics')
    assert status == 200
    assert payload['available'] is True and payload['status'] == '已诊断'
    assert payload['new_optimizer_updates'] == 0       # the probe must not have trained anything
    assert payload['not_a_canonical_evaluation'] is True
    checkpoint = payload['checkpoint']
    assert checkpoint['identity']['arm'] == 'S0_TriMask_HS'
    assert checkpoint['sha256_unchanged'] is True
    assert checkpoint['parameter_state_unchanged'] is True
    assert checkpoint['run_status_unchanged'] is True
    shares = payload['variance']['share_of_total']
    assert shares['level'] + shares['profile'] + shares['interaction'] == pytest.approx(1.0)
    # the per-query rank arrays of the variants are pruned from the response, the summary numbers stay
    normal = payload['replacements']['variants']['NORMAL']
    assert normal['L3_I2T']['ce'] == pytest.approx(1.1)
    assert 'per_query_rank' not in normal['L3_I2T']
    assert 'per_query_rank' not in normal['L2_I2T']
    assert payload['geometry']['stats']['valid_fraction'] == pytest.approx(1.0)
    assert payload['hard_queries'][0]['strongest_negative_label'] == '300/400'
    assert payload['tie_rule'].startswith('rank = 1 +')
    assert payload['not_run']


def test_diagnostics_never_reads_a_client_supplied_path(tmp_path):
    run_dir = build_run(tmp_path)
    write_probe(run_dir)
    with Server(registry_for(run_dir), free_port()) as server:
        for path in ('/api/run/../diagnostics', '/api/run/%2e%2e%2fdiagnostics',
                     '/api/run/unknown_run/diagnostics', '/api/run/hs_test/diagnostics/extra'):
            status, payload = server.get(path)
            assert status == 404, path
            assert 'error' in payload


def test_diagnostics_only_opens_whitelisted_files_and_writes_nothing(tmp_path, monkeypatch):
    run_dir = build_run(tmp_path)
    write_probe(run_dir)
    # a stray file inside diagnostics/ must never be served
    (run_dir / 'diagnostics' / 'anything_else.json').write_text('{"secret": 1}', encoding='utf-8')
    opened = []
    real_open = open

    def tracking_open(path, *args, **kwargs):
        opened.append(str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr('builtins.open', tracking_open)
    with Server(registry_for(run_dir), free_port()) as server:
        status, _ = server.get('/api/run/hs_test/diagnostics')
    assert status == 200
    assert not any('anything_else' in path for path in opened)
    # the service is read-only: nothing was created or modified inside the run directory
    before = sorted(os.listdir(str(run_dir)))
    with Server(registry_for(run_dir), free_port()) as server:
        server.get('/api/run/hs_test/diagnostics')
    assert sorted(os.listdir(str(run_dir))) == before


def test_diagnostics_does_not_change_run_status_or_the_verdict(tmp_path):
    status_payload = {'phase': 'complete', 'completed_steps': 500, 'arm': 'S0_TriMask_HS'}
    run_dir = build_run(tmp_path, status=status_payload, evaluation=True)
    write_probe(run_dir)
    # the real @500 evaluation (0.6006 / 0.41208) FAILS the frozen gate; the new endpoint must not
    # be able to change that verdict, neither by writing anything nor by miscounting the file
    canonical = run_dir / 'evaluation' / 'hs_test_canonical.json'
    canonical.write_text(json.dumps({'canonical': {'S0_TriMask_HS@500': {
        'coco_val2017': {'image2text_R1': 0.6006, 'image2text_R5': 0.824, 'image2text_R10': 0.887,
                         'text2image_R1': 0.41208, 'text2image_R5': 0.671,
                         'text2image_R10': 0.766},
        'checkpoint_sha256': 'cafe'}}}), encoding='utf-8')
    status_file = run_dir / 'run_status.json'
    before = status_file.read_text(encoding='utf-8')
    canonical_before = canonical.read_text(encoding='utf-8')
    with Server(registry_for(run_dir), free_port()) as server:
        _, evaluation_before = server.get('/api/run/hs_test/evaluation')
        status, _ = server.get('/api/run/hs_test/diagnostics')
        _, evaluation_after = server.get('/api/run/hs_test/evaluation')
    assert status == 200
    assert status_file.read_text(encoding='utf-8') == before
    assert canonical.read_text(encoding='utf-8') == canonical_before
    assert evaluation_before['verdict']['verdict'] == 'FAIL'
    assert evaluation_after == evaluation_before


def test_service_still_never_imports_torch_with_the_diagnostics_endpoint(tmp_path):
    run_dir = build_run(tmp_path)
    write_probe(run_dir)
    script = (
        'import json, sys, urllib.request\n'
        'sys.path.insert(0, %r)\n'
        'from dashboard_data import RunRegistry\n'
        'from serve_training_dashboard import build_server\n'
        'registry = RunRegistry()\n'
        'registry.register("hs_test", %r)\n'
        'server, data, port = build_server(registry, "127.0.0.1", 0, %r)\n'
        'import threading\n'
        'threading.Thread(target=server.serve_forever, daemon=True).start()\n'
        'with urllib.request.urlopen("http://127.0.0.1:%%d/api/run/hs_test/diagnostics" %% port) as r:\n'
        '    payload = json.loads(r.read())\n'
        'assert payload["available"] is True\n'
        'assert "torch" not in sys.modules, sorted(m for m in sys.modules if "torch" in m)\n'
        'print("NO_TORCH_OK")\n'
    ) % (TOOLS_DIR, str(run_dir), WEB_DIR)
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert 'NO_TORCH_OK' in result.stdout
