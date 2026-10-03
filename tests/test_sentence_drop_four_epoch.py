"""Protect complete-stream validation and reporting before aggregate statistics exist."""
import copy
import json

import pytest

from experiments.nest_clip_v1.balanced_rdrop_4epoch_v1.run import compare_streams, STREAM_KEYS
from experiments.nest_clip_v1.balanced_rdrop_4epoch_v1.report import write_report, DATASETS
from tests.test_nested_resume import payloads
from train.train_nested_semantic_mask import validate_resume_payload


def test_sentence_drop500_can_continue_only_with_same_horizon_and_data_code():
    previous, current = payloads()
    for cfg in (previous['config'], current):
        cfg.update(epochs=4, horizon=4868, remainder_mode='sentence_drop', four_epoch_followup=True)
    previous['scheduler_horizon'] = 4868
    current['max_updates'] = 4868
    assert validate_resume_payload(previous, current, 'old-trainer') == 500
    changed = copy.deepcopy(current)
    changed['remainder_mode'] = 'compact'
    with pytest.raises(AssertionError, match='remainder_mode'):
        validate_resume_payload(previous, changed, 'old-trainer')
    changed = copy.deepcopy(current)
    changed['code_sha256']['train/nested_semantic_data.py'] = 'changed'
    with pytest.raises(AssertionError, match='code mismatch'):
        validate_resume_payload(previous, changed, 'old-trainer')


def stream(tmp_path, name, records):
    path = tmp_path/name
    path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    return path


def entries():
    return [dict(step=step, nonfinite=0,
                 rank_health=[dict(rank=rank, gradients_finite=True,
                                   sampling={key:f'{step}:{rank}:{key}' for key in STREAM_KEYS})
                              for rank in range(4)]) for step in (1, 2)]


def test_complete_stream_chain_rejects_drift_and_duplicate_steps(tmp_path):
    records = entries()
    parent = stream(tmp_path, 'parent', records[:1])
    continuation = stream(tmp_path, 'continuation', records[1:])
    reference = stream(tmp_path, 'reference', records)
    assert compare_streams([parent, continuation], [reference], expected_updates=2)['passed']
    with pytest.raises(AssertionError):
        compare_streams([parent, continuation], [reference])
    bad = copy.deepcopy(records)
    bad[1]['rank_health'][2]['sampling']['K'] = 'changed'
    reference = stream(tmp_path, 'reference', bad)
    with pytest.raises(AssertionError, match='Matched stream drift step2: K'):
        compare_streams([parent, continuation], [reference], expected_updates=2)
    with pytest.raises(AssertionError, match='Noncontiguous'):
        compare_streams([parent, parent], [parent, parent], expected_updates=2)


def test_report_callback_accepts_native_result_before_statistics(tmp_path):
    metrics = {ds:{dr:{f'R@{k}':.5 for k in (1,5,10)} for dr in ('I2T','T2I')} for ds in DATASETS}
    record = dict(metrics=metrics, scores=dict(Score5_R1=.5, J_long3=.5, J_long=.5),
                  checkpoint_sha256='full', bare_sha256='bare')
    state = dict(status='running', baseline=record, parent500=record, rdrop_trial='trial',
                 trials={'trial':dict(budgets={'4868':record})})
    write_report(state, tmp_path)
    assert 'pending aggregation' in (tmp_path/'RDROP_4EPOCH_REPORT.md').read_text()
    record['sentence_drop_statistics'] = {'valid_R_samples':10}
    record['resource_summary'] = {'max_seconds':2.5}
    record['stream_comparison'] = {'passed':True}
    state['status'] = 'completed'
    write_report(state, tmp_path)
    assert 'pending aggregation' not in (tmp_path/'RDROP_4EPOCH_REPORT.md').read_text()
    assert json.loads((tmp_path/'sentence_drop_statistics.json').read_text()) == record['sentence_drop_statistics']
