import json

from streamlit.testing.v1 import AppTest
import pytest

from tools.data.audit_sharegpt4v import sha256_file
from tools.data.full_data_gate import require_full_data
from tools.said_dashboard.data_integrity_page import gate_passes


def artifacts(tmp_path):
    dataset = tmp_path / 'data.json'
    dataset.write_text('[]')
    split = tmp_path / 'sharegpt4v_split_manifest.json'
    split.write_text('{}')
    audit = dict(scan_complete=True, records=1002, resolved_records=1002, val_records=1000,
                 train_records=2, missing_records=0, corrupt_records=0, unreadable_records=0,
                 unknown_source_records=0, overlap_count=0, data_root=str(tmp_path),
                 json_sha256=sha256_file(dataset), manifest_sha256=sha256_file(split), sources={})
    shards = {'json_sha256': audit['json_sha256'],
              'shards': [dict(id='%06d' % i, image_verified=True, hash_verified=True) for i in range(51)],
              'union': dict(required_total=100, union_total=100, intersection_total=100,
                            missing_required=0, extra_ids=0, cross_shard_duplicates=0)}
    for name, value in [('sharegpt4v_full_audit', audit), ('sam_shards', shards)]:
        (tmp_path / (name + '.json')).write_text(json.dumps(value))
    return audit, shards


def test_gate_rejects_missing_incomplete_union_and_wrong_identity(tmp_path):
    audit, shards = artifacts(tmp_path)
    assert gate_passes(audit, shards)
    assert not gate_passes(dict(audit, missing_records=1), shards)
    assert not gate_passes(audit, dict(shards, json_sha256='different'))
    assert not gate_passes(audit, dict(shards, shards=shards['shards'][:-1]))
    shards['union']['cross_shard_duplicates'] = 1
    assert not gate_passes(audit, shards)


def test_strict_full_gate_binds_json_root_and_manifest(tmp_path):
    audit, _ = artifacts(tmp_path)
    path = tmp_path / 'sharegpt4v_full_audit.json'
    data = tmp_path / 'data.json'
    assert require_full_data(path, data, tmp_path)['records'] == 1002
    with pytest.raises(RuntimeError, match='root differs'):
        require_full_data(path, data, tmp_path / 'different')
    (tmp_path / 'sharegpt4v_split_manifest.json').write_text('{"changed":1}')
    with pytest.raises(RuntimeError, match='manifest SHA256 mismatch'):
        require_full_data(path, data, tmp_path)
    data.write_text('[{}]')
    with pytest.raises(RuntimeError, match='JSON SHA256 differs'):
        require_full_data(path, data, tmp_path)


@pytest.mark.parametrize('state', ['missing', 'fail', 'pass'])
def test_chinese_dashboard_missing_fail_pass(tmp_path, monkeypatch, state):
    monkeypatch.setenv('SAID_DATA_AUDIT_ROOT', str(tmp_path))
    if state != 'missing':
        audit, _ = artifacts(tmp_path)
        if state == 'fail':
            audit['missing_records'] = 1
            (tmp_path / 'sharegpt4v_full_audit.json').write_text(json.dumps(audit))
    app = AppTest.from_string('from tools.said_dashboard.data_integrity_page import main\nmain()').run()
    assert not app.exception
    if state == 'pass':
        assert app.success[0].value == '完整数据已通过训练 Gate'
    else:
        assert app.error[0].value == '完整数据尚未通过训练 Gate'
    if state == 'missing':
        assert '审计文件尚不可用' in app.warning[0].value
