"""Data-only audit: fixtures exercise failures rather than requiring downloaded data."""
import json
from pathlib import Path

from PIL import Image
import pytest

from tools.data.audit_sharegpt4v import (
    audit_records, full_data_gate, inspect_image, safe_path, source_for, split_manifest,
)


@pytest.mark.parametrize('path,source', [
    ('coco/train2017/000000125860.jpg', 'COCO'),
    ('llava/llava_pretrain/images/00120/001200924.jpg', 'LLaVA'),
    ('sam/images/sa_545504.jpg', 'SAM'),
    ('sam/other/image.jpg', 'unknown_source'),
    ('unseen/images/a.jpg', 'unknown_source'),
])
def test_source_patterns(path, source):
    assert source_for(path) == source


def test_audit_all_failure_categories_and_duplicate_weighting(tmp_path):
    good = tmp_path / 'coco/train2017/000000125860.jpg'
    good.parent.mkdir(parents=True)
    Image.new('RGB', (10, 12)).save(good)
    corrupt = tmp_path / 'sam/images/sa_2.jpg'
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b'not an image')
    denied = tmp_path / 'sam/images/sa_3.jpg'
    Image.new('RGB', (2, 2)).save(denied)
    denied.chmod(0)
    paths = [good.relative_to(tmp_path).as_posix()] * 2 + [
        'sam/images/sa_1.jpg', 'sam/images/sa_2.jpg', 'sam/images/sa_3.jpg', 'unknown/x.jpg']
    result = audit_records([{'image': p} for p in paths], tmp_path)
    denied.chmod(0o644)
    assert result['resolved_records'] == 2
    assert result['missing_records'] == 2
    assert result['corrupt_records'] == 1
    assert result['unreadable_records'] == 1
    assert result['duplicate_path_records'] == 1
    assert result['unknown_source_records'] == 1
    assert result['sources']['COCO']['unique_images'] == 1
    assert set(result['error_samples']) == {'missing', 'corrupt', 'unreadable'}
    assert not full_data_gate(result)


def test_split_deterministic_and_overlap(tmp_path):
    records = [{'image': 'sam/images/sa_%d.jpg' % i} for i in range(1002)]
    records[1001] = records[2].copy()
    manifest = split_manifest(records, tmp_path, 'hash')
    assert manifest == split_manifest(records, tmp_path, 'hash')
    assert manifest['val_records'] == 1000
    assert manifest['train_records'] == 2
    assert manifest['entries'][999]['split'] == 'validation'
    assert manifest['entries'][1000]['split'] == 'training'
    assert manifest['overlap_count'] == 1
    assert manifest['overlap_samples'] == ['sam/images/sa_2.jpg']
    assert len(records) == 1002


def test_symlink_overlap_and_path_escape(tmp_path):
    target = tmp_path / 'sam/images/sa_1.jpg'
    target.parent.mkdir(parents=True)
    Image.new('RGB', (2, 2)).save(target)
    (target.parent / 'sa_2.jpg').symlink_to(target)
    records = [{'image': 'sam/images/sa_1.jpg'}] * 1000 + [{'image': 'sam/images/sa_2.jpg'}]
    assert split_manifest(records, tmp_path, 'hash')['overlap_count'] == 1
    with pytest.raises(ValueError):
        safe_path(tmp_path, '../outside.jpg')


def test_image_reopens_and_decodes_truncated_jpeg(tmp_path):
    path = tmp_path / 'image.jpg'
    Image.new('RGB', (100, 100)).save(path)
    path.write_bytes(path.read_bytes()[:-30])
    assert inspect_image(path)[0] == 'corrupt'


def test_gate_requires_complete_matching_counts():
    good = dict(scan_complete=True, records=3, resolved_records=3, missing_records=0,
                corrupt_records=0, unreadable_records=0, unknown_source_records=0)
    assert full_data_gate(good)
    for key in ('missing_records', 'corrupt_records', 'unreadable_records', 'unknown_source_records'):
        assert not full_data_gate(dict(good, **{key: 1}))
    assert not full_data_gate(dict(good, scan_complete=False))
    assert not full_data_gate(dict(good, resolved_records=2))
    assert not full_data_gate({})


def test_spawn_audit_matches_serial(tmp_path):
    path = tmp_path / 'sam/images/sa_1.jpg'
    path.parent.mkdir(parents=True)
    Image.new('RGB', (7, 9)).save(path)
    records = [{'image': 'sam/images/sa_1.jpg'}, {'image': 'sam/images/sa_2.jpg'}]
    assert audit_records(records, tmp_path, workers=2) == audit_records(records, tmp_path)
