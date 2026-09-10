import io
import tarfile
from pathlib import Path

from PIL import Image
import pytest

from tools.data.prepare_hf_sa1b import (
    SHARDS, canonical_id, required_ids, scan_tar, selective_extract, union_stats, verify_hash,
)
from tools.data.audit_sharegpt4v import sha256_file
from tools.data.transfer_hf_sa1b import sftp_batch


def make_tar(path, ids, corrupt=None):
    with tarfile.open(path, 'w') as archive:
        for key in ids:
            buf = io.BytesIO()
            Image.new('RGB', (13, 17)).save(buf, 'JPEG')
            raw = b'broken' if key == corrupt else buf.getvalue()
            member = tarfile.TarInfo('./%s.jpg' % key)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))


def test_51_shard_union_exact_and_negative_cases():
    shards = {shard: ['sa_%d' % i] for i, shard in enumerate(SHARDS)}
    required = {'sa_%d' % i for i in range(51)}
    good = union_stats(shards, required)
    assert good['required_total'] == good['union_total'] == good['intersection_total'] == 51
    assert good['missing_required'] == good['extra_ids'] == good['cross_shard_duplicates'] == 0
    shards['000050'] = ['sa_0', 'sa_999']
    bad = union_stats(shards, required)
    assert bad['missing_required'] == bad['extra_ids'] == bad['cross_shard_duplicates'] == 1


def test_per_shard_id_audit(tmp_path):
    tar = tmp_path / 'shard.tar'
    make_tar(tar, ['sa_1', 'sa_1', 'sa_2', 'sa_999'])
    result = scan_tar(tar, {'sa_1', 'sa_2'}, {'sa_2'})
    assert result['image_count'] == 4
    assert result['required_intersection'] == 2
    assert result['duplicate_ids'] == ['sa_1']
    assert result['hf_only_ids'] == ['sa_999']
    assert result['already_seen_duplicate_ids'] == ['sa_2']


def test_selective_extract_and_resume_preserves_valid_bytes(tmp_path):
    tar = tmp_path / 'shard.tar'
    make_tar(tar, ['sa_1', 'sa_2', 'sa_999'])
    out = tmp_path / 'images'
    selective_extract(tar, out, {'sa_1'})
    target = out / 'sa_1.jpg'
    before = (target.stat().st_mtime_ns, target.read_bytes())
    resumed = selective_extract(tar, out, {'sa_1', 'sa_2'})
    assert resumed == {'newly_extracted': 1, 'existing_valid': 1, 'validated_images': 2}
    assert before == (target.stat().st_mtime_ns, target.read_bytes())
    assert not (out / 'sa_999.jpg').exists()


def test_corrupt_member_stops_atomically_and_resumes(tmp_path):
    tar, out = tmp_path / 'shard.tar', tmp_path / 'images'
    make_tar(tar, ['sa_1', 'sa_2'], corrupt='sa_2')
    with pytest.raises(RuntimeError, match='corrupt'):
        selective_extract(tar, out, {'sa_1', 'sa_2'})
    assert (out / 'sa_1.jpg').exists()
    assert not (out / 'sa_2.jpg').exists()
    assert not list(out.glob('*.partial'))
    make_tar(tar, ['sa_1', 'sa_2'])
    assert selective_extract(tar, out, {'sa_1', 'sa_2'})['existing_valid'] == 1


def test_hash_mismatch_stops_and_preserves_file(tmp_path):
    tar = tmp_path / 'shard.tar'
    make_tar(tar, ['sa_1'])
    with pytest.raises(RuntimeError, match='SHA256 mismatch'):
        verify_hash(tar, '0' * 64)
    assert verify_hash(tar, sha256_file(tar)) == sha256_file(tar)


@pytest.mark.parametrize('name', ['../sa_1.jpg', '/sa_1.jpg', 'bad.jpg', 'sa_1.jpg/../../x'])
def test_unsafe_canonical_names(name):
    with pytest.raises(ValueError):
        canonical_id(name)


def test_required_extraction_only_sam():
    assert required_ids([{'image': 'sam/images/sa_9.jpg'}, {'image': 'coco/train2017/sa_8.jpg'}]) == {'sa_9'}


def test_native_sftp_new_target_and_partial_resume():
    assert sftp_batch('C:/with space/a.tar', '/remote/a.partial', 0).startswith('put "C:/with space/a.tar"')
    assert sftp_batch('C:/a.tar', '/remote/a.partial', 1024).startswith('reput ')
    with pytest.raises(ValueError):
        sftp_batch('C:/a.tar\nrm /other', '/remote/a.partial', 0)
