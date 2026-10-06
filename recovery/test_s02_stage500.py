import hashlib
import threading
import pytest
from recovery import s02_stage500 as stage


def snapshot(memory=10 * stage.GIB, oom=0, mempsi=0, iopsi=0):
    return dict(memory_current=memory, memory_events=dict(oom=0, oom_kill=oom, under_oom=0),
                memory_PSI=dict(some=dict(avg10=mempsi)), io_PSI=dict(full=dict(avg10=iopsi)))


def test_stream_atomic_copy_hash_and_no_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(stage, 'LOCAL', tmp_path / 'ssd')
    source, target = tmp_path / 'source', tmp_path / 'ssd/images/example.jpg'
    source.write_bytes(bytes(range(256)) * 8200)
    result = stage.atomic_copy(source, target, source.stat().st_size)
    assert result['source_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert target.read_bytes() == source.read_bytes()
    assert not target.with_name('example.jpg.incomplete').exists()
    assert not stage.atomic_copy(source, target, source.stat().st_size)['copied']
    target.write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match='mismatch'):
        stage.atomic_copy(source, target, source.stat().st_size)


def test_pause_preserves_partial_without_publishing(tmp_path, monkeypatch):
    monkeypatch.setattr(stage, 'LOCAL', tmp_path / 'ssd')
    source, target = tmp_path / 'source', tmp_path / 'ssd/image.jpg'
    source.write_bytes(b'image')
    stop = threading.Event()
    stop.set()
    with pytest.raises(InterruptedError):
        stage.atomic_copy(source, target, 5, stop)
    assert not target.exists()
    assert target.with_name('image.jpg.incomplete').exists()


def test_symlink_and_source_size_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(stage, 'LOCAL', tmp_path / 'ssd')
    stage.LOCAL.mkdir()
    source = tmp_path / 'source'
    source.write_bytes(b'image')
    target = stage.LOCAL / 'image.jpg'
    target.symlink_to(source)
    with pytest.raises(RuntimeError, match='Unsafe'):
        stage.atomic_copy(source, target, 5)
    with pytest.raises(RuntimeError, match='size changed'):
        stage.atomic_copy(source, stage.LOCAL / 'other.jpg', 4)


def test_high_memory_requires_continuous_five_minutes():
    stop = threading.Event()
    guard = stage.Guard(snapshot(), 100, stop)
    assert guard.check(snapshot(471 * stage.GIB), 0) is None
    assert guard.check(snapshot(471 * stage.GIB), 299) is None
    assert guard.check(snapshot(), 300) is None
    assert guard.check(snapshot(471 * stage.GIB), 301) is None
    assert guard.check(snapshot(471 * stage.GIB), 601)
    assert stop.is_set()


@pytest.mark.parametrize('current,kernel', [
    (snapshot(oom=1), []), (snapshot(mempsi=21), []), (snapshot(iopsi=51), []),
    (snapshot(), [(101, 'nfs: server not responding')]),
])
def test_immediate_resource_pause(current, kernel):
    stop = threading.Event()
    guard = stage.Guard(snapshot(), 100, stop)
    assert guard.check(current, 10, kernel)
    assert stop.is_set()


def test_historical_kernel_noise_and_cache_limit_hits_do_not_mean_oom():
    stop = threading.Event()
    guard = stage.Guard(snapshot(), 100, stop)
    current = snapshot()
    current['memory_events']['limit_hits_cumulative'] = 999999
    assert guard.check(current, 10, [(99, 'nfs: server not responding')]) is None
    assert not stop.is_set()


@pytest.mark.parametrize('escape', [False, True])
def test_local_dataset_rejects_missing_or_symlink_before_native_read(tmp_path, monkeypatch, escape):
    import json
    from recovery.s02_train500 import StrictLocalDataset
    from train.nested_semantic_data import NestedDataset
    mirror = tmp_path / 'ssd'
    mirror.mkdir()
    if escape:
        outside = tmp_path / 'nfs-image.jpg'
        outside.write_bytes(b'image')
        (mirror / 'image.jpg').symlink_to(outside)
    dataset = object.__new__(StrictLocalDataset)
    payload = json.dumps(dict(image='image.jpg')).encode()
    dataset._records, dataset._offsets = payload, [0, len(payload)]
    dataset.image_root = mirror
    calls = []
    monkeypatch.setattr(NestedDataset, '__getitem__', lambda *_: calls.append(True))
    with pytest.raises(RuntimeError, match='NFS fallback forbidden'):
        dataset[0]
    assert calls == []


def test_local_dataset_preserves_native_result(tmp_path, monkeypatch):
    import json
    from recovery.s02_train500 import StrictLocalDataset
    from train.nested_semantic_data import NestedDataset
    (tmp_path / 'image.jpg').write_bytes(b'image')
    dataset = object.__new__(StrictLocalDataset)
    payload = json.dumps(dict(image='image.jpg')).encode()
    dataset._records, dataset._offsets = payload, [0, len(payload)]
    dataset.image_root, dataset._path_proofs = tmp_path, 8
    expected = dict(image=object(), sample_id=1000)
    monkeypatch.setattr(NestedDataset, '__getitem__', lambda *_: expected)
    assert dataset[0] is expected
