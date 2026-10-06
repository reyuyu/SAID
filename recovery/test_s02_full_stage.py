import hashlib
from pathlib import Path
import threading

import pytest

from recovery import s02_full_stage as stage


def test_single_pass_resume_retained_partial(tmp_path,monkeypatch):
    root=tmp_path/'local'
    root.mkdir()
    source=tmp_path/'source.jpg'
    payload=b'abc123'*10000
    source.write_bytes(payload)
    dest=root/'coco/train2017/a.jpg'
    dest.parent.mkdir(parents=True)
    partial=dest.with_name(dest.name+'.incomplete')
    partial.write_bytes(payload[:9876])
    original=Path.open
    opens=[]
    def counted(path,*args,**kwargs):
        if path==source:
            opens.append(path)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',counted)
    monkeypatch.setattr(stage.os,'fsync',lambda _:pytest.fail('per-file fsync forbidden'))
    result=stage.atomic_copy(source,dest,len(payload),threading.Event(),root)
    assert len(opens)==1
    assert result['resumed_prefix_bytes']==9876
    assert result['source_sha256']==hashlib.sha256(payload).hexdigest()
    assert dest.read_bytes()==payload
    assert not partial.exists()


def test_stop_keeps_partial_never_installs_final(tmp_path):
    root=tmp_path/'local'
    root.mkdir()
    source=tmp_path/'source.jpg'
    source.write_bytes(b'abc')
    dest=root/'sam/images/a.jpg'
    stop=threading.Event()
    stop.set()
    with pytest.raises(InterruptedError):
        stage.atomic_copy(source,dest,3,stop,root)
    assert not dest.exists()
    assert dest.with_name(dest.name+'.part').exists()


def test_corrupt_prefix_preserved(tmp_path):
    root=tmp_path/'local'
    root.mkdir()
    source=tmp_path/'source.jpg'
    source.write_bytes(b'abc123')
    dest=root/'sam/images/a.jpg'
    dest.parent.mkdir(parents=True)
    part=dest.with_name(dest.name+'.part')
    part.write_bytes(b'bad')
    with pytest.raises(RuntimeError,match='prefix differs'):
        stage.atomic_copy(source,dest,6,threading.Event(),root)
    assert part.read_bytes()==b'bad' and not dest.exists()


def test_existing_final_not_recopied(tmp_path):
    dest=tmp_path/'coco/train2017/a.jpg'
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b'original')
    with pytest.raises(RuntimeError,match='missing files only'):
        stage.atomic_copy(tmp_path/'absent',dest,8,threading.Event(),tmp_path)
    assert dest.read_bytes()==b'original'


def test_local_resolver_rejects_NFS_symlink_and_traversal(tmp_path):
    root=tmp_path/'local'
    root.mkdir()
    external=tmp_path/'NFS'
    external.mkdir()
    (root/'sam').symlink_to(external)
    for name in ['../coco/a.jpg','/coco/a.jpg','sam/images/a.jpg']:
        with pytest.raises((RuntimeError,ValueError)):
            stage.local_path(name,root)


def test_worker_selection_prefers_lower_when_gain_under15percent():
    def b(w,s,healthy=True):
        return dict(workers=w,MiB_s=s,healthy=healthy)
    assert stage.choose_workers([b(2,20),b(4,26),b(6,29)])==4
    assert stage.choose_workers([b(2,20),b(4,26),b(6,31),b(8,32)])==6
    assert stage.choose_workers([b(2,20),b(4,30,False)])==2


def test_guard_distinguishes_reclaimable_cache_from_actual_OOM():
    guard=stage.Guard()
    def s(memory=350,oom=0,psi=0):
        return dict(memory_current=memory*stage.GIB,memory_events=dict(oom_kill=oom),
            memory_PSI=dict(full=dict(avg10=psi)),io_PSI=dict(full=dict(avg10=0)))
    assert guard.check(s(),0) is None
    assert guard.check(s(memory=200),59) is None
    assert guard.check(s(),60) is None
    assert guard.check(s(),121)=='memory>300GiB sustained'
    guard=stage.Guard()
    assert guard.check(s(memory=20,oom=1),0).startswith('Actual cgroup OOM')
    guard=stage.Guard()
    assert guard.check(s(memory=20,psi=11),0) is None
    assert guard.check(s(memory=20,psi=11),61)=='severe memory PSI sustained'
