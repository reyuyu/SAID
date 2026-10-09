import copy
from pathlib import Path
import pytest

from recovery import hns_s12_2434_validation as r


def test_exact_two_nodes_continuation_not_fresh_or_third_arm():
    assert r.TARGETS==(1217,2434)
    assert r.predecessor(1217)==(r.PARENT,500)
    assert r.predecessor(2434)==(r.segment(1217)/'training/step001217.pt',1217)
    for target in r.TARGETS:
        command=r.training_command(target)
        assert command[command.index('--resume')+1]==str(r.predecessor(target)[0])
        assert command[command.index('--max-updates')+1]==str(target)
        assert command[command.index('--image-root')+1]=='/root/said_s02_stage500/ShareGPT4V'
        assert command[command.index('--index-dir')+1]=='/root/said_s02_stage500/data_index'
        assert command[command.index('--init-state')+1]==str(r.STEP0)
        assert '--expected-parent-trainer-sha256' not in command
    for invalid in (500,501,3651,4868):
        with pytest.raises(AssertionError):r.predecessor(invalid)


@pytest.mark.parametrize('key,value',[
    ('lambda_align',8),('lambda_sparse',1),('lambda_hierarchy',4),
    ('inclusion_max',1),('sampling_mode','nested_detail_kr234'),
    ('view_weights',[1,1,1]),('workers',4),('seed',1),('hns_enabled',False)])
def test_frozen_rejects_scientific_and_loader_drift(key,value):
    config=r.common.read(r.BASE_EXP/'config.json');r.frozen(config)
    config[key]=value
    with pytest.raises(AssertionError):r.frozen(config)


def test_resume_gate_requires_every_rank_all_five_batches(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'RUN',tmp_path/'runtime');monkeypatch.setattr(r,'EXP',tmp_path/'exp')
    node=r.segment(1217);node.mkdir(parents=True)
    r.dump(node/'RESTORE_AUDIT.json',dict(passed=True))
    for step in range(501,506):
        for rank in range(4):
            for kind in ('batch','lr'):r.dump(node/f'{kind}-{step}-rank{rank}.json',dict(passed=True,step=step,rank=rank))
    assert r.resume_gate(1217)['records']==5120
    (node/'lr-505-rank3.json').unlink()
    with pytest.raises(FileNotFoundError):r.resume_gate(1217)


def test_queue_evaluates_and_reports_before_next_segment(monkeypatch,tmp_path):
    from unittest.mock import Mock
    from tools import eval_five_parallel
    monkeypatch.setattr(r,'RUN',tmp_path/'runtime');monkeypatch.setattr(r,'EXP',tmp_path/'exp')
    monkeypatch.setattr(r.local,'IMAGES',tmp_path/'images')
    r.RUN.mkdir();r.dump(r.EXP/'STATE.json',dict(status='PREPARED'));r.dump(r.EXP/'CPU_TESTS.json',dict(passed=True))
    events=[]
    class Supervisor:
        def execute(self,name,command,training=False):events.append((name,tuple(command)))
    monkeypatch.setattr(r.local,'Supervisor',Supervisor)
    monkeypatch.setattr(eval_five_parallel,'require_gpu_idle',lambda _:None)
    for n in r.TARGETS:
        r.segment(n).mkdir();r.dump(r.segment(n)/'training/acceptance.json',dict(passed=True,ranks=[
            dict(completed_updates=n,updates_this_run=n-r.predecessor(n)[1],max_parameter_difference_from_rank0=0)]*4))
    monkeypatch.setattr(r,'identity',lambda p,n:dict(sha256=r.PARENT_SHA))
    monkeypatch.setattr(r,'resume_gate',lambda n:events.append(('gate',n)))
    monkeypatch.setattr(r,'evaluate',lambda s,n:events.append(('eval',n)) or {})
    monkeypatch.setattr(r,'report',lambda n,s,v:events.append(('report',n)))
    monkeypatch.setattr(r,'combined',lambda n:None)
    monkeypatch.setattr(r,'publish',lambda:events.append(('publish',None)))
    r.run()
    assert [e[0] for e in events]==['train','gate','gradient','eval','report','publish']*2
    assert r.common.read(r.RUN/'completed.json')['stop']==2434


def test_first_segment_evaluation_failure_blocks_second(monkeypatch,tmp_path):
    from tools import eval_five_parallel
    monkeypatch.setattr(r,'RUN',tmp_path/'runtime');monkeypatch.setattr(r,'EXP',tmp_path/'exp')
    monkeypatch.setattr(r.local,'IMAGES',tmp_path/'images')
    r.RUN.mkdir();r.dump(r.EXP/'STATE.json',dict(status='PREPARED'));r.dump(r.EXP/'CPU_TESTS.json',dict(passed=True))
    commands=[]
    class Supervisor:
        def execute(self,name,command,training=False):commands.append(name)
    monkeypatch.setattr(r.local,'Supervisor',Supervisor)
    monkeypatch.setattr(eval_five_parallel,'require_gpu_idle',lambda _:None)
    r.segment(1217).mkdir();r.dump(r.segment(1217)/'training/acceptance.json',dict(passed=True,ranks=[
        dict(completed_updates=1217,updates_this_run=717,max_parameter_difference_from_rank0=0)]*4))
    monkeypatch.setattr(r,'identity',lambda p,n:dict(sha256=r.PARENT_SHA))
    monkeypatch.setattr(r,'resume_gate',lambda _:None)
    def fail(*args):raise RuntimeError('evaluator failed')
    monkeypatch.setattr(r,'evaluate',fail)
    with pytest.raises(RuntimeError,match='evaluator failed'):r.run()
    assert commands==['train','gradient']
    assert r.common.read(r.EXP/'STATE.json')['status']=='STOPPED_WITH_EVIDENCE'
