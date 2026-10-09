import copy
import pytest
from recovery import hns_s12_uniform_e1_1217 as r


def test_only_uniform_existing500_native_resume1217():
    r.configure();assert r.protocol.TARGETS==(1217,)
    cmd=r.protocol.training_command(1217)
    for flag,value in [('--resume',str(r.PARENT)),('--max-updates','1217'),('--run-type','formal'),
                       ('--init-state',str(r.STEP0)),('--image-root','/root/said_s02_stage500/ShareGPT4V'),
                       ('--index-dir','/root/said_s02_stage500/data_index')]:
        assert cmd[cmd.index(flag)+1]==value
    assert cmd[cmd.index('-m')+1]==r.ENTRY
    assert '--expected-parent-trainer-sha256' not in cmd and '--legacy-b0' not in cmd
    assert r.predecessor(1217)==(r.PARENT,500)
    for value in (500,505,2434,3651,4868):
        with pytest.raises(AssertionError):r.predecessor(value)


@pytest.mark.parametrize('key,value',[
    ('lambda_align',8),('lambda_sparse',1),('lambda_hierarchy',.5),
    ('view_sparsity_weights',[1,2,2]),('view_weights',[1,1,1]),
    ('inclusion_max',1),('hns_enabled',False),('sampling_mode','nested_detail_kr234'),
    ('workers',4),('seed',1),('sampling_seed',1),('shuffle_seed',1)])
def test_frozen_rejects_any_method_config_change(key,value):
    cfg=r.protocol.common.read(r.BASE_EXP/'config.json');r.frozen(cfg)
    bad=copy.deepcopy(cfg);bad[key]=value
    with pytest.raises(AssertionError):r.frozen(bad)


def test_resume_gate_requires_post_update_every_rank(tmp_path,monkeypatch):
    r.configure();monkeypatch.setattr(r,'EXP',tmp_path/'experiment')
    monkeypatch.setattr(r.protocol,'EXP',r.EXP);monkeypatch.setattr(r.protocol,'RUN',tmp_path/'runtime')
    node=r.protocol.segment(1217);node.mkdir(parents=True)
    r.dump(node/'RESTORE_AUDIT.json',dict(passed=True))
    for s in range(501,506):
        for rank in range(4):
            for kind in ('batch','lr'):r.dump(node/f'{kind}-{s}-rank{rank}.json',dict(passed=True,step=s,rank=rank))
            r.dump(node/f'update-{s}-rank{rank}.json',dict(passed=True,step=s,optimizer_counters=[s],parameter_difference=0))
    assert r.resume_gate(1217)['no_repeated_first500_optimizer_updates']
    (node/'update-505-rank3.json').unlink()
    with pytest.raises(FileNotFoundError):r.resume_gate(1217)


def suffix():
    return [dict(step=s,epoch=0,actual_lrs={'backbone':float(s)},nonfinite=0,
        rank_health=[dict(rank=rank,updates=s-500,batch=180 if s==1217 else 256,
            gradients_finite=True,stream_sha256=f'{s}:{rank}',sampling={'tokens':f'{s}:{rank}'})
            for rank in range(4)]) for s in range(501,1218)]


def test_suffix_proof_includes_all_updates_and_original_short_tail():
    a=suffix();proof=r.stream_proof(a,copy.deepcopy(a))
    assert proof['updates']==717 and proof['records']==733904
    a[-1]['rank_health'][3]['batch']=256
    with pytest.raises(AssertionError):r.stream_proof(a,suffix())


def test_suffix_proof_rejects_text_or_lr_drift():
    for kind in ('text','lr'):
        a=suffix();b=copy.deepcopy(a)
        if kind=='text':a[50]['rank_health'][2]['sampling']['tokens']='drift'
        else:a[50]['actual_lrs']['backbone']=0
        with pytest.raises(AssertionError):r.stream_proof(a,b)


@pytest.mark.parametrize('failure',[False,True])
def test_one_training_evaluation_then_stop_no_automatic_retry(tmp_path,monkeypatch,failure):
    from tools import eval_five_parallel
    r.configure();p=r.protocol
    monkeypatch.setattr(p,'RUN',tmp_path/'runtime');monkeypatch.setattr(p,'EXP',tmp_path/'experiment')
    monkeypatch.setattr(p.local,'IMAGES',tmp_path/'images')
    p.RUN.mkdir();r.dump(p.EXP/'STATE.json',dict(status='PREPARED'));r.dump(p.EXP/'CPU_TESTS.json',dict(passed=True))
    p.segment(1217).mkdir();r.dump(p.segment(1217)/'training/acceptance.json',dict(passed=True,ranks=[
        dict(completed_updates=1217,updates_this_run=717,max_parameter_difference_from_rank0=0)]*4))
    events=[]
    class Supervisor:
        def execute(self,name,command,training=False):events.append(name)
    monkeypatch.setattr(p.local,'Supervisor',Supervisor)
    monkeypatch.setattr(eval_five_parallel,'require_gpu_idle',lambda _:None)
    monkeypatch.setattr(p,'identity',lambda *args:dict(sha256=r.PARENT_SHA))
    monkeypatch.setattr(p,'resume_gate',lambda _:events.append('gate'))
    def evaluate(*args):
        events.append('evaluate')
        if failure:raise RuntimeError('evaluator failure')
        return {}
    monkeypatch.setattr(p,'evaluate',evaluate)
    monkeypatch.setattr(p,'report',lambda *args:events.append('report'))
    monkeypatch.setattr(p,'combined',lambda _:events.append('combined'))
    monkeypatch.setattr(p,'publish',lambda:events.append('publish'))
    if failure:
        with pytest.raises(RuntimeError,match='evaluator failure'):p.run()
        assert events==['train','gate','gradient','evaluate']
        assert p.common.read(p.EXP/'STATE.json')['status']=='STOPPED_WITH_EVIDENCE'
    else:
        p.run();assert events==['train','gate','gradient','evaluate','report','combined','publish']
        assert p.common.read(p.RUN/'completed.json')['stop']==1217
        assert p.common.read(p.EXP/'STATE.json')['status']=='COMPLETED_GPU_IDLE'
