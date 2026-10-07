"""INC0 objective/gradient/schedule exclusion, full stream gate and report contracts."""
import copy
import json

import pytest
import torch

from model import balanced_hparam_search as objective
from recovery import nested_d3_inc0_500 as experiment
from recovery import nested_d3_local_search as search
from recovery import nested_d3_followup500 as runner
from recovery.s02_nfs500 import dump
from tests import test_balanced_hparams as reference


@pytest.fixture
def configured(monkeypatch):
    for name in ('EXP','RUN_ROOT','BRANCH','ARMS','ENTRY_MODULE','PHASE_PREFIX','EXTRA_SOURCES',
                 'ARM','RUN','PHASE','ARM_EXP','CONFIG','ANCHOR_EXP','ANCHOR_RUN'):
        monkeypatch.setattr(search,name,getattr(search,name))
    for name in ('EXP','RUN_ROOT','BRANCH','ARMS','ENTRY','PUBLISH_MESSAGE','MAIN_LOG','IDENTITY','configure','summarize'):
        monkeypatch.setattr(runner,name,getattr(runner,name))
    experiment.configure()


def test_only_inclusion_changes(configured):
    base=json.loads((experiment.BASE_EXP/'config.json').read_text());cfg=search.arm_config('INC0')
    assert {k for k in base.keys()|cfg.keys() if base.get(k)!=cfg.get(k)}=={'inclusion_max'}
    assert cfg['inclusion_max']==0 and search.arm_sparsity('INC0')==[1.,2.,2.]
    assert cfg['sampling_mode']=='nested_detail_d3' and cfg['view_weights']==[1.35,1.35,.30]
    search.frozen_config(cfg,'INC0')
    for key,value in [('inclusion_max',.5),('sampling_mode','nested_detail_kr234'),('workers',4),
                      ('view_sparsity_weights',[1,2,3]),('view_weights',[1,1,1]),('sparsity_scale',.5)]:
        with pytest.raises(AssertionError):search.frozen_config(dict(cfg,**{key:value}),'INC0')
    assert list(runner.ARMS)==['INC0'] and runner.summarize is experiment.summarize
    assert runner.ENTRY=='recovery.nested_d3_inc0_500'


@pytest.mark.parametrize('completed',[0,137,499])
def test_no_inclusion_graph_schedule_or_gradient(monkeypatch,completed):
    torch.manual_seed(8837)
    base=reference.make(dict(view_weights=[1.35,1.35,.3]));base.inclusion_hierarchy='detail_chain'
    off=copy.deepcopy(base);off.search_hparams['inclusion_max']=0.
    explicit=copy.deepcopy(off);images,tokens,valid=reference.inputs()
    old,old_logs=base(images,*tokens,valid,completed)
    called=[];original=objective.detail_chain_inclusion
    def telemetry(*args):
        assert not torch.is_grad_enabled()
        value=original(*args);assert not value.requires_grad
        called.append(True);return value
    def forbidden_schedule(*args):raise AssertionError('INC0 must bypass inclusion schedule')
    monkeypatch.setattr(objective,'detail_chain_inclusion',telemetry)
    monkeypatch.setattr(objective,'inclusion_weight',forbidden_schedule)
    actual,logs=off(images,*tokens,valid,completed)
    assert called==[True] and logs['inc_weight']==logs['inclusion_loss']==0
    assert logs['inclusion_enabled'] is False and torch.isfinite(actual)
    # Independent pair-score/CE/sparsity reference; a constant diagnostic zero
    # creates no inclusion autograd path.
    monkeypatch.setattr(reference,'inclusion',lambda pf,po,pe:torch.zeros_like(pf[:,0],requires_grad=False))
    manual=reference.weighted_reference(explicit,images,tokens,valid,completed)
    torch.testing.assert_close(actual,manual,atol=3e-4,rtol=3e-5)
    torch.testing.assert_close(old-actual,old_logs['inc_weight']*old_logs['inc'],atol=3e-4,rtol=3e-5)
    for k in ('inc','F_sparse','O_sparse','E_sparse','F_Dall_mask_iou','Dall_Ds_mask_iou',
              'Dall_F_hard_violation','Ds_Dall_hard_violation','F_i2t','O_t2i','E_i2t'):
        torch.testing.assert_close(logs[k],old_logs[k],atol=0,rtol=0)
    actual.backward();manual.backward()
    for name,p in off.named_parameters():
        q=dict(explicit.named_parameters())[name];assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=5e-4,rtol=5e-5,msg=name)


def test_zero_allowed_only_for_inclusion():
    assert objective.hparams({'inclusion_max':0})['inclusion_max']==0
    for config in ({'inclusion_max':-1},{'inclusion_max':float('nan')},{'inclusion_max':float('inf')},
                   {'sparsity_scale':0},{'fusion_lr':0}):
        with pytest.raises(AssertionError):objective.hparams(config)


def frozen_step():
    health=[]
    for rank in range(4):
        ids=list(range(rank*256+1000,(rank+1)*256+1000));ns=[6]*256
        choices=[search.expected_indices(n,sid,'INC0') for n,sid in zip(ns,ids)]
        s={k:'frozen:'+k for k in search.FULL_STREAM}
        s.update(sample_ids=ids,n=ns,K=[len(c) for c in choices],nested_d3_exact=True,
                 lowest_selected_sentence_indices_sha256=search.indices_digest(ids,choices))
        health.append(dict(rank=rank,sampling=s,gradients_finite=True,batch=256,updates=1))
    return dict(step=1,epoch=0,actual_lrs=[1e-6,1e-3,1e-3,2e-4],loss=1.,nonfinite=0,
                inc_weight=0.,inclusion_loss=0.,inclusion_enabled=False,rank_health=health)


def test_fixed_K3_stream_and_no_hidden_loss_gate(configured):
    a=frozen_step();b=copy.deepcopy(a)
    for h in b['rank_health']:
        s=h['sampling'];s['D3_selected_sentence_indices_sha256']=s.pop('lowest_selected_sentence_indices_sha256')
    assert search.matched_stream([a],[b],'INC0')['records']==1024
    for k,v in [('inc_weight',.01),('inclusion_loss',.01),('inclusion_enabled',True)]:
        with pytest.raises(AssertionError):search.matched_stream([dict(a,**{k:v})],[b],'INC0')
    for key in search.FULL_STREAM+('K','lowest_selected_sentence_indices_sha256'):
        bad=copy.deepcopy(a);bad['rank_health'][0]['sampling'][key]='drift'
        with pytest.raises(AssertionError):search.matched_stream([bad],[b],'INC0')
    bad=copy.deepcopy(a);bad['actual_lrs'][0]*=2
    with pytest.raises(AssertionError):search.matched_stream([bad],[b],'INC0')


def test_checkpoint_runtime_only_inclusion_difference(configured,monkeypatch):
    monkeypatch.setattr(search,'ARM','INC0');cfg=search.arm_config('INC0')
    cfg.update(start_updates=0,resume=None,init_sha256=search.STEP0_SHA,max_updates=500,run_type='formal',
        runtime_model=dict(search_hparams=objective.hparams(cfg)),
        code_sha256={p:search.sha(search.ROOT/p) for p in search.EDITED})
    old=copy.deepcopy(cfg);old['runtime_model']['search_hparams']['inclusion_max']=1
    old['code_sha256']={p:'old' for p in search.EDITED}
    opt=dict(param_groups=[],state={0:{'step':torch.tensor(5.)}})
    payload=dict(config=cfg,completed_steps=5,scheduler_horizon=4868,optimizer=opt,
        model={'w':torch.ones(2)},adapter={'w':torch.ones(2)})
    assert search.checkpoint_invariants(payload,dict(payload,config=old))['passed']
    bad=copy.deepcopy(payload);bad['config']['runtime_model']['search_hparams']['inclusion_max']=.5
    with pytest.raises(AssertionError):search.checkpoint_invariants(bad,dict(payload,config=old))


@pytest.mark.parametrize('deltas,violation,status',[
    ({},.01,'INCLUSION_NEUTRAL'),
    ({'Score5':.3,'J_long3':.1},.01,'INCLUSION_OVERCONSTRAINED'),
    ({'Score5':-.3,'J_long3':-.3},.01,'INCLUSION_NECESSARY'),
    ({'Score5':-.3,'J_long3':.3},.01,'TRADEOFF'),
])
def test_classification(deltas,violation,status):
    base=dict(Score5=71.0636,J_long3=74.954666,J_long=84.025001,Short4=65.227,Urban_I2T=90.9,Urban_T2I=89.7)
    q={k:v+deltas.get(k,0) for k,v in base.items()}
    old={'last50':dict(Dall_F_hard_violation=.01,D3_Dall_hard_violation=.017)}
    masks=copy.deepcopy(old);masks['last50']['D3_Dall_hard_violation']+=violation
    assert experiment.classify(q,base,masks,old)==status


def test_report_summarization_uses_actual_anchor_schema(configured,tmp_path,monkeypatch):
    monkeypatch.setattr(experiment,'EXP',tmp_path)
    base=experiment.BASE_EXP
    result=json.loads((base/'RESULTS.json').read_text());result['quality_delta_vs_anchor_pp']={}
    diag=json.loads((base/'TRAINING_DIAGNOSTICS.json').read_text())
    masks=json.loads((base/'MASK_HIERARCHY_AUDIT.json').read_text())
    masks.update(keep_ratio={v:d['keep_ratio'] for v,d in diag['last50_views'].items()},
        delta_vs_anchor={k:0. for k in masks['last50']},keep_ratio_delta_vs_anchor={v:0. for v in diag['last50_views']})
    grad=json.loads((base/'GRADIENT_SPOTCHECK.json').read_text())
    grad['weighted_lowest_Dall_ratio']=grad['weighted_mean_gradient_norms']['D3']/grad['weighted_mean_gradient_norms']['Dall']
    for name,value in [('RESULTS',result),('TRAINING_DIAGNOSTICS',diag),('MASK_HIERARCHY_AUDIT',masks),
                       ('GRADIENT_SPOTCHECK',grad),('VALIDATION',{'passed':True})]:dump(tmp_path/(name+'.json'),value)
    (tmp_path/'REPORT.md').write_text('Synthetic report fixture\n')
    experiment.summarize()
    value=json.loads((tmp_path/'RESULTS.json').read_text())
    assert value['classification']=='INCLUSION_NEUTRAL'
    assert value['gradient_delta_vs_anchor']['weighted_D3_Dall_ratio']==0
    assert (tmp_path/'SEARCH_SUMMARY.md').is_file()
