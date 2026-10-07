"""Literal half inclusion, frozen stream/schedule, three-point diagnostics/report."""
import copy
import json

import pytest
import torch

from model import balanced_hparam_search as objective
from recovery import nested_d3_inc05_500 as experiment
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


def test_only_inclusion_max_changes_and_one_fresh_arm(configured):
    base=json.loads((experiment.BASE_EXP/'config.json').read_text());cfg=search.arm_config(experiment.ARM)
    assert {k for k in base.keys()|cfg.keys() if base.get(k)!=cfg.get(k)}=={'inclusion_max'}
    assert cfg['inclusion_max']==.5 and search.arm_sparsity(experiment.ARM)==[1.,2.,2.]
    assert cfg['sampling_mode']=='nested_detail_d3' and cfg['view_weights']==[1.35,1.35,.30]
    search.frozen_config(cfg,experiment.ARM)
    for key,value in [('inclusion_max',0),('inclusion_max',.75),('sampling_mode','nested_detail_kr234'),('workers',4),
                      ('view_sparsity_weights',[1,2,3]),('view_weights',[1,1,1]),('sparsity_scale',.5)]:
        with pytest.raises(AssertionError):search.frozen_config(dict(cfg,**{key:value}),experiment.ARM)
    assert list(runner.ARMS)==['INC0.5'] and runner.summarize is experiment.summarize
    assert runner.ENTRY=='recovery.nested_d3_inc05_500'


@pytest.mark.parametrize('completed',[0,137,199,200,499])
def test_half_inclusion_only_loss_gradient_change_and_original_ramp(monkeypatch,completed):
    torch.manual_seed(5522)
    base=reference.make(dict(view_weights=[1.35,1.35,.3]));base.inclusion_hierarchy='detail_chain'
    half=copy.deepcopy(base);half.search_hparams['inclusion_max']=.5
    explicit=copy.deepcopy(half);images,tokens,valid=reference.inputs()
    old,old_logs=base(images,*tokens,valid,completed)
    actual,logs=half(images,*tokens,valid,completed)
    assert logs['inclusion_enabled'] is True
    assert logs['inc_weight']==.5*old_logs['inc_weight']==.5*min(1.,completed/200.)
    torch.testing.assert_close(logs['inclusion_loss'],logs['inc_weight']*logs['inc'],atol=0,rtol=0)
    monkeypatch.setattr(reference,'inclusion',objective.detail_chain_inclusion)
    manual=reference.weighted_reference(explicit,images,tokens,valid,completed)
    torch.testing.assert_close(actual,manual,atol=3e-4,rtol=3e-5)
    torch.testing.assert_close(actual-old,-.5*old_logs['inc_weight']*old_logs['inc'],atol=3e-4,rtol=3e-5)
    for k in ('inc','F_sparse','O_sparse','E_sparse','F_Dall_mask_iou','Dall_Ds_mask_iou',
              'Dall_F_hard_violation','Ds_Dall_hard_violation','F_i2t','F_t2i','O_i2t','O_t2i','E_i2t','E_t2i'):
        torch.testing.assert_close(logs[k],old_logs[k],atol=0,rtol=0)
    actual.backward();manual.backward()
    for name,p in half.named_parameters():
        q=dict(explicit.named_parameters())[name];assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=5e-4,rtol=5e-5,msg=name)
    assert half.optimizer_groups()[0]['peak_lr']==base.optimizer_groups()[0]['peak_lr']


def test_detached_child_two_edges_unchanged():
    f=torch.tensor([[.1,.4]],requires_grad=True);d=torch.tensor([[.3,.2]],requires_grad=True)
    low=torch.tensor([[.5,.6]],requires_grad=True)
    (.5*objective.detail_chain_inclusion(f,d,low).sum()).backward()
    assert low.grad is None
    torch.testing.assert_close(f.grad,torch.tensor([[-.125,0.]]))
    torch.testing.assert_close(d.grad,torch.tensor([[-.125,-.125]]))


def frozen_pair():
    health=[]
    for rank in range(4):
        ids=list(range(rank*256+1000,(rank+1)*256+1000));ns=[6]*256
        choices=[search.expected_indices(n,sid,experiment.ARM) for n,sid in zip(ns,ids)]
        s={k:'frozen:'+k for k in search.FULL_STREAM}
        s.update(sample_ids=ids,n=ns,K=[len(c) for c in choices],nested_d3_exact=True,
                 lowest_selected_sentence_indices_sha256=search.indices_digest(ids,choices))
        health.append(dict(rank=rank,sampling=s,gradients_finite=True,batch=256,updates=1))
    a=dict(step=1,epoch=0,actual_lrs=[1e-6,1e-3,1e-3,2e-4],loss=1.,nonfinite=0,
        inc=.02,inc_weight=0.,inclusion_loss=0.,inclusion_enabled=True,rank_health=health)
    b=copy.deepcopy(a)
    for h in b['rank_health']:
        s=h['sampling'];s['D3_selected_sentence_indices_sha256']=s.pop('lowest_selected_sentence_indices_sha256')
    return a,b


def test_fixed_K3_stream_and_half_ramp_gate(configured):
    a,b=frozen_pair();assert search.matched_stream([a],[b],experiment.ARM)['records']==1024
    for key in search.FULL_STREAM+('K','lowest_selected_sentence_indices_sha256'):
        bad=copy.deepcopy(a);bad['rank_health'][0]['sampling'][key]='drift'
        with pytest.raises(AssertionError):search.matched_stream([bad],[b],experiment.ARM)
    for key,value in [('inc_weight',.5),('inclusion_loss',.1),('inclusion_enabled',False)]:
        with pytest.raises(AssertionError):search.matched_stream([dict(a,**{key:value})],[b],experiment.ARM)
    bad=copy.deepcopy(a);bad['actual_lrs'][0]*=2
    with pytest.raises(AssertionError):search.matched_stream([bad],[b],experiment.ARM)
    for step in (100,200,201,500):
        a['step']=b['step']=step
        for row in (a,b):
            for h in row['rank_health']:h['updates']=step
        b['inc_weight']=min(1.,(step-1)/200.)
        a['inc_weight']=.5*b['inc_weight'];a['inclusion_loss']=a['inc_weight']*a['inc']
        assert search.matched_stream([a],[b],experiment.ARM)['passed']


def test_checkpoint_only_inclusion_metadata_changes_optimizer_scheduler_frozen(configured,monkeypatch):
    monkeypatch.setattr(search,'ARM',experiment.ARM);cfg=search.arm_config(experiment.ARM)
    cfg.update(start_updates=0,resume=None,init_sha256=search.STEP0_SHA,max_updates=500,run_type='formal',
        runtime_model=dict(search_hparams=objective.hparams(cfg)),
        code_sha256={p:search.sha(search.ROOT/p) for p in search.EDITED})
    old=copy.deepcopy(cfg);old['runtime_model']['search_hparams']['inclusion_max']=1
    old['code_sha256']={p:'old' for p in search.EDITED}
    opt=dict(param_groups=[],state={0:{'step':torch.tensor(5.)}})
    payload=dict(config=cfg,completed_steps=5,scheduler_horizon=4868,optimizer=opt,
        scheduler=dict(completed_steps=5,horizon=4868),data_cursor=dict(next_epoch=0,next_batch=5),
        model={'w':torch.ones(2)},adapter={'w':torch.ones(2)})
    ref=dict(payload,config=old);assert search.checkpoint_invariants(payload,ref)['passed']
    bad=copy.deepcopy(payload);bad['scheduler']['horizon']=500
    with pytest.raises(AssertionError):search.checkpoint_invariants(bad,ref)
    bad=copy.deepcopy(payload);bad['config']['runtime_model']['search_hparams']['inclusion_max']=0
    with pytest.raises(AssertionError):search.checkpoint_invariants(bad,ref)


def hierarchy_fixture():
    old={'last50':dict(Dall_F_hard_violation=.0104,D3_Dall_hard_violation=.0168,F_Dall_mask_iou=.9659,Dall_D3_mask_iou=.9001)}
    off={'last50':dict(Dall_F_hard_violation=.0274,D3_Dall_hard_violation=.0413,F_Dall_mask_iou=.9210,Dall_D3_mask_iou=.8499)}
    current={'last50':dict(Dall_F_hard_violation=.015,D3_Dall_hard_violation=.023,F_Dall_mask_iou=.95,Dall_D3_mask_iou=.88),
             'keep_ratio':dict(F=.83,Dall=.82,D3=.77)}
    return experiment.hierarchy_position(current,old,off)


@pytest.mark.parametrize('changes,strong,status',[
    ({'Score5':71.10,'J_long3':74.95,'Urban_T2I':89.6,'Short4':65.30},True,'SOFT_INCLUSION_STRONG_POSITIVE'),
    ({'Score5':71.08,'J_long3':74.95,'Urban_T2I':89.7},False,'SOFT_INCLUSION_POSITIVE'),
    ({'Score5':71.09,'J_long3':74.90,'Urban_T2I':89.4,'Short4':65.5},False,'SOFT_INCLUSION_TRADEOFF'),
    ({'Score5':70.9,'J_long3':74.8,'J_long':83.9,'Short4':65.1,'Urban_I2T':90.8,'Urban_T2I':89.3},False,'NO_IMPROVEMENT'),
])
def test_classification(changes,strong,status):
    base=dict(Score5=71.0636,J_long3=74.954666,J_long=84.025001,Short4=65.227,Urban_I2T=90.9,Urban_T2I=89.7)
    off=dict(Score5=71.158309,J_long3=74.978516,J_long=83.995003,Short4=65.428,Urban_I2T=91.1,Urban_T2I=89.4)
    q=dict(base,**changes);p=hierarchy_fixture();p['clearly_stronger_than_INC0']=strong
    assert experiment.classify(q,base,off,p)==status


def test_three_point_report_uses_actual_reference_schemas(configured,tmp_path,monkeypatch):
    monkeypatch.setattr(experiment,'EXP',tmp_path)
    base=experiment.BASE_EXP;result=json.loads((base/'RESULTS.json').read_text())
    result['quality_delta_vs_anchor_pp']={k:0 for k in experiment.QUALITY}
    diag=json.loads((base/'TRAINING_DIAGNOSTICS.json').read_text());masks=json.loads((base/'MASK_HIERARCHY_AUDIT.json').read_text())
    masks.update(keep_ratio={v:d['keep_ratio'] for v,d in diag['last50_views'].items()},
        delta_vs_anchor={k:0 for k in masks['last50']},inclusion_max=.5,ramp200=True,detached_child=True,
        inclusion_edges=['Dall->F','lowest->Dall'])
    masks['last50']['inc_weight']=.5
    grad=json.loads((base/'GRADIENT_SPOTCHECK.json').read_text());grad['weighted_lowest_Dall_ratio']=grad['weighted_mean_gradient_norms']['D3']/grad['weighted_mean_gradient_norms']['Dall']
    for name,value in [('RESULTS',result),('TRAINING_DIAGNOSTICS',diag),('MASK_HIERARCHY_AUDIT',masks),
                       ('GRADIENT_SPOTCHECK',grad),('VALIDATION',{'passed':True})]:dump(tmp_path/(name+'.json'),value)
    (tmp_path/'REPORT.md').write_text('Synthetic report fixture\n')
    actual=[dict(step=u,inc_weight=.5*min(1.,(u-1)/200.)) for u in range(1,501)]
    monkeypatch.setattr(experiment,'rows',lambda path:actual)
    monkeypatch.setattr(search,'matched_stream',lambda *args:dict(passed=True,records=512000))
    experiment.summarize()
    v=json.loads((tmp_path/'RESULTS.json').read_text())
    assert 'quality_delta_vs_INC0_pp' in v and len(v['recall_delta_vs_INC0_pp'])==5
    assert v['inclusion_loss_audit']['weight_by_update']['201']==.5
    assert v['gradient_delta_vs_references']['Anchor']['weighted_D3_Dall_ratio']==0
    assert v['references']['INC0']['result_sha256']
    assert (tmp_path/'SEARCH_SUMMARY.md').is_file()
