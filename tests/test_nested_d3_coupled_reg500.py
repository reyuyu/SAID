"""Routing, scale, complete objective gradients, frozen stream and report audits."""
import copy
import json

import pytest
import torch
from torch.nn import functional as F

from model import balanced_hparam_search as objective
from model.nested_semantic_mask import hard_st
from recovery import nested_d3_coupled_reg500 as experiment
from recovery import nested_d3_local_search as search
from recovery import nested_d3_followup500 as runner
from recovery.s02_nfs500 import dump
from tests import test_balanced_hparams as reference
from tests.test_nested_fusion import explicit_inputs,explicit_logits


@pytest.fixture
def configured(monkeypatch):
    for name in ('EXP','RUN_ROOT','BRANCH','ARMS','ENTRY_MODULE','PHASE_PREFIX','EXTRA_SOURCES',
                 'ARM','RUN','PHASE','ARM_EXP','CONFIG','ANCHOR_EXP','ANCHOR_RUN'):
        monkeypatch.setattr(search,name,getattr(search,name))
    for name in ('EXP','RUN_ROOT','BRANCH','ARMS','ENTRY','PUBLISH_MESSAGE','MAIN_LOG','IDENTITY',
                 'configure','summarize','REPORT_NAMES'):
        monkeypatch.setattr(runner,name,getattr(runner,name))
    experiment.configure()


def test_outside_child_detached_parent_gradient_and_inside_reverse():
    audit=experiment.routing_example()
    assert audit['outside_child_gradient'] is None and audit['inside_parent_gradient'] is None
    assert audit['outside_parent_gradient']==[[-.5,0.]]
    assert audit['inside_child_gradient'][0][1]>audit['inside_child_gradient'][0][0]>0


def test_relative_low_parent_support_and_broad_high_support():
    parent=torch.tensor([[.999,.999,.001]])
    low=torch.tensor([[0.,0.,1.]])
    high=torch.tensor([[1.,1.,0.]])
    narrow=torch.tensor([[1.,0.,0.]])
    low_in,_=objective.nested_edge_terms(low,parent)
    high_in,_=objective.nested_edge_terms(high,parent)
    narrow_in,_=objective.nested_edge_terms(narrow,parent)
    assert low_in.item()<.001 and high_in.item()>.99 and high_in>narrow_in


@pytest.mark.parametrize('weight',[0.,.495,.995,1.])
def test_outside_exact_anchor_total_scale_and_no_child_sparse_arguments(weight):
    pf=torch.tensor([[.1,.6],[.7,.2]],requires_grad=True)
    d=torch.tensor([[.4,.2],[.8,.4]],requires_grad=True)
    low=torch.tensor([[.7,.1],[.9,.8]],requires_grad=True)
    valid=torch.ones(2,dtype=torch.bool);sf=torch.tensor(.75,requires_grad=True)
    total,terms=objective.coupled_nested_terms(sf,pf,d,low,valid,weight,1,2)
    old=weight*objective.detail_chain_inclusion(pf,d,low).mean()
    torch.testing.assert_close(terms['total_outside_penalty'],old,atol=0,rtol=0)
    inside1=(pf.detach()*d).sum(-1)/(pf.detach().sum(-1)+1e-6)
    inside2=(d.detach()*low).sum(-1)/(d.detach().sum(-1)+1e-6)
    torch.testing.assert_close(total,sf/3+2/3*(inside1.mean()+inside2.mean())+old)


def manual_objective(module,images,views,valid,completed):
    pairs=[];probabilities=[]
    for i,tokens in enumerate(views if valid.sum()>=2 else views[:1]):
        z,t,zv,zt=explicit_inputs(module,images,tokens)
        prob=explicit_logits(module,zv,zt).sigmoid();mask=hard_st(prob)
        scores=100*(F.normalize(z[None]*mask,dim=-1,eps=1e-6)*F.normalize(t,dim=-1,eps=1e-6)[:,None]).sum(-1)
        enabled=torch.ones_like(valid) if i==0 else valid
        labels=torch.arange(len(z))[enabled]
        ce=F.cross_entropy(scores.T[enabled].masked_fill(~enabled[None],-torch.inf),labels)
        ce+=F.cross_entropy(scores[enabled].masked_fill(~enabled[None],-torch.inf),labels)
        pairs.append((ce,mask.diagonal(dim1=0,dim2=1).T[enabled].abs().mean()))
        probabilities.append(prob.diagonal(dim1=0,dim2=1).T)
    if valid.sum()<2:return 10*pairs[0][0]+pairs[0][1]
    pf,pd,pl=probabilities
    reg=pairs[0][1]/3
    for child,parent in [(pd,pf),(pl,pd)]:
        inside=((parent.detach()*child).sum(-1)/(parent.detach().sum(-1)+1e-6))[valid].mean()
        outside=torch.relu(child.detach()-parent).mean(-1)[valid].mean()
        reg+=2/3*inside+.5*min(1.,completed/200.)*outside
    return 10/3*sum(w*pair[0] for w,pair in zip([1.35,1.35,.30],pairs))+reg


@pytest.mark.parametrize('completed,valid_count',[(0,2),(199,2),(200,2),(499,2),(499,1),(499,0)])
def test_full_objective_all_parameter_gradients_and_original_fallback(completed,valid_count):
    torch.manual_seed(90211)
    module=reference.make(dict(view_weights=[1.35,1.35,.30]))
    module.inclusion_hierarchy='detail_chain';module.regularizer_mode='coupled_nested'
    manual=copy.deepcopy(module);images,views,valid=reference.inputs()
    valid.zero_();valid[:valid_count]=True
    actual,logs=module(images,*views,valid,completed)
    expected=manual_objective(manual,images,views,valid,completed)
    torch.testing.assert_close(actual,expected,atol=3e-4,rtol=3e-5)
    actual.backward();expected.backward()
    for name,p in module.named_parameters():
        q=dict(manual.named_parameters())[name]
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=5e-4,rtol=5e-5,msg=name)
    if valid_count>=2:
        assert logs['independent_child_sparse_applied']==logs['independent_inclusion_applied']==0
        torch.testing.assert_close(logs['total_outside_penalty'],logs['inclusion_loss'])
        ce=sum(w*(logs[p+'_i2t']+logs[p+'_t2i']) for w,p in zip([1.35,1.35,.3],['F','O','E']))*10/3
        torch.testing.assert_close(logs['loss'],ce+logs['total_nested_regularizer'])


def test_old_child_global_sparsity_cannot_change_objective(monkeypatch):
    torch.manual_seed(553)
    module=reference.make(dict(view_weights=[1.35,1.35,.30]))
    module.inclusion_hierarchy='detail_chain';module.regularizer_mode='coupled_nested'
    images,views,valid=reference.inputs();first,_=module(images,*views,valid,200)
    fn=objective.fusion_view_terms;calls=[]
    def changed(*args,**kwargs):
        result=list(fn(*args,**kwargs));calls.append(True)
        if len(calls)>1:result[1]=result[1]+10000.
        return tuple(result)
    monkeypatch.setattr(objective,'fusion_view_terms',changed)
    second,logs=module(images,*views,valid,200)
    torch.testing.assert_close(first,second,atol=0,rtol=0)
    assert logs['old_global_sparse_counterfactual']>1000


def test_DDP_scaling_for_uneven_valid_pairs():
    torch.manual_seed(11)
    pf=torch.rand(6,4,requires_grad=True);pd=torch.rand(6,4,requires_grad=True);pl=torch.rand(6,4,requires_grad=True)
    valid=torch.tensor([1,1,0,0,1,0],dtype=torch.bool)
    omega=torch.rand(6,requires_grad=True)
    full,_=objective.coupled_nested_terms(omega.mean(),pf,pd,pl,valid,.5,1,3)
    parts=[]
    for i in range(3):
        s=slice(i*2,i*2+2)
        part,_=objective.coupled_nested_terms(omega[s].mean(),pf[s],pd[s],pl[s],valid[s],.5,3,3)
        parts.append(part)
    averaged=sum(parts)/3
    torch.testing.assert_close(full,averaged)
    gf=torch.autograd.grad(full,(omega,pf,pd,pl),retain_graph=True)
    gd=torch.autograd.grad(averaged,(omega,pf,pd,pl))
    for a,b in zip(gf,gd):torch.testing.assert_close(a,b)


def test_only_regularizer_config_changes_and_single_fresh_arm(configured):
    base=json.loads((experiment.BASE_EXP/'config.json').read_text());cfg=search.arm_config(experiment.ARM)
    assert {k for k in base.keys()|cfg.keys() if base.get(k)!=cfg.get(k)}=={'regularizer_mode'}
    search.frozen_config(cfg,experiment.ARM)
    assert list(runner.ARMS)==['CoupledNested'] and runner.summarize is experiment.summarize
    assert 'REGULARIZER_AUDIT.json' in runner.REPORT_NAMES
    for key,value in [('inclusion_max',.5),('regularizer_mode','independent'),('workers',4),
                      ('view_weights',[1,1,1]),('view_sparsity_weights',[1,2,3])]:
        with pytest.raises(AssertionError):search.frozen_config(dict(cfg,**{key:value}),experiment.ARM)


def test_constructor_rejects_unreviewed_combinations():
    from tests.test_nested_fusion import TinyFusionCLIP
    cfg=dict(view_weights=[1.35,1.35,.3])
    for kwargs in [dict(inclusion_hierarchy='siblings'),dict(view_sparsity_weights=[1,2,3]),
                   dict(search_hparams=dict(cfg,inclusion_max=.5))]:
        options=dict(search_hparams=cfg,inclusion_hierarchy='detail_chain',regularizer_mode='coupled_nested',
                     fusion='balanced_stack',visual='patch',checkpoint_encoders=False)
        options.update(kwargs)
        with pytest.raises(AssertionError):objective.BalancedSearch(TinyFusionCLIP(32),**options)


def test_full_frozen_trajectory_gate_rejects_text_index_RNG_LR_drift(configured):
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
    a.update(regularizer_mode='coupled_nested',inclusion_folded_into_regularizer=True,
        independent_child_sparse_applied=0.,independent_inclusion_applied=0.,total_outside_penalty=0.,
        Omega_F=.9,total_inside_sparsity=.4,total_nested_regularizer=.7)
    assert search.matched_stream([a],[b],experiment.ARM)['records']==1024
    for key in search.FULL_STREAM+('K','lowest_selected_sentence_indices_sha256'):
        bad=copy.deepcopy(a);bad['rank_health'][0]['sampling'][key]='drift'
        with pytest.raises(AssertionError):search.matched_stream([bad],[b],experiment.ARM)
    for key,value in [('independent_inclusion_applied',.1),('inc_weight',.5),('total_nested_regularizer',.9)]:
        with pytest.raises(AssertionError):search.matched_stream([dict(a,**{key:value})],[b],experiment.ARM)


def test_checkpoint_regularizer_identity_optimizer_and_scheduler_frozen(configured,monkeypatch):
    monkeypatch.setattr(search,'ARM',experiment.ARM)
    cfg=search.arm_config(experiment.ARM)
    cfg.update(start_updates=0,resume=None,init_sha256=search.STEP0_SHA,max_updates=500,run_type='formal',
        runtime_model=dict(search_hparams=objective.hparams(cfg),regularizer_mode='coupled_nested'),
        code_sha256={p:search.sha(search.ROOT/p) for p in search.EDITED})
    old=copy.deepcopy(cfg);old.pop('regularizer_mode');old['runtime_model'].pop('regularizer_mode')
    old['code_sha256']={p:'old' for p in search.EDITED}
    opt=dict(param_groups=[],state={0:{'step':torch.tensor(5.)}})
    payload=dict(config=cfg,completed_steps=5,scheduler_horizon=4868,optimizer=opt,
        model={'w':torch.ones(2)},adapter={'w':torch.ones(2)})
    ref=dict(payload,config=old)
    assert search.checkpoint_invariants(payload,ref)['passed']
    bad=copy.deepcopy(payload);bad['config']['runtime_model']['regularizer_mode']='independent'
    with pytest.raises(AssertionError):search.checkpoint_invariants(bad,ref)


def fake_steps():
    actual=[];ref=[]
    for step in range(1,501):
        lam=min(1.,(step-1)/200.)
        a={k:.2 for k in experiment.REG_KEYS}
        a.update(step=step,valid_global=1000,inc_weight=lam,independent_child_sparse_applied=0.,
            independent_inclusion_applied=0.,F_sparse=.8,O_sparse=.7,E_sparse=.6,Omega_F=.8,
            total_inside_sparsity=4*.2/3,total_outside_penalty=.2*lam,inclusion_loss=.2*lam,
            old_global_sparse_counterfactual=(.8+1.4+1.2)/3)
        a['total_nested_regularizer']=.8/3+a['total_inside_sparsity']+a['total_outside_penalty']
        actual.append(a);ref.append(dict(a))
    return actual,ref


def test_all500_regularizer_identity_audit_and_double_count_rejected():
    a,b=fake_steps();audit=experiment.regularizer_audit(a,b)
    assert audit['passed'] and len(audit['matched_all500'])==500
    a[10]['independent_inclusion_applied']=.001
    with pytest.raises(AssertionError):experiment.regularizer_audit(a,b)


@pytest.mark.parametrize('changes,coverage,status',[
    ({'Score5':71.16,'J_long3':75.,'Urban_T2I':89.6,'Short4':65.3},True,'COUPLED_NESTED_STRONG_POSITIVE'),
    ({'Score5':71.12,'J_long3':75.02,'Urban_T2I':89.7},True,'COUPLED_NESTED_POSITIVE'),
    ({},True,'REGULARIZER_SIMPLIFICATION_POSITIVE'),
    ({'Score5':71.2,'Urban_T2I':89.3},True,'COUPLED_NESTED_TRADEOFF'),
    ({'Score5':70.9,'J_long3':74.8,'Urban_T2I':89.3},True,'COUPLED_NESTED_NEGATIVE'),
    ({},False,'COUPLED_NESTED_TRADEOFF')])
def test_frozen_classification(changes,coverage,status):
    base=dict(Score5=71.0636,J_long3=74.954666,J_long=84.025001,Short4=65.227,Urban_I2T=90.9,Urban_T2I=89.7)
    assert experiment.classify(dict(base,**changes),base,coverage)==status


def test_four_reference_report_uses_actual_baseline_schemas(configured,tmp_path,monkeypatch):
    monkeypatch.setattr(experiment,'EXP',tmp_path)
    base=experiment.BASE_EXP;result=json.loads((base/'RESULTS.json').read_text())
    diag=json.loads((base/'TRAINING_DIAGNOSTICS.json').read_text())
    masks=json.loads((base/'MASK_HIERARCHY_AUDIT.json').read_text())
    masks['keep_ratio']={v:d['keep_ratio'] for v,d in diag['last50_views'].items()}
    grad=json.loads((base/'GRADIENT_SPOTCHECK.json').read_text())
    grad['weighted_lowest_Dall_ratio']=grad['weighted_mean_gradient_norms']['D3']/grad['weighted_mean_gradient_norms']['Dall']
    for name,value in [('RESULTS',result),('TRAINING_DIAGNOSTICS',diag),('MASK_HIERARCHY_AUDIT',masks),
                       ('GRADIENT_SPOTCHECK',grad),('VALIDATION',{'passed':True})]:dump(tmp_path/(name+'.json'),value)
    (tmp_path/'REPORT.md').write_text('Synthetic fixture\n')
    a,b=fake_steps();monkeypatch.setattr(experiment,'rows',lambda p:a if 'CoupledNested' in str(p) else b)
    monkeypatch.setattr(search,'matched_stream',lambda *a:dict(records=512000))
    experiment.summarize()
    value=json.loads((tmp_path/'RESULTS.json').read_text())
    assert set(value['references'])=={'Anchor','INC0','INC0.5'}
    assert len(value['recall_delta_vs_references_pp']['INC0'])==5
    assert (tmp_path/'REGULARIZER_AUDIT.json').stat().st_size<1024*1024
    assert value['regularizer_audit_passed'] and (tmp_path/'SEARCH_SUMMARY.md').exists()
