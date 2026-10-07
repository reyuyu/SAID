"""BBNS routing, hinges, independent full gradients, audit/config/stream/report."""
import copy
import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from model import balanced_hparam_search as objective
from model.nested_support_band import bbns_regularizer,support_band_edge,validate_bands
from model.nested_semantic_mask import hard_st
from recovery import nested_d3_bbns500 as experiment
from recovery import nested_d3_local_search as search
from recovery import nested_d3_followup500 as runner
from recovery.nested_d3_support_audit import distribution,bands_from_distributions,support_vectors,KEYS
from recovery.s02_nfs500 import dump
from tests import test_balanced_hparams as reference
from tests.test_nested_fusion import explicit_inputs,explicit_logits

BAND=dict(kappa=.8,tau_low=.4,tau_high=.6)
BANDS={k:dict(BAND) for k in ('Dall_F','D3_Dall')}


@pytest.fixture
def configured(monkeypatch):
    for name in ('EXP','RUN_ROOT','BRANCH','ARMS','ENTRY_MODULE','PHASE_PREFIX','EXTRA_SOURCES',
                 'ARM','RUN','PHASE','ARM_EXP','CONFIG','ANCHOR_EXP','ANCHOR_RUN'):
        monkeypatch.setattr(search,name,getattr(search,name))
    for name in ('EXP','RUN_ROOT','BRANCH','ARMS','ENTRY','PUBLISH_MESSAGE','MAIN_LOG','IDENTITY',
                 'configure','summarize','REPORT_NAMES'):
        monkeypatch.setattr(runner,name,getattr(runner,name))
    experiment.configure()


def test_manual_coverage_refinement_gradient_routing():
    audit=experiment.routing_audit();assert audit['passed']
    for case in audit['cases'].values():
        assert case['cover_child_gradient'] is None and case['refine_parent_gradient'] is None
    assert audit['cases']['upper']['cover_loss']>0
    assert audit['cases']['inside']['cover_loss']==0


@pytest.mark.parametrize('child,parent,cover,refine,direction',[
    (.2,.9,False,True,-1),(.9,.3,True,True,1),(.5,.9,False,False,0)])
def test_coverage_and_band_loss_signs(child,parent,cover,refine,direction):
    c=torch.full((2,5),child,requires_grad=True);p=torch.full((2,5),parent,requires_grad=True)
    v=support_band_edge(c,p,BAND)
    assert bool((v['cover_loss']>0).all())==cover
    r=v['lower_band_loss']+v['upper_band_loss']
    assert bool((r>0).all())==refine
    c_grad,p_grad=torch.autograd.grad(r.sum(),(c,p),allow_unused=True)
    assert p_grad is None
    assert torch.equal(c_grad.sign(),torch.full_like(c_grad,direction))


def manual_objective(module,images,views,valid):
    pairs=[];probabilities=[]
    for i,tokens in enumerate(views if valid.sum()>=2 else views[:1]):
        z,t,zv,zt=explicit_inputs(module,images,tokens)
        prob=explicit_logits(module,zv,zt).sigmoid();mask=hard_st(prob)
        scores=100*(F.normalize(z[None]*mask,dim=-1,eps=1e-6)*F.normalize(t,dim=-1,eps=1e-6)[:,None]).sum(-1)
        enabled=torch.ones_like(valid) if i==0 else valid;labels=torch.arange(len(z))[enabled]
        ce=F.cross_entropy(scores.T[enabled].masked_fill(~enabled[None],-torch.inf),labels)
        ce+=F.cross_entropy(scores[enabled].masked_fill(~enabled[None],-torch.inf),labels)
        pairs.append((ce,mask.diagonal(dim1=0,dim2=1).T[enabled].abs().mean()))
        probabilities.append(prob.diagonal(dim1=0,dim2=1).T)
    if valid.sum()<2:return 10*pairs[0][0]+pairs[0][1]/3
    pf,pd,pl=probabilities;reg=pairs[0][1]/3
    for name,c,p in [('Dall_F',pd,pf),('D3_Dall',pl,pd)]:
        band=module.support_bands[name]
        C=(c.detach()*p).sum(-1)/(c.detach().sum(-1)+1e-6)
        R=(c*p.detach()).sum(-1)/(p.detach().sum(-1)+1e-6)
        penalty=torch.relu(band['kappa']-C).square()+torch.relu(band['tau_low']-R).square()+torch.relu(R-band['tau_high']).square()
        reg+=penalty[valid].mean()
    return 10/3*sum(w*pair[0] for w,pair in zip([1.35,1.35,.30],pairs))+reg


@pytest.mark.parametrize('valid_count,completed',[(2,0),(2,199),(2,499),(1,499),(0,499)])
def test_full_objective_all_parameter_gradients_and_no_old_schedule(monkeypatch,valid_count,completed):
    torch.manual_seed(50211)
    module=reference.make(dict(view_weights=[1.35,1.35,.30]))
    module.inclusion_hierarchy='detail_chain';module.regularizer_mode='bbns';module.support_bands=BANDS
    explicit=copy.deepcopy(module);images,views,valid=reference.inputs();valid.zero_();valid[:valid_count]=True
    def forbidden(*args):raise AssertionError('BBNS must never evaluate old inclusion ramp')
    monkeypatch.setattr(objective,'inclusion_weight',forbidden)
    actual,logs=module(images,*views,valid,completed)
    expected=manual_objective(explicit,images,views,valid)
    torch.testing.assert_close(actual,expected,atol=3e-4,rtol=3e-5)
    assert logs['inc_weight']==logs['inclusion_loss']==0 and logs['inclusion_enabled'] is False
    assert logs['independent_child_sparse_applied']==logs['independent_inclusion_applied']==0
    actual.backward();expected.backward()
    for name,p in module.named_parameters():
        q=dict(explicit.named_parameters())[name];assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=5e-4,rtol=5e-5,msg=name)


def test_DDP_normalization_uneven_valid_and_F_over_three():
    torch.manual_seed(10)
    pf=torch.rand(6,4,requires_grad=True);pd=torch.rand(6,4,requires_grad=True);pl=torch.rand(6,4,requires_grad=True)
    omega=torch.rand(6,requires_grad=True);valid=torch.tensor([1,0,0,0,1,1],dtype=torch.bool)
    full,_=bbns_regularizer(omega.mean(),pf,pd,pl,valid,BANDS,1,3)
    parts=[]
    for i in range(3):
        s=slice(2*i,2*i+2)
        p,_=bbns_regularizer(omega[s].mean(),pf[s],pd[s],pl[s],valid[s],BANDS,3,3)
        parts.append(p)
    merged=sum(parts)/3;torch.testing.assert_close(full,merged)
    gf=torch.autograd.grad(full,(omega,pf,pd,pl),retain_graph=True);gd=torch.autograd.grad(merged,(omega,pf,pd,pl))
    for a,b in zip(gf,gd):torch.testing.assert_close(a,b)


def test_old_child_sparse_and_inclusion_are_not_additional_losses(monkeypatch):
    torch.manual_seed(519)
    module=reference.make(dict(view_weights=[1.35,1.35,.30]))
    module.inclusion_hierarchy='detail_chain';module.regularizer_mode='bbns';module.support_bands=BANDS
    images,views,valid=reference.inputs();first,_=module(images,*views,valid,200)
    fn=objective.fusion_view_terms;calls=[]
    def changed(*args,**kwargs):
        result=list(fn(*args,**kwargs));calls.append(True)
        if len(calls)>1:result[1]=result[1]+10000.
        return tuple(result)
    def telemetry(*args):
        assert not torch.is_grad_enabled()
        return torch.full((3,),10000.)
    monkeypatch.setattr(objective,'fusion_view_terms',changed)
    monkeypatch.setattr(objective,'detail_chain_inclusion',telemetry)
    second,_=module(images,*views,valid,200)
    torch.testing.assert_close(first,second,atol=0,rtol=0)


def test_audit_population_stats_soft_probabilities_and_percentiles():
    d=distribution(np.arange(101)/100)
    assert d['P20']==.2 and d['P80']==.8 and d['median']==.5
    torch.testing.assert_close(torch.tensor(d['std']),torch.tensor(np.std(np.arange(101)/100)).float())
    f=torch.tensor([[.9,.2]]);dall=torch.tensor([[.5,.4]]);d3=torch.tensor([[.2,.3]])
    v=support_vectors(f,dall,d3);assert v.shape==(1,len(KEYS))
    values=dict(zip(KEYS,v[0].tolist()))
    assert values['C_Dall_F']==pytest.approx((.45+.08)/(.9+1e-6))
    assert values['R_Dall_F']==pytest.approx((.45+.08)/(1.1+1e-6))
    assert values['keep_Dall']==.5 and values['keep_D3']==0


def test_exact_P20_P80_config_and_one_fresh_arm(configured):
    audit=json.loads((experiment.EXP/'ANCHOR_SUPPORT_AUDIT.json').read_text())
    cfg=search.arm_config('BBNS');base=json.loads((experiment.BASE_EXP/'config.json').read_text())
    assert {k for k in base.keys()|cfg.keys() if base.get(k)!=cfg.get(k)}=={'regularizer_mode','support_bands'}
    assert cfg['support_bands']==bands_from_distributions(audit['soft_distributions'])==audit['frozen_bands']
    assert audit['records']==16384 and audit['loader_exhausted_normally']
    assert cfg==json.loads(experiment.CONFIG.read_text())
    assert list(runner.ARMS)==['BBNS'] and runner.summarize is experiment.summarize
    search.frozen_config(cfg,'BBNS')
    altered=copy.deepcopy(cfg);altered['support_bands']['Dall_F']['kappa']+=.01
    with pytest.raises(AssertionError):search.frozen_config(altered,'BBNS')
    for key,value in [('workers',4),('sampling_mode','nested_detail_kr234'),('view_weights',[1,1,1])]:
        with pytest.raises(AssertionError):search.frozen_config(dict(cfg,**{key:value}),'BBNS')


def test_bands_reject_invalid_or_unfrozen_shapes():
    for band in ({}, {'Dall_F':BAND},dict(BANDS,Dall_F=dict(BAND,tau_low=.9)),
                 dict(BANDS,D3_Dall=dict(BAND,kappa=float('nan')))):
        with pytest.raises(AssertionError):validate_bands(band)


def frozen_step():
    health=[]
    for rank in range(4):
        ids=list(range(rank*256+1000,(rank+1)*256+1000));ns=[6]*256
        choices=[search.expected_indices(n,sid,'BBNS') for n,sid in zip(ns,ids)]
        s={k:'frozen:'+k for k in search.FULL_STREAM}
        s.update(sample_ids=ids,n=ns,K=[len(c) for c in choices],nested_d3_exact=True,
                 lowest_selected_sentence_indices_sha256=search.indices_digest(ids,choices))
        health.append(dict(rank=rank,sampling=s,gradients_finite=True,batch=256,updates=1))
    a=dict(step=1,epoch=0,actual_lrs=[1e-6,1e-3,1e-3,2e-4],loss=1.,nonfinite=0,inc_weight=0.,
        inclusion_loss=0.,inclusion_enabled=False,regularizer_mode='bbns',old_inclusion_schedule_applied=False,
        independent_child_sparse_applied=0.,independent_inclusion_applied=0.,Omega_F=.9,total_band_edges=.1,
        total_nested_regularizer=.4,rank_health=health)
    b=copy.deepcopy(a)
    for h in b['rank_health']:
        s=h['sampling'];s['D3_selected_sentence_indices_sha256']=s.pop('lowest_selected_sentence_indices_sha256')
    return a,b


def test_full_stream_and_no_old_loss_gate(configured):
    a,b=frozen_step();assert search.matched_stream([a],[b],'BBNS')['records']==1024
    for key in search.FULL_STREAM+('K','lowest_selected_sentence_indices_sha256'):
        bad=copy.deepcopy(a);bad['rank_health'][0]['sampling'][key]='drift'
        with pytest.raises(AssertionError):search.matched_stream([bad],[b],'BBNS')
    for key,value in [('inc_weight',1),('independent_child_sparse_applied',.1),('total_nested_regularizer',.5)]:
        with pytest.raises(AssertionError):search.matched_stream([dict(a,**{key:value})],[b],'BBNS')


def test_checkpoint_BBNS_metadata_only_optimizer_frozen(configured,monkeypatch):
    monkeypatch.setattr(search,'ARM','BBNS');cfg=search.arm_config('BBNS')
    cfg.update(start_updates=0,resume=None,init_sha256=search.STEP0_SHA,max_updates=500,run_type='formal',
        runtime_model=dict(search_hparams=objective.hparams(cfg),regularizer_mode='bbns',support_bands=cfg['support_bands']),
        code_sha256={p:search.sha(search.ROOT/p) for p in search.EDITED|{'model/nested_support_band.py'}})
    old=copy.deepcopy(cfg);old['runtime_model'].pop('regularizer_mode');old['runtime_model'].pop('support_bands')
    old['code_sha256']={p:'old' for p in search.EDITED}
    opt=dict(param_groups=[],state={0:{'step':torch.tensor(5.)}})
    payload=dict(config=cfg,completed_steps=5,scheduler_horizon=4868,optimizer=opt,
        model={'w':torch.ones(2)},adapter={'w':torch.ones(2)})
    assert search.checkpoint_invariants(payload,dict(payload,config=old))['passed']


def fake_steps():
    output=[]
    for step in range(1,501):
        row=dict(step=step,regularizer_mode='bbns',inc_weight=0.,inclusion_loss=0.,inclusion_enabled=False,
            independent_child_sparse_applied=0.,independent_inclusion_applied=0.,Omega_F=.8)
        for e in ('Dall_F','D3_Dall'):
            row.update({e+'_'+k:.0 for k in experiment.EDGE_KEYS})
            row.update({e+'_coverage':.7,e+'_relative_support':.65,e+'_zero_refine_band':.7,e+'_zero_edge_band':.6})
            row[e+'_percent_inside_zero_loss_band']=60.
            row[e+'_percent_inside_refinement_band']=70.
        row['total_band_edges']=0.;row['total_nested_regularizer']=.8/3;output.append(row)
    return output


def test_all500_support_audit_identity_and_double_count_rejected():
    a=fake_steps();assert experiment.band_audit(a,BANDS)['passed']
    a[200]['independent_inclusion_applied']=.1
    with pytest.raises(AssertionError):experiment.band_audit(a,BANDS)


@pytest.mark.parametrize('changes,band,status',[
    ({'Score5':71.16,'J_long3':75.,'Urban_T2I':89.6,'Short4':65.3},True,'BBNS_STRONG_POSITIVE'),
    ({'Score5':71.12,'J_long3':75.02,'Urban_T2I':89.7},True,'BBNS_POSITIVE'),
    ({},True,'BBNS_REGULARIZATION_SUCCESS'),
    ({'Score5':71.2,'Urban_T2I':89.3},True,'BBNS_TRADEOFF'),
    ({'Score5':70.9,'J_long3':74.8,'Urban_T2I':89.3},True,'BBNS_NEGATIVE'),
    ({},False,'BBNS_TRADEOFF')])
def test_frozen_classification(changes,band,status):
    base=dict(Score5=71.0636,J_long3=74.954666,J_long=84.025001,Short4=65.227,Urban_I2T=90.9,Urban_T2I=89.7)
    keep=dict(F=.85,Dall=.84,D3=.79)
    assert experiment.classify(dict(base,**changes),base,keep,band)==status


def test_four_reference_final_summary_actual_schemas(configured,tmp_path,monkeypatch):
    base=experiment.BASE_EXP;result=json.loads((base/'RESULTS.json').read_text())
    diag=json.loads((base/'TRAINING_DIAGNOSTICS.json').read_text());masks=json.loads((base/'MASK_HIERARCHY_AUDIT.json').read_text())
    masks['keep_ratio']={v:d['keep_ratio'] for v,d in diag['last50_views'].items()}
    grad=json.loads((base/'GRADIENT_SPOTCHECK.json').read_text());grad['weighted_lowest_Dall_ratio']=grad['weighted_mean_gradient_norms']['D3']/grad['weighted_mean_gradient_norms']['Dall']
    anchor=json.loads((experiment.EXP/'ANCHOR_SUPPORT_AUDIT.json').read_text())
    for name,value in [('RESULTS',result),('TRAINING_DIAGNOSTICS',diag),('MASK_HIERARCHY_AUDIT',masks),
        ('GRADIENT_SPOTCHECK',grad),('VALIDATION',{'passed':True}),('ANCHOR_SUPPORT_AUDIT',anchor),
        ('GRADIENT_ROUTING_AUDIT',experiment.routing_audit()),('config',search.arm_config('BBNS'))]:dump(tmp_path/(name+'.json'),value)
    (tmp_path/'REPORT.md').write_text('Synthetic fixture\n')
    monkeypatch.setattr(experiment,'EXP',tmp_path)
    monkeypatch.setattr(experiment,'rows',lambda p:fake_steps())
    monkeypatch.setattr(search,'matched_stream',lambda *a:dict(records=512000))
    experiment.summarize()
    value=json.loads((tmp_path/'RESULTS.json').read_text())
    assert set(value['references'])=={'Anchor','INC0','Coupled v1'}
    assert len(value['recall_delta_vs_references_pp']['INC0'])==5
    assert value['support_band_audit_passed'] and (tmp_path/'SUPPORT_BAND_AUDIT.json').stat().st_size<1024*1024
    assert (tmp_path/'SEARCH_SUMMARY.md').exists()
