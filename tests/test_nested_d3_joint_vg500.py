"""Joint V/G formulas, gradients, feasible regions, full loss and frozen gates."""
import copy
import json

import pytest
import torch
from torch.nn import functional as F

from model import balanced_hparam_search as objective
from model.nested_joint_vg import joint_vg_edge,joint_vg_regularizer,vg_ratios,validate_regions
from model.nested_semantic_mask import hard_st
from recovery import nested_d3_joint_vg500 as experiment
from recovery import nested_d3_local_search as search
from recovery import nested_d3_followup500 as runner
from recovery.nested_d3_vg_audit import regions_from_distributions,support_vectors,KEYS
from recovery.s02_nfs500 import dump
from tests import test_balanced_hparams as reference
from tests.test_nested_fusion import explicit_inputs,explicit_logits

REGION=dict(eps_v=.05,gamma_low=.1,gamma_high=.3)
REGIONS={e:dict(REGION) for e in ('Dall_F','D3_Dall')}


@pytest.fixture
def configured(monkeypatch):
    for name in ('EXP','RUN_ROOT','BRANCH','ARMS','ENTRY_MODULE','PHASE_PREFIX','EXTRA_SOURCES',
                 'ARM','RUN','PHASE','ARM_EXP','CONFIG','ANCHOR_EXP','ANCHOR_RUN'):
        monkeypatch.setattr(search,name,getattr(search,name))
    for name in ('EXP','RUN_ROOT','BRANCH','ARMS','ENTRY','PUBLISH_MESSAGE','MAIN_LOG','IDENTITY',
                 'configure','summarize','REPORT_NAMES'):
        monkeypatch.setattr(runner,name,getattr(runner,name))
    experiment.configure()


def test_ratios_direct_definition_and_soft_audit():
    c=torch.tensor([[.8,.2]],requires_grad=True);p=torch.tensor([[.6,.4]],requires_grad=True)
    v,g=vg_ratios(c,p)
    assert v.item()==pytest.approx(.2/(1+1e-6)) and g.item()==pytest.approx(.2/(1+1e-6))
    gv=torch.autograd.grad(v.sum(),(c,p),retain_graph=True)
    gg=torch.autograd.grad(g.sum(),(c,p))
    assert all(torch.count_nonzero(t)>0 for t in (*gv,*gg))
    values=dict(zip(KEYS,support_vectors(p,c,.5*c)[0].tolist()))
    assert values['V_Dall_F']==pytest.approx(v.item()) and values['G_Dall_F']==pytest.approx(g.item())
    assert vg_ratios(.5*p,p)[0].item()==0
    assert vg_ratios(.2*p,p)[1].item()>vg_ratios(.5*p,p)[1].item()


def test_manual_joint_gradients_and_exact_equality_kink():
    audit=experiment.gradient_audit();assert audit['passed'] and audit['no_stop_gradient']
    equal=audit['cases']['identical']
    assert equal['values']['G_lower_penalty']>0 and equal['values']['G']==0
    assert not torch.count_nonzero(torch.tensor(equal['child_gradient']))
    assert not torch.count_nonzero(torch.tensor(equal['parent_gradient']))
    for name in ('upward','lower','upper'):
        for key in ('child_gradient','parent_gradient'):
            assert torch.count_nonzero(torch.tensor(audit['cases'][name][key]))


@pytest.mark.parametrize('child,parent,positive,sign',[(.59,.60,True,1),(.45,.60,False,0),(.2,.60,True,-1)])
def test_G_bands_and_joint_gradient_direction(child,parent,positive,sign):
    c=torch.full((2,5),child,requires_grad=True);p=torch.full((2,5),parent,requires_grad=True)
    terms=joint_vg_edge(c,p,REGION);loss=terms['G_lower_penalty']+terms['G_upper_penalty']
    assert bool((loss>0).all())==positive
    gc,gp=torch.autograd.grad(loss.sum(),(c,p))
    assert torch.equal(gc.sign(),torch.full_like(gc,sign))
    assert torch.equal(gp.sign(),torch.full_like(gp,-sign))


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
        band=module.vg_regions[name]
        V=(c-p).clamp_min(0).sum(-1)/(c.sum(-1)+1e-6)
        G=(p-c).clamp_min(0).sum(-1)/(p.sum(-1)+1e-6)
        penalty=(V-band['eps_v']).clamp_min(0).square()+(band['gamma_low']-G).clamp_min(0).square()+(G-band['gamma_high']).clamp_min(0).square()
        reg+=penalty[valid].mean()
    return 10/3*sum(w*pair[0] for w,pair in zip([1.35,1.35,.30],pairs))+reg


@pytest.mark.parametrize('valid_count,completed',[(2,0),(2,99),(2,199),(2,499),(1,499),(0,499)])
def test_full_loss_all_parameter_gradients_and_old_ramp_forbidden(monkeypatch,valid_count,completed):
    torch.manual_seed(50211)
    module=reference.make(dict(view_weights=[1.35,1.35,.30]))
    module.inclusion_hierarchy='detail_chain';module.regularizer_mode='joint_vg';module.vg_regions=REGIONS
    explicit=copy.deepcopy(module);images,views,valid=reference.inputs();valid.zero_();valid[:valid_count]=True
    def forbidden(*args):raise AssertionError('Joint-VG must never evaluate old inclusion ramp')
    monkeypatch.setattr(objective,'inclusion_weight',forbidden)
    actual,logs=module(images,*views,valid,completed);expected=manual_objective(explicit,images,views,valid)
    torch.testing.assert_close(actual,expected,atol=3e-4,rtol=3e-5)
    assert logs['inc_weight']==logs['inclusion_loss']==0 and logs['inclusion_enabled'] is False
    assert logs['independent_child_sparse_applied']==logs['independent_inclusion_applied']==0
    actual.backward();expected.backward()
    for name,p in module.named_parameters():
        q=dict(explicit.named_parameters())[name];assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=5e-4,rtol=5e-5,msg=name)


def test_DDP_uneven_valid_normalization_and_endpoint_gradients():
    torch.manual_seed(10)
    pf=torch.rand(6,4,requires_grad=True);pd=torch.rand(6,4,requires_grad=True);pl=torch.rand(6,4,requires_grad=True)
    omega=torch.rand(6,requires_grad=True);valid=torch.tensor([1,0,0,0,1,1],dtype=torch.bool)
    full,_=joint_vg_regularizer(omega.mean(),pf,pd,pl,valid,REGIONS,1,3)
    parts=[]
    for i in range(3):
        s=slice(2*i,2*i+2)
        p,_=joint_vg_regularizer(omega[s].mean(),pf[s],pd[s],pl[s],valid[s],REGIONS,3,3);parts.append(p)
    merged=sum(parts)/3;torch.testing.assert_close(full,merged)
    gf=torch.autograd.grad(full,(omega,pf,pd,pl),retain_graph=True);gd=torch.autograd.grad(merged,(omega,pf,pd,pl))
    for a,b in zip(gf,gd):torch.testing.assert_close(a,b)


def test_old_child_sparse_and_inclusion_not_in_loss(monkeypatch):
    torch.manual_seed(519)
    module=reference.make(dict(view_weights=[1.35,1.35,.30]))
    module.inclusion_hierarchy='detail_chain';module.regularizer_mode='joint_vg';module.vg_regions=REGIONS
    images,views,valid=reference.inputs();first,_=module(images,*views,valid,200)
    fn=objective.fusion_view_terms;calls=[]
    def changed(*args,**kwargs):
        result=list(fn(*args,**kwargs));calls.append(True)
        if len(calls)>1:result[1]=result[1]+10000.
        return tuple(result)
    def telemetry(*args):
        assert not torch.is_grad_enabled()
        return torch.full((3,),10000.)
    monkeypatch.setattr(objective,'fusion_view_terms',changed);monkeypatch.setattr(objective,'detail_chain_inclusion',telemetry)
    second,_=module(images,*views,valid,200);torch.testing.assert_close(first,second,atol=0,rtol=0)


def test_frozen_quantiles_single_arm_and_config_drift(configured):
    audit=json.loads((experiment.EXP/'ANCHOR_VG_AUDIT.json').read_text());cfg=search.arm_config(experiment.ARM)
    base=json.loads((experiment.BASE_EXP/'config.json').read_text())
    assert {k for k in base.keys()|cfg.keys() if base.get(k)!=cfg.get(k)}=={'regularizer_mode','vg_regions'}
    assert cfg['vg_regions']==regions_from_distributions(audit['soft_distributions'])==audit['frozen_regions']
    assert audit['records']==16384 and audit['loader_exhausted_normally']
    assert cfg==json.loads(experiment.CONFIG.read_text())
    assert list(runner.ARMS)==[experiment.ARM] and runner.summarize is experiment.summarize
    search.frozen_config(cfg,experiment.ARM)
    for key,value in [('workers',4),('sampling_mode','nested_detail_kr234'),('view_weights',[1,1,1]),('inclusion_max',.5)]:
        with pytest.raises(AssertionError):search.frozen_config(dict(cfg,**{key:value}),experiment.ARM)


def test_reject_zero_lower_band_and_invalid_regions():
    for regions in ({},{'Dall_F':REGION},dict(REGIONS,Dall_F=dict(REGION,gamma_low=0)),
        dict(REGIONS,Dall_F=dict(REGION,gamma_low=.4)),dict(REGIONS,D3_Dall=dict(REGION,eps_v=float('nan')))):
        with pytest.raises(AssertionError):validate_regions(regions)


def test_checkpoint_metadata_optimizer_and_sources_frozen(configured,monkeypatch):
    monkeypatch.setattr(search,'ARM',experiment.ARM);cfg=search.arm_config(experiment.ARM)
    cfg.update(start_updates=0,resume=None,init_sha256=search.STEP0_SHA,max_updates=500,run_type='formal',
        runtime_model=dict(search_hparams=objective.hparams(cfg),regularizer_mode='joint_vg',vg_regions=cfg['vg_regions']),
        code_sha256={p:search.sha(search.ROOT/p) for p in search.EDITED|{'model/nested_support_band.py','model/nested_joint_vg.py'}})
    old=copy.deepcopy(cfg);old['runtime_model'].pop('regularizer_mode');old['runtime_model'].pop('vg_regions')
    old['code_sha256']={p:'old' for p in search.EDITED}
    opt=dict(param_groups=[],state={0:{'step':torch.tensor(5.)}})
    payload=dict(config=cfg,completed_steps=5,scheduler_horizon=4868,optimizer=opt,model={'w':torch.ones(2)},adapter={'w':torch.ones(2)})
    assert search.checkpoint_invariants(payload,dict(payload,config=old))['passed']


def fake_steps():
    output=[]
    for step in range(1,501):
        row=dict(step=step,regularizer_mode='joint_vg',inc_weight=0.,inclusion_loss=0.,inclusion_enabled=False,
            old_inclusion_schedule_applied=False,independent_child_sparse_applied=0.,independent_inclusion_applied=0.)
        row.update({k:0. for k in experiment.VG_KEYS})
        row.update({k:0. for k in ('F_i2t','F_t2i','O_i2t','O_t2i','E_i2t','E_t2i',
            'F_keep_ratio','O_keep_ratio','E_keep_ratio','F_Dall_mask_iou','Dall_Ds_mask_iou','Dall_F_hard_violation','Ds_Dall_hard_violation')})
        row['Omega_F']=.8;row['total_nested_regularizer']=.8/3;output.append(row)
    return output


def test_vg_audit_raw_mask_names_and_double_counting_rejected():
    output=fake_steps()
    assert experiment.vg_audit(output,REGIONS)['passed']
    output[200]['independent_inclusion_applied']=.1
    with pytest.raises(AssertionError):experiment.vg_audit(output,REGIONS)


@pytest.mark.parametrize('changes,stable,trivial,status',[
    ({'Score5':71.17,'J_long3':75.,'Urban_T2I':89.6,'Short4':65.3},True,False,'JOINT_VG_STRONG_POSITIVE'),
    ({'Score5':71.12,'J_long3':75.02},False,False,'JOINT_VG_POSITIVE'),
    ({},True,False,'JOINT_VG_STRUCTURAL_SUCCESS'),
    ({},True,True,'JOINT_VG_NEGATIVE'),
    ({'Score5':70.8,'J_long3':74.7},False,False,'JOINT_VG_NEGATIVE'),
    ({'Score5':71.2,'Urban_T2I':89.3},True,False,'JOINT_VG_TRADEOFF')])
def test_classification_declared_before_training(changes,stable,trivial,status):
    q=dict(Score5=71.063600,J_long3=74.954666,J_long=84.025001,Short4=65.227,Urban_I2T=90.9,Urban_T2I=89.7)
    assert experiment.classify(dict(q,**changes),q,stable,trivial)==status


def test_stream_gate_rejects_text_indices_LR_and_hidden_losses(configured):
    health=[]
    for rank in range(4):
        ids=list(range(rank*256+1000,(rank+1)*256+1000));ns=[6]*256
        choices=[search.expected_indices(n,sid,experiment.ARM) for n,sid in zip(ns,ids)]
        s={k:'frozen:'+k for k in search.FULL_STREAM}
        s.update(sample_ids=ids,n=ns,K=[len(c) for c in choices],nested_d3_exact=True,
            lowest_selected_sentence_indices_sha256=search.indices_digest(ids,choices))
        health.append(dict(rank=rank,sampling=s,gradients_finite=True,batch=256,updates=1))
    a=dict(fake_steps()[0],epoch=0,actual_lrs=[1e-6,1e-3,1e-3,2e-4],loss=1.,nonfinite=0,rank_health=health)
    b=copy.deepcopy(a)
    for h in b['rank_health']:
        s=h['sampling'];s['D3_selected_sentence_indices_sha256']=s.pop('lowest_selected_sentence_indices_sha256')
    assert search.matched_stream([a],[b],experiment.ARM)['records']==1024
    for key in search.FULL_STREAM+('K','lowest_selected_sentence_indices_sha256'):
        bad=copy.deepcopy(a);bad['rank_health'][0]['sampling'][key]='drift'
        with pytest.raises(AssertionError):search.matched_stream([bad],[b],experiment.ARM)
    for key,value in [('actual_lrs',[1.]),('inc_weight',1),('independent_child_sparse_applied',.1),('total_nested_regularizer',.5)]:
        with pytest.raises(AssertionError):search.matched_stream([dict(a,**{key:value})],[b],experiment.ARM)


def test_final_summary_real_schemas_all_four_references(configured,tmp_path,monkeypatch):
    base=experiment.BASE_EXP;result=json.loads((base/'RESULTS.json').read_text())
    diag=json.loads((base/'TRAINING_DIAGNOSTICS.json').read_text());masks=json.loads((base/'MASK_HIERARCHY_AUDIT.json').read_text())
    masks['keep_ratio']={v:d['keep_ratio'] for v,d in diag['last50_views'].items()}
    grad=json.loads((base/'GRADIENT_SPOTCHECK.json').read_text())
    grad['weighted_lowest_Dall_ratio']=grad['weighted_mean_gradient_norms']['D3']/grad['weighted_mean_gradient_norms']['Dall']
    anchor=json.loads((experiment.EXP/'ANCHOR_VG_AUDIT.json').read_text())
    for name,value in [('RESULTS',result),('TRAINING_DIAGNOSTICS',diag),('MASK_HIERARCHY_AUDIT',masks),
        ('GRADIENT_SPOTCHECK',grad),('VALIDATION',{'passed':True}),('ANCHOR_VG_AUDIT',anchor),
        ('config',search.arm_config(experiment.ARM))]:dump(tmp_path/(name+'.json'),value)
    (tmp_path/'REPORT.md').write_text('Synthetic fixture\n')
    monkeypatch.setattr(experiment,'EXP',tmp_path);monkeypatch.setattr(experiment,'rows',lambda p:fake_steps())
    monkeypatch.setattr(search,'matched_stream',lambda *a:dict(records=512000))
    experiment.summarize();value=json.loads((tmp_path/'RESULTS.json').read_text())
    assert set(value['references'])=={'Anchor','INC0','Coupled v1','BBNS'}
    assert len(value['recall_delta_vs_references_pp']['BBNS'])==5
    assert value['vg_audit_passed'] and (tmp_path/'VG_AUDIT.json').stat().st_size<1024*1024
    assert json.loads((tmp_path/'GRADIENT_AUDIT.json').read_text())['joint_edge_gradient_routing']['passed']
    assert set(json.loads((tmp_path/'VG_AUDIT.json').read_text())['diagnostic_steps'])=={'1','100','200','500'}
    assert json.loads((tmp_path/'MASK_HIERARCHY_AUDIT.json').read_text())['inclusion_loss_active'] is False
