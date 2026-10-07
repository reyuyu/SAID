"""Hard-ST binary truth table, joint gradients, actual INC0 equivalence and isolation."""
import copy
import json
import subprocess
import types

import pytest
import torch
from torch.nn import functional as F

from model.hard_nested_sparsity import hard_violation,hierarchy_weight,hns_terms,hard_telemetry
from model.nested_semantic_mask import hard_st
from tests.test_balanced_hparams import inputs
from tests.test_nested_fusion import explicit_inputs,explicit_logits
from recovery.hns_ddp_correctness import make
from recovery.hns_preflight import BASE_SHA,EXP,ROOT
from train.train_nested_semantic_mask import build_optimizer


@pytest.mark.parametrize('c,p,expected',[(0,0,0),(0,1,0),(1,1,0),(1,0,1)])
def test_binary_and_actual_st_gradient(c,p,expected):
    cl=torch.tensor([[-2. if c==0 else 2.]],requires_grad=True)
    pl=torch.tensor([[-2. if p==0 else 2.]],requires_grad=True)
    cp,pp=cl.sigmoid(),pl.sigmoid()
    value=hard_violation(hard_st(cp),hard_st(pp)).sum()
    assert value.item()==expected
    gc,gp=torch.autograd.grad(value,(cl,pl))
    if expected:
        assert gc.item()>0 and gp.item()<0
        torch.testing.assert_close(gc,cp*(1-cp));torch.testing.assert_close(gp,-pp*(1-pp))
    else:assert gc.item()==gp.item()==0


def test_product_wrong_at_equal_zero():
    c=torch.tensor([[.1]],requires_grad=True);p=torch.tensor([[.1]],requires_grad=True)
    wrong=hard_st(c)*(1-hard_st(p))
    assert torch.autograd.grad(wrong.sum(),c)[0].item()!=0
    assert torch.autograd.grad(hard_violation(hard_st(c),hard_st(p)).sum(),c)[0].item()==0


@pytest.mark.parametrize('completed,weight',[(0,0),(99,.495),(199,.995),(200,1),(299,1),(499,1)])
def test_ramp(completed,weight):assert hierarchy_weight(completed)==weight


def base_forward():
    source=subprocess.check_output(['git','show',BASE_SHA+':model/balanced_hparam_search.py'],cwd=ROOT,text=True)
    namespace={'__name__':'model._fetched_inc0_for_test','__package__':'model'}
    exec(compile(source,'fetched-INC0-balanced_hparam_search.py','exec'),namespace)
    return namespace['BalancedSearch'].forward


def test_lambda0_actual_fetched_INC0_all_gradients_and_AdamW_exact():
    torch.manual_seed(7879);current=make();baseline=copy.deepcopy(current)
    baseline.forward=types.MethodType(base_forward(),baseline)
    images,views,valid=inputs()
    a,al=current(images,*views,valid,0);b,bl=baseline(images,*views,valid,0)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    for key in bl:
        if torch.is_tensor(bl[key]):torch.testing.assert_close(al[key],bl[key],atol=0,rtol=0,msg=key)
        else:assert al[key]==bl[key]
    a.backward();b.backward()
    for name,p in current.named_parameters():
        q=dict(baseline.named_parameters())[name]
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0,msg=name)
    build_optimizer(current).step();build_optimizer(baseline).step()
    for name,p in current.named_parameters():torch.testing.assert_close(p,dict(baseline.named_parameters())[name],atol=0,rtol=0,msg=name)


def explicit_hns(module,images,views,valid,completed):
    values=[];masks=[]
    for i,tokens in enumerate(views):
        z,t,zv,zt=explicit_inputs(module,images,tokens)
        m=hard_st(explicit_logits(module,zv,zt).sigmoid())
        scores=100*(F.normalize(z[None]*m,dim=-1,eps=1e-6)*F.normalize(t,dim=-1,eps=1e-6)[:,None]).sum(-1)
        enabled=torch.ones_like(valid) if i==0 else valid
        labels=torch.arange(len(z))[enabled]
        ce=F.cross_entropy(scores.T[enabled].masked_fill(~enabled[None],-torch.inf),labels)+F.cross_entropy(scores[enabled].masked_fill(~enabled[None],-torch.inf),labels)
        pos=m.diagonal(dim1=0,dim2=1).T;masks.append(pos)
        values.append((ce,pos[enabled].abs().mean()))
    loss=10/3*sum(w*x[0] for w,x in zip((1.35,1.35,.3),values))
    loss+=(values[0][1]+2*values[1][1]+2*values[2][1])/3
    loss+=hierarchy_weight(completed)*2/3*(hard_violation(masks[1],masks[0])[valid].mean()+hard_violation(masks[2],masks[1])[valid].mean())
    return loss


@pytest.mark.parametrize('completed',[0,99,199,499])
def test_independent_actual_positive_mask_objective_all_gradients(completed):
    torch.manual_seed(7939);module=make();reference=copy.deepcopy(module)
    images,views,valid=inputs()
    actual,logs=module(images,*views,valid,completed)
    expected=explicit_hns(reference,images,views,valid,completed)
    torch.testing.assert_close(actual,expected,atol=3e-4,rtol=3e-5)
    actual.backward();expected.backward()
    for name,p in module.named_parameters():
        q=dict(reference.named_parameters())[name]
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=8e-4,rtol=8e-5,msg=name)
    assert logs['inc_weight']==logs['inclusion_loss']==0 and not logs['inclusion_enabled']


def test_uneven_valid_normalization_and_counts():
    torch.manual_seed(7913);masks=[torch.randint(0,2,(8,32)).float().requires_grad_() for _ in range(3)]
    valid=torch.tensor([0,0,1,1,1,0,1,0],dtype=torch.bool)
    expected,_=hns_terms(*masks,valid,199,1,4)
    parts=[hns_terms(*(m[r*2:r*2+2] for m in masks),valid[r*2:r*2+2],199,4,4)[0] for r in range(4)]
    torch.testing.assert_close(sum(parts)/4,expected,atol=6e-8,rtol=2e-7)
    telemetry,width=hard_telemetry(*masks,valid)
    assert width==32 and telemetry['DF_violation_count']==telemetry['DF_child_only_count']


def test_frozen_config_and_preflight():
    base=json.loads((ROOT/'experiments/nest_clip_v1/nested_d3_inc0_500_v1/config.json').read_text())
    current=json.loads((EXP/'config.json').read_text());assert current.pop('hns_enabled') is True
    assert current==base and base['inclusion_max']==0
    evidence=json.loads((EXP/'MATCHED_PREFLIGHT.json').read_text());assert evidence['passed']


def test_runner_single_arm_and_smoke_independent(monkeypatch):
    from recovery.nested_d3_hns500 import configure,ARMS,SMOKE,RUN_ROOT,ENTRY
    from recovery import nested_d3_followup500 as runner,nested_d3_local_search as search
    from recovery import s02_local500 as local
    for name in ('EXP','RUN_ROOT','BRANCH','ARMS','ENTRY_MODULE','PHASE_PREFIX','EXTRA_SOURCES',
                 'ARM','RUN','PHASE','ARM_EXP','CONFIG','ANCHOR_EXP','ANCHOR_RUN','EDITED','Supervisor'):
        monkeypatch.setattr(search,name,getattr(search,name))
    for name in ('EXP','RUN_ROOT','BRANCH','ARMS','ENTRY','PUBLISH_MESSAGE','MAIN_LOG','IDENTITY',
                 'configure','summarize','publication_paths'):
        monkeypatch.setattr(runner,name,getattr(runner,name))
    for name in ('RUN','PHASE','CONFIG'):monkeypatch.setattr(local,name,getattr(local,name))
    configure();search.activate('HNS')
    assert list(ARMS)==['HNS'] and SMOKE!=search.RUN and SMOKE!=RUN_ROOT
    assert runner.ENTRY==ENTRY and search.EDITED=={'model/balanced_hparam_search.py','train/train_nested_semantic_mask.py'}
    expected=search.arm_config('HNS');assert expected['hns_enabled'] and expected['inclusion_max']==0


def test_raw_quality_and_collapse_outcome_does_not_stop_training():
    from recovery.nested_d3_hns500 import quality_raw,assess
    result=json.loads((ROOT/'experiments/nest_clip_v1/nested_d3_inc0_500_v1/RESULTS.json').read_text())
    q=quality_raw(result);assert q['Urban_T2I']==result['metrics']['Urban-1k']['T2I']['R@1']
    old={**{'HNS_'+v+'_keep':.7 for v in ('F','Dall','D3')},
         'HNS_DF_hard_violation_ratio':.02,'HNS_3D_hard_violation_ratio':.04}
    new={**old,**{'HNS_'+v+'_keep':.9 for v in ('F','Dall','D3')},
        'HNS_gap_F_D':0.,'HNS_gap_D_D3':0.}
    equality={k:0. for k in ('HNS_DF_exact_equality_ratio','HNS_3D_exact_equality_ratio','HNS_triple_exact_equality_ratio')}
    equality.update({'HNS_'+v+'_keep':.2 for v in ('F','Dall','D3')})
    decision=assess(q,q,new,old,equality)
    assert decision['classification']=='NEGATIVE' and decision['collapse_warning']
    assert not decision['automatic_full'] and not decision['automatic_other_arms']


def test_completed_report_pipeline_schema_with_synthetic_run(tmp_path,monkeypatch):
    """Exercise final reporting with real baseline schemas and synthetic run evidence."""
    from recovery import nested_d3_hns500 as hns,nested_d3_local_search as search
    from experiments.nest_clip_v1.balanced_hparam_search_v1 import search as native
    from recovery.s02_nfs500 import dump
    run=tmp_path/'runtime';run.mkdir();exp=tmp_path/'experiment';exp.mkdir()
    base=ROOT/'experiments/nest_clip_v1/nested_d3_inc0_500_v1'
    for name in ('RESULTS.json','MASK_HIERARCHY_AUDIT.json','TRAINING_DIAGNOSTICS.json','RUNTIME_STATS.json'):
        dump(exp/name,json.loads((base/name).read_text()))
    dump(exp/'VALIDATION.json',{'passed':True})
    dump(exp/'references/JOINT_VG_RESULTS.json',json.loads((EXP/'references/JOINT_VG_RESULTS.json').read_text()))
    result=json.loads((exp/'RESULTS.json').read_text())
    values=dict(HNS_F_keep=.80,HNS_Dall_keep=.78,HNS_D3_keep=.74,HNS_gap_F_D=.02,HNS_gap_D_D3=.04,
        HNS_DF_hard_violation_ratio=.01,HNS_3D_hard_violation_ratio=.02,HNS_DF_exact_equality_ratio=.01,
        HNS_3D_exact_equality_ratio=.01,HNS_triple_exact_equality_ratio=.0)
    matched={k:0. for k in values}
    dump(exp/'GRADIENT_AUDIT.json',dict(passed=True,matched_cohort_delta_vs_INC0=matched,
        endpoint_gradient_direction={},group_cosines={}))
    dump(run/'supervisor-result.json',dict(commands=[],started_utc='synthetic',ended_utc='synthetic',
        acceptance={'passed':True}))
    for name in ('launch-provenance.json','first-five-gate.json','full-stream-proof.json','commands.json'):
        dump(run/name,{'passed':True,'synthetic_fixture':True})
    (run/'HNS-gradient-audit500.log').write_text('Synthetic unit-test fixture\n')
    monkeypatch.setattr(hns,'EXP',exp);monkeypatch.setattr(search,'RUN',run)
    monkeypatch.setattr(hns,'SMOKE',tmp_path/'smoke');monkeypatch.setattr(hns,'SMOKE_PHASE',tmp_path/'phase')
    monkeypatch.setattr(hns,'rows',lambda path:[dict(step=s,lambda_h=min(1,(s-1)/200),**values) for s in range(1,501)])
    monkeypatch.setattr(native,'native_metrics',lambda path:(result['metrics'],result['native_evaluation_provenance'],{}))
    hns.summarize()
    actual=json.loads((exp/'RESULTS.json').read_text())
    assert set(actual['all_baseline_comparisons'])=={'INC0','Anchor','KR234','Joint-VG'}
    assert all(v==0 for v in actual['all_baseline_comparisons']['INC0']['quality_delta_pp'].values())
    assert len(list((exp/'evaluations').glob('*.json')))==5
    assert (exp/'DECISION.json').exists() and 'Scientific questions' in (exp/'REPORT.md').read_text()
