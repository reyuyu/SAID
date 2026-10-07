"""Production beta routing, exact frozen-default and beta0 equivalence, selection."""
import copy
import json
import subprocess
import types

import pytest
import torch

from recovery.hns_ddp_correctness import make
from model.hard_nested_sparsity import hns_terms,hard_violation,hierarchy_weight
from model.nested_semantic_mask import hard_st
from tests.test_balanced_hparams import inputs
from tests.test_nested_d3_hns500 import base_forward,explicit_hns
from recovery import hns_final_two500 as final
from train.train_nested_semantic_mask import build_optimizer


def compare_backward(a,b):
    for name,p in a.named_parameters():
        q=dict(b.named_parameters())[name]
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0,msg=name)


@pytest.mark.parametrize('beta',[(1.,1.),(1.,2.),(0.,0.),(2.,2.)])
@pytest.mark.parametrize('completed',[0,99,199,499])
def test_production_decomposition_and_every_gradient(beta,completed):
    torch.manual_seed(9793);module=make(beta);baseline=copy.deepcopy(module)
    baseline.forward=types.MethodType(base_forward(),baseline)
    module.capture_hns_graph=True
    images,views,valid=inputs();actual,logs=module(images,*views,valid,completed)
    inc0,_=baseline(images,*views,valid,completed)
    # Independently add surcharge to the fetched original INC0 graph, using the
    # masks produced by the real frozen fusion implementation.
    from tests.test_nested_fusion import explicit_inputs,explicit_logits
    masks=[]
    for tokens in views:
        z,t,zv,zt=explicit_inputs(baseline,images,tokens)
        m=hard_st(explicit_logits(baseline,zv,zt).sigmoid())
        masks.append(m.diagonal(dim1=0,dim2=1).T)
    expected=inc0+hierarchy_weight(completed)*(beta[0]*hard_violation(masks[1],masks[0])[valid].mean()+beta[1]*hard_violation(masks[2],masks[1])[valid].mean())/3
    torch.testing.assert_close(actual,expected,atol=3e-4,rtol=3e-5)
    actual.backward();expected.backward()
    for name,p in module.named_parameters():
        q=dict(baseline.named_parameters())[name]
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=8e-4,rtol=8e-5,msg=name)
    assert [logs['HNS_beta_DF'],logs['HNS_beta_3D']]==list(beta)
    assert logs['inclusion_loss']==logs['inc_weight']==0
    assert module.hns_graph['beta']==beta


def test_zero_beta_restores_INC0_all_gradients_and_AdamW_exact():
    torch.manual_seed(7879);a=make((0.,0.));b=copy.deepcopy(a)
    b.forward=types.MethodType(base_forward(),b);images,views,valid=inputs()
    la,_=a(images,*views,valid,499);lb,_=b(images,*views,valid,499)
    torch.testing.assert_close(la,lb,atol=0,rtol=0);la.backward();lb.backward();compare_backward(a,b)
    build_optimizer(a).step();build_optimizer(b).step()
    for n,p in a.named_parameters():torch.testing.assert_close(p,dict(b.named_parameters())[n],atol=0,rtol=0)


def test_default_beta_fetched_HNS_v1_all_gradients_exact():
    source=subprocess.check_output(['git','show',final.BASE_SHA+':model/balanced_hparam_search.py'],cwd=final.ROOT,text=True)
    old_terms=subprocess.check_output(['git','show',final.BASE_SHA+':model/hard_nested_sparsity.py'],cwd=final.ROOT,text=True)
    ns={'__name__':'model._fetched_hns','__package__':'model'};exec(compile(source,'fetchedHNS','exec'),ns)
    old_ns={};exec(compile(old_terms,'fetchedHNSterms','exec'),old_ns);ns['hns_terms']=old_ns['hns_terms']
    torch.manual_seed(9793);a=make();b=copy.deepcopy(a);b.forward=types.MethodType(ns['BalancedSearch'].forward,b)
    images,views,valid=inputs();la,_=a(images,*views,valid,499);lb,_=b(images,*views,valid,499)
    torch.testing.assert_close(la,lb,atol=0,rtol=0);la.backward();lb.backward();compare_backward(a,b)


@pytest.mark.parametrize('beta',[(1.,1.),(1.,2.)])
def test_uneven_valid_scaling_and_joint_signs(beta):
    logits=[torch.tensor([[-2.,2.,-2.,2.]]*8,requires_grad=True) for _ in range(3)]
    logits[1]=torch.tensor([[2.,2.,-2.,-2.]]*8,requires_grad=True)
    masks=[hard_st(l.sigmoid()) for l in logits];valid=torch.tensor([0,0,1,1,1,0,1,0],dtype=torch.bool)
    all_loss,_=hns_terms(*masks,valid,199,1,4,beta)
    parts=[hns_terms(*(m[2*r:2*r+2] for m in masks),valid[2*r:2*r+2],199,4,4,beta)[0] for r in range(4)]
    torch.testing.assert_close(sum(parts)/4,all_loss,atol=6e-8,rtol=2e-7)
    df=hard_violation(masks[1],masks[0])[valid].mean();gf,gd=torch.autograd.grad(df,(logits[0],logits[1]))
    assert (gf<0).any() and (gd>0).any() and not (gf>0).any() and not (gd<0).any()


def test_actual_arm_config_only_beta_and_fresh_queue(monkeypatch):
    from recovery import nested_d3_local_search as search,nested_d3_followup500 as pub,hns_preflight as pf,s02_local500 as local,nested_d3_hns500 as old
    for mod in (search,pub,pf,local,old):
        for n in list(vars(mod)):
            if not n.startswith('__'):monkeypatch.setattr(mod,n,getattr(mod,n))
    assert list(final.ARMS)==['HNS-Weak','HNS-InnerFocus']
    base=json.loads((final.BASE_EXP/'config.json').read_text())
    for arm in final.ARMS:
        final.configure(arm);cfg=search.arm_config(arm)
        assert cfg==dict(base,hns_beta=final.ARMS[arm]['beta'])
        assert json.loads((search.CONFIG).read_text())==cfg
        assert local.RUN!=old.SMOKE and local.CONFIG==search.CONFIG
        assert search.ANCHOR_RUN==final.BASE_RUN
    source=(final.ROOT/'recovery/hns_final_two500.py').read_text()
    assert source.index('receipt=publication.publish();completed.append(arm)')<source.index("state('BOTH_COMPLETED_AND_SYNCED_GPU_IDLE'")
    assert "receipt['remote_HEAD_matches_local']" in source and "matched_preflight(arm);publish_preflight(arm)" in source


def test_raw_tie_champion_and_clear_score_advantage():
    q={n:dict(Score5=.710,J_long3=.750,J_long=.84,Urban_T2I=.897,Short4=.652) for n in ('INC0','v1','weak','inner')}
    h={n:.05 for n in q}
    q['weak']['Score5']=.7106;q['v1']['J_long3']=.752
    c=final.champion(q,h);assert c['BEST_SCORE5_CANDIDATE']=='weak' and c['BEST_LONG_CANDIDATE']=='v1'
    q['weak']['Score5']=.7104;c=final.champion(q,h)
    assert c['raw_Score5_maximum']=='weak' and c['BEST_SCORE5_CANDIDATE']=='v1'
    assert c['automatic_full'] is False


def test_report_pipeline_real_schemas_synthetic_training(tmp_path,monkeypatch):
    from recovery import nested_d3_local_search as search,nested_d3_hns500 as old
    from recovery.s02_nfs500 import dump
    from experiments.nest_clip_v1.balanced_hparam_search_v1 import search as native
    exp=tmp_path/'exp';run=tmp_path/'run';run.mkdir();exp.mkdir()
    for name in ('RESULTS.json','TRAINING_DIAGNOSTICS.json','MASK_HIERARCHY_AUDIT.json','RUNTIME_STATS.json'):
        dump(exp/name,json.loads((final.BASE_EXP/name).read_text()))
    audit=json.loads((final.BASE_EXP/'GRADIENT_AUDIT.json').read_text())
    audit.update(beta=[1.,1.],hierarchy_gradient_norm_ratio_vs_HNS_v1={},weighted_inner_outer_gradient_norm_ratio={})
    dump(exp/'GRADIENT_AUDIT.json',audit);dump(exp/'VALIDATION.json',{'passed':True})
    dump(exp/'SEARCH_PLAN.json',{'selection_rule':'synthetic fixture'})
    dump(run/'supervisor-result.json',dict(acceptance={'passed':True},commands=[],started_utc='synthetic',ended_utc='synthetic'))
    for name in ('launch-provenance.json','first-five-gate.json','full-stream-proof.json','commands.json'):
        dump(run/name,{'synthetic_fixture':True})
    (run/'HNS-gradient-audit500.log').write_text('Synthetic fixture\n')
    result=json.loads((exp/'RESULTS.json').read_text())
    values=dict(json.loads((final.BASE_EXP/'TRAINING_DIAGNOSTICS.json').read_text())['HNS']['last50'])
    values.update(HNS_beta_DF=1.,HNS_beta_3D=1.)
    monkeypatch.setattr(search,'ARM_EXP',exp);monkeypatch.setattr(search,'RUN',run)
    monkeypatch.setattr(old,'SMOKE',tmp_path/'smoke');monkeypatch.setattr(old,'SMOKE_PHASE',tmp_path/'phase')
    monkeypatch.setattr(final,'rows',lambda p:[dict(step=s,**values) for s in range(1,501)])
    monkeypatch.setattr(native,'native_metrics',lambda p:(result['metrics'],result['native_evaluation_provenance'],{}))
    final.summarize_arm('HNS-Weak')
    actual=json.loads((exp/'RESULTS.json').read_text())
    assert actual['beta']==[1.,1.] and actual['quality_raw_fraction']['Score5']==old.quality_raw(result)['Score5']
    assert set(actual['all_baseline_comparisons'])=={'INC0','HNS-v1'}
    assert all(x==0 for x in actual['all_baseline_comparisons']['HNS-v1']['quality_delta_pp'].values())
    assert len(list((exp/'evaluations').glob('*.json')))==5
    assert 'parent expansion and child contraction' in (exp/'REPORT.md').read_text()


def test_sequential_publish_failure_cannot_start_E2(tmp_path,monkeypatch):
    from recovery import nested_d3_local_search as search
    calls=[]
    monkeypatch.setattr(final,'RUN_ROOT',tmp_path/'run')
    monkeypatch.setattr(final,'configure',lambda arm:calls.append(('configure',arm)))
    monkeypatch.setattr(final,'git',lambda *a:final.ARMS['HNS-Weak']['branch'])
    monkeypatch.setattr(final,'state',lambda *a,**kw:calls.append(('state',a[0])))
    class Fake:
        def run(self):calls.append(('trained',500))
    monkeypatch.setattr(final,'Supervisor',Fake)
    monkeypatch.setattr(final,'summarize_arm',lambda arm:calls.append(('report',arm)))
    def reject():
        calls.append(('publish','failed'));raise RuntimeError('Synthetic network failure')
    monkeypatch.setattr(final.publication,'publish',reject)
    monkeypatch.setattr(final.signal,'signal',lambda *a:None)
    with pytest.raises(RuntimeError,match='Synthetic network failure'):final.run_queue()
    assert calls.count(('configure','HNS-Weak'))==1
    assert ('configure','HNS-InnerFocus') not in calls
    assert calls.index(('report','HNS-Weak'))<calls.index(('publish','failed'))
