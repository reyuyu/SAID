"""One combination only: fetched KR234 semantics and historical soft W20 math."""
import copy
import json
import random
import subprocess
import types

import numpy as np
import pytest
import torch

from recovery import w20_kr234_500 as experiment
from recovery import nested_d3_local_search as search, nested_d3_followup500 as runner
from train import nested_semantic_data as data
from train.train_nested_semantic_mask import build_optimizer
from tests.test_balanced_hparams import make,inputs
from tests.test_nested_d3_local_search import explicit_objective


@pytest.fixture
def configured(monkeypatch):
    for mod in (search,runner,search.local):
        for name in list(vars(mod)):
            if not name.startswith('__'):monkeypatch.setattr(mod,name,getattr(mod,name))
    experiment.configure();search.activate(experiment.ARM)


def test_only_two_authorized_config_changes(configured):
    anchor=json.loads((experiment.ANCHOR_EXP/'config.json').read_text())
    expected=experiment.expected_config()
    assert {k for k in anchor.keys()|expected.keys() if anchor.get(k)!=expected.get(k)}=={'view_weights','sampling_mode'}
    assert expected==json.loads(experiment.CANONICAL_CONFIG.read_text())==search.arm_config(experiment.ARM)
    assert expected['view_weights']==[1.4,1.4,.2] and sum(expected['view_weights'])==3.
    assert expected['inclusion_max']==1. and expected['workers']==8
    assert search.arm_sparsity(experiment.ARM)==[1.,2.,2.]
    experiment.frozen_config(expected,experiment.ARM)
    assert search.frozen_lowest_reference(experiment.ARM)
    assert list(search.ARMS)==[experiment.ARM]


@pytest.mark.parametrize('key,value',[
    ('hns_enabled',True),('inclusion_max',0),('view_weights',[1.35,1.35,.2]),
    ('sparsity_scale',2),('view_sparsity_weights',[1,2,3]),('workers',4),
    ('sampling_seed',1),('batch_size',128),('summary_t2i_weight',.5)])
def test_forbidden_change_rejected(configured,key,value):
    cfg=dict(experiment.expected_config(),**{key:value})
    with pytest.raises(AssertionError):experiment.frozen_config(cfg,experiment.ARM)


@pytest.mark.parametrize('completed',[0,99,199,499])
def test_actual_loss_every_gradient_AdamW_exact_fetched_KR234_no_HNS(completed):
    path=experiment.ROOT/'model/balanced_hparam_search.py'
    source=experiment.git_bytes(experiment.REMOTE_SEARCH,path)
    ns={'__name__':'model._fetched_kr234_W20_test','__package__':'model'}
    exec(compile(source,'fetchedKR234','exec'),ns)
    torch.manual_seed(8127)
    a=make(dict(view_weights=[1.4,1.4,.2]));a.inclusion_hierarchy='detail_chain'
    b=copy.deepcopy(a);b.forward=types.MethodType(ns['BalancedSearch'].forward,b)
    images,tokens,valid=inputs()
    la,loga=a(images,*tokens,valid,completed);lb,logb=b(images,*tokens,valid,completed)
    assert a.hns_enabled is False and not any(k.startswith('HNS_') for k in loga)
    torch.testing.assert_close(la,lb,atol=0,rtol=0)
    for k,v in logb.items():
        if torch.is_tensor(v):torch.testing.assert_close(loga[k],v,atol=0,rtol=0)
        else:assert loga[k]==v
    la.backward();lb.backward()
    for name,p in a.named_parameters():
        q=dict(b.named_parameters())[name]
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0,msg=name)
    oa,ob=build_optimizer(a),build_optimizer(b)
    assert [{k:v for k,v in g.items() if k!='params'} for g in oa.param_groups]==[
        {k:v for k,v in g.items() if k!='params'} for g in ob.param_groups]
    oa.step();ob.step()
    for name,p in a.named_parameters():torch.testing.assert_close(p,dict(b.named_parameters())[name],atol=0,rtol=0)


@pytest.mark.parametrize('completed',[0,99,200,499])
def test_independent_W20_objective_reconstruction_and_detached_child(completed):
    torch.manual_seed(513)
    a=make(dict(view_weights=[1.4,1.4,.2]));a.inclusion_hierarchy='detail_chain'
    b=copy.deepcopy(a);images,tokens,valid=inputs()
    actual,logs=a(images,*tokens,valid,completed);expected=explicit_objective(b,images,tokens,valid,completed)
    torch.testing.assert_close(actual,expected,atol=3e-4,rtol=3e-5)
    experiment.reconstruction(dict(step=completed+1,**{k:float(v) if torch.is_tensor(v) else v for k,v in logs.items()}))
    actual.backward();expected.backward()
    for name,p in a.named_parameters():
        q=dict(b.named_parameters())[name];assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=5e-4,rtol=5e-5,msg=name)


@pytest.mark.parametrize('caption',[
    'Only summary','x '*300,'Summary. A','Summary. A. B','Summary. A. B. C',
    'Summary. A. B. C. D','Summary. A. B. C. D. E. F. G. H'])
def test_fetched_KR234_RNG_fallback_order_text_tokens_exact(caption):
    old=experiment.fetched_data(experiment.REMOTE_SEARCH)
    before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    for epoch in (0,1,3):
        for sid in (0,123,887654):
            a=data.sampled_text_views(caption,'nested_detail_kr234',0,epoch,sid)
            b=old['sampled_text_views'](caption,'nested_detail_kr234',0,epoch,sid)
            for k in ('views','K','n','detail_indices','dall_indices','valid','reason'):
                assert a[k]==b[k]
            for k in ('tokens_f','tokens_o','tokens_e'):torch.testing.assert_close(a[k],b[k],atol=0,rtol=0)
    after=np.random.get_state()
    assert random.getstate()==before[0] and torch.equal(torch.get_rng_state(),before[2])
    assert before[1][0]==after[0] and np.array_equal(before[1][1],after[1]) and before[1][2:]==after[2:]


@pytest.mark.parametrize('score,long3,status,candidate',[
    (.7110790090368924,.7504650150614873,'NEGATIVE',False),
    (.7110790090368925,.7504650150614873,'STRONG_POSITIVE',True),
    (.712,.749,'STRONG_POSITIVE',True),(.712,.745,'POSITIVE',True),
    (.710,.751,'MIXED',False),(.710,.740,'NEGATIVE',False)])
def test_raw_score_only_candidacy_no_invented_negative_gate(score,long3,status,candidate):
    base=dict(Score5=.7110790090368924,J_long3=.7504650150614873,J_long=.8412,Short4=.652)
    q=dict(base,Score5=score,J_long3=long3)
    assert experiment.classify(q,base,dict(base,Score5=.710636),dict(base,Score5=.710883))==(status,candidate)


def test_no_full_third_arm_or_unbounded_publication(configured):
    assert list(runner.ARMS)==[experiment.ARM]
    assert runner.ENTRY=='recovery.w20_kr234_500'
    from recovery.check_stage500_publish import ALLOWED
    paths=experiment.publication_paths()
    assert all(str(p.relative_to(experiment.ROOT)) in ALLOWED for p in paths)
    assert all(p.suffix in ('.py','.md','.json') for p in paths)
    assert len([p for p in paths if p.parent.name=='evaluations'])==5
    assert not any(p.suffix=='.pt' for p in paths)


def test_complete_report_real_baseline_schemas_synthetic_fixture_only(tmp_path,monkeypatch):
    from recovery.s02_nfs500 import dump
    from experiments.nest_clip_v1.balanced_hparam_search_v1 import search as native
    exp=tmp_path/'exp';run=tmp_path/'run';exp.mkdir();run.mkdir()
    for name in ('RESULTS.json','TRAINING_DIAGNOSTICS.json','GRADIENT_SPOTCHECK.json','RUNTIME_STATS.json'):
        dump(exp/name,json.loads((experiment.KR_EXP/name).read_text()))
    dump(exp/'VALIDATION.json',dict(passed=True,synthetic_fixture=True))
    raw=tmp_path/'sample.json';raw.write_text('Synthetic fixture only\n')
    (tmp_path/'local-path-proof-5000.json').write_text('Synthetic fixture only\n')
    dump(exp/'MATCHED_PREFLIGHT.json',dict(passed=True,raw_evidence=dict(path=str(raw))))
    dump(exp/'SEARCH_PLAN.json',dict(prepared_utc='synthetic'))
    dump(exp/'BASELINE_PROVENANCE.json',dict(synthetic_fixture=True))
    dump(run/'commands.json',[])
    fixture=json.loads((exp/'RESULTS.json').read_text())
    sources={}
    for ds in fixture['metrics']:
        p=tmp_path/(ds+'.json');dump(p,dict(synthetic_fixture=True));sources[ds]=p
    monkeypatch.setattr(experiment,'EXP',exp);monkeypatch.setattr(search,'RUN',run)
    monkeypatch.setattr(native,'native_metrics',lambda path:(fixture['metrics'],{},sources))
    monkeypatch.setattr(experiment.subprocess,'check_output',lambda *args,**kwargs:'')
    experiment.summarize()
    actual=json.loads((exp/'RESULTS.json').read_text())
    assert actual['classification']=='NEGATIVE' and not actual['next_full_candidate']
    assert actual['comparisons']['KR234']['quality_delta_pp']['Score5']==0
    assert set(actual['comparisons'])=={'Anchor','W20','KR234'}
    assert actual['GPU_idle_after_evaluation'] and not actual['automatic_full']
    assert len(list((exp/'evaluations').glob('*.json')))==5
    assert 'HNS is disabled' in (exp/'REPORT.md').read_text()
