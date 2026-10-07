"""Private KR2M1 RNG, absolute WeakSparse math and fail-closed sequential queue."""
import copy
import json
import math
import random
import sys

import numpy as np
import pytest
import torch

from recovery import nested_d3_followup500 as followup
from recovery import nested_d3_local_search as search
from train import nested_semantic_data as data
from model.balanced_hparam_search import BalancedSearch
from tests.test_nested_fusion import TinyFusionCLIP
from tests import test_balanced_hparams as reference
from tests.test_nested_d3_local_search import explicit_objective


@pytest.fixture
def configured(monkeypatch):
    for name in ('EXP','RUN_ROOT','BRANCH','ARMS','ENTRY_MODULE','PHASE_PREFIX','EXTRA_SOURCES',
                 'ARM','RUN','PHASE','ARM_EXP','CONFIG'):
        monkeypatch.setattr(search,name,getattr(search,name))
    followup.configure()


@pytest.mark.parametrize('m',[0,1,2,3,4,5,9,15])
def test_kr2m1_order_strict_subset_uniformity_and_no_global_rng(m):
    before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    allowed=list(range(2,m)) if m>=3 else [1] if m else [0]
    counts={k:0 for k in allowed}
    for sid in range(4000):
        chosen=data.sample_kr2m1_detail_indices(m+1,0,2,sid)
        assert chosen==data.sample_kr2m1_detail_indices(m+1,0,2,sid)
        assert chosen==sorted(set(chosen)) and all(1<=j<=m for j in chosen)
        assert len(chosen) in allowed
        if m>=2:assert len(chosen)<m
        if len(chosen)==3:assert chosen==data.sample_partial_detail_indices(m+1,0,2,sid)
        counts[len(chosen)]+=1
    expected=4000/len(allowed)
    assert max(abs(v-expected) for v in counts.values())<6*math.sqrt(expected)
    after=np.random.get_state()
    assert before[0]==random.getstate() and torch.equal(before[2],torch.get_rng_state())
    assert before[1][0]==after[0] and np.array_equal(before[1][1],after[1]) and before[1][2:]==after[2:]


@pytest.mark.parametrize('caption',['Only summary','x '*300,'Summary. A',
    'Summary. A. B','Summary. A. B. C',
    'Summary. A. B. C. D. E. F. G. H. I. J'])
def test_kr2m1_F_Dall_tokens_fallback_and_runtime_observer(caption):
    fixed=data.sampled_text_views(caption,'nested_detail_d3',0,0,444)
    kr=data.sampled_text_views(caption,'nested_detail_kr2m1',0,0,444)
    assert kr['views'][:2]==fixed['views'][:2] and kr['valid']==fixed['valid']
    assert torch.equal(kr['tokens_f'],fixed['tokens_f']) and torch.equal(kr['tokens_o'],fixed['tokens_o'])
    if not kr['valid']:assert kr['views']==fixed['views'] and torch.equal(kr['tokens_e'],fixed['tokens_e'])
    else:
        text='. '.join(kr['views'][0].split('. ')[j] for j in kr['detail_indices'])
        assert text==kr['views'][2]
        assert torch.equal(kr['tokens_e'],data.longclip.tokenize([text],truncate=False)[0])
    batch=data.collate([dict(image=torch.zeros(3,2,2),sample_id=444,image_id=0,**kr)])
    assert batch['random_detail_k_rule']=='kr2m1'
    assert search.observe_selection(batch)['nested_d3_exact']


def test_kr2m1_identity_includes_seed_epoch_sample():
    base=[data.sample_kr2m1_detail_indices(12,0,0,s) for s in range(20)]
    assert base!=[data.sample_kr2m1_detail_indices(12,1,0,s) for s in range(20)]
    assert base!=[data.sample_kr2m1_detail_indices(12,0,1,s) for s in range(20)]


@pytest.mark.parametrize('arm',list(followup.ARMS))
def test_configs_single_axis_frozen_no_combination(configured,arm):
    anchor=json.loads((search.ANCHOR_EXP/'config.json').read_text());cfg=search.arm_config(arm)
    delta={k for k in anchor.keys()|cfg.keys() if anchor.get(k)!=cfg.get(k)}
    assert delta==({'sampling_mode'} if arm=='KR2M1' else {'view_sparsity_weights'})
    search.frozen_config(cfg,arm)
    assert cfg['view_weights']==[1.35,1.35,.3] and cfg['seed']==0 and cfg['workers']==8
    if arm!='KR2M1':assert search.arm_sparsity(arm)==[.5,1.,1.5] and sum(search.arm_sparsity(arm))==3
    other=dict(cfg,sampling_mode='nested_detail_kr2m1') if arm!='KR2M1' else dict(cfg,view_sparsity_weights=[.5,1.,1.5])
    with pytest.raises(AssertionError):search.frozen_config(other,arm)
    for key,value in [('workers',4),('inclusion_max',2),('fusion_lr',1e-4),('batch_size',128)]:
        with pytest.raises(AssertionError):search.frozen_config(dict(cfg,**{key:value}),arm)


def test_weak_sparse_constructor_keeps_absolute_mass_and_objective_gradients():
    torch.manual_seed(853)
    base=reference.make(dict(view_weights=[1.35,1.35,.3]));base.inclusion_hierarchy='detail_chain'
    variant=BalancedSearch(TinyFusionCLIP(32),search_hparams=dict(view_weights=[1.35,1.35,.3]),
        inclusion_hierarchy='detail_chain',view_sparsity_weights=[.5,1.,1.5],
        fusion='balanced_stack',visual='patch',text_tokens=6,checkpoint_encoders=False,image_chunk=2,text_chunk=2)
    variant.load_state_dict(base.state_dict(),strict=True)
    assert variant.view_sparsity_weights==[.5,1.,1.5]
    manual=copy.deepcopy(variant);images,tokens,valid=reference.inputs()
    old,old_logs=base(images,*tokens,valid,137);loss,logs=variant(images,*tokens,valid,137)
    target=explicit_objective(manual,images,tokens,valid,137)
    delta=sum((w-b)*logs[p+'_sparse'] for w,b,p in zip([.5,1.,1.5],[1.,2.,2.],['F','O','E']))/3
    torch.testing.assert_close(loss-old,delta,atol=3e-4,rtol=3e-5)
    torch.testing.assert_close(loss,target,atol=3e-4,rtol=3e-5)
    for key in ('inc','inc_weight','F_sparse','O_sparse','E_sparse','F_Dall_mask_iou',
                'Dall_Ds_mask_iou','Dall_F_hard_violation','Ds_Dall_hard_violation'):
        torch.testing.assert_close(logs[key],old_logs[key],atol=0,rtol=0)
    loss.backward();target.backward()
    for name,p in variant.named_parameters():
        q=dict(manual.named_parameters())[name]
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=5e-4,rtol=5e-5)


@pytest.mark.parametrize('hist,median',[({'2':2,'4':2},3.),({'1':1,'3':4},3.),({},None)])
def test_exact_histogram_median(hist,median):
    assert followup.frequency_median(hist)==median


@pytest.mark.parametrize('fail_first',[False,True])
def test_sequential_queue_second_only_after_first_complete_report(configured,monkeypatch,tmp_path,fail_first):
    events=[]
    monkeypatch.setattr(followup,'RUN_ROOT',tmp_path/'runtime')
    monkeypatch.setattr(followup,'EXP',tmp_path/'queue')
    monkeypatch.setattr(followup,'configure',lambda:None)
    monkeypatch.setattr(search,'prepare',lambda:events.append('prepared'))
    for arm in followup.ARMS:
        spec=dict(search.ARMS[arm],experiment_dir=str(tmp_path/arm))
        monkeypatch.setitem(search.ARMS,arm,spec)
    monkeypatch.setattr(search,'activate',lambda arm:monkeypatch.setattr(search,'ARM',arm))
    class FakeSupervisor:
        def run(self):
            arm=search.ARM;events.append('start:'+arm)
            assert search.arm_config(arm)['view_weights']==[1.35,1.35,.3]
            if fail_first:raise RuntimeError('deliberate first-arm failure')
            p=search.experiment_dir(arm);p.mkdir()
            (p/'VALIDATION.json').write_text('{"passed":true}')
            (p/'REPORT.md').write_text('complete review/report')
            events.append('review/report:'+arm)
    monkeypatch.setattr(search,'Supervisor',FakeSupervisor)
    monkeypatch.setattr(followup,'augment_diagnostics',lambda arm:events.append('diagnostics:'+arm))
    monkeypatch.setattr(followup,'summarize',lambda:events.append('summary'))
    monkeypatch.setattr(followup,'publish',lambda:events.append('publish') or {'passed':True})
    monkeypatch.setattr(followup.signal,'signal',lambda *args:None)
    monkeypatch.setattr(sys,'argv',['runner'])
    if fail_first:
        with pytest.raises(RuntimeError,match='deliberate'):followup.main()
        assert events==['prepared','start:KR2M1']
        assert json.loads((followup.EXP/'QUEUE_STATE.json').read_text())['status']=='STOPPED_WITH_EVIDENCE'
    else:
        followup.main()
        assert events==['prepared','start:KR2M1','review/report:KR2M1','diagnostics:KR2M1',
            'start:WeakSparse-051015','review/report:WeakSparse-051015','diagnostics:WeakSparse-051015','summary','publish']
        assert json.loads((followup.EXP/'QUEUE_STATE.json').read_text())['status']=='COMPLETED_AND_SYNCED'
