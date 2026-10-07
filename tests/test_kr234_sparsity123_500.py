"""Literal mass6 objective, frozen full KR234 stream, and declared result gates."""
import copy
import json

import pytest
import torch

from recovery import kr234_sparsity123_500 as experiment
from recovery import nested_d3_local_search as search
from recovery import nested_d3_followup500 as runner
from model.balanced_hparam_search import BalancedSearch
from tests.test_nested_fusion import TinyFusionCLIP
from tests import test_balanced_hparams as reference
from tests.test_nested_d3_local_search import explicit_objective


@pytest.fixture
def configured(monkeypatch):
    for name in ('EXP','RUN_ROOT','BRANCH','ARMS','ENTRY_MODULE','PHASE_PREFIX','EXTRA_SOURCES',
                 'ARM','RUN','PHASE','ARM_EXP','CONFIG','ANCHOR_EXP','ANCHOR_RUN'):
        monkeypatch.setattr(search,name,getattr(search,name))
    for name in ('EXP','RUN_ROOT','BRANCH','ARMS','ENTRY','PUBLISH_MESSAGE','MAIN_LOG','IDENTITY','configure','summarize'):
        monkeypatch.setattr(runner,name,getattr(runner,name))
    experiment.configure()


def test_only_config_delta_is_literal_sparsity(configured):
    baseline=json.loads((experiment.BASE_EXP/'config.json').read_text())
    cfg=search.arm_config(experiment.ARM)
    assert {k for k in baseline.keys()|cfg.keys() if baseline.get(k)!=cfg.get(k)}=={'view_sparsity_weights'}
    assert cfg['view_sparsity_weights']==[1.,2.,3.] and sum(cfg['view_sparsity_weights'])==6
    assert cfg['sparsity_scale']==baseline['sparsity_scale']==1
    assert search.frozen_lowest_reference(experiment.ARM)
    assert search.ANCHOR_RUN==experiment.BASE_RUN
    search.frozen_config(cfg,experiment.ARM)
    for key,value in [('sampling_mode','nested_detail_kr2m1'),('view_sparsity_weights',[5/6,10/6,15/6]),
                      ('inclusion_max',2),('sparsity_scale',1.2),('view_weights',[1.,1.,1.]),('workers',4)]:
        with pytest.raises(AssertionError):search.frozen_config(dict(cfg,**{key:value}),experiment.ARM)


def test_literal_sparsity_only_changes_lowest_loss_and_exact_gradients():
    torch.manual_seed(1483)
    base=reference.make(dict(view_weights=[1.35,1.35,.3]));base.inclusion_hierarchy='detail_chain'
    variant=BalancedSearch(TinyFusionCLIP(32),search_hparams=dict(view_weights=[1.35,1.35,.3]),
        inclusion_hierarchy='detail_chain',view_sparsity_weights=[1.,2.,3.],
        fusion='balanced_stack',visual='patch',text_tokens=6,checkpoint_encoders=False,image_chunk=2,text_chunk=2)
    variant.load_state_dict(base.state_dict(),strict=True);assert variant.view_sparsity_weights==[1.,2.,3.]
    manual=copy.deepcopy(variant);images,tokens,valid=reference.inputs()
    old,old_logs=base(images,*tokens,valid,137);loss,logs=variant(images,*tokens,valid,137)
    target=explicit_objective(manual,images,tokens,valid,137)
    torch.testing.assert_close(loss-old,logs['E_sparse']/3,atol=3e-4,rtol=3e-5)
    torch.testing.assert_close(loss,target,atol=3e-4,rtol=3e-5)
    for key in ('F_i2t','F_t2i','O_i2t','O_t2i','E_i2t','E_t2i','inc','inc_weight',
                'F_sparse','O_sparse','E_sparse','F_Dall_mask_iou','Dall_Ds_mask_iou',
                'Dall_F_hard_violation','Ds_Dall_hard_violation'):
        torch.testing.assert_close(logs[key],old_logs[key],atol=0,rtol=0)
    loss.backward();target.backward()
    for name,p in variant.named_parameters():
        q=dict(manual.named_parameters())[name];assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=5e-4,rtol=5e-5)


def frozen_step():
    health=[]
    for rank in range(4):
        ids=list(range(rank*256+1000,(rank+1)*256+1000));ns=[6]*256
        choices=[search.expected_indices(n,sid,experiment.ARM) for n,sid in zip(ns,ids)]
        s={k:'frozen:'+k for k in search.FULL_STREAM}
        s.update(sample_ids=ids,n=ns,K=[len(c) for c in choices],nested_d3_exact=True,
            lowest_selected_sentence_indices_sha256=search.indices_digest(ids,choices))
        health.append(dict(rank=rank,sampling=s,gradients_finite=True,batch=256,updates=1))
    return dict(step=1,epoch=0,actual_lrs=[1e-6,1e-3,1e-3,2e-4],loss=1.,nonfinite=0,rank_health=health)


def test_all_K_text_tokens_indices_LR_are_required(configured):
    a=frozen_step();assert search.matched_stream([a],[copy.deepcopy(a)],experiment.ARM)['records']==1024
    assert search.matched_stream([a],[a],experiment.ARM)['all_lowest_strings_tokens_exact_anchor']


@pytest.mark.parametrize('field',['local_views_sha256','split_sha256','full_view_sha256',
    'summary_view_sha256','lowest_selected_sentence_indices_sha256','K','sample_ids','LR'])
def test_actual_stream_drift_rejected(configured,field):
    a=frozen_step();b=copy.deepcopy(a)
    if field=='LR':b['actual_lrs'][0]*=2
    elif field in ('K','sample_ids'):b['rank_health'][0]['sampling'][field][0]+=1
    else:b['rank_health'][0]['sampling'][field]='drift'
    with pytest.raises(AssertionError):search.matched_stream([a],[b],experiment.ARM)


@pytest.mark.parametrize('deltas,drop,status',[
    ({'Score5':.1},.02,'KR234_S123_STRONG_POSITIVE'),
    ({'J_long3':.1,'Urban_T2I':-.3},.02,'SPARSITY_TRADEOFF'),
    ({'J_long':.1,'Short4':-.2},.02,'SPARSITY_TRADEOFF'),
    ({'Score5':-.1,'J_long3':-.1,'Urban_T2I':-.1},.02,'SPARSITY_OVERREGULARIZED'),
    ({},0.,'NO_IMPROVEMENT'),
])
def test_classification_thresholds(deltas,drop,status):
    base=dict(Score5=71.107901,J_long3=75.046502,J_long=84.120002,Short4=65.2,Urban_I2T=91.6,Urban_T2I=89.2)
    q={k:v+deltas.get(k,0) for k,v in base.items()}
    old=dict(keep_ratio=dict(F=.852,Dall=.842,Dk=.78236),last50=dict(Dk_Dall_hard_violation=.0227))
    mask=copy.deepcopy(old);mask['keep_ratio']['Dk']-=drop
    assert experiment.classify(q,base,mask,old)==status


def test_only_one_background_arm_and_full_publication_reports(configured):
    assert list(runner.ARMS)==['KR234-S123']
    assert runner.summarize is experiment.summarize
    assert runner.ENTRY=='recovery.kr234_sparsity123_500'
    assert runner.PUBLISH_MESSAGE=='Report isolated KR234 literal sparsity1/2/3 local500 experiment'
    assert search.arm_config(experiment.ARM)['sampling_mode']=='nested_detail_kr234'
    assert experiment.EXP/'GRADIENT_SPOTCHECK.json' in runner.publication_paths()
