"""Weight-only D3 isolation, actual gradient scaling and predeclared gates."""
import copy
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from model.balanced_hparam_search import detail_chain_inclusion
from train import nested_semantic_data as data
from recovery import nested_detail_d3_balanced500 as experiment
from tests import test_balanced_hparams as reference


def test_only_declared_alignment_weights_change():
    old=json.loads((experiment.EQUAL_EXP/'config.json').read_text())
    new=json.loads(experiment.CONFIG.read_text())
    assert {k for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}=={'view_weights'}
    experiment.changed_config(new)
    experiment.changed_config(dict(new,code_sha256={},resume=None,completed_steps=5))
    for key,value in [('sampling_seed',1),('workers',4),('sparsity_scale',2),('inclusion_max',2)]:
        with pytest.raises(AssertionError,match='Config drift'):experiment.changed_config(dict(new,**{key:value}))


def test_loss_gradient_change_only_weighted_alignment(monkeypatch):
    torch.manual_seed(1010)
    module=reference.make(dict(view_weights=[1.35,1.35,.30]));module.inclusion_hierarchy='detail_chain'
    explicit=copy.deepcopy(module);equal=copy.deepcopy(module);equal.search_hparams['view_weights']=[1.,1.,1.]
    images,tokens,valid=reference.inputs()
    monkeypatch.setattr(reference,'inclusion',detail_chain_inclusion)
    manual=reference.weighted_reference(explicit,images,tokens,valid,499)
    actual,logs=module(images,*tokens,valid,499);previous,old_logs=equal(images,*tokens,valid,499)
    torch.testing.assert_close(actual,manual,atol=3e-4,rtol=3e-5)
    for key in ('inc','inc_weight','F_sparse','O_sparse','E_sparse',
        'F_Dall_mask_iou','Dall_Ds_mask_iou','Dall_F_hard_violation','Ds_Dall_hard_violation'):
        torch.testing.assert_close(logs[key],old_logs[key],atol=0,rtol=0)
    raw=[logs[p+'_i2t']+logs[p+'_t2i'] for p in ('F','O','E')]
    delta=10/3*sum((a-1)*v for a,v in zip([1.35,1.35,.30],raw))
    torch.testing.assert_close(actual-previous,delta,atol=3e-4,rtol=3e-5)
    actual.backward();manual.backward()
    for name,p in module.named_parameters():
        q=dict(explicit.named_parameters())[name]
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=5e-4,rtol=5e-5,msg=name)


def test_indices_observer_read_only_and_rng_isolated(monkeypatch):
    monkeypatch.setattr(experiment,'ORIGINAL_SAMPLING',data.sampling_diagnostics)
    captions=['Summary. First detail. Second detail. Third detail. Fourth detail','One sentence']
    samples=[dict(image=torch.zeros(3,2,2),sample_id=i,image_id=i,
        **data.sampled_text_views(c,'nested_detail_d3',0,0,i)) for i,c in enumerate(captions)]
    batch=data.collate(samples);before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    tensors={k:v.clone() for k,v in batch.items() if torch.is_tensor(v)}
    original=data.sampling_diagnostics(batch);current=experiment.selection_diagnostics(batch)
    assert {k:v for k,v in current.items() if k!='D3_selected_sentence_indices_sha256'}==original
    assert current['D3_selected_sentence_indices_sha256']==experiment.digest_indices([0,1],batch['detail_indices'])
    assert random.getstate()==before[0] and torch.equal(before[2],torch.get_rng_state())
    after=np.random.get_state()
    assert before[1][0]==after[0] and np.array_equal(before[1][1],after[1]) and before[1][2:]==after[2:]
    assert all(torch.equal(v,batch[k]) for k,v in tensors.items())


def test_full_stream_rejects_changed_indices_digest():
    health=[]
    for rank in range(4):
        ids=list(range(rank*256,(rank+1)*256));n=[5]*256
        s={k:'same' for k in experiment.STREAM_KEYS};s.update(sample_ids=ids,n=n,nested_d3_exact=True)
        choices=[data.sample_partial_detail_indices(5,0,0,sid) for sid in ids]
        s['D3_selected_sentence_indices_sha256']=experiment.digest_indices(ids,choices)
        health.append(dict(rank=rank,sampling=s,gradients_finite=True,batch=256,updates=1))
    row=dict(step=1,actual_lrs=[.001],loss=1.,nonfinite=0,rank_health=health)
    assert experiment.matched_rows([row],[copy.deepcopy(row)])==1024
    bad=copy.deepcopy(row);bad['rank_health'][0]['sampling']['D3_selected_sentence_indices_sha256']='changed'
    with pytest.raises(AssertionError):experiment.matched_rows([bad],[row])


def test_actual_weighted_gradient_norms_include_common_factor():
    g=dict(mean_gradient_norms=dict(F=10.,Dall=14.,D3=28.),
        batches=[dict(gradient_norms=dict(F=10.,Dall=14.,D3=28.)) for _ in range(8)])
    p=experiment.gradient_pressure(g)
    assert p['weighted_mean_norms']==dict(F=45.,Dall=63.,D3=28.)
    assert p['weighted_ratios']['D3_Dall']==pytest.approx(.30/1.35*2)
    assert p['D3_larger_than_F_plus_Dall_batches']==8
    assert p['weighted_D3_larger_than_F_plus_Dall_batches']==0
    assert p['no_longer_dominant'] and not p['dominant']


@pytest.mark.parametrize('score,long3,long2,short,urban,not_dominant,status',[
    (70.750319,74.4,83.5,65.3,88.5,True,'D3_BALANCED_STRONG_POSITIVE'),
    (70.7,74.4,83.5,65.3,88.5,False,'D3_BALANCED_POSITIVE'),
    (70.5,74.6,83.5,64.8,88.5,True,'D3_LONG_TRADEOFF'),
    (70.5,74.2,83.0,65.,88.3,True,'D3_BALANCED_NEGATIVE'),
    (70.6,74.35,83.2,65.,88.3,True,'D3_BALANCED_INCONCLUSIVE'),
])
def test_user_classification_gates(score,long3,long2,short,urban,not_dominant,status):
    r=experiment.classify(dict(Score5=score,J_long3=long3,J_long=long2,Short4=short),
        {'Urban-1k':{'I2T':{'R@1':.91},'T2I':{'R@1':urban/100}}},
        dict(no_longer_dominant=not_dominant,gradient_balance_improved=True))
    assert r['status']==status and not r['automatic_continuation'] and not r['automatic_new_experiments']
