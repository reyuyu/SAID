"""Single-axis isolation, coefficient/gradient math, private K RNG and selection gates."""
import copy
import json
import math
import random

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from recovery import nested_d3_local_search as search
from recovery import nested_d3_local_search_evidence as evidence
from train import nested_semantic_data as data
from model.balanced_hparam_search import BalancedSearch,detail_chain_inclusion
from model.nested_semantic_mask import hard_st
from tests import test_balanced_hparams as reference


@pytest.mark.parametrize('arm',list(search.ARMS))
def test_each_arm_changes_only_its_declared_axis(arm):
    anchor=json.loads((search.ANCHOR_EXP/'config.json').read_text());cfg=search.arm_config(arm)
    changed={k for k in anchor.keys()|cfg.keys() if anchor.get(k)!=cfg.get(k)}
    expected={'view_weights'} if arm.startswith('W') else {'view_sparsity_weights'} if arm.startswith('S') else {'sampling_mode'}
    assert changed==expected
    search.frozen_config(cfg,arm)
    for key,value in [('workers',4),('inclusion_max',2),('sparsity_scale',2),('sampling_seed',1),('fusion_lr',1e-4)]:
        with pytest.raises(AssertionError):search.frozen_config(dict(cfg,**{key:value}),arm)
    mixed=copy.deepcopy(cfg)
    if arm.startswith('S'):mixed['view_weights']=[1.4,1.4,.2]
    else:mixed['view_sparsity_weights']=search.sparsity_coefficients(2.5)
    with pytest.raises(AssertionError):search.frozen_config(mixed,arm)


@pytest.mark.parametrize('r',[2.,2.5,3.])
def test_sparsity_mass_and_parent_ratio_exact(r):
    c=search.sparsity_coefficients(r)
    assert sum(c)==pytest.approx(5,abs=1e-12)
    assert c[1]/c[0]==pytest.approx(2) and c[2]/c[0]==pytest.approx(r)
    assert math.isclose(sum(c),5.,abs_tol=1e-12)


def explicit_objective(module,images,tokens,valid,completed):
    values=[];probabilities=[]
    for j,view in enumerate(tokens):
        z,t,zv,zt=reference.explicit_inputs(module,images,view)
        p=reference.explicit_logits(module,zv,zt).sigmoid();mask=hard_st(p)
        scores=100*(F.normalize(z[None]*mask,dim=-1,eps=1e-6)*F.normalize(t,dim=-1,eps=1e-6)[:,None]).sum(-1)
        enabled=torch.ones_like(valid) if j==0 else valid
        labels=torch.arange(len(z))[enabled]
        ce=F.cross_entropy(scores.T[enabled].masked_fill(~enabled[None],-torch.inf),labels)+F.cross_entropy(scores[enabled].masked_fill(~enabled[None],-torch.inf),labels)
        sparse=mask.diagonal(dim1=0,dim2=1).T[enabled].abs().mean()
        values.append((ce,sparse));probabilities.append(p.diagonal(dim1=0,dim2=1).T)
    weights=module.search_hparams['view_weights'];c=module.view_sparsity_weights
    return (10/3*sum(w*v[0] for w,v in zip(weights,values))+
            sum(w*v[1] for w,v in zip(c,values))/3+
            min(1,completed/200)*detail_chain_inclusion(*probabilities)[valid].mean())


@pytest.mark.parametrize('arm',['W20','W25','W35','W40','S25','S30'])
def test_actual_loss_and_gradients_match_only_declared_change(arm):
    torch.manual_seed(774)
    base=reference.make(dict(view_weights=[1.35,1.35,.3]));base.inclusion_hierarchy='detail_chain'
    variant=copy.deepcopy(base);variant.search_hparams['view_weights']=search.ARMS[arm]['weights']
    variant.view_sparsity_weights=search.sparsity_coefficients(search.ARMS[arm]['r'])
    manual=copy.deepcopy(variant);images,tokens,valid=reference.inputs()
    old,old_logs=base(images,*tokens,valid,137);loss,logs=variant(images,*tokens,valid,137)
    target=explicit_objective(manual,images,tokens,valid,137)
    torch.testing.assert_close(loss,target,atol=3e-4,rtol=3e-5)
    for key in ('inc','inc_weight','F_sparse','O_sparse','E_sparse','F_Dall_mask_iou','Dall_Ds_mask_iou','Dall_F_hard_violation','Ds_Dall_hard_violation'):
        torch.testing.assert_close(old_logs[key],logs[key],atol=0,rtol=0)
    if arm.startswith('W'):
        delta=10/3*sum((w-b)*(logs[p+'_i2t']+logs[p+'_t2i']) for w,b,p in zip(search.ARMS[arm]['weights'],[1.35,1.35,.3],['F','O','E']))
    else:
        delta=sum((w-b)*logs[p+'_sparse'] for w,b,p in zip(variant.view_sparsity_weights,[1.,2.,2.],['F','O','E']))/3
    torch.testing.assert_close(loss-old,delta,atol=3e-4,rtol=3e-5)
    loss.backward();target.backward()
    for name,p in variant.named_parameters():
        q=dict(manual.named_parameters())[name];assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=5e-4,rtol=5e-5,msg=name)


@pytest.mark.parametrize('m,valid',[(-1,[0]),(0,[0]),(1,[1]),(2,[1]),(3,[2]),(4,[2,3]),(5,[2,3,4]),(9,[2,3,4])])
def test_KR234_rules_order_without_replacement_and_uniformity(m,valid):
    before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone());counts={k:0 for k in valid}
    for sid in range(2400):
        indices=data.sample_random_partial_detail_indices(m+1,0,0,sid)
        assert indices==data.sample_random_partial_detail_indices(m+1,0,0,sid)
        assert indices==sorted(set(indices)) and all(1<=j<=m for j in indices)
        assert len(indices) in valid;counts[len(indices)]+=1
        if m>=2:assert len(indices)<m
        if len(indices)==3:assert indices==data.sample_partial_detail_indices(m+1,0,0,sid)
    if len(valid)>1:
        assert max(abs(n-2400/len(valid)) for n in counts.values())<100
    after=np.random.get_state()
    assert before[0]==random.getstate() and torch.equal(before[2],torch.get_rng_state())
    assert before[1][0]==after[0] and np.array_equal(before[1][1],after[1]) and before[1][2:]==after[2:]


@pytest.mark.parametrize('caption',['Only summary','x '*300,'Summary. A. B','Summary. A. B. C. D. E. F'])
def test_KR234_text_packing_tokens_fallback_and_observer(caption):
    fixed=data.sampled_text_views(caption,'nested_detail_d3',0,0,123)
    kr=data.sampled_text_views(caption,'nested_detail_kr234',0,0,123)
    assert fixed['views'][:2]==kr['views'][:2] and fixed['valid']==kr['valid']
    assert torch.equal(fixed['tokens_f'],kr['tokens_f']) and torch.equal(fixed['tokens_o'],kr['tokens_o'])
    if kr['valid']:
        text='. '.join(kr['views'][0].split('. ')[j] for j in kr['detail_indices'])
        assert text==kr['views'][2]
        assert torch.equal(kr['tokens_e'],data.longclip.tokenize([text],truncate=False)[0])
    else:assert fixed['views']==kr['views'] and torch.equal(fixed['tokens_e'],kr['tokens_e'])
    sample=dict(image=torch.zeros(3,2,2),sample_id=123,image_id=0,**kr)
    batch=data.collate([sample]);assert batch['random_detail_k']
    original={k:v.clone() for k,v in batch.items() if torch.is_tensor(v)}
    s=search.observe_selection(batch);assert s['nested_d3_exact']
    assert all(torch.equal(v,batch[k]) for k,v in original.items())


def result(score=71.063600,long3=74.954666,short=65.227000,urban=89.7):
    return dict(scores_percent=dict(Score5=score,J_long3=long3,J_long=84.025001,Short4=short),
        metrics={'Urban-1k':{'I2T':{'R@1':.909},'T2I':{'R@1':urban/100}}})


@pytest.mark.parametrize('score,long3,short,urban,status',[
    (71.1,74.954666,65.077,89.7,'NEW_500_BEST'),
    (71.1,74.9,65.227,89.8,'PARETO_POSITIVE'),
    (71.1,74.954666,65.0,89.7,'NO_IMPROVEMENT'),
    (71.0636,74.954666,65.227,89.7,'NO_IMPROVEMENT'),
    (71.02,75.1,65.227,89.7,'PARETO_POSITIVE'),
])
def test_winner_and_small_tradeoff_rules(score,long3,short,urban,status):
    assert evidence.classify(result(score,long3,short,urban),True)==status


def test_Pareto_dominance_includes_Short4():
    anchor=evidence.quality(result());better=evidence.quality(result(score=71.2))
    tradeoff=evidence.quality(result(score=71.3,short=65.1))
    assert evidence.dominates(better,anchor) and not evidence.dominates(anchor,better)
    assert not evidence.dominates(tradeoff,better) and not evidence.dominates(better,tradeoff)


def test_sparsity_first5_gate_allows_only_runtime_coefficient_metadata(monkeypatch):
    monkeypatch.setattr(search,'ARM','S25')
    cfg=search.arm_config('S25')
    cfg.update(start_updates=0,resume=None,init_sha256=search.STEP0_SHA,max_updates=500,run_type='formal',
        runtime_model=dict(search_hparams=dict(view_weights=[1.35,1.35,.3]),view_sparsity_weights=search.sparsity_coefficients(2.5)),
        code_sha256={p:search.sha(search.ROOT/p) for p in search.EDITED})
    ref_cfg=copy.deepcopy(cfg);ref_cfg['runtime_model'].pop('view_sparsity_weights')
    ref_cfg['code_sha256']={p:'old' for p in search.EDITED}
    opt=dict(param_groups=[],state={0:{'step':torch.tensor(5.)}})
    payload=dict(config=cfg,completed_steps=5,scheduler_horizon=4868,optimizer=opt,
        model={'w':torch.ones(2)},adapter={'w':torch.ones(2)})
    reference_payload=dict(payload,config=ref_cfg)
    assert search.checkpoint_invariants(payload,reference_payload)['passed']
    bad=copy.deepcopy(payload);bad['config']['runtime_model']['view_sparsity_weights']=[1.,2.,2.]
    with pytest.raises(AssertionError):search.checkpoint_invariants(bad,reference_payload)
