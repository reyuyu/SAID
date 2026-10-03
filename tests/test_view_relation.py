"""Sibling stop-gradient, zero margin, original-loss recovery and local reuse."""
import copy
import json

import pytest
import torch

from model.view_relation import sibling_terms, RelationCollapseMonitor
from model.nested_semantic_mask import hard_st
from tests.test_balanced_hparams import make, inputs
from train.train_nested_semantic_mask import build_optimizer


def example(direction):
    z=torch.tensor([[1.,2.,3.]],requires_grad=True)
    pp=torch.tensor([[.8,.8,.2]],requires_grad=True)
    pr=torch.tensor([[.2,.8,.8]],requires_grad=True)
    mp,mr=hard_st(pp),hard_st(pr)
    tp=torch.tensor([[1.,2.,0.]],requires_grad=True)
    tr=torch.tensor([[0.,2.,3.]],requires_grad=True)
    if direction=='P':tp=torch.tensor([[.1,1.,2.]],requires_grad=True)
    if direction=='R':tr=torch.tensor([[2.,1.,.1]],requires_grad=True)
    return z,tp,tr,pp,pr,mp,mr


def test_satisfied_relations_have_exact_zero_loss_and_zero_extra_gradients():
    z,tp,tr,pp,pr,mp,mr=example('none');loss,values=sibling_terms(z,tp,tr,mp,mr)
    assert values['c_PP']>values['c_RP'] and values['c_RR']>values['c_PR']
    assert float(loss)==0
    loss.sum().backward()
    for value in (z,tp,tr,pp,pr):assert torch.equal(value.grad,torch.zeros_like(value))


@pytest.mark.parametrize('direction',['P','R'])
def test_only_own_mask_receives_violation_gradient_and_z_text_remain_trainable(direction):
    z,tp,tr,pp,pr,mp,mr=example(direction);loss,values=sibling_terms(z,tp,tr,mp,mr)
    assert values[direction+'_violation']>0
    assert values[('R' if direction=='P' else 'P')+'_violation']==0
    loss.sum().backward()
    own,cross,text=(pp,pr,tp) if direction=='P' else (pr,pp,tr)
    assert z.grad.abs().sum()>0 and text.grad.abs().sum()>0 and own.grad.abs().sum()>0
    assert torch.equal(cross.grad,torch.zeros_like(cross))
    other=tr if direction=='P' else tp
    assert torch.equal(other.grad,torch.zeros_like(other))


def test_equal_cosines_have_zero_margin_and_zero_gradient():
    z,tp,tr,pp,pr,mp,mr=example('none');loss,_=sibling_terms(z,tp,tr,mp,mp)
    assert float(loss)==0
    loss.sum().backward()
    assert torch.equal(z.grad,torch.zeros_like(z))


@pytest.mark.parametrize('completed',[0,100,199,200,499])
def test_disabled_sibling_recovers_all_gradients_adamw_parameters_and_states_bitwise(completed):
    torch.manual_seed(1771);baseline=make(dict(fusion_lr=2e-4));variant=copy.deepcopy(baseline)
    variant.view_relation=True;variant.sibling_coefficient=0
    images,views,valid=inputs();a,_=baseline(images,*views,valid,completed);b,logs=variant(images,*views,valid,completed)
    torch.testing.assert_close(a,b,atol=0,rtol=0);assert logs['sib_loss']==0
    a.backward();b.backward();oa,ob=build_optimizer(baseline),build_optimizer(variant)
    for name,p in baseline.named_parameters():
        q=dict(variant.named_parameters())[name];assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0,msg=name)
    oa.step();ob.step()
    for name,p in baseline.named_parameters():
        q=dict(variant.named_parameters())[name];torch.testing.assert_close(p,q,atol=0,rtol=0,msg=name)
        assert (p in oa.state)==(q in ob.state)
        if p in oa.state:
            for key,value in oa.state[p].items():torch.testing.assert_close(value,ob.state[q][key],atol=0,rtol=0,msg=name+key)


@pytest.mark.parametrize('valid_count',[0,1,2,3])
def test_no_extra_encoder_calls_and_f_only_fallback(valid_count):
    torch.manual_seed(1773);module=make(dict(fusion_lr=2e-4));module.view_relation=True
    counters={'image':0,'text':0};old_image,old_text=module.clip.encode_image,module.clip.encode_text
    def image(*args,**kwargs):counters['image']+=1;return old_image(*args,**kwargs)
    def text(*args,**kwargs):counters['text']+=1;return old_text(*args,**kwargs)
    module.clip.encode_image=image;module.clip.encode_text=text
    images,views,valid=inputs();valid[:]=False;valid[:valid_count]=True
    loss,logs=module(images,*views,valid,499);assert torch.isfinite(loss)
    assert counters=={'image':1,'text':3 if valid_count>=2 else 1}
    assert logs['F_candidates']==3 and logs['O_candidates']==(valid_count if valid_count>=2 else 0)
    if valid_count<2:assert logs['sib_loss']==logs['relation_loss']==logs['inc']==0
    else:
        torch.testing.assert_close(logs['relation_loss'],logs['inc']+logs['sib_loss'],atol=1e-7,rtol=1e-6)
        for key in ['c_PP_mean','c_RP_mean','c_RR_mean','c_PR_mean']:assert -1.000001<=float(logs[key])<=1.000001


def test_only_two_requested_constraints_and_unscaled_cosine():
    values=sibling_terms(*[example('P')[i] for i in (0,1,2,5,6)])
    loss,c=values
    expected=.5*(torch.relu(c['c_RP']-c['c_PP'])+torch.relu(c['c_PR']-c['c_RR']))
    torch.testing.assert_close(loss,expected,atol=0,rtol=0)
    assert loss.item()<=2


def test_collapse_monitor_preserves_healthy_baseline_and_stops_persistent_extreme(tmp_path):
    path=tmp_path/'reference.jsonl';path.write_text(''.join(json.dumps(dict(step=s,oe_iou=.8))+'\n' for s in range(1,50)))
    cfg=dict(relation_baseline_steps=str(path),relation_guard_window=3,relation_guard_patience=2)
    monitor=RelationCollapseMonitor(cfg)
    healthy=dict(oe_iou=.75,F_positive_keep_ratio=.6,O_positive_keep_ratio=.6,E_positive_keep_ratio=.6)
    for step in range(1,10):assert monitor.update(step,healthy) is None
    collapsed=dict(healthy,oe_iou=.01)
    reason=None
    for step in range(10,20):
        reason=monitor.update(step,collapsed)
        if reason:break
    assert 'IoU collapsed' in reason
