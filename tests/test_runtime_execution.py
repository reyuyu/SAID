"""Mathematical guardrails for audit, ordered fused views, cache and Hard-ST scoring."""
import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from model.runtime_execution import configure_checkpointing, reduced_masked_score
from model.nested_semantic_mask import hard_st
from model.nested_fusion_mask import pair_logits
from tests.test_balanced_hparams import make, inputs
from train.runtime_audit import audit_step


def test_sparse_audit_schedule_and_benchmark_policy():
    required=(1,5,100,200,500,1000,1500)
    assert all(audit_step(s,'sparse') for s in required)
    assert not any(audit_step(s,'sparse') for s in (2,4,6,101,501))
    for key in ('epoch_boundary','checkpoint','final'):
        assert audit_step(17,'sparse',**{key:True})
    assert audit_step(2,'full') and not audit_step(1,'benchmark',final=True,checkpoint=True)


@pytest.mark.parametrize('closed',[False,True])
def test_reduced_score_preserves_hard_st_mask_gradient_even_when_all_closed(closed):
    torch.manual_seed(1763)
    z=torch.randn(7,32,requires_grad=True);t=torch.randn(5,32,requires_grad=True)
    logits=(torch.full((5,7,32),-2.) if closed else torch.randn(5,7,32)).requires_grad_()
    mask=hard_st(logits.sigmoid());normalized=F.normalize(t,dim=-1,eps=1e-6)
    old=100*(F.normalize(z[None]*mask,dim=-1,eps=1e-6)*normalized[:,None]).sum(-1)
    new=reduced_masked_score(z,normalized,mask)
    torch.testing.assert_close(old,new,atol=3e-5,rtol=2e-6)
    a=torch.autograd.grad(old.sum(),(z,t,logits),retain_graph=True)
    b=torch.autograd.grad(new.sum(),(z,t,logits))
    for x,y in zip(a,b):
        assert torch.isfinite(y).all()
        torch.testing.assert_close(x,y,atol=3e-5,rtol=3e-6)


def test_cache_slices_keep_text_first_visual_second_and_all_gradients():
    module=make();torch.manual_seed(15)
    with torch.no_grad():module.fusion_branch.gate.weight.normal_(0,.1)
    visual=(torch.randn(4),torch.randn(4,32,requires_grad=True))
    text=(torch.randn(3),torch.randn(3,32,requires_grad=True))
    old=pair_logits(module,visual,text)
    weight=module.fusion_branch.gate.weight
    cached_visual=(*visual,F.linear(visual[1],weight[:,32:]))
    cached_text=(*text,F.linear(text[1],weight[:,:32]))
    new=pair_logits(module,cached_visual,cached_text)
    torch.testing.assert_close(old,new,atol=0,rtol=0)
    a=torch.autograd.grad(old.square().sum(),(visual[1],text[1],weight),retain_graph=True)
    b=torch.autograd.grad(new.square().sum(),(visual[1],text[1],weight))
    for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=0,rtol=0)


@pytest.mark.parametrize('valid_count',[0,1,2,3])
@pytest.mark.parametrize('backward_strategy',['batched','per_view_recompute'])
def test_fused_views_preserve_loss_and_gradients_and_f_only_fallback(valid_count,backward_strategy):
    torch.manual_seed(37);base=make(dict(fusion_lr=2e-4));new=copy.deepcopy(base)
    new.configure_runtime(dict(fused_text_views=True,fused_text_backward=backward_strategy,
        cache_gate_projection=True,gate_projection_chunk=2,cache_text_normalization=True,reduced_pair_score=True))
    images,views,valid=inputs();valid[:]=False;valid[:valid_count]=True
    if valid_count<2:
        def forbidden(*args):raise AssertionError('Fallback may not encode P/R')
        new.encode_views=forbidden
    old_loss,_=base(images,*views,valid,500);new_loss,_=new(images,*views,valid,500)
    torch.testing.assert_close(old_loss,new_loss,atol=3e-4,rtol=1e-5)
    old_loss.backward();new_loss.backward()
    for name,p in base.named_parameters():
        q=dict(new.named_parameters())[name]
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=3e-4,rtol=2e-4,msg=name)


@pytest.mark.parametrize('strategy',['full','none','partial2','partial3'])
def test_checkpoint_segments_preserve_every_block_and_its_gradient(strategy):
    class Transformer(nn.Module):
        def __init__(self):
            super().__init__();self.resblocks=nn.Sequential(*[nn.Linear(8,8) for _ in range(6)])
        def forward(self,x):return self.resblocks(x)
    class Clip(nn.Module):
        def __init__(self):
            super().__init__();self.transformer=Transformer();self.visual=nn.Module();self.visual.transformer=Transformer()
    torch.manual_seed(7);a=Clip();b=copy.deepcopy(a);configure_checkpointing(b,strategy)
    x=torch.randn(5,8);old=a.transformer(x)+a.visual.transformer(x);new=b.transformer(x)+b.visual.transformer(x)
    torch.testing.assert_close(old,new,atol=0,rtol=0);old.sum().backward();new.sum().backward()
    for name,p in a.named_parameters():torch.testing.assert_close(p.grad,dict(b.named_parameters())[name].grad,atol=0,rtol=0)
