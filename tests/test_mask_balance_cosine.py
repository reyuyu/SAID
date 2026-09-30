"""Explicit fusion references and bounded cosine conditions for the two repair arms."""
import copy
import math

import pytest
import torch
from torch.nn import functional as F

from model.nested_fusion_mask import NestedFusionMask, pair_logits, fusion_scores
from tests.test_nested_fusion import TinyFusionCLIP, explicit_inputs, explicit_logits, reference_loss
from train.train_nested_semantic_mask import build_optimizer


@pytest.mark.parametrize('fusion,visual', [('balanced_stack', 'patch'), ('cosine_crossscore', 'cls')])
def test_direct_formula_logits_gradients_and_adamw(fusion, visual):
    torch.manual_seed(873)
    model=NestedFusionMask(TinyFusionCLIP(32), fusion=fusion, visual=visual,
                           text_tokens=6, checkpoint_encoders=False, image_chunk=2, text_chunk=2)
    if fusion=='balanced_stack':
        with torch.no_grad():
            model.fusion_branch.gate.weight.normal_(std=.02)
    reference=copy.deepcopy(model)
    images=torch.randn(3,8)
    views=[torch.randint(0,31,(3,6)) for _ in range(3)]
    valid=torch.tensor([1,0,1],dtype=torch.bool)
    z,v=model.encode_visual(images);t,c=model.encode_view(views[0])
    expected=explicit_logits(reference,*explicit_inputs(reference,images,views[0])[2:])
    actual=pair_logits(model,v,c)
    torch.testing.assert_close(actual,expected,atol=3e-6,rtol=3e-5)
    paired=pair_logits(model,v,c,paired=True)
    torch.testing.assert_close(paired,actual.diagonal(dim1=0,dim2=1).T,atol=3e-6,rtol=3e-5)
    loss,logs=model(images,*views,valid,61)
    reference_value=reference_loss(reference,images,views,valid,61)
    torch.testing.assert_close(loss,reference_value,atol=3e-4,rtol=3e-5)
    loss.backward();reference_value.backward()
    left,right=dict(model.named_parameters()),dict(reference.named_parameters())
    for name in left:
        assert (left[name].grad is None)==(right[name].grad is None),name
        if left[name].grad is not None:
            torch.testing.assert_close(left[name].grad,right[name].grad,atol=1.5e-3,rtol=1e-4,msg=name)
    # The complete two-rank worker records actual AdamW differences and near-zero amplification.
    build_optimizer(model).step();build_optimizer(reference).step()
    assert logs['inc_weight']==61/200


def test_balanced_zero_gate_exact_half_and_nonzero_condition_changes():
    torch.manual_seed(877)
    clip=TinyFusionCLIP(32)
    rng=torch.get_rng_state().clone()
    model=NestedFusionMask(clip,fusion='balanced_stack',visual='patch',text_tokens=6,checkpoint_encoders=False)
    assert torch.equal(torch.get_rng_state(),rng)
    assert model.fusion_branch.gate.bias is None
    assert model.fusion_branch.gate.weight.count_nonzero()==0
    images=torch.randn(3,8,requires_grad=True);tokens=torch.randint(0,31,(3,6))
    _,v=model.encode_visual(images);_,t=model.encode_view(tokens)
    gate=model.fusion_branch.balanced_gate(v[1],t[1])
    torch.testing.assert_close(gate,torch.full_like(gate,.5),atol=0,rtol=0)
    actual=pair_logits(model,v,t)
    torch.testing.assert_close(actual,.5*t[1][:,None]+.5*v[1][None],atol=0,rtol=0)
    actual.sum().backward()
    assert images.grad is None and clip.token_embedding.weight.grad is None
    assert model.fusion_branch.gate.weight.grad.abs().sum()>0
    for blocks in (clip.mask_net.resblocks,model.fusion_branch.visual_blocks):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in blocks.parameters())
    assert model.fusion_branch.visual_adapter.weight.grad.abs().sum()>0
    assert not torch.equal(actual,pair_logits(model,tuple(x.roll(1,0) for x in v),t))
    assert not torch.equal(actual,pair_logits(model,v,tuple(x.roll(1,0) for x in t)))


def test_cosine_bounds_rescaling_standard_linear_and_no_temperature():
    torch.manual_seed(881)
    model=NestedFusionMask(TinyFusionCLIP(32),fusion='cosine_crossscore',visual='cls',
                           text_tokens=6,checkpoint_encoders=False)
    branch=model.fusion_branch
    zv,zt=torch.randn(3,1,32),torch.randn(4,6,32)
    q,k=branch.projected_queries(zv),branch.projected_keys(zt)
    r=torch.einsum('iak,btk->biat',q,k)
    assert r.abs().max()<=1+1e-6
    contracted=branch.cross_logits(q,branch.contract_keys(k))
    explicit=branch.readout(r.flatten(-2))
    torch.testing.assert_close(contracted,explicit,atol=2e-7,rtol=2e-6)
    torch.testing.assert_close(branch.projected_queries(zv*1000),q,atol=2e-7,rtol=2e-6)
    torch.testing.assert_close(branch.projected_keys(zt*.01),k,atol=2e-7,rtol=2e-6)
    assert not branch.readout.weight.count_nonzero()==0
    assert branch.readout.weight.abs().max()<=1/math.sqrt(6)
    assert branch.readout.bias.abs().max()<=1/math.sqrt(6)
    assert not model.clip.mask_net.attn_pool.attention.weight.requires_grad


@pytest.mark.parametrize('fusion,visual',[('balanced_stack','patch'),('cosine_crossscore','cls')])
def test_diagnostics_boundary_and_checkpoint_graph(fusion,visual):
    torch.manual_seed(887)
    model=NestedFusionMask(TinyFusionCLIP(32),fusion=fusion,visual=visual,text_tokens=6,
                           checkpoint_encoders=False,image_chunk=2,text_chunk=2)
    checkpointed=copy.deepcopy(model);checkpointed.checkpoint_pair_blocks=True
    images=torch.randn(3,8);views=[torch.randint(0,31,(3,6)) for _ in range(3)]
    valid=torch.tensor([1,0,1],dtype=torch.bool)
    loss,logs=model(images,*views,valid,0)
    other,otherlogs=checkpointed(images,*views,valid,0)
    loss.backward();other.backward()
    torch.testing.assert_close(loss,other,atol=0,rtol=0)
    for name,p in model.named_parameters():
        q=dict(checkpointed.named_parameters())[name]
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0,msg=name)
    assert all(not torch.is_tensor(v) or not v.requires_grad for v in logs.values())
    if fusion=='balanced_stack':
        assert float(logs['F_g_mean'])==.5 and float(logs['F_g_variance'])==0
        assert float(logs['g_F_P_abs_difference'])==0
        assert float(logs['O_g_q50'])==.5
    else:
        assert float(logs['F_qk_abs_max'])<=1+1e-6
        assert float(logs['F_sigmoid_saturation'])<.01
    for validity in (torch.zeros_like(valid),torch.tensor([1,0,0],dtype=torch.bool)):
        fallback,data=model(images,*views,validity,0)
        assert torch.isfinite(fallback) and data['O_candidates']==0
