"""HybridF loss mixing, compatibility and gradient tests."""
import copy
import io
from pathlib import Path
import subprocess
import types

import pytest
import torch
from torch.nn import functional as F

import model.nested_semantic_mask as hybrid
from model.said_cls_cvssl import said_mask_from_hidden
from tests.test_nested_semantic_mask import TinyCLIP, literal_scores
from tests.test_nested_resume import payloads
from train.train_nested_semantic_mask import build_optimizer, validate_resume_payload


def reference(clip, images, tokens, valid, s, eta=.25):
    """Independent full-batch matrix reference; mix CE, not logits."""
    z=clip.encode_image(images)
    terms=[];ps=[]
    active=tokens if int(valid.sum())>=2 else tokens[:1]
    for i,tok in enumerate(active):
        t,h=clip.encode_text(tok,return_full=True)
        m,p,_=said_mask_from_hidden(clip.mask_net,h);ps.append(p)
        sel=torch.ones_like(valid) if i==0 else valid
        q=literal_scores(z[sel],t[sel],m[sel]);labels=torch.arange(len(q),device=q.device)
        align=F.cross_entropy(q,labels)+F.cross_entropy(q.T,labels)
        if i==0:
            n=100*(F.normalize(z,dim=-1,eps=1e-6)@F.normalize(t,dim=-1,eps=1e-6).T)
            native=F.cross_entropy(n,labels)+F.cross_entropy(n.T,labels)
            align=(1-eta)*align+eta*native
        terms.append((align,m[sel].abs().mean()))
    if len(terms)==1:return 10*terms[0][0]+terms[0][1]
    inc=.5*(F.relu(ps[1].detach()-ps[0]).mean(-1)+F.relu(ps[2].detach()-ps[0]).mean(-1))
    return (10/3)*sum(v[0] for v in terms)+(terms[0][1]+2*terms[1][1]+2*terms[2][1])/3+min(1,s/200)*inc[valid].mean()


def inputs(n=4):
    return torch.randn(n,8),[torch.randint(0,30,(n,6)) for _ in range(3)]


@pytest.mark.parametrize('validity', [[1,1,1,0],[0,0,0,0],[0,1,0,0]])
def test_zero_mix_exact_reviewed_path(validity,monkeypatch):
    source=subprocess.check_output(['git','show','25a5d12:model/nested_semantic_mask.py'],cwd=Path(__file__).resolve().parents[1],text=True)
    old=types.ModuleType('model._reviewed_nested');old.__package__='model';exec(compile(source,'reviewed_nested','exec'),old.__dict__)
    torch.manual_seed(13);a=TinyCLIP();b=copy.deepcopy(a)
    ma=old.NestedSemanticMask(a,arm='A3',checkpoint_encoders=False)
    mb=hybrid.NestedSemanticMask(b,arm='A3',checkpoint_encoders=False)
    monkeypatch.setattr(hybrid,'native_full_terms',lambda *x:(_ for _ in ()).throw(AssertionError('native evaluated at zero mix')))
    images,tokens=inputs();valid=torch.tensor(validity,dtype=torch.bool)
    before=torch.get_rng_state().clone()
    la,loga=ma(images,*tokens,valid,200);la.backward()
    lb,logb=mb(images,*tokens,valid,200);lb.backward()
    assert torch.equal(before,torch.get_rng_state()) and torch.equal(la,lb)
    assert loga.keys()==logb.keys()
    for k in loga:
        assert torch.equal(loga[k],logb[k]) if torch.is_tensor(loga[k]) else loga[k]==logb[k]
    for p,q in zip(a.parameters(),b.parameters()):assert torch.equal(p.grad,q.grad)
    for m in (ma,mb):build_optimizer(m).step()
    assert a.state_dict().keys()==b.state_dict().keys()
    for k,v in a.state_dict().items():assert torch.equal(v,b.state_dict()[k])


def test_native_reference_all_ones_and_direct_gradients():
    torch.manual_seed(17);clip=TinyCLIP();images,tokens=inputs()
    z=clip.encode_image(images);t=clip.encode_text(tokens[0])
    q=100*(F.normalize(z,dim=-1,eps=1e-6)[:,None]*F.normalize(t,dim=-1,eps=1e-6)[None]).sum(-1)
    torch.testing.assert_close(hybrid.native_scores(z,t),q,atol=2e-5,rtol=3e-6)
    torch.testing.assert_close(hybrid.masked_scores(z,t,torch.ones_like(t)),q,atol=2e-5,rtol=3e-6)
    loss,logs=hybrid.native_full_terms(z,t)
    labels=torch.arange(4)
    expected=F.cross_entropy(q,labels)+F.cross_entropy(q.T,labels)
    torch.testing.assert_close(loss,expected,atol=2e-5,rtol=3e-6)
    loss.backward()
    assert clip.visual.weight.grad.abs().sum()>0 and clip.token_embedding.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in clip.mask_net.parameters())
    assert logs['candidates']==4


@pytest.mark.parametrize('validity',[[1,1,1,0],[0,0,0,0],[0,1,0,0]])
def test_mixed_formula_and_encoder_call_count(validity):
    torch.manual_seed(19);clip=TinyCLIP();ref=copy.deepcopy(clip);images,tokens=inputs()
    module=hybrid.NestedSemanticMask(clip,arm='A3',checkpoint_encoders=False,full_native_mix=.25)
    counts={'image':0,'text':0};oi,ot=clip.encode_image,clip.encode_text
    def image(x):counts['image']+=1;return oi(x)
    def text(x,**kw):counts['text']+=1;return ot(x,**kw)
    clip.encode_image=image;clip.encode_text=text
    valid=torch.tensor(validity,dtype=torch.bool)
    actual,logs=module(images,*tokens,valid,200);expected=reference(ref,images,tokens,valid,200)
    torch.testing.assert_close(actual,expected,atol=1e-4,rtol=1e-5)
    assert counts=={'image':1,'text':3 if sum(validity)>=2 else 1}
    assert logs['full_native_mix']==.25 and logs['loss_reconstruction_abs_error']<1e-4
    assert not any(v.requires_grad for v in logs.values() if torch.is_tensor(v))
    actual.backward();expected.backward()
    for (name,p),(_,q) in zip(clip.named_parameters(),ref.named_parameters()):
        torch.testing.assert_close(p.grad,q.grad,atol=8e-4,rtol=5e-5,msg=name)


def test_checkpoint_native_state_structure_and_resume_mix_guard():
    torch.manual_seed(7);clip=TinyCLIP()
    old=hybrid.NestedSemanticMask(copy.deepcopy(clip),arm='A3',checkpoint_encoders=False)
    new=hybrid.NestedSemanticMask(clip,arm='A3',checkpoint_encoders=False,full_native_mix=.25)
    assert old.state_dict().keys()==new.state_dict().keys()
    optimizer=build_optimizer(new);images,tokens=inputs()
    loss,_=new(images,*tokens,torch.ones(4,dtype=torch.bool),0);loss.backward();optimizer.step()
    payload=io.BytesIO();torch.save(dict(model=clip.state_dict(),optimizer=optimizer.state_dict(),mix=.25),payload);payload.seek(0)
    state=torch.load(payload,weights_only=False);bare=TinyCLIP();bare.load_state_dict(state['model'],strict=True)
    opt=build_optimizer(old);opt.load_state_dict(state['optimizer'])
    with torch.no_grad():
        torch.testing.assert_close(clip.encode_image(images),bare.encode_image(images),atol=0,rtol=0)
        torch.testing.assert_close(clip.encode_text(tokens[0]),bare.encode_text(tokens[0]),atol=0,rtol=0)
    previous,current=payloads();current['full_native_mix']=.25
    with pytest.raises(AssertionError,match='full_native_mix'):validate_resume_payload(previous,current,'old-trainer')
    previous['config']['full_native_mix']=.25
    assert validate_resume_payload(previous,current,'old-trainer')==500


def test_inclusion_still_only_direct_full_probability_gradient():
    pf,pp,pr=[torch.tensor([[x]],requires_grad=True) for x in (.1,.7,.8)]
    hybrid.inclusion(pf,pp,pr).sum().backward()
    assert pf.grad.item()<0 and pp.grad is None and pr.grad is None
