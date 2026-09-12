"""Independent counterexamples for fix_v2; audit CPU results are not GPU benchmarks."""
import copy
import pytest
import torch
import torch.nn.functional as F
from test_said_token_v1 import build_env, EOT_ID
from model.said_token_v1 import (ranking_sums, soft_from_grid, hard_from_grid,
    complement_pool, caption_identity, EPS, text_content_mask)


def independent_soft(c, a, valid=None, eta=.1):
    # Literal weighted exponential formula, independent of production log-domain reductions.
    c = c.detach()
    w = torch.cat([torch.ones_like(a[..., :1]), a.sigmoid()], -1)
    if valid is None:
        valid = torch.ones_like(c[..., 0, :], dtype=torch.bool)
    v = (w*c.masked_fill(~valid[..., None, :], -torch.inf).max(-1).values).sum(-1)/w.sum(-1)
    t = eta * ((c/eta).exp()*w[..., :, None]).sum(-2).div(w.sum(-1)[..., None]).log()
    return v + (t*valid).sum(-1)/valid.sum(-1)


def test_hinge_sign_shift_and_sum_reduction():
    s = torch.full((3,3), .05, dtype=torch.float64)
    s.fill_diagonal_(.1)
    s.requires_grad_()
    shift = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
    x = s+shift
    legal = ~torch.eye(3, dtype=torch.bool)
    a,b,_ = ranking_sums(x, x.diagonal(), x.diagonal(), legal)
    loss = (a+b)/legal.sum()
    gs,gc = torch.autograd.grad(loss,(s,shift))
    assert loss.item() == pytest.approx(.3)
    assert (gs.diagonal()<0).all() and (gs[legal]>0).all()
    assert abs(gc.item())<1e-14
    a,b,_ = ranking_sums(s,s.diagonal(),s.diagonal(),torch.zeros_like(legal))
    zero_grad = torch.autograd.grad(a+b,s)[0]
    assert torch.equal(zero_grad,torch.zeros_like(s))


@pytest.mark.parametrize('constant',[0.,-.7,.4])
def test_constant_grid_has_no_gate_preference(constant):
    c = torch.full((2,3,6,5),constant,dtype=torch.float64,requires_grad=True)
    a = torch.randn(2,3,5,dtype=torch.float64,requires_grad=True)
    value = soft_from_grid(c,a)
    gc,ga = torch.autograd.grad(value.sum(),(c,a),allow_unused=True)
    torch.testing.assert_close(value,torch.full_like(value,2*constant),atol=1e-14,rtol=1e-13)
    assert gc is None and ga.abs().max()<1e-13


def test_continuous_proxy_fp64_finite_difference_and_independent_reference():
    torch.manual_seed(8)
    c=torch.randn(2,2,5,4,dtype=torch.float64,requires_grad=True)*.2
    a=torch.randn(2,2,4,dtype=torch.float64,requires_grad=True)
    valid=torch.tensor([True,False,True,True]).expand(2,2,4)
    result=soft_from_grid(c,a,valid)
    expected=independent_soft(c,a,valid)
    torch.testing.assert_close(result,expected,atol=1e-13,rtol=1e-12)
    g=torch.autograd.grad(result.sum(),a,retain_graph=True)[0]
    ge=torch.autograd.grad(expected.sum(),a)[0]
    torch.testing.assert_close(g,ge,atol=1e-13,rtol=1e-11)
    assert g.abs().max()>1e-5
    assert torch.autograd.gradcheck(lambda z:soft_from_grid(c,z,valid),(a,),eps=1e-6,atol=1e-7,rtol=1e-5)


def test_masks_exclude_highest_scores_and_empty_text_uses_eos_only():
    c=torch.tensor([[[-.8,1.],[-.9,1.],[1.,1.]]])
    gate=torch.tensor([[1.,0.]])
    valid=torch.tensor([[True,False]])
    score,v,t=hard_from_grid(c,gate,valid)
    assert v.item()==pytest.approx(-.85) and t.item()==pytest.approx(-.8)
    assert score.item()==pytest.approx(-1.65)


def independent_score(v,t,router,tvalid):
    output=[]
    for i in range(len(v)):
        row=[]
        for j in range(len(t)):
            a=(F.normalize(router.visual(v[i,1:]),dim=-1,eps=EPS)*
               F.normalize(router.text(t[j,0]),dim=-1,eps=EPS)).sum(-1)/.07
            idx=torch.argsort(-a,stable=True)[:16]+1
            idx=torch.cat([idx.new_zeros(1),idx])
            c=F.normalize(v[i],dim=-1,eps=EPS) @ F.normalize(t[j],dim=-1,eps=EPS).T
            kept=c[idx][:,tvalid[j]]
            hard=kept.max(-1).values.mean()+kept.max(-2).values.mean()
            soft=independent_soft(c,a,tvalid[j])
            row.append(hard+(soft-soft.detach()))
        output.append(torch.stack(row))
    return torch.stack(output)


def test_matched_pairs_single_grid_and_gradients(monkeypatch):
    _,_,_,module,_=build_env()
    torch.manual_seed(42)
    v=torch.randn(3,33,16,requires_grad=True)
    t=torch.randn(3,33,16,requires_grad=True)
    valid=torch.ones(3,33,dtype=torch.bool)
    valid[1,1:]=False
    from model import said_token_v1 as core
    original=core._similarity_grid
    calls=[]
    def counted(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(core,'_similarity_grid',counted)
    full=module.scorer.score(v,t,valid)['score'].diagonal()
    assert len(calls)==1
    matched=module.scorer.score_matched_pairs(v,t,valid)['score']
    assert len(calls)==1, 'matched path must use b paired matrices, not b x b'
    expected=independent_score(v,t,module.router,valid).diagonal()
    torch.testing.assert_close(full,expected,atol=2e-6,rtol=1e-5)
    torch.testing.assert_close(matched,expected,atol=2e-6,rtol=1e-5)
    params=[v,t,*module.router.parameters()]
    reference=torch.autograd.grad(expected.sum(),params,retain_graph=True)
    for value in [full,matched]:
        grads=torch.autograd.grad(value.sum(),params,retain_graph=True)
        for g,r in zip(grads,reference):
            torch.testing.assert_close(g,r,atol=2e-6,rtol=3e-5)


def full_reference(module,images,text,ids):
    gi,patch=module.encode_image_tokens(images)
    gt,tlocal=module.encode_text_tokens(text)
    valid,empty=text_content_mask(text,EOT_ID)
    vs=module.image_aggregator(patch)[0]
    ts=module.text_aggregator(tlocal,valid)[0]
    gi,gt=F.normalize(gi,dim=-1,eps=EPS),F.normalize(gt,dim=-1,eps=EPS)
    v=torch.cat([gi[:,None],vs],1)
    t=torch.cat([gt[:,None],ts],1)
    valid_slots=torch.cat([torch.ones_like(empty[:,None]),(~empty[:,None]).expand(-1,32)],1)
    scores=independent_score(v,t,module.router,valid_slots)
    cap=[tuple(row[:row.argmax()+1].tolist()) for row in text]
    terms=[]
    for i in range(len(ids)):
        for j in range(len(ids)):
            if ids[i]!=ids[j] and cap[i]!=cap[j]:
                terms.append(F.relu(.2+scores[i,j]-scores[i,i])+F.relu(.2+scores[j,i]-scores[i,i]))
    said=torch.stack(terms).mean() if terms else scores.sum()*0
    gates=[]
    for i in range(len(ids)):
        a=module.router.logits(v[i:i+1,1:],t[i:i+1,0])[0,0]
        gate=torch.zeros_like(a)
        gate[torch.argsort(-a,stable=True)[:16]]=1
        gates.append(gate)
    m=1-torch.stack(gates).detach()
    raw=(vs*m[...,None]).sum(1)/m.sum(-1).clamp_min(1)[:,None]
    valid_rec=(raw.norm(dim=-1)>EPS)&(m.sum(-1)>0)
    prediction=module.decoder(F.normalize(raw,dim=-1,eps=EPS),gt.detach())
    rec=(1-(F.normalize(prediction,dim=-1,eps=EPS)*module.reference_global(images)).sum(-1))
    rec=(rec*valid_rec).sum()/valid_rec.sum().clamp_min(1)
    return said+.1*rec


@pytest.mark.parametrize('chunk,checkpoint',[((1,2),True),((3,3),False)])
def test_optimized_module_matches_independent_full_reference(chunk,checkpoint):
    _,images,text,module,ids=build_env(chunk=chunk,captions=['cat','dog',''])
    module.checkpoint_pairwise=checkpoint
    expected=full_reference(module,images,text,ids)
    expected.backward()
    grads={n:None if p.grad is None else p.grad.clone() for n,p in module.named_parameters()}
    module.zero_grad(set_to_none=True)
    actual=module(images,text,ids,EOT_ID)['loss_total_for_backward']
    actual.backward()
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
    for n,p in module.named_parameters():
        if grads[n] is None:
            assert p.grad is None,n
        else:
            diff=(p.grad-grads[n]).abs().max()
            assert diff<=2e-6+3e-5*grads[n].abs().max(),(n,diff)


def test_duplicates_padding_no_negatives_and_invalid_u():
    _,images,text,module,ids=build_env(captions=['cat','cat','cat'])
    padded=F.pad(text,(0,5),value=123)
    assert caption_identity(padded,1).unique().numel()==1
    out=module(images,text,ids,EOT_ID)
    assert out['loss_said_pairs_total']==0 and out['loss_said']==0
    out['loss_said'].backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad)==0 for p in module.parameters())
    for scale in [0.,1e-9]:
        slots=torch.full((2,32,4),scale,requires_grad=True)
        raw,norm,valid,m=complement_pool(slots,torch.zeros(2,32))
        assert not valid.any() and torch.isfinite(norm).all()
        (norm.sum()*valid.sum()).backward()
        assert torch.equal(slots.grad,torch.zeros_like(slots))
    raw,norm,valid,m=complement_pool(torch.ones(2,32,4),torch.ones(2,32))
    assert not valid.any()


def test_explicit_fp32_cores_under_bf16_autocast():
    _,images,text,module,ids=build_env()
    with torch.autocast('cpu',dtype=torch.bfloat16):
        out=module(images,text,ids,EOT_ID)
    for key,value in out['core_dtypes'].items():
        if not key.startswith('student_'):
            assert value=='torch.float32',(key,value)
