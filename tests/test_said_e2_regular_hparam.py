from types import SimpleNamespace
import copy
import pytest
import torch
from model import nested_fusion_mask as m
from model.balanced_hparam_search import macro_terms
from recovery import said_e2_regular_hparam_fourarm500 as c


def test_fixed_four_one_scalar_arms():
    assert c.ARMS=={'P1-Temp95':(95.,1.2),'P2-Temp105':(105.,1.2),'P3-Sparse110':(100.,1.1),'P4-Sparse130':(100.,1.3)}
    original=c.read(c.BASE_EXP/'config.json')
    for a,(scale,sparse) in c.ARMS.items():
        cfg=c.config(a);c.frozen(cfg,a)
        assert {k for k,v in original.items() if cfg[k]!=v}==({'lambda_sparse'} if sparse!=1.2 else set())
        assert cfg['contrastive_logit_scale']==scale


def test_scores_temperature_keeps_mask_and_uses_same_symmetric_CE(monkeypatch):
    torch.manual_seed(0)
    pair=torch.randn(3,4,8)
    monkeypatch.setattr(m,'pair_logits',lambda *args:pair)
    z=torch.randn(4,8,requires_grad=True);text=torch.randn(3,8,requires_grad=True)
    active=torch.ones(3,4,dtype=torch.bool)
    for scale in (95.,100.,105.):
        module=SimpleNamespace(contrastive_logit_scale=scale,fusion='balanced_stack')
        score,summary,extra=m.score_block(module,z,text,(),(),active,False)
        expected=scale*(torch.nn.functional.normalize(z[None]*m.hard_st(pair.sigmoid()),dim=-1,eps=1e-6)*torch.nn.functional.normalize(text,dim=-1,eps=1e-6)[:,None]).sum(-1)
        assert torch.equal(score,expected)
        if scale==95.:ref_summary,ref_extra=summary,extra
        assert torch.equal(summary,ref_summary) and torch.equal(extra,ref_extra)
        assert torch.equal(m.directional_ce(score.sum(),score.square().sum(),score.sum()*0),score.sum()+score.square().sum()+score.sum()*0)
    absent=m.score_block(SimpleNamespace(fusion='balanced_stack'),z,text,(),(),active,False)[0]
    explicit=m.score_block(SimpleNamespace(contrastive_logit_scale=100.,fusion='balanced_stack'),z,text,(),(),active,False)[0]
    assert torch.equal(absent,explicit)
    g1=torch.autograd.grad(absent.sum(),(z,text),retain_graph=True)
    g2=torch.autograd.grad(explicit.sum(),(z,text))
    assert all(torch.equal(a,b) for a,b in zip(g1,g2))


@pytest.mark.parametrize('sparse',[1.1,1.2,1.3])
@pytest.mark.parametrize('ramp',[0.,.5,1.])
def test_sparse_macro_multiplies_only_once(sparse,ramp):
    a=torch.tensor(2.,requires_grad=True);s=torch.tensor(3.,requires_grad=True);h=torch.tensor(4.,requires_grad=True)
    wa,ws,wh=macro_terms(a,s,ramp*h,dict(lambda_align=10.,lambda_sparse=sparse,lambda_hierarchy=1.))
    assert wa is a and torch.equal(ws,s*sparse) and torch.equal(wh,h*ramp)
    grads=torch.autograd.grad(wa+ws+wh,(a,s,h))
    assert torch.equal(grads[0],torch.tensor(1.)) and torch.equal(grads[1],torch.tensor(sparse)) and torch.equal(grads[2],torch.tensor(ramp))


def test_stream_checks_original_LR_and_each_declared_scalar():
    reference=c.rows(c.BASE_RUN/'step500/steps.jsonl')[:2]
    for a,(scale,sparse) in c.ARMS.items():
        rows=copy.deepcopy(reference)
        for r in rows:
            r['contrastive_logit_scale']=scale;r['macro_lambda_sparse']=sparse
            r['macro_weighted_sparse']=sparse*r['macro_raw_sparse']
            r['loss']=sum(r['macro_weighted_'+k] for k in ('align','sparse','hierarchy'))
        assert c.matched_stream(rows,reference,a)['records']==2048
        bad=copy.deepcopy(rows);bad[-1]['actual_lrs']['fusion_adapter']*=1.15
        with pytest.raises(AssertionError):c.matched_stream(bad,reference,a)
        bad=copy.deepcopy(rows);bad[-1]['rank_health'][0]['sampling']['K'][0]=999
        with pytest.raises(AssertionError):c.matched_stream(bad,reference,a)
        bad=copy.deepcopy(rows);bad[-1]['macro_weighted_sparse']*=sparse
        with pytest.raises(AssertionError):c.matched_stream(bad,reference,a)
