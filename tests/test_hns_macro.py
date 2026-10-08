import copy
import types
import pytest
import torch
from model.balanced_hparam_search import MACRO_DEFAULTS,macro_hparams,macro_terms
from recovery.hns_macro_equivalence import make,legacy_forward,compare_gradients


@pytest.mark.parametrize('completed',[0,99,199,499])
@pytest.mark.parametrize('valid_count',[0,1,2,6])
def test_default_fetched_HNS_forward_all_gradients_exact(completed,valid_count):
    torch.manual_seed(55221);a=make();b=copy.deepcopy(a)
    b.forward=types.MethodType(legacy_forward(),b)
    images=torch.randn(6,8);views=[torch.randint(0,31,(6,6)) for _ in range(3)]
    valid=torch.arange(6)<valid_count
    x,logs=a(images,*views,valid,completed);y,old=b(images,*views,valid,completed)
    torch.testing.assert_close(x,y,atol=0,rtol=0)
    for key in old:
        if torch.is_tensor(old[key]):torch.testing.assert_close(logs[key],old[key],atol=0,rtol=0,msg=key)
        else:assert logs[key]==old[key],key
    x.backward();y.backward();compare_gradients(a,b)
    assert logs['lambda_h']==min(1.,completed/200.)
    oa=torch.optim.AdamW(a.parameters(),lr=1e-4);ob=torch.optim.AdamW(b.parameters(),lr=1e-4)
    oa.step();ob.step()
    for pa,pb in zip(a.parameters(),b.parameters()):torch.testing.assert_close(pa,pb,atol=0,rtol=0)


@pytest.mark.parametrize('key,value,index',[('lambda_align',12.,0),('lambda_sparse',.8,1),('lambda_hierarchy',.75,2),('lambda_hierarchy',1.25,2)])
def test_component_isolation_scaling_once(key,value,index):
    a,s,h=[torch.tensor(v,requires_grad=True) for v in (20.,.8,.1)]
    hp=dict(MACRO_DEFAULTS,**{key:value})
    result=macro_terms(a,s,h,hp);base=(a,s,h)
    for i,(x,y) in enumerate(zip(result,base)):
        scale=value/(10 if index==0 else 1) if i==index else 1
        torch.testing.assert_close(x,y*scale,rtol=0,atol=0)
    grads=torch.autograd.grad(sum(result),(a,s,h))
    assert [float(g) for g in grads]==pytest.approx([value/(10 if index==0 else 1) if i==index else 1 for i in range(3)])


@pytest.mark.parametrize('value',[0,-1,float('nan'),float('inf')])
def test_invalid_scale_rejected(value):
    with pytest.raises(AssertionError):macro_hparams({'lambda_align':value})
