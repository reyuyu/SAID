import copy
import types
import pytest
import torch

from model.balanced_hparam_search import BalancedSearch,MACRO_DEFAULTS
from recovery.hns_balanced_macro_equivalence import make,legacy_forward,compare_gradients,COMPLETED


@pytest.mark.parametrize('method',['HNS','Balanced'])
@pytest.mark.parametrize('completed',COMPLETED)
@pytest.mark.parametrize('valid_count',[0,1,2,6])
def test_fetched_defaults_loss_all_parameter_gradients_AdamW_exact(method,completed,valid_count):
    torch.manual_seed(55221);a=make(method);b=copy.deepcopy(a)
    b.forward=types.MethodType(legacy_forward(method),b)
    images=torch.randn(6,8);views=[torch.randint(0,31,(6,6)) for _ in range(3)]
    valid=torch.arange(6)<valid_count
    x,logs=a(images,*views,valid,completed);y,old=b(images,*views,valid,completed)
    torch.testing.assert_close(x,y,atol=0,rtol=0)
    for key in old:
        if torch.is_tensor(old[key]):torch.testing.assert_close(logs[key],old[key],atol=0,rtol=0,msg=key)
        else:assert logs[key]==old[key],key
    x.backward();y.backward();compare_gradients(a,b)
    oa=torch.optim.AdamW(a.parameters(),lr=1e-4);ob=torch.optim.AdamW(b.parameters(),lr=1e-4);oa.step();ob.step()
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,atol=0,rtol=0)
    assert logs.get('HNS_enabled',False)==(method=='HNS')
    if method=='Balanced':assert 'HNS_surcharge' not in logs


@pytest.mark.parametrize('method',['HNS','Balanced'])
@pytest.mark.parametrize('key,value',[('lambda_align',8.),('lambda_sparse',1.2),('lambda_hierarchy',1.25)])
def test_actual_forward_component_isolation_and_graph_gradients(method,key,value):
    torch.manual_seed(173);base=make(method);new=copy.deepcopy(base)
    new.macro_hparams=dict(MACRO_DEFAULTS,**{key:value});new.capture_hns_graph=base.capture_hns_graph=True
    images=torch.randn(6,8);views=[torch.randint(0,31,(6,6)) for _ in range(3)]
    valid=torch.ones(6,dtype=torch.bool)
    old,oldlog=base(images,*views,valid,99);loss,logs=new(images,*views,valid,99)
    component=key.removeprefix('lambda_');scale=value/(10 if component=='align' else 1)
    for part in ('align','sparse','hierarchy'):
        torch.testing.assert_close(new.hns_graph['weighted_'+part],base.hns_graph['weighted_'+part]*(scale if part==component else 1),rtol=0,atol=0)
    torch.testing.assert_close(loss,(new.hns_graph['weighted_align']+new.hns_graph['weighted_sparse'])+new.hns_graph['weighted_hierarchy'],rtol=0,atol=0)
    params=[p for p in new.parameters() if p.requires_grad]
    total=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    parts=[torch.autograd.grad(new.hns_graph['weighted_'+k],params,retain_graph=True,allow_unused=True) for k in ('align','sparse','hierarchy')]
    for p,g,*gg in zip(params,total,*parts):
        if g is not None:torch.testing.assert_close(g,sum((torch.zeros_like(p) if v is None else v for v in gg)),rtol=3e-5,atol=2e-5)
    assert logs.get('HNS_enabled',False)==(method=='HNS')
    if method=='Balanced':assert logs['inc_weight']==.495 and 'HNS_surcharge' not in logs


@pytest.mark.parametrize('field,value',[('sparsity_scale',1.2),('inclusion_max',1.25)])
def test_balanced_opt_in_refuses_double_scaling(field,value):
    from tests.test_nested_fusion import TinyFusionCLIP
    cfg=dict(view_weights=[1.35,1.35,.3],inclusion_max=1,**MACRO_DEFAULTS);cfg[field]=value
    with pytest.raises(AssertionError):BalancedSearch(TinyFusionCLIP(32),search_hparams=cfg,
        inclusion_hierarchy='detail_chain',fusion='balanced_stack',visual='patch',checkpoint_encoders=False)


def test_hns_and_soft_inclusion_cannot_both_be_enabled():
    from tests.test_nested_fusion import TinyFusionCLIP
    with pytest.raises(AssertionError):BalancedSearch(TinyFusionCLIP(32),search_hparams=dict(
        view_weights=[1.35,1.35,.3],inclusion_max=1,**MACRO_DEFAULTS),hns_enabled=True,
        inclusion_hierarchy='detail_chain',fusion='balanced_stack',visual='patch',checkpoint_encoders=False)
