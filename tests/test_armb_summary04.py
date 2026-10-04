"""Normalized S0.4 weighting changes only alignment, checked with explicit CE."""
import copy
import torch
from tests.test_balanced_hparams import make,inputs,weighted_reference
from experiments.nest_clip_v1.gradient_composition_audit_v1.components import production_components
from experiments.nest_clip_v1.armb_summary04_500_v1.run import WEIGHTS


def test_weighted_loss_and_gradients_match_explicit_global_reference():
    torch.manual_seed(842)
    model=make({'view_weights':WEIGHTS});reference=copy.deepcopy(model)
    images,views,valid=inputs();valid.fill_(True)
    actual,logs=model(images,*views,valid,200)
    expected=weighted_reference(reference,images,views,valid,200)
    assert sum(WEIGHTS)==3 and model.summary_t2i_weight==1
    torch.testing.assert_close(actual,expected,atol=3e-4,rtol=3e-5)
    actual.backward();expected.backward()
    for name,p in model.named_parameters():
        q=dict(reference.named_parameters())[name]
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=2e-4,rtol=8e-5,msg=name)


def test_auxiliary_losses_and_directional_weights_unchanged():
    torch.manual_seed(843);old=make({'view_weights':[1.2,.6,1.2]});new=copy.deepcopy(old);new.search_hparams['view_weights']=WEIGHTS
    images,views,valid=inputs()
    a,la=production_components(old,images,views,valid,100);b,lb=production_components(new,images,views,valid,100)
    for key in ['F_i2t','F_t2i','O_i2t','O_t2i','E_i2t','E_t2i','sparse','inc']:
        torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
    expected_delta=10/3*sum((w-v)*a[p+'_combined'] for w,v,p in zip(WEIGHTS,[1.2,.6,1.2],['F','O','E']))
    torch.testing.assert_close(b['total']-a['total'],expected_delta,atol=1e-4,rtol=2e-5)
