"""Exact production exposure, zero-native auxiliaries and bounded Gram correctness."""
import copy
import pytest
import torch
from tests.test_balanced_hparams import make,inputs
from experiments.nest_clip_v1.gradient_composition_audit_v1.components import production_components,native_components,OBJECTIVES
from experiments.nest_clip_v1.gradient_composition_audit_v1.stats import gram_matrices,derive,linearity_errors,parameter_group

@pytest.mark.parametrize('weights',[[1.,1.,1.],[1.2,.6,1.2]])
def test_capture_production_default_loss_gradients_bitwise(weights):
    torch.manual_seed(987)
    old=make({'view_weights':weights});new=copy.deepcopy(old)
    images,views,valid=inputs()
    lo,logs=old(images,*views,valid,200)
    components,captured=production_components(new,images,views,valid,200)
    torch.testing.assert_close(lo,components['total'],atol=0,rtol=0)
    assert all(torch.equal(logs[k],captured[k]) if torch.is_tensor(logs[k]) else logs[k]==captured[k] for k in logs)
    lo.backward();components['total'].backward()
    for n,p in old.named_parameters():
        q=dict(new.named_parameters())[n]
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)


def test_FP32_components_and_weighted_gradients_linear():
    torch.manual_seed(988);model=make({'view_weights':[1.2,.6,1.2]});images,views,valid=inputs()
    losses,logs=production_components(model,images,views,valid,200)
    for label,prefix in [('F','F'),('O','O'),('E','E')]:
        torch.testing.assert_close(losses[label+'_i2t'],logs[prefix+'_i2t'],atol=0,rtol=0)
        torch.testing.assert_close(losses[label+'_t2i'],logs[prefix+'_t2i'],atol=0,rtol=0)
    named={n:p for n,p in model.named_parameters() if p.requires_grad};gradients={}
    for key in ['F_combined','O_combined','E_combined','align_actual']:
        values=torch.autograd.grad(losses[key],tuple(named.values()),retain_graph=True,allow_unused=True)
        gradients[key]=dict(zip(named,values))
    errors=linearity_errors(named,gradients,[1.2,.6,1.2]);assert errors['relative_L2']<2e-6


def test_native_reference_has_no_auxiliary_gradients():
    torch.manual_seed(989);model=make();images,views,valid=inputs()
    losses,logs,checks=native_components(model,images,views[0],True,precision='fp32')
    assert checks['shape']==[3,3] and checks['direct_top1_rankings_equal']
    named={n:p for n,p in model.named_parameters() if p.requires_grad}
    values=torch.autograd.grad(losses['native_combined'],tuple(named.values()),allow_unused=True)
    for n,g in zip(named,values):
        if parameter_group(n) not in ['G1_vision_backbone','G2_text_backbone']:assert g is None


def test_per_parameter_Gram_matches_direct_dot_and_NA():
    a=torch.nn.Parameter(torch.zeros(2,3));b=torch.nn.Parameter(torch.zeros(4));named={'clip.visual.proj':a,'clip.text_projection':b}
    keys=OBJECTIVES;torch.manual_seed(990)
    gradients={key:{name:torch.randn_like(p) for name,p in named.items()} for key in keys}
    grams,support=gram_matrices(named,gradients,keys,chunk=2)
    for i,key in enumerate(keys):
        for j,other in enumerate(keys):
            expected=sum((gradients[key][name].double()*gradients[other][name].double()).sum().item() for name in named)
            assert abs(grams['native_backbone_total'][i][j]-expected)<1e-12
    result=derive(grams,support,keys,[1.,1.,1.]);assert result['G3_text_mask_blocks']['pairs']['F_combined__native_combined']['cosine'] is None
