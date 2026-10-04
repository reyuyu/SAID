"""Zero Summary alignment leaves the production forward and auxiliaries intact."""
import ast
import copy
from pathlib import Path
import subprocess
import types
import pytest
import torch
from model.balanced_hparam_search import hparams
from tests.test_balanced_hparams import make,inputs,weighted_reference
from experiments.nest_clip_v1.gradient_composition_audit_v1.components import production_components

@pytest.mark.parametrize('weights',[[1.4,.2,1.4],[1.5,0.,1.5]])
def test_registered_losses_and_gradients_match_explicit_CE(weights):
    torch.manual_seed(981);model=make({'view_weights':weights});reference=copy.deepcopy(model)
    images,views,valid=inputs();valid.fill_(True)
    actual,_=model(images,*views,valid,200);expected=weighted_reference(reference,images,views,valid,200)
    torch.testing.assert_close(actual,expected,atol=3e-4,rtol=3e-5)
    actual.backward();expected.backward()
    for name,p in model.named_parameters():
        q=dict(reference.named_parameters())[name]
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=2e-4,rtol=8e-5,msg=name)


def test_zero_S_CE_gradient_but_S_auxiliary_gradient_retained():
    torch.manual_seed(982);model=make({'view_weights':[1.5,0.,1.5]});images,views,valid=inputs();valid.fill_(True)
    encoded=[];original=model.encode_view
    def observe(self,tokens):
        result=original(tokens);encoded.append(result);return result
    model.encode_view=types.MethodType(observe,model)
    losses,logs=production_components(model,images,views,valid,200)
    assert len(encoded)==3 and logs['O_candidates']==3
    Stext,Scondition=encoded[1]
    alignment=torch.autograd.grad(losses['align_actual'],[Stext,Scondition[1]],retain_graph=True,allow_unused=True)
    assert all(g is None or torch.count_nonzero(g)==0 for g in alignment)
    total=torch.autograd.grad(losses['total'],Scondition[1],retain_graph=True)[0]
    sparse=torch.autograd.grad(losses['sparse'],Scondition[1],retain_graph=True)[0]
    assert torch.count_nonzero(total)>0 and torch.count_nonzero(sparse)>0
    # Inclusion still references Summary probabilities through the unchanged target.
    assert torch.isfinite(losses['inc']) and torch.isfinite(logs['O_i2t']) and torch.isfinite(logs['O_t2i'])


def test_S_weight_changes_alignment_only():
    torch.manual_seed(983);old=make({'view_weights':[1.3,.4,1.3]});new=copy.deepcopy(old)
    new.search_hparams['view_weights']=[1.5,0.,1.5]
    images,views,valid=inputs();a,_=production_components(old,images,views,valid,100);b,_=production_components(new,images,views,valid,100)
    for key in ['F_i2t','F_t2i','O_i2t','O_t2i','E_i2t','E_t2i','sparse','inc']:
        torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
    delta=10/3*(.2*a['F_combined']-.4*a['O_combined']+.2*a['E_combined'])
    torch.testing.assert_close(b['total']-a['total'],delta,atol=1e-4,rtol=3e-5)

@pytest.mark.parametrize('weights',[[-.1,1,1],[0,0,0],[1,float('nan'),1],[1,float('inf'),1]])
def test_invalid_weights_still_rejected(weights):
    with pytest.raises(AssertionError):hparams({'view_weights':weights})


def test_all_production_math_except_validation_AST_unchanged():
    root=Path(__file__).resolve().parents[1];path='model/balanced_hparam_search.py'
    old=ast.parse(subprocess.check_output(['git','show','bbe5daa:'+path],cwd=root,text=True));new=ast.parse((root/path).read_text())
    nodes=lambda tree:[ast.dump(n,include_attributes=False) for n in tree.body if not(isinstance(n,ast.FunctionDef) and n.name=='hparams')]
    assert nodes(old)==nodes(new)
