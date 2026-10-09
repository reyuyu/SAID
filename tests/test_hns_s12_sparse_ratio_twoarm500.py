import math
import subprocess
import sys
import json
import pytest
import torch
from model.balanced_hparam_search import BalancedSearch
from tests.test_nested_fusion import TinyFusionCLIP

from recovery import hns_s12_sparse_ratio_twoarm500 as r


def test_requested_ratios_have_equal_mass_and_coefficients():
    assert r.ARMS['E1-AlignMatched']['sparsity_weights']==[2.25,2.25,.5]
    assert all(math.isclose(x,5/3,abs_tol=1e-12) for x in r.ARMS['E2-Uniform']['sparsity_weights'])
    for spec in r.ARMS.values():
        assert math.isclose(sum(spec['sparsity_weights']),5.,abs_tol=1e-12)


def test_commands_are_fresh_common0_and_local_only():
    r.configure()
    for arm in r.ARMS:
        for smoke in (True,False):
            cmd=r.runner.training_command(arm,smoke)
            assert '--resume' not in cmd
            assert cmd[cmd.index('--max-updates')+1]==('5' if smoke else '500')
            assert cmd[cmd.index('--image-root')+1]=='/root/said_s02_stage500/ShareGPT4V'
            assert cmd[cmd.index('-m')+1]==r.ENTRY


def test_equivalence_receipt_subprocess(tmp_path):
    out=tmp_path/'equivalence.json'
    subprocess.run([sys.executable,'-m','recovery.hns_sparse_ratio_equivalence','--output',str(out)],check=True)
    assert out.exists()
    receipt=json.loads(out.read_text())
    assert 'default_hns_behavior_preserved' not in receipt
    assert 'gradient' not in receipt


def tiny(weights):
    return BalancedSearch(TinyFusionCLIP(32),search_hparams=dict(view_weights=[1.35,1.35,.3],
        inclusion_max=0,lambda_align=10,lambda_sparse=1.2,lambda_hierarchy=1),
        view_sparsity_weights=weights,hns_enabled=True,inclusion_hierarchy='detail_chain',
        fusion='balanced_stack',visual='patch',text_tokens=6,checkpoint_encoders=False,image_chunk=2,text_chunk=2)


@pytest.mark.parametrize('weights',[[1.,2.,2.],[2.25,2.25,.5],[5/3,5/3,5/3]])
@pytest.mark.parametrize('completed',[0,99,200,499])
def test_actual_forward_only_sparse_changes_and_no_double_scale(weights,completed):
    torch.manual_seed(943);model=tiny([1.,2.,2.]);model.capture_hns_graph=True
    images=torch.randn(6,8);tokens=[torch.randint(0,31,(6,6)) for _ in range(3)]
    valid=torch.tensor([1,1,0,1,1,0],dtype=torch.bool)
    _,_=model(images,*tokens,valid,completed);base=model.hns_graph
    params=[p for p in model.parameters() if p.requires_grad]
    expected={k:torch.autograd.grad(base[k],params,allow_unused=True,retain_graph=True) for k in ('weighted_align','weighted_hierarchy')}
    model.view_sparsity_weights=weights
    loss,logs=model(images,*tokens,valid,completed);graph=model.hns_graph
    for k in expected:
        torch.testing.assert_close(graph[k],base[k],atol=0,rtol=0)
        actual=torch.autograd.grad(graph[k],params,allow_unused=True,retain_graph=True)
        for x,y in zip(actual,expected[k]):
            assert (x is None)==(y is None)
            if x is not None:torch.testing.assert_close(x,y,atol=0,rtol=0)
    sf,sd,s3=graph['raw_sparse_views']
    sparse=(sf+2*sd+2*s3)/3 if weights==[1.,2.,2.] else (weights[0]*sf+weights[1]*sd+weights[2]*s3)/3
    torch.testing.assert_close(graph['weighted_sparse'],1.2*sparse,atol=0,rtol=0)
    torch.testing.assert_close(loss,(graph['weighted_align']+1.2*sparse)+graph['weighted_hierarchy'],atol=0,rtol=0)
    assert not logs['inclusion_enabled'] and logs['inc_weight']==0
    assert graph['lambda_h']==min(1,completed/200)


def test_unreviewed_hns_allocation_rejected():
    with pytest.raises(AssertionError,match='Unreviewed HNS'):tiny([2.,2.,1.])
