"""Weight-only config/gradient isolation and predeclared ablation decision gates."""
import copy
import json
from pathlib import Path

import pytest
import torch

from model.balanced_hparam_search import detail_chain_inclusion
from recovery.nested_detail_equal_weight500 import changed_config,classify,gradient_pressure,SCORES
from tests import test_balanced_hparams as reference


def test_only_alignment_weights_change():
    root = Path(__file__).resolve().parents[1]
    old = json.loads((root/'experiments/nest_clip_v1/nested_detail_500_v1/config.json').read_text())
    new = json.loads((root/'experiments/nest_clip_v1/nested_detail_equal_weight_500_v1/config.json').read_text())
    assert {k for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}=={'view_weights'}
    changed_config(new)
    bad = dict(new,sampling_seed=1)
    with pytest.raises(AssertionError,match='Config drift'):
        changed_config(bad)


def test_equal_weight_loss_and_gradient_match_explicit_chain_reference(monkeypatch):
    torch.manual_seed(1007)
    module = reference.make(dict(view_weights=[1.,1.,1.]))
    module.inclusion_hierarchy='detail_chain'
    explicit = copy.deepcopy(module)
    old_weights = copy.deepcopy(module);old_weights.search_hparams['view_weights']=[1.4,1.4,.2]
    images,tokens,valid = reference.inputs()
    monkeypatch.setattr(reference,'inclusion',detail_chain_inclusion)
    manual = reference.weighted_reference(explicit,images,tokens,valid,499)
    actual,logs = module(images,*tokens,valid,499)
    previous,previous_logs = old_weights(images,*tokens,valid,499)
    torch.testing.assert_close(actual,manual,atol=3e-4,rtol=3e-5)
    for key in ('inc','inc_weight','F_sparse','O_sparse','E_sparse',
                'F_Dall_mask_iou','Dall_Ds_mask_iou','Dall_F_hard_violation','Ds_Dall_hard_violation'):
        torch.testing.assert_close(logs[key],previous_logs[key],atol=0,rtol=0)
    raw = [logs[p+'_i2t']+logs[p+'_t2i'] for p in ('F','O','E')]
    delta = 10/3*sum((a-b)*v for a,b,v in zip([1,1,1],[1.4,1.4,.2],raw))
    torch.testing.assert_close(actual-previous,delta,atol=3e-4,rtol=3e-5)
    assert module.state_dict().keys()==old_weights.state_dict().keys()
    actual.backward();manual.backward()
    for name,p in module.named_parameters():
        q = dict(explicit.named_parameters())[name]
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:
            torch.testing.assert_close(p.grad,q.grad,atol=5e-4,rtol=5e-5,msg=name)


@pytest.mark.parametrize('urban,score_delta,long_delta,dominant,status',[
    (88.7,0,0,False,'EQUAL_WEIGHT_STRONG_POSITIVE'),
    (88.6,-.19,-.19,True,'EQUAL_WEIGHT_POSITIVE'),
    (88.6,-.21,0,True,'TRADEOFF'),
    (88.4,-.01,0,True,'ATOMIC_DETAIL_OVERWEIGHTED'),
    (88.5,0,0,False,'EQUAL_WEIGHT_NO_CLEAR_GAIN'),
])
def test_decision(urban,score_delta,long_delta,dominant,status):
    result = classify(dict(Score5=SCORES['Score5']+score_delta,J_long3=SCORES['J_long3']+long_delta),
        {'Urban-1k':{'I2T':{'R@1':.909},'T2I':{'R@1':urban/100}}},{'dominant':dominant})
    assert result['status']==status and not result['automatic_continuation']


@pytest.mark.parametrize('votes,expected',[(5,False),(6,True),(8,True)])
def test_gradient_dominance_requires_consistency(votes,expected):
    result = gradient_pressure(dict(mean_gradient_norms=dict(F=1.,Dall=1.,Ds=3.),
        batches=[dict(gradient_norms=dict(F=1.,Dall=1.,Ds=3. if i<votes else 1.)) for i in range(8)]))
    assert result['dominant']==expected
    assert result['ratios']==dict(Ds_Dall=3.,Ds_F=3.)
