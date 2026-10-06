"""Visible-only semantics, private RNG and detached chain gradient routing."""
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from model import longclip
from model.balanced_hparam_search import detail_chain_inclusion
from model.nested_semantic_mask import inclusion
from train import nested_semantic_data as data


@pytest.mark.parametrize('caption', [
    'First. Second.', 'First. Second. Third. Fourth. Fifth.',
    'A street. A bus. A shop. ' + 'invisible '*300,
    'Only one.', 'word '*300, 'word '*300 + '. A cat.',
])
def test_visible_hierarchy_and_frozen_F(caption):
    for epoch in (0,1,3):
        for sid in (1000,73,512000):
            old = data.sampled_text_views(caption,'summary_random_detail',0,epoch,sid)
            new = data.sampled_text_views(caption,'nested_detail',0,epoch,sid)
            assert torch.equal(new['tokens_f'],old['tokens_f'])
            assert new['views'][0] == old['views'][0] and new['valid'] == old['valid']
            assert new['reference_views'] == old['reference_views']
            if new['valid']:
                parts = new['views'][0].split('. ')
                j = new['detail_indices'][0]
                assert new['dall_indices'] == list(range(1,len(parts)))
                assert new['views'][1] == '. '.join(parts[1:])
                assert 1 <= j < len(parts) and new['views'][2] == parts[j]
                assert torch.equal(new['tokens_o'],old['reference_tokens_e'])
                assert torch.equal(new['tokens_e'],longclip.tokenize([parts[j]],truncate=False)[0])
            else:
                assert new['views'][1:] == [None,None]
                assert not new['tokens_o'].any() and not new['tokens_e'].any()
                assert new['detail_indices'] == new['dall_indices'] == []


def test_stateless_uniform_and_global_rng():
    before_py,before_np,before_torch = random.getstate(),np.random.get_state(),torch.get_rng_state().clone()
    draws = [data.sample_atomic_detail_index(6,0,0,i) for i in range(10000)]
    assert draws == [data.sample_atomic_detail_index(6,0,0,i) for i in range(10000)]
    assert draws != [data.sample_atomic_detail_index(6,0,1,i) for i in range(10000)]
    assert all(1800 <= draws.count(i) <= 2200 for i in range(1,6))
    assert before_py == random.getstate() and torch.equal(before_torch,torch.get_rng_state())
    current = np.random.get_state()
    assert current[0] == before_np[0] and np.array_equal(current[1],before_np[1]) and current[2:] == before_np[2:]


def test_chain_child_detach_and_no_direct_edge():
    f = torch.tensor([[.1,.9]],requires_grad=True)
    a = torch.tensor([[.4,.3]],requires_grad=True)
    s = torch.tensor([[.8,.7]],requires_grad=True)
    actual = detail_chain_inclusion(f,a,s)
    assert torch.allclose(actual,torch.tensor([.275]))
    actual.sum().backward()
    assert torch.equal(f.grad,torch.tensor([[-.25,0.]]))
    assert torch.equal(a.grad,torch.tensor([[-.25,-.25]]))
    assert s.grad is None
    # The baseline sibling formula remains distinct and unchanged.
    assert torch.allclose(inclusion(f,a,s),torch.tensor([.25]))


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_metadata_with_device_tokens_and_cpu_references(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    samples = [dict(image=torch.zeros(3,2,2),image_id=i,sample_id=i,
        **data.sampled_text_views(c,'nested_detail',sample_id=i))
        for i,c in enumerate(('One. Two. Three. Four.','Only one.'))]
    batch = data.collate(samples)
    for k in ('tokens_f','tokens_o','tokens_e','valid'):
        batch[k] = batch[k].to(device)
    diag = data.sampling_diagnostics(batch)
    assert diag['nested_detail_exact'] and diag['nested_detail_statistics']['valid_samples'] == 1
    assert 'summary_baseline_exact' not in diag


def test_frozen_config_except_requested_method_changes():
    root = Path(__file__).resolve().parents[1]
    old = json.loads((root/'recovery/configs/summary02_local500.json').read_text())
    new = json.loads((root/'experiments/nest_clip_v1/nested_detail_500_v1/config.json').read_text())
    assert {k for k in old.keys()|new.keys() if old.get(k)!=new.get(k)} == {
        'sampling_mode','inclusion_hierarchy','view_weights'}
    assert new['view_weights'] == [1.4,1.4,.2] and new['sparsity_scale'] == new['inclusion_max'] == 1


def test_chain_forward_matches_explicit_objective_and_telemetry_has_no_gradient(monkeypatch):
    import copy
    from tests import test_balanced_hparams as reference
    torch.manual_seed(924)
    module = reference.make(dict(view_weights=[1.4,1.4,.2]))
    module.inclusion_hierarchy = 'detail_chain'
    manual = copy.deepcopy(module)
    images,views,valid = reference.inputs()
    monkeypatch.setattr(reference,'inclusion',detail_chain_inclusion)
    expected = reference.weighted_reference(manual,images,views,valid,499)
    actual,logs = module(images,*views,valid,499)
    torch.testing.assert_close(actual,expected,atol=3e-4,rtol=3e-5)
    assert {'F_Dall_mask_iou','Dall_Ds_mask_iou','Ds_Dall_hard_violation','Dall_F_hard_violation'} <= logs.keys()
    assert all(not v.requires_grad for v in logs.values() if torch.is_tensor(v))
    actual.backward(); expected.backward()
    for name,p in module.named_parameters():
        q = dict(manual.named_parameters())[name]
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            torch.testing.assert_close(p.grad,q.grad,atol=5e-4,rtol=5e-5,msg=name)


def test_resume_rejects_changed_hierarchy():
    from tests.test_nested_resume import payloads
    from model.balanced_hparam_search import hparams
    from train.train_nested_semantic_mask import validate_resume_payload
    old,current = payloads()
    old['config'].update(hparam_search=True,inclusion_hierarchy='detail_chain',**hparams({}))
    current.update(hparam_search=True,inclusion_hierarchy='siblings',**hparams({}))
    with pytest.raises(AssertionError,match='hierarchy'):
        validate_resume_payload(old,current,'old-trainer')
