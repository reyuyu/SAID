import copy
import math
import subprocess
import types
import pytest
import torch
from recovery.hns_v1_pure3 import (BASE, ROOT, HORIZON, analytic_lrs, expected_config,
                                   normalize_json, validate_config)
from recovery.hns_ddp_correctness import make
from tests.test_balanced_hparams import inputs
from train.train_nested_semantic_mask import optimizer_learning_rates, training_horizon


def test_only_declared_schedule_and_save_control_changes():
    old = __import__('json').loads(subprocess.check_output(['git', 'show', BASE +
        ':experiments/nest_clip_v1/nested_d3_hns_full_v1/config.json'], cwd=ROOT))
    new = expected_config(); validate_config(new)
    assert {k for k in new if new[k] != old[k]} == {'epochs', 'four_epoch_followup', 'save_initial_checkpoint'}
    assert training_horizon(new, 1217) == HORIZON
    assert training_horizon(old, 1217) == 4868


@pytest.mark.parametrize('key,value', [('view_weights', [1,1,1]), ('view_sparsity_weights',[5/3]*3),
    ('hns_beta',[1,2]), ('lambda_sparse',1.2), ('hns_enabled',False), ('epochs',4),
    ('hns_detach_child',True), ('batch_size',128), ('fusion_lr',1e-4)])
def test_method_drift_rejected(key, value):
    c = expected_config(); c[key] = value
    with pytest.raises(AssertionError): validate_config(c)


def test_full_lr_sequence_and_terminal_index():
    module = make(); module.search_hparams['fusion_lr'] = 2e-4
    for s in range(HORIZON + 1):
        assert list(optimizer_learning_rates(module, s, HORIZON)) == analytic_lrs(s)
    assert analytic_lrs(0) == [5e-9, 1e-3, 1e-3, 2e-4]
    assert analytic_lrs(199)[0] == analytic_lrs(200)[0] == 1e-6
    assert all(v > 0 for v in analytic_lrs(3650))
    assert analytic_lrs(3651) == [0,0,0,0]
    assert optimizer_learning_rates(module, 500, 3651) != optimizer_learning_rates(module, 500, 4868)


@pytest.mark.parametrize('completed', [0,99,199,500])
def test_fetched_original_loss_logs_and_every_gradient_exact(completed):
    source = subprocess.check_output(['git','show',BASE+':model/balanced_hparam_search.py'],cwd=ROOT,text=True)
    namespace = {'__name__':'model._pure3_frozen_ref','__package__':'model'}
    exec(compile(source, 'HNS-v1-frozen', 'exec'), namespace)
    torch.manual_seed(20261011)
    new = make(); old = copy.deepcopy(new)
    old.forward = types.MethodType(namespace['BalancedSearch'].forward, old)
    images, views, valid = inputs()
    a, al = new(images,*views,valid,completed); b, bl = old(images,*views,valid,completed)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert al.keys() == bl.keys()
    for k in al:
        if torch.is_tensor(al[k]): torch.testing.assert_close(al[k],bl[k],atol=0,rtol=0,msg=k)
        else: assert al[k] == bl[k]
    a.backward(); b.backward()
    for name,p in new.named_parameters():
        q = dict(old.named_parameters())[name]
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None: torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0,msg=name)


def test_complete_json_normalization_fail_closed():
    a = {'histogram': {3: 256}, 'nested': {'K': [1,2,3]}}
    b = {'histogram': {'3': 256}, 'nested': {'K': [1,2,3]}}
    assert normalize_json(a) == normalize_json(b)
    assert normalize_json(a) != normalize_json(dict(b, extra=0))
    assert normalize_json(a) != normalize_json({'histogram': {'3':255},'nested':{'K':[1,2,3]}})
    with pytest.raises(ValueError): normalize_json({1:1,'1':2})
    with pytest.raises(ValueError): normalize_json({'invalid':float('nan')})
