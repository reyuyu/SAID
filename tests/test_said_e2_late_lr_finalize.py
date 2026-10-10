import copy

import pytest

from recovery import said_e2_late_lr_finalize as f


def test_urban_mean_does_not_mistake_one_direction_gain_for_joint_gain():
    baseline = f.c.read(f.c.BASE_EXP / 'step4868/RESULTS.json')
    trial = copy.deepcopy(baseline)
    urban = next(ds for ds in trial['metrics'] if 'Urban' in ds)
    trial['metrics'][urban]['I2T']['R@1'] += .001
    trial['metrics'][urban]['T2I']['R@1'] -= .002
    out = f.comparison(trial, baseline)
    assert out['delta_pp']['Urban_I2T'] == pytest.approx(.1)
    assert out['delta_pp']['Urban_T2I'] == pytest.approx(-.2)
    assert out['delta_pp']['Urban_Mean'] == pytest.approx(-.05)
    assert not out['Urban_both_directions_improved']
    assert not out['Score5_and_Urban_Mean_improved']


def test_report_module_never_constructs_model_or_optimizer():
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(f))
    calls = [n.func.attr for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert not set(calls).intersection({'AdamW', 'SGD', 'backward', 'step', 'load_state_dict'})
