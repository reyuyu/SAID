"""Four-epoch schedule, exact boundary continuation and frozen parent selection."""
import copy
import math
from pathlib import Path

import pytest

from experiments.nest_clip_v1.balanced_hparam_search_v1.search import BASE_CONFIG, Search
from experiments.nest_clip_v1.three_followup_v1 import four_epoch
from model.balanced_hparam_search import hparams, trial_id
from tests.test_balanced_hparams import make
from tests.test_nested_resume import payloads
from train.train_nested_semantic_mask import training_horizon, validate_resume_payload, optimizer_learning_rates


def test_four_epoch_config_is_explicit_and_does_not_change_three_epoch_defaults(tmp_path):
    hp = hparams(dict(fusion_lr=2e-4))
    config = dict(BASE_CONFIG, epochs=4, four_epoch_followup=True)
    runner = Search(run_dir=tmp_path/'runtime', experiment_dir=tmp_path/'evidence', base_config=config)
    tid, path = runner.config(hp)
    actual = four_epoch.load(path)
    assert actual['epochs'] == 4 and actual['four_epoch_followup']
    assert actual['fusion_lr'] == 2e-4 and actual['inclusion_max'] == 1
    assert tid == trial_id(hp)
    assert BASE_CONFIG['epochs'] == 3 and 'four_epoch_followup' not in BASE_CONFIG
    assert training_horizon(actual,1217) == 4868
    assert training_horizon(BASE_CONFIG,1217) == 3651
    for field,value in [('four_epoch_followup',False),('fusion','crossscore_flat'),('epochs',5)]:
        wrong = dict(actual, **{field:value})
        with pytest.raises(AssertionError):
            training_horizon(wrong,1217)


def test_four_epoch_resume_preserves_boundary_and_rejects_original_three_epoch_checkpoint():
    previous,current = payloads()
    previous['config'].update(epochs=4,horizon=4868,four_epoch_followup=True)
    current.update(epochs=4,horizon=4868,four_epoch_followup=True,max_updates=4868)
    previous.update(completed_steps=3651,scheduler_horizon=4868,next_epoch=3,next_batch=0)
    assert validate_resume_payload(previous,current,'old-trainer') == 3651
    original = copy.deepcopy(previous)
    original['config'].update(epochs=3,horizon=3651,four_epoch_followup=False)
    original['scheduler_horizon'] = 3651
    with pytest.raises(AssertionError):
        validate_resume_payload(original,current,'old-trainer')


def test_four_epoch_lr_is_continuous_with_same_peaks_and_different_trajectory():
    model = make(dict(fusion_lr=2e-4))
    rates = optimizer_learning_rates(model,3651,4868)
    factor = .5*(1+math.cos(math.pi*3651/4868))
    assert rates[1:] == (1e-3*factor,1e-3*factor,2e-4*factor)
    assert rates[0] > 0 and rates[1] > 0
    assert rates != optimizer_learning_rates(model,0,4868)
    assert rates != optimizer_learning_rates(model,3651,3651)
    assert optimizer_learning_rates(model,4867,4868)[3] > 0


def test_third_starts_at_step0_and_resumes_its_own_four_epoch_checkpoint(monkeypatch):
    runner = four_epoch.FourEpoch.__new__(four_epoch.FourEpoch)
    hp = hparams(dict(fusion_lr=2e-4))
    tid = trial_id(hp)
    experiment = dict(name='Four-epoch fusion-only',trial_id=tid,status='pending')
    runner.state = dict(four_epoch_experiment=experiment,experiment3=dict(trial_id=tid),
                        trials={tid:dict(hparams=hp,budgets={})})
    previous = dict(status='awaiting_epoch_clarification',experiment3=dict(trial_id=tid),
                    experiments=[dict(status='completed'),dict(status='completed')])
    monkeypatch.setattr(four_epoch,'load',lambda path: dict(horizon=4868)
                        if str(path).endswith('config.json') else previous)
    monkeypatch.setattr(four_epoch.shutil,'copytree',lambda *args,**kwargs: None)
    calls = []

    def train(hp,stop,resume=None):
        calls.append((stop,resume))
        return Path('/four-epoch')/f'step{stop}'

    def evaluate(tid,root,stop):
        runner.state['trials'][tid]['budgets'][str(stop)] = dict(checkpoint=str(root/f'step{stop:06d}.pt'))

    monkeypatch.setattr(runner,'train',train)
    monkeypatch.setattr(runner,'evaluate',evaluate)
    monkeypatch.setattr(runner,'save',lambda: None)
    monkeypatch.setattr(runner,'sync',lambda message: None)
    runner.run()
    assert calls == [(3651,None),(4868,Path('/four-epoch/step3651/step003651.pt'))]
    assert runner.state['status'] == 'completed'
    assert runner.state['experiment3']['status'] == 'completed'
