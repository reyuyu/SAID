"""Runtime isolation and full-budget execution despite weak screening scores."""
from pathlib import Path

import pytest

from experiments.nest_clip_v1.balanced_hparam_search_v1.search import Search, TrialFailure
from experiments.nest_clip_v1.three_followup_v1.run import EXPERIMENTS, Followup
from model.balanced_hparam_search import trial_id


def test_search_paths_are_isolated_by_instance(tmp_path):
    first = Search(run_dir=tmp_path/'first', experiment_dir=tmp_path/'exp1')
    second = Search(run_dir=tmp_path/'second', experiment_dir=tmp_path/'exp2')
    first.enroll(EXPERIMENTS[0][1])
    second.enroll(EXPERIMENTS[1][1])
    first.publish()
    second.publish()
    assert first.state_path != second.state_path
    assert set(first.state['trials']).isdisjoint(second.state['trials'])
    assert (tmp_path/'exp1/SEARCH_STATE.json').exists()
    assert (tmp_path/'exp2/SEARCH_STATE.json').exists()
    assert len(list((tmp_path/'first/configs').glob('*.json'))) == 1
    assert len(list((tmp_path/'second/configs').glob('*.json'))) == 1


@pytest.mark.parametrize('fail_first', [False, True])
def test_full_budget_and_same_trial_resume_are_sequential(monkeypatch, fail_first):
    runner = Followup.__new__(Followup)
    runner.state = dict(trials={}, experiments=[])
    for label, hp in EXPERIMENTS:
        tid = trial_id(hp)
        runner.state['trials'][tid] = dict(hparams=hp, budgets={})
        runner.state['experiments'].append(dict(name=label, trial_id=tid, status='pending'))
    calls = []

    def train(hp, stop, kind='formal', resume=None):
        tid = trial_id(hp)
        calls.append((tid, kind, stop, resume))
        if fail_first and tid == trial_id(EXPERIMENTS[0][1]):
            raise TrialFailure('nonfinite smoke')
        return Path('/runtime')/tid/f'step{stop}'

    def evaluate(tid, root, stop):
        runner.state['trials'][tid]['budgets'][str(stop)] = dict(
            checkpoint=str(root/f'step{stop:06d}.pt'), scores=dict(Score5_R1=0.01))

    monkeypatch.setattr(runner, 'train', train)
    monkeypatch.setattr(runner, 'evaluate', evaluate)
    monkeypatch.setattr(runner, 'save', lambda: None)
    monkeypatch.setattr(runner, 'publish', lambda: None)
    monkeypatch.setattr(runner, 'sync', lambda message: None)
    runner.run()
    for index, (_, hp) in enumerate(EXPERIMENTS):
        tid = trial_id(hp)
        actual = [c for c in calls if c[0] == tid]
        if fail_first and index == 0:
            assert len(actual) == 1
            assert runner.state['experiments'][index]['status'] == 'failed'
        else:
            assert [c[2] for c in actual] == [5,500,3651]
            assert actual[0][3] is None and actual[1][3] is None
            assert actual[2][3] == Path('/runtime')/tid/'step500/step000500.pt'
            assert runner.state['experiments'][index]['status'] == 'completed'
    assert runner.state['status'] == 'awaiting_epoch_clarification'
    assert all(c[0] == trial_id(EXPERIMENTS[0][1]) for c in calls[:1 if fail_first else 3])
