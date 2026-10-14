import math

import pytest

from train.train_nested_semantic_mask import learning_rates, training_horizon


BASE = dict(
    epochs=5, five_epoch_followup=True, four_epoch_followup=False,
    hparam_search=True, base_model='ViT-L/14', condition_mode='dual_branch',
    fusion='balanced_stack', visual='patch',
    experiment_name='HNS-v1-ViT-L14-Full5',
    trial_id='hns-v1-l14-full5epoch6085-fromscratch-v1')


def test_only_explicit_l14_five_epoch_is_authorized():
    assert training_horizon(BASE, 1217) == 6085
    for bad in (
        dict(BASE, base_model='ViT-B/16'),
        dict(BASE, five_epoch_followup=False),
        dict(BASE, four_epoch_followup=True),
        dict(BASE, fusion='crossscore_flat'),
        dict(BASE, trial_id='other'),
    ):
        with pytest.raises(AssertionError):
            training_horizon(bad, 1217)


def test_five_epoch_scheduler_boundaries_are_continuous():
    for step in (0, 200, 1217, 2434, 3651, 4868, 6084):
        rates = learning_rates(step, 6085, include_adapter=True)
        assert all(rate >= 0 for rate in rates)
    assert learning_rates(0, 6085, True)[0] == 5e-9
    assert learning_rates(199, 6085, True)[0] == 1e-6
    assert learning_rates(201, 6085, True)[0] < 1e-6
    assert learning_rates(6084, 6085, True)[1] > 0
    assert learning_rates(6085, 6085, True)[1] == 0
    # The five-epoch curve is intentionally different from the historical 4868 curve.
    assert learning_rates(4868, 6085, True) != learning_rates(4868, 4868, True)


def test_epoch_boundaries_and_sample_count():
    assert [1217 * i for i in range(1, 6)] == [1217, 2434, 3651, 4868, 6085]
    assert 6085 * 1024 == 6231040
