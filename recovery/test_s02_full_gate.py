import copy
import importlib

import pytest
import torch


gate = importlib.import_module("experiments.nest_clip_v1.armb_summary02_4epoch_v1.recovery_train_gate")


def prefix():
    health = [dict(rank=rank, gradients_finite=True, batch=256, stream_sha256=str(rank),
                   sampling=dict(sample_ids=[rank], full_view_sha256="F", local_views_sha256="SD", split_sha256="split"))
              for rank in range(4)]
    result = []
    for step in range(1, 6):
        ranks = copy.deepcopy(health)
        for item in ranks:
            item["updates"] = step
        result.append(dict(step=step, loss=20., actual_lrs={"backbone": 1e-8}, rank_health=ranks))
    return result


def test_exact_prefix_passes():
    smoke = prefix()
    assert gate.compare_prefix(copy.deepcopy(smoke), smoke)["passed"]


@pytest.mark.parametrize("mutation", ["ids", "tokens", "loss", "lr", "extra_update", "missing_rank"])
def test_drift_stops_before_six(mutation):
    smoke = prefix()
    full = copy.deepcopy(smoke)
    if mutation == "ids":
        full[0]["rank_health"][0]["sampling"]["sample_ids"] = [100]
    elif mutation == "tokens":
        full[0]["rank_health"][0]["stream_sha256"] = "drift"
    elif mutation == "loss":
        full[0]["loss"] = 21.
    elif mutation == "lr":
        full[0]["actual_lrs"]["backbone"] = 1e-3
    elif mutation == "extra_update":
        full.append(dict(step=6))
    else:
        full[0]["rank_health"].pop()
    with pytest.raises(RuntimeError):
        gate.compare_prefix(full, smoke)


def checkpoint():
    return dict(completed_steps=5, scheduler_horizon=4868,
                config=dict(start_updates=0, resume=None, init_sha256=gate.STEP0_SHA),
                model={"weight": torch.tensor([1.])}, adapter={"weight": torch.tensor([0.])},
                optimizer=dict(param_groups=[dict(lr=1e-8, params=[0])],
                               state={0: dict(step=torch.tensor(5.), exp_avg=torch.tensor([.1]),
                                              exp_avg_sq=torch.tensor([.01]))}))


def test_bitwise_step5_optimizer_and_model_pass():
    smoke = checkpoint()
    result = gate.compare_checkpoint(copy.deepcopy(smoke), smoke)
    assert result["bitwise_equal_tensors"] == result["compared_tensors"]
    assert result["optimizer_steps"] == [5]


@pytest.mark.parametrize("mutation", ["model", "moments", "counter", "resume", "horizon"])
def test_model_or_optimizer_drift_rejected(mutation):
    smoke = checkpoint()
    full = copy.deepcopy(smoke)
    if mutation == "model":
        full["model"]["weight"] += 1.
    elif mutation == "moments":
        full["optimizer"]["state"][0]["exp_avg"] += 1.
    elif mutation == "counter":
        full["optimizer"]["state"][0]["step"] += 1.
    elif mutation == "resume":
        full["config"]["resume"] = "step5.pt"
    else:
        full["scheduler_horizon"] = 5
    with pytest.raises((RuntimeError, AssertionError)):
        gate.compare_checkpoint(full, smoke)
