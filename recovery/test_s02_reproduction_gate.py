import copy
import math

import pytest
import torch

from experiments.nest_clip_v1.armb_summary02_4epoch_v1 import reproduction_full as pipeline
from experiments.nest_clip_v1.armb_summary02_4epoch_v1 import reproduction_train_gate as gate
from recovery.test_s02_full_gate import checkpoint, prefix


def reference_prefix():
    result = prefix()
    for row in result:
        row["nonfinite"] = 0
    return result


def test_finite_numerical_variation_does_not_reject_prefix():
    smoke = reference_prefix()
    actual = copy.deepcopy(smoke)
    actual[-1]["loss"] += .1
    assert gate.compare_prefix(actual, smoke)["passed"]


@pytest.mark.parametrize("mutation", ["ids", "tokens", "lr", "rank", "grad", "nan", "extra"])
def test_hard_prefix_invariants_fail_closed(mutation):
    smoke = reference_prefix()
    actual = copy.deepcopy(smoke)
    if mutation == "ids":
        actual[0]["rank_health"][0]["sampling"]["sample_ids"] = [999]
    elif mutation == "tokens":
        actual[0]["rank_health"][0]["stream_sha256"] = "changed"
    elif mutation == "lr":
        actual[0]["actual_lrs"]["backbone"] = .1
    elif mutation == "rank":
        actual[0]["rank_health"].pop()
    elif mutation == "grad":
        actual[0]["rank_health"][0]["gradients_finite"] = False
    elif mutation == "nan":
        actual[0]["loss"] = math.nan
    else:
        actual.append(dict(step=6))
    with pytest.raises(RuntimeError):
        gate.compare_prefix(actual, smoke)


def reference_checkpoint():
    result = checkpoint()
    result["config"]["view_weights"] = [1.4, .2, 1.4]
    return result


def test_cross_run_model_and_moment_variation_allowed():
    reference = reference_checkpoint()
    actual = copy.deepcopy(reference)
    actual["model"]["weight"] += .1
    actual["optimizer"]["state"][0]["exp_avg"] += .1
    assert gate.checkpoint_invariants(actual, reference)["passed"]


@pytest.mark.parametrize("mutation", ["counter", "weights", "resume", "horizon", "nonfinite", "groups"])
def test_checkpoint_hard_invariants(mutation):
    reference = reference_checkpoint()
    actual = copy.deepcopy(reference)
    if mutation == "counter":
        actual["optimizer"]["state"][0]["step"] += 1
    elif mutation == "weights":
        actual["config"]["view_weights"][1] = .3
    elif mutation == "resume":
        actual["config"]["resume"] = "old-step5.pt"
    elif mutation == "horizon":
        actual["scheduler_horizon"] = 500
    elif mutation == "nonfinite":
        actual["model"]["weight"][:] = math.inf
    else:
        actual["optimizer"]["param_groups"][0]["lr"] = .1
    with pytest.raises(RuntimeError):
        gate.checkpoint_invariants(actual, reference)


def historical_metrics():
    return pipeline.load(pipeline.EXP / "PARENT_500.json")["metrics"]


def test_historical_retrieval_passes():
    metrics = historical_metrics()
    assert pipeline.reproduction_gate(pipeline.REFERENCE, metrics, metrics)["passed"]


@pytest.mark.parametrize("name", ["Score5_R1", "J_long3", "Urban_I2T", "Urban_T2I"])
def test_threshold_is_inclusive_and_no_boundary_tolerance_bypass(name):
    metrics = historical_metrics()
    actual = copy.deepcopy(metrics)
    percent = dict(pipeline.REFERENCE)
    threshold = pipeline.THRESHOLDS[name]
    if name.startswith("Urban_"):
        actual["Urban-1k"][name.split("_")[1]]["R@1"] = threshold / 100
    else:
        percent[name] = threshold
    assert pipeline.reproduction_gate(percent, actual, metrics)["passed"]
    if name.startswith("Urban_"):
        actual["Urban-1k"][name.split("_")[1]]["R@1"] -= .000001
    else:
        percent[name] -= .0001
    result = pipeline.reproduction_gate(percent, actual, metrics)
    assert result["status"] == "REPRODUCTION_FAIL_AT_500"
    assert result["near_boundary"][name]


def test_obvious_dataset_collapse_rejected_even_if_aggregate_given_high():
    metrics = historical_metrics()
    actual = copy.deepcopy(metrics)
    actual["COCO"]["I2T"]["R@1"] = .01
    assert not pipeline.reproduction_gate(pipeline.REFERENCE, actual, metrics)["passed"]


def test_exact_resume_state_comparison_is_distinct_from_cross_run_comparison():
    state = reference_checkpoint()
    assert gate.exact_state(state, copy.deepcopy(state))
    changed = copy.deepcopy(state)
    changed["optimizer"]["state"][0]["exp_avg"] += .00001
    assert not gate.exact_state(state, changed)


def test_scheduler_and_scaler_metadata_do_not_introduce_new_training_math():
    payload = reference_checkpoint()
    payload["optimizer"]["param_groups"][0]["name"] = "backbone"
    payload["config"]["shuffle_seed"] = 0
    payload.update(next_epoch=0, next_batch=5)
    model = payload["model"]
    optimizer = payload["optimizer"]
    result = gate.resumable_metadata(payload)
    assert result["model"] is model and result["optimizer"] is optimizer
    assert result["scheduler"]["horizon"] == 4868
    assert result["scaler"] == dict(enabled=False, dtype="bfloat16", state=None)
    assert result["data_cursor"] == dict(next_epoch=0, next_batch=5)
