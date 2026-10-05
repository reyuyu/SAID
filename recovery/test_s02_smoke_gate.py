import math
import copy

import pytest

import s02_smoke_gate
from s02_smoke_gate import check_config, check_steps, require


def test_gate_fails_closed():
    with pytest.raises(RuntimeError, match="blocked"):
        require(False, "blocked")


@pytest.mark.parametrize("steps", [[], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]])
def test_exactly_five_steps_required(steps):
    with pytest.raises(RuntimeError, match="exactly"):
        check_steps([dict(step=step) for step in steps], {})


@pytest.mark.parametrize("config", [{}, {"view_weights": [1.4, 1.0, 1.4]}, {"horizon": 5}])
def test_incomplete_or_wrong_protocol_config_rejected(config):
    with pytest.raises(RuntimeError, match="frozen S02"):
        check_config(config)


def test_nan_is_not_finite():
    with pytest.raises(RuntimeError, match="nonfinite"):
        require(math.isfinite(float("nan")), "nonfinite")


def valid_steps():
    rows = []
    for step in range(1, 6):
        factor = .5 * (1 + math.cos(math.pi * (step - 1) / 4868))
        row = dict(step=step, duplicate_image_ids=[], nonfinite=0, loss=61.,
                   inc=0., inc_weight=0., actual_lrs=dict(backbone=1e-6 * step / 200,
                       text_mask_and_shared_pool=1e-3 * factor, visual_mask=1e-3 * factor,
                       fusion_adapter=2e-4 * factor),
                   rank_health=[dict(rank=rank, batch=256, updates=step, gradients_finite=True,
                                     gradient_norms={"backbone": 1.}) for rank in range(4)])
        for prefix in ("F", "O", "E"):
            row.update({prefix + "_i2t": 3., prefix + "_t2i": 3.,
                        prefix + "_sparse": .6, prefix + "_keep_ratio": .6})
        rows.append(row)
    acceptance = dict(passed=True, ranks=[dict(rank=rank, completed_updates=5, updates_this_run=5,
                      max_parameter_difference_from_rank0=0., final_nccl_all_reduce=10.) for rank in range(4)])
    return rows, acceptance


def test_historical_weighted_loss_and_lr_pass():
    rows, acceptance = valid_steps()
    metrics = check_steps(rows, acceptance)
    assert metrics[0]["weighted_alignment"]["S"] == 4.
    assert metrics[0]["sparsity"] == 1.


@pytest.mark.parametrize("mutation", ["nan", "wrong_lr", "wrong_weighted_loss", "rank_duplicate", "ddp_mismatch", "six_optimizer_updates"])
def test_corrupt_acceptance_rejected(mutation):
    rows, acceptance = copy.deepcopy(valid_steps())
    if mutation == "nan":
        rows[0]["loss"] = float("nan")
    elif mutation == "wrong_lr":
        rows[1]["actual_lrs"]["fusion_adapter"] = .0001
    elif mutation == "wrong_weighted_loss":
        rows[0]["loss"] = 100.
    elif mutation == "rank_duplicate":
        rows[0]["rank_health"][1]["rank"] = 0
    elif mutation == "ddp_mismatch":
        acceptance["ranks"][0]["max_parameter_difference_from_rank0"] = .01
    else:
        acceptance["ranks"][0]["completed_updates"] = 6
    with pytest.raises(RuntimeError):
        check_steps(rows, acceptance)


@pytest.mark.parametrize("passed", [True, False])
def test_publishing_finished_gate_never_authorizes_more_training(tmp_path, monkeypatch, passed):
    import json
    monkeypatch.setattr(s02_smoke_gate, "EVIDENCE", tmp_path)
    policy = tmp_path / "recovery-operation-policy.json"
    policy.write_text(json.dumps(dict(smoke_authorized=True, formal_training_authorized=False)))
    result = dict(passed=passed, executed=True, status="READY" if passed else "NOT_READY",
                  blockers=[], steps_completed=5)
    s02_smoke_gate.publish_readiness(result)
    published = json.loads(policy.read_text())
    assert not published["smoke_authorized"]
    assert not published["conditional_smoke_authorized"]
    assert not published["formal_training_authorized"]
    assert json.loads((tmp_path / "smoke-audit.json").read_text())["passed"] == passed


def test_publishing_cannot_accept_six_updates():
    with pytest.raises(RuntimeError, match="exactly five"):
        s02_smoke_gate.publish_readiness(dict(passed=True, steps_completed=6))
