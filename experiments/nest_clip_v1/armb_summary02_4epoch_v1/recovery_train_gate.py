"""Run the unchanged trainer, synchronously validate update5 before update6."""

import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist

from train import train_nested_semantic_mask as trainer


ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parent
RUN = ROOT / "runtime/SAID-nest-clip-v1/armb_summary02_4epoch_recovery_v1"
TRAIN = RUN / "train"
SMOKE_COMMAND = ROOT / "recovery/evidence/s02-smoke-command.json"
STEP0_SHA = "54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def require_live_supervisor(pid):
    require(pid > 0, "Missing full supervisor identity; refusing unmonitored training")
    try:
        os.kill(pid, 0)
    except ProcessLookupError as error:
        raise RuntimeError("Full supervisor disappeared; refusing further optimizer updates") from error


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def compare_prefix(actual, expected):
    require([row["step"] for row in actual] == [1, 2, 3, 4, 5], "Full prefix does not contain exactly updates1..5")
    require([row["step"] for row in expected] == [1, 2, 3, 4, 5], "Smoke prefix is incomplete")
    losses = []
    for full, smoke in zip(actual, expected):
        require(full["actual_lrs"] == smoke["actual_lrs"], "Optimizer/scheduler LR drift")
        require(math.isfinite(full["loss"]) and
                math.isclose(full["loss"], smoke["loss"], rel_tol=1e-5, abs_tol=1e-4),
                f"Unexplained loss drift at update{full['step']}: {full['loss']} vs {smoke['loss']}")
        full_health = {item["rank"]: item for item in full["rank_health"]}
        smoke_health = {item["rank"]: item for item in smoke["rank_health"]}
        require(set(full_health) == set(smoke_health) == {0, 1, 2, 3}, "Missing first-five rank")
        for rank, health in full_health.items():
            reference = smoke_health[rank]
            require(health["sampling"]["sample_ids"] == reference["sampling"]["sample_ids"], "Sample ID drift")
            require(health["stream_sha256"] == reference["stream_sha256"], "F/S/D string or token ID drift")
            for key in ("full_view_sha256", "local_views_sha256", "split_sha256"):
                require(health["sampling"][key] == reference["sampling"][key], "Sampling/text digest drift")
            require(health["gradients_finite"] and health["batch"] == 256 and health["updates"] == full["step"],
                    "Invalid first-five optimizer/gradient health")
        losses.append(dict(step=full["step"], loss=full["loss"], smoke_loss=smoke["loss"],
                           absolute_difference=abs(full["loss"] - smoke["loss"])))
    return dict(passed=True, samples=5120, exact_sample_ids=True,
                exact_FSD_string_and_token_stream=True, exact_LR=True, losses=losses)


def compare_checkpoint(full, smoke):
    require(full["completed_steps"] == smoke["completed_steps"] == 5 and
            full["scheduler_horizon"] == smoke["scheduler_horizon"] == 4868,
            "Step5 optimizer/scheduler checkpoint mismatch")
    require(full["config"]["start_updates"] == 0 and full["config"]["resume"] is None and
            full["config"]["init_sha256"] == STEP0_SHA, "Full did not start from common step0")
    require(full["optimizer"]["param_groups"] == smoke["optimizer"]["param_groups"],
            "Optimizer hyperparameters, parameter ordering or LR drift")
    require(full["optimizer"]["state"].keys() == smoke["optimizer"]["state"].keys(), "Optimizer state keys drift")
    comparisons, exact, maximum = 0, 0, 0.
    pairs = [(full["model"], smoke["model"]), (full["adapter"], smoke["adapter"])]
    pairs.extend((full["optimizer"]["state"][parameter], smoke["optimizer"]["state"][parameter])
                 for parameter in full["optimizer"]["state"])
    for left, right in pairs:
        require(left.keys() == right.keys(), "Step5 state_dict keys drift")
        for name, tensor in left.items():
            require(torch.is_tensor(tensor) and tensor.shape == right[name].shape and
                    tensor.dtype == right[name].dtype and bool(torch.isfinite(tensor).all()), "Invalid step5 tensor")
            comparisons += 1
            if torch.equal(tensor, right[name]):
                exact += 1
            else:
                maximum = max(maximum, float((tensor.float() - right[name].float()).abs().max()))
                torch.testing.assert_close(tensor, right[name], rtol=1e-5, atol=1e-7)
    require({int(state["step"]) for state in full["optimizer"]["state"].values()} == {5}, "AdamW counter is not5")
    return dict(passed=True, compared_tensors=comparisons, bitwise_equal_tensors=exact,
                maximum_absolute_difference=maximum, optimizer_steps=[5], horizon=4868,
                floating_tolerance=dict(rtol=1e-5, atol=1e-7),
                scaler="Historical BF16 autocast has no GradScaler", resume=None)


def synchronous_gate(module, optimizer):
    dist.barrier()
    config = json.loads((TRAIN / "config.json").read_text())
    trainer.save_checkpoint(module, optimizer, config, 5, TRAIN)
    result = None
    if dist.get_rank() == 0:
        try:
            command = json.loads(SMOKE_COMMAND.read_text())
            smoke = Path(command["quarantined_output_dir"])
            result = compare_prefix(rows(TRAIN / "steps.jsonl"), rows(smoke / "steps.jsonl"))
            full_checkpoint = torch.load(TRAIN / "step000005.pt", map_location="cpu", weights_only=False)
            smoke_checkpoint = torch.load(smoke / "step000005.pt", map_location="cpu", weights_only=False)
            result["checkpoint"] = compare_checkpoint(full_checkpoint, smoke_checkpoint)
            del full_checkpoint, smoke_checkpoint
            result.update(gate="BEFORE_UPDATE6", formal_stop_updates=4868,
                          full_config_horizon=config["horizon"], full_start_updates=config["start_updates"])
            require(config["horizon"] == config["max_updates"] == 4868 and
                    config["run_type"] == "formal", "Incorrect full horizon/stop")
        except Exception as error:
            result = dict(passed=False, gate="BEFORE_UPDATE6", error=type(error).__name__ + ": " + str(error))
        (RUN / "first-five-gate.json").write_text(json.dumps(result, indent=2) + "\n")
        (EXP / "FIRST_FIVE_GATE.json").write_text(json.dumps(result, indent=2) + "\n")
        if not result["passed"]:
            failed = dict(status="BLOCKED_PREFIX_GATE", completed_updates=5,
                          stop_updates=4868, formal_training_started=True,
                          formal_training_authorized=False, gate=result,
                          error=result["error"], resume=None)
            for path in (RUN / "state.json", EXP / "FULL_PROGRESS.json", EXP / "FULL_RESULTS.json"):
                path.write_text(json.dumps(failed, indent=2) + "\n")
        print(json.dumps(dict(event="first_five_gate", **result)), flush=True)
    payload = [result]
    dist.broadcast_object_list(payload, src=0)
    require(payload[0]["passed"], "Full stopped before update6: " + str(payload[0].get("error")))
    dist.barrier()


def main():
    original_optimizer = trainer.build_optimizer
    original_learning_rates = trainer.optimizer_learning_rates
    context = dict(optimizer=None, checked=False)
    supervisor_pid = int(os.environ.get("SAID_FULL_SUPERVISOR_PID", "0"))
    require_live_supervisor(supervisor_pid)

    def capture_optimizer(module):
        optimizer = original_optimizer(module)
        context["optimizer"] = optimizer
        return optimizer

    def guarded_learning_rates(module, completed, horizon):
        require_live_supervisor(supervisor_pid)
        require(horizon == 4868, "Scheduler horizon drift")
        if completed == 5 and not context["checked"]:
            synchronous_gate(module, context["optimizer"])
            context["checked"] = True
        require(completed < 4868, "Attempt to execute optimizer update4869")
        return original_learning_rates(module, completed, horizon)

    require(os.environ.get("WORLD_SIZE") == "4", "Full requires four-rank torchrun")
    trainer.build_optimizer = capture_optimizer
    trainer.optimizer_learning_rates = guarded_learning_rates
    try:
        trainer.main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
