"""Read-only invariants around the unchanged native trainer, horizon4868."""

import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from train import train_nested_semantic_mask as trainer
from . import recovery_train_gate as historical


ROOT = historical.ROOT
EXP = historical.EXP
RUN = ROOT / "runtime/SAID-nest-clip-v1/armb_summary02_500gate_recovery_v1"
STEP0_SHA = historical.STEP0_SHA


def require(condition, message):
    historical.require(condition, message)


def exact_state(actual, expected):
    if torch.is_tensor(expected):
        return (torch.is_tensor(actual) and actual.dtype == expected.dtype and
                torch.equal(actual.detach().cpu(), expected.detach().cpu()))
    if isinstance(expected, np.ndarray):
        return isinstance(actual, np.ndarray) and np.array_equal(actual, expected)
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and actual.keys() == expected.keys() and
                all(exact_state(actual[key], value) for key, value in expected.items()))
    if isinstance(expected, (list, tuple)):
        return (isinstance(actual, type(expected)) and len(actual) == len(expected) and
                all(exact_state(left, right) for left, right in zip(actual, expected)))
    return actual == expected


def compare_prefix(actual, expected):
    require([row["step"] for row in actual] == [1, 2, 3, 4, 5], "Incorrect first-five sequence")
    require([row["step"] for row in expected] == [1, 2, 3, 4, 5], "Incomplete smoke reference")
    for current, reference in zip(actual, expected):
        require(current["actual_lrs"] == reference["actual_lrs"], "LR drift")
        require(math.isfinite(current["loss"]) and current["nonfinite"] == 0, "Nonfinite loss")
        health = {row["rank"]: row for row in current["rank_health"]}
        baseline = {row["rank"]: row for row in reference["rank_health"]}
        require(set(health) == set(baseline) == {0, 1, 2, 3}, "Missing rank")
        for rank, item in health.items():
            require(item["stream_sha256"] == baseline[rank]["stream_sha256"], "F/S/D strings/tokens changed")
            for key in ("sample_ids", "full_view_sha256", "local_views_sha256", "split_sha256"):
                require(item["sampling"][key] == baseline[rank]["sampling"][key], "Sample/text drift: " + key)
            require(item["gradients_finite"] and item["updates"] == current["step"] and
                    item["batch"] == 256, "AdamW/gradient health failed")
    return dict(passed=True, samples=5120, exact_sample_ids=True, exact_FSD_strings_tokens=True,
                exact_LR=True, losses=[row["loss"] for row in actual], cross_run_numeric_comparison=False)


def checkpoint_invariants(current, reference):
    require(current["completed_steps"] == 5 and current["scheduler_horizon"] == 4868, "Step/horizon drift")
    config = current["config"]
    require(config["start_updates"] == 0 and config["resume"] is None and
            config["init_sha256"] == STEP0_SHA, "Not a fresh common-step0 trajectory")
    frozen = json.loads((ROOT / "recovery/configs/summary02.json").read_text())
    for key in frozen:
        if key in reference["config"]:
            require(config.get(key) == reference["config"][key], "Frozen hyperparameter drift: " + key)
    for key in ("seed", "sampling_seed", "shuffle_seed", "view_weights", "hparams", "trial_id",
                "code_sha256", "component_initialization", "data", "horizon", "batch_size",
                "world_size", "accumulation", "optimizer_groups"):
        if key in reference["config"]:
            require(config.get(key) == reference["config"][key], "Frozen config/source drift: " + key)
    require(config["view_weights"] == [1.4, .2, 1.4], "S02 weights changed")
    require(current["optimizer"]["param_groups"] == reference["optimizer"]["param_groups"], "Optimizer group drift")
    require(current["optimizer"]["state"].keys() == reference["optimizer"]["state"].keys(), "Optimizer state keys drift")
    pairs = [(current["model"], reference["model"]), (current["adapter"], reference["adapter"])]
    pairs.extend((state, reference["optimizer"]["state"][key]) for key, state in current["optimizer"]["state"].items())
    for actual, expected in pairs:
        require(actual.keys() == expected.keys(), "Tensor keys drift")
        for name, tensor in actual.items():
            require(torch.is_tensor(tensor) and tensor.shape == expected[name].shape and
                    tensor.dtype == expected[name].dtype and bool(torch.isfinite(tensor).all()), "Invalid tensor: " + name)
    require({int(state["step"]) for state in current["optimizer"]["state"].values()} == {5}, "AdamW must update5 times")
    return dict(passed=True, optimizer_steps=[5], scheduler_horizon=4868,
                cross_run_model_and_moment_comparison=False, scaler="Historical BF16; no GradScaler")


def resumable_metadata(payload):
    completed = payload["completed_steps"]
    require(payload["scheduler_horizon"] == 4868, "Checkpoint horizon drift")
    payload.update(global_step=completed,
                   scheduler=dict(type="native_manual_learning_rate_function", horizon=4868,
                                  completed_steps=completed, last_update_index=completed - 1,
                                  optimizer_lrs={group["name"]: group["lr"] for group in payload["optimizer"]["param_groups"]}),
                   scaler=dict(enabled=False, dtype="bfloat16", state=None),
                   sampler=dict(type="DistributedSampler", seed=payload["config"]["shuffle_seed"],
                                world_size=4, batch_size=256, batches_per_epoch=1217),
                   data_cursor=dict(next_epoch=payload["next_epoch"], next_batch=payload["next_batch"]),
                   trajectory_root=str(RUN))
    return payload


def write(path, value):
    temporary = Path(str(path) + ".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def first_five_gate(module, optimizer, output):
    difference = trainer.parameter_agreement(module)
    agreement = [None] * 4
    dist.all_gather_object(agreement, dict(rank=dist.get_rank(), difference=difference))
    config = json.loads((output / "config.json").read_text())
    trainer.save_checkpoint(module, optimizer, config, 5, output)
    result = None
    if dist.get_rank() == 0:
        try:
            smoke = Path(json.loads(historical.SMOKE_COMMAND.read_text())["quarantined_output_dir"])
            result = compare_prefix(historical.rows(output / "steps.jsonl"), historical.rows(smoke / "steps.jsonl"))
            current = torch.load(output / "step000005.pt", map_location="cpu", weights_only=False)
            reference = torch.load(smoke / "step000005.pt", map_location="cpu", weights_only=False)
            result["checkpoint"] = checkpoint_invariants(current, reference)
            require(all(item["difference"] == 0 for item in agreement), "Within-run DDP disagreement")
            require(config["horizon"] == 4868 and config["max_updates"] == 500 and config["run_type"] == "formal",
                    "Incorrect full-trajectory first-stage stop")
            result.update(rank_parameter_agreement=agreement, gate="BEFORE_UPDATE6", stop_updates=500,
                          horizon=4868, source_sha256=config["code_sha256"])
        except Exception as error:
            result = dict(passed=False, gate="BEFORE_UPDATE6", error=type(error).__name__ + ": " + str(error))
        write(RUN / "first-five-gate.json", result)
        write(EXP / "FIRST_FIVE_GATE.json", result)
        print(json.dumps(dict(event="minimal_first_five_gate", **result)), flush=True)
    results = [result]
    dist.broadcast_object_list(results, src=0)
    require(results[0]["passed"], "Minimal hard gate failed: " + str(results[0].get("error")))
    dist.barrier()


def main():
    supervisor = int(os.environ.get("SAID_FULL_SUPERVISOR_PID", "0"))
    historical.require_live_supervisor(supervisor)
    stage = os.environ["SAID_S02_STAGE"]
    require(stage in ("step500", "step4868") and os.environ.get("WORLD_SIZE") == "4", "Invalid stage/world")
    output = RUN / stage
    original_optimizer = trainer.build_optimizer
    original_lrs = trainer.optimizer_learning_rates
    original_save = trainer.atomic_save
    original_validate = trainer.validate_resume_payload
    original_restore = trainer.restore_rng_state
    context = dict(optimizer=None, checked=False, resumed=None, restored_rng=False)

    def capture_optimizer(module):
        optimizer = original_optimizer(module)
        context["optimizer"] = optimizer
        return optimizer

    def save(payload, path):
        if "model" in payload and "optimizer" in payload:
            payload = resumable_metadata(payload)
        return original_save(payload, path)

    def validate(previous, current, *args):
        require(stage == "step4868" and previous["completed_steps"] == 500 and
                current["resume"] == str(RUN / "step500/step000500.pt"), "Only this trajectory's step500 may resume")
        require(previous.get("trajectory_root") == str(RUN) and previous["global_step"] == 500 and
                previous["scheduler"]["horizon"] == 4868 and previous["scaler"]["enabled"] is False,
                "Missing resumable scheduler/scaler metadata")
        require(json.loads((EXP / "CONTINUATION_GATE.json").read_text())["status"] == "REPRODUCTION_PASS",
                "No authorization to continue past500")
        context["resumed"] = previous
        return original_validate(previous, current, *args)

    def restore(state):
        original_restore(state)
        require(exact_state(trainer.rng_state(), {key: state[key] for key in ("python", "numpy", "cpu", "cuda")}),
                "CPU/CUDA/Python/NumPy RNG restoration drift")
        context["restored_rng"] = True

    def learning_rates(module, completed, horizon):
        historical.require_live_supervisor(supervisor)
        require(horizon == 4868 and completed < (500 if stage == "step500" else 4868), "Step/horizon exceeded")
        if stage == "step500" and completed == 5 and not context["checked"]:
            first_five_gate(module, context["optimizer"], output)
            context["checked"] = True
        if stage == "step4868" and completed == 500 and not context["checked"]:
            previous = context["resumed"]
            require(previous is not None and context["restored_rng"], "No validated full-state resume")
            require(exact_state(module.clip.state_dict(), previous["model"]) and
                    exact_state(trainer.auxiliary_module(module).state_dict(), previous["adapter"]), "Resume model mismatch")
            require(exact_state(context["optimizer"].state_dict(), previous["optimizer"]), "Resume AdamW/moments mismatch")
            require(torch.equal(module._loader_epoch_generator_state,
                                previous["rng_per_rank"][dist.get_rank()]["loader_generator"]), "Sampler loader RNG mismatch")
            difference = trainer.parameter_agreement(module)
            proof = dict(rank=dist.get_rank(), passed=True, model_optimizer_exact_restore=True,
                         rng_restored=True, data_cursor=previous["data_cursor"], horizon=4868,
                         scaler=previous["scaler"], parameter_difference=difference)
            proofs = [None] * 4
            dist.all_gather_object(proofs, proof)
            require(all(item["parameter_difference"] == 0 for item in proofs), "Resume DDP disagreement")
            if dist.get_rank() == 0:
                write(EXP / "STEP500_RESUME_AUDIT.json", dict(passed=True, ranks=proofs, next_update=501))
            context.update(checked=True, resumed=None)
        return original_lrs(module, completed, horizon)

    trainer.build_optimizer = capture_optimizer
    trainer.atomic_save = save
    trainer.validate_resume_payload = validate
    trainer.restore_rng_state = restore
    trainer.optimizer_learning_rates = learning_rates
    try:
        trainer.main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
