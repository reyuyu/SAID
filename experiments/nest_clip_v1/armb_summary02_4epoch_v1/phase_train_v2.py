"""Instrument unchanged native training; diagnostic only, hard cap40."""

import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from train import train_nested_semantic_mask as trainer
from recovery.resource_stall_v2 import Nvml, system_snapshot
from . import reproduction_train_gate as invariants


class PhaseRecorder:
    def __init__(self, local, rank):
        self.path = Path(local) / f"rank{rank}.jsonl"
        self.rank = rank
        self.nvml = Nvml()
        self.current = None
        self.counter = 0
        self.phase = None
        self.collective_depth = 0

    def begin(self):
        self.finish()
        self.counter += 1
        self.current = dict(step=self.counter, rank=self.rank, started=time.perf_counter(),
                            phase_wall_s={}, events={}, explicit_collective_wall_s={},
                            system_before=system_snapshot(), gpu_before=self.nvml.sample(), h2d_calls=0)

    def invoke(self, name, function, *args, **kwargs):
        if self.current is None:
            return function(*args, **kwargs)
        started = time.perf_counter()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        previous = self.phase
        self.phase = name
        try:
            return function(*args, **kwargs)
        finally:
            end.record()
            self.current["phase_wall_s"][name] = time.perf_counter() - started
            self.current["events"][name] = (begin, end)
            self.phase = previous
            if name == "backward":
                self.current["gradient_checks_begin"] = time.perf_counter()
                self.current["gradient_checks_gpu_begin"] = end

    def finish(self):
        if self.current is None:
            return
        current = self.current
        self.current = None
        if "optimizer" not in current["events"]:
            return
        record = {key: value for key, value in current.items() if key not in
                  ("events", "started", "h2d_gpu_begin", "gradient_checks_gpu_begin", "gradient_checks_begin")}
        record["wall_cycle_before_checkpoint_s"] = time.perf_counter() - current["started"]
        for name, (begin, end) in current["events"].items():
            record[name + "_s"] = begin.elapsed_time(end) / 1000 if end.query() else None
        record["ddp_sync_s"] = current["explicit_collective_wall_s"].get("outside_model", 0.)
        record["backward_includes_DDP"] = True
        record["system_after"] = system_snapshot()
        record["gpu_after"] = self.nvml.sample()
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")


class TimedIterator:
    def __init__(self, iterator, recorder):
        self.iterator, self.recorder = iterator, recorder

    def __iter__(self):
        return self

    def __next__(self):
        self.recorder.begin()
        started = time.perf_counter()
        try:
            value = next(self.iterator)
        except StopIteration:
            self.recorder.current = None
            raise
        self.recorder.current["data_wait_s"] = time.perf_counter() - started
        return value

    def __getattr__(self, name):
        return getattr(self.iterator, name)


def main():
    local = Path(os.environ["SAID_PHASE_LOCAL"])
    rank = int(os.environ["RANK"])
    supervisor = int(os.environ["SAID_PHASE_SUPERVISOR_PID"])
    invariants.historical.require_live_supervisor(supervisor)
    recorder = PhaseRecorder(local, rank)
    original_iter = DataLoader.__iter__
    original_cuda = torch.Tensor.cuda
    original_backward = torch.Tensor.backward
    original_forward = DistributedDataParallel.forward
    original_optimizer = trainer.build_optimizer
    original_rates = trainer.optimizer_learning_rates
    original_checkpoint = trainer.save_checkpoint
    original_sync = torch.cuda.synchronize
    context = dict(optimizer=None, prefix_checked=False)

    def iterator(loader):
        return TimedIterator(original_iter(loader), recorder)

    def cuda(tensor, *args, **kwargs):
        current = recorder.current
        if current is None or current["h2d_calls"] >= 5:
            return original_cuda(tensor, *args, **kwargs)
        if current["h2d_calls"] == 0:
            current["h2d_wall_begin"] = time.perf_counter()
            current["h2d_gpu_begin"] = torch.cuda.Event(enable_timing=True)
            current["h2d_gpu_begin"].record()
        result = original_cuda(tensor, *args, **kwargs)
        current["h2d_calls"] += 1
        if current["h2d_calls"] == 5:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            current["events"]["h2d"] = (current["h2d_gpu_begin"], end)
            current["phase_wall_s"]["h2d"] = time.perf_counter() - current.pop("h2d_wall_begin")
        return result

    def forward(module, *args, **kwargs):
        return recorder.invoke("forward", original_forward, module, *args, **kwargs)

    def backward(tensor, *args, **kwargs):
        return recorder.invoke("backward", original_backward, tensor, *args, **kwargs)

    def optimizer(module):
        instance = original_optimizer(module)
        original_step = instance.step
        def step(*args, **kwargs):
            if recorder.current:
                current = recorder.current
                current["gradient_checks_wall_s"] = time.perf_counter() - current["gradient_checks_begin"]
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                current["events"]["gradient_checks"] = (current["gradient_checks_gpu_begin"], end)
            return recorder.invoke("optimizer", original_step, *args, **kwargs)
        instance.step = step
        context["optimizer"] = instance
        return instance

    def checkpoint(module, instance, config, completed, output):
        recorder.finish()
        return original_checkpoint(module, instance, config, completed, output)

    def rates(module, completed, horizon):
        invariants.historical.require_live_supervisor(supervisor)
        invariants.require(horizon == 4868 and completed < 40, "Diagnostic cap/horizon violation")
        if completed == 5 and not context["prefix_checked"]:
            started = time.perf_counter()
            difference = trainer.parameter_agreement(module)
            differences = [None] * 4
            dist.all_gather_object(differences, difference)
            output = Path(json.loads((local / "paths.json").read_text())["output_dir"])
            config = json.loads((output / "config.json").read_text())
            original_checkpoint(module, context["optimizer"], config, 5, output)
            result = None
            if rank == 0:
                try:
                    smoke = Path(json.loads(invariants.historical.SMOKE_COMMAND.read_text())["quarantined_output_dir"])
                    result = invariants.compare_prefix(invariants.historical.rows(output / "steps.jsonl"), invariants.historical.rows(smoke / "steps.jsonl"))
                    actual = torch.load(output / "step000005.pt", map_location="cpu", weights_only=False)
                    reference = torch.load(smoke / "step000005.pt", map_location="cpu", weights_only=False)
                    result["checkpoint"] = invariants.checkpoint_invariants(actual, reference)
                    invariants.require(all(value == 0 for value in differences), "Within-run DDP disagreement")
                    result["rank_parameter_differences"] = differences
                    del actual, reference
                except Exception as error:
                    result = dict(passed=False, error=str(error))
                invariants.write(local / "prefix-gate.json", result)
            payload = [result]
            dist.broadcast_object_list(payload, src=0)
            invariants.require(payload[0]["passed"], "Diagnostic prefix invariant failure")
            context["prefix_checked"] = True
            recorder.current["prefix_gate_wall_s"] = time.perf_counter() - started
        return original_rates(module, completed, horizon)

    def synchronize(*args, **kwargs):
        started = time.perf_counter()
        result = original_sync(*args, **kwargs)
        if recorder.current is not None:
            key = recorder.phase or "outside_model"
            values = recorder.current["explicit_collective_wall_s"]
            values["cuda_synchronize/" + key] = values.get("cuda_synchronize/" + key, 0.) + time.perf_counter() - started
        return result

    def collective(name, function):
        def invoke(*args, **kwargs):
            started = time.perf_counter()
            recorder.collective_depth += 1
            try:
                return function(*args, **kwargs)
            finally:
                recorder.collective_depth -= 1
                if recorder.current is not None and recorder.collective_depth == 0:
                    elapsed = time.perf_counter() - started
                    key = recorder.phase or "outside_model"
                    values = recorder.current["explicit_collective_wall_s"]
                    values[key] = values.get(key, 0.) + elapsed
                    values[name + "/" + key] = values.get(name + "/" + key, 0.) + elapsed
        return invoke

    DataLoader.__iter__ = iterator
    torch.Tensor.cuda = cuda
    torch.Tensor.backward = backward
    DistributedDataParallel.forward = forward
    torch.cuda.synchronize = synchronize
    trainer.build_optimizer = optimizer
    trainer.optimizer_learning_rates = rates
    trainer.save_checkpoint = checkpoint
    for name in ("all_reduce", "all_gather", "all_gather_object", "broadcast", "broadcast_object_list", "barrier"):
        setattr(dist, name, collective(name, getattr(dist, name)))
    try:
        trainer.main()
    finally:
        recorder.finish()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
