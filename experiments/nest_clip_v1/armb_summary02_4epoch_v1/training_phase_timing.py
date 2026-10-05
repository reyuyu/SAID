"""Read-only timing around the native trainer; no scheduler/resource-gate changes."""

import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from train import train_nested_semantic_mask as trainer
from .phase_train_v2 import PhaseRecorder, TimedIterator


def install(local, rank):
    recorder = PhaseRecorder(local, rank)
    original_iter = DataLoader.__iter__
    original_cuda = torch.Tensor.cuda
    original_backward = torch.Tensor.backward
    original_forward = DistributedDataParallel.forward
    original_optimizer = trainer.build_optimizer
    original_checkpoint = trainer.save_checkpoint
    original_rates = trainer.optimizer_learning_rates
    original_sync = torch.cuda.synchronize

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
        return instance

    def checkpoint(*args, **kwargs):
        if recorder.current and "optimizer" in recorder.current["events"]:
            recorder.finish()
        return original_checkpoint(*args, **kwargs)

    def rates(module, completed, horizon):
        started = time.perf_counter()
        if recorder.current:
            recorder.current["step"] = completed + 1
        result = original_rates(module, completed, horizon)
        if recorder.current:
            recorder.current["control_gate_wall_s"] = time.perf_counter() - started
        return result

    def synchronize(*args, **kwargs):
        started = time.perf_counter()
        result = original_sync(*args, **kwargs)
        if recorder.current:
            key = "cuda_synchronize/" + (recorder.phase or "outside_model")
            values = recorder.current["explicit_collective_wall_s"]
            values[key] = values.get(key, 0.) + time.perf_counter() - started
        return result

    def collective(name, function):
        def invoke(*args, **kwargs):
            started = time.perf_counter()
            recorder.collective_depth += 1
            try:
                return function(*args, **kwargs)
            finally:
                recorder.collective_depth -= 1
                if recorder.current and recorder.collective_depth == 0:
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
    trainer.build_optimizer = optimizer
    trainer.save_checkpoint = checkpoint
    trainer.optimizer_learning_rates = rates
    torch.cuda.synchronize = synchronize
    for name in ("all_reduce", "all_gather", "all_gather_object", "broadcast", "broadcast_object_list", "barrier"):
        setattr(dist, name, collective(name, getattr(dist, name)))
    return recorder
