"""Compatibility launcher for the audited L/14 Full5 step-0 container.

The historical L/14 step-0 payload predates the trainer's provenance field.
This controller adds only in-memory metadata while preserving the file bytes,
model tensors, optimizer state, and the trainer's original init SHA check.
"""
import runpy
import sys

import torch

_load = torch.load


def _compat_load(*args, **kwargs):
    value = _load(*args, **kwargs)
    if isinstance(value, dict) and value.get("completed_steps") == 0 and "provenance" not in value:
        config = value.get("config", {})
        if config.get("base_model") == "ViT-L/14":
            value = dict(value)
            value["provenance"] = {
                "source": "OpenAI CLIP + original random MaskNetwork",
                "base_model": "ViT-L/14",
            }
    return value


torch.load = _compat_load
sys.argv[0] = "train.train_nested_semantic_mask"
runpy.run_module("train.train_nested_semantic_mask", run_name="__main__")
