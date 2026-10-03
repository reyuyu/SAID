"""Audit policy and observational profiler instrumentation; no objective changes."""
from contextlib import contextmanager, nullcontext
from collections import Counter
import functools
import hashlib
import json
import time

import torch
import torch.distributed as dist


AUDIT_LEVELS = ('full', 'sparse', 'benchmark')


def audit_step(step, level, *, epoch_boundary=False, checkpoint=False, final=False):
    if level not in AUDIT_LEVELS:
        raise ValueError(level)
    return level == 'full' or (level == 'sparse' and (
        step in (1, 5, 100, 200, 500) or step % 500 == 0 or
        epoch_boundary or checkpoint or final))


def finite_gradients(module):
    gradients = [p.grad for p in module.parameters() if p.grad is not None]
    if not gradients:
        return torch.zeros((), device=next(module.parameters()).device, dtype=torch.int32)
    norms = torch._foreach_norm(gradients)
    return torch.isfinite(torch.stack(norms)).all().int()


def update_rolling_hash(digest, batch):
    # The caller retains these CPU tensors before H2D; no token GPU->CPU copies.
    for key in ('sample_id', 'n', 'K', 'valid'):
        digest.update(key.encode())
        digest.update(batch[key].numpy().tobytes())
    return digest.hexdigest()


class ProfileRanges:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.view = 'F'
        self.calls = 0
        self.originals = []

    def range(self, name):
        return torch.profiler.record_function(name) if self.enabled else nullcontext()

    def wrap(self, owner, name, label):
        original = getattr(owner, name)
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            with self.range(label() if callable(label) else label):
                return original(*args, **kwargs)
        setattr(owner, name, wrapped)
        self.originals.append((owner, name, original))

    def install(self, module, fusion_module, semantic_module, balanced_module):
        if not self.enabled:
            return
        self.wrap(module.clip, 'encode_image', 'image_encoder_forward')
        self.wrap(module.clip, 'encode_text', lambda: 'text_encoder_'+self.view)
        original = module.encode_view
        def encode_view(tokens):
            self.view = ('F', 'P', 'R')[self.calls % 3]
            self.calls += 1
            return original(tokens)
        module.encode_view = encode_view
        self.originals.append((module, 'encode_view', original))
        self.wrap(module.fusion_branch, 'encode_visual', 'visual_mask_branch')
        self.wrap(module.clip.mask_net.resblocks, 'forward', lambda: 'text_mask_branch_'+self.view)
        self.wrap(fusion_module, 'pool_summary', 'shared_attention_pool')
        self.wrap(fusion_module, 'fusion_scores', lambda: 'pair_score_'+self.view)
        original_terms = balanced_module.fusion_view_terms
        self.wrap(balanced_module, 'fusion_view_terms', lambda: 'loss_'+self.view)
        for owner in (fusion_module, semantic_module, balanced_module):
            for name in ('gather', 'global_sum'):
                if hasattr(owner, name):
                    self.wrap(owner, name, 'collectives_gather')

    def close(self):
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)


def profiler_summary(profiler):
    averages = []
    for event in profiler.key_averages():
        averages.append(dict(name=event.key, count=event.count,
            cpu_total_us=event.cpu_time_total, cpu_self_us=event.self_cpu_time_total,
            cuda_total_us=event.device_time_total, cuda_self_us=event.self_device_time_total))
    kernels = Counter()
    for event in profiler.events():
        if str(event.device_type).endswith('CUDA'):
            kernels[event.name] += event.time_range.elapsed_us()
    return dict(operators=sorted(averages, key=lambda x:x['cuda_self_us'], reverse=True),
                cuda_kernels_total_us=sum(kernels.values()),
                cuda_kernels=[dict(name=k, cumulative_us=v) for k,v in kernels.most_common(100)],
                note='Range CUDA totals are inclusive; overlapping ranges must not be summed as exclusive wall time.')
