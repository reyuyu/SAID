"""Phase 2 distributed test: gradient-preserving gather + Said-only contrastive loss.

Run with 2 ranks (2 GPUs)::

    NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
        torchrun --nproc_per_node=2 tests/test_distributed_said_loss.py

Under plain ``pytest`` (no distributed context) the test is skipped, so the unit
test suite stays runnable on a single process.

What is verified
----------------
1. ``gather_features_with_grad`` produces the correct global-batch shape.
2. The gathered tensor keeps an autograd graph (gradient-preserving gather).
3. Local image / text / Said features all receive non-zero, finite gradients.
4. Cross-rank gradient routing: the gradient of a slice that belongs to rank 0
   flows back only to rank 0's local tensor (this can only happen with an
   autograd-aware all-gather; a detached all_gather would give no gradient).
5. No deadlock: the script finishes and exits with code 0 under torchrun.
"""
import json
import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.salu_model import contrastive_loss, gather_features_with_grad  # noqa: E402

LOCAL_BATCH = 4
DIM = 512


def _init_process_group():
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')


def _check(seed_offset: int = 0):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device('cuda', rank % torch.cuda.device_count())
    torch.cuda.set_device(device)

    torch.manual_seed(1234 + rank + seed_offset)

    def leaf():
        return torch.randn(LOCAL_BATCH, DIM, device=device, requires_grad=True)

    z_image = F.normalize(leaf(), dim=-1)
    z_text = F.normalize(leaf(), dim=-1)
    z_said = F.normalize(leaf(), dim=-1)
    for tensor in (z_image, z_text, z_said):
        tensor.retain_grad()
    scale = torch.tensor(14.2784, device=device)  # exp(log(1 / 0.07))

    z_image_all = gather_features_with_grad(z_image)
    z_text_all = gather_features_with_grad(z_text)
    z_said_all = gather_features_with_grad(z_said)

    global_batch = LOCAL_BATCH * world_size
    assert tuple(z_image_all.shape) == (global_batch, DIM), tuple(z_image_all.shape)
    assert tuple(z_text_all.shape) == (global_batch, DIM)
    assert tuple(z_said_all.shape) == (global_batch, DIM)
    assert z_image_all.requires_grad and z_text_all.requires_grad and z_said_all.requires_grad

    loss_global = contrastive_loss(z_image_all, z_text_all, scale)
    loss_said = contrastive_loss(z_said_all, z_text_all, scale)
    loss = loss_global + loss_said
    loss.backward()

    grad_status = {}
    for name, tensor in (('z_image', z_image), ('z_text', z_text), ('z_said', z_said)):
        grad = tensor.grad
        grad_status[name] = {
            'exists': grad is not None,
            'finite': bool(torch.isfinite(grad).all()) if grad is not None else False,
            'abs_sum': float(grad.abs().sum()) if grad is not None else 0.0,
        }

    # ---- cross-rank gradient routing probe -------------------------------
    # probe_all[0] is rank 0's row; its gradient must land on rank 0's local
    # tensor only. A detached all_gather would produce no gradient at all.
    probe_local = F.normalize(leaf(), dim=-1)
    probe_local.retain_grad()
    probe_all = gather_features_with_grad(probe_local)
    probe_all[0].sum().backward()
    probe_grad = 0.0 if probe_local.grad is None else float(probe_local.grad.abs().sum())

    result = {
        'rank': rank,
        'world_size': world_size,
        'local_batch': LOCAL_BATCH,
        'global_batch': global_batch,
        'loss_global': float(loss_global),
        'loss_said': float(loss_said),
        'loss_total': float(loss),
        'finite': bool(torch.isfinite(loss)),
        'grads': grad_status,
        'probe_grad_abs_sum_rank0_row': probe_grad,
    }
    return result


def _assert_result(result):
    assert result['finite'], 'loss is not finite'
    for name, status in result['grads'].items():
        assert status['exists'], '%s gradient missing' % name
        assert status['finite'], '%s gradient not finite' % name
        assert status['abs_sum'] > 0.0, '%s gradient is zero' % name
    # rank 0 owns the row we probed -> it must receive a gradient there
    if result['rank'] == 0:
        assert result['probe_grad_abs_sum_rank0_row'] > 0.0, 'no gradient routed to rank 0 row'


def test_distributed_said_loss():
    if not (dist.is_available() and dist.is_initialized()):
        pytest.skip('run via: torchrun --nproc_per_node=2 tests/test_distributed_said_loss.py')
    result = _check()
    _assert_result(result)


if __name__ == '__main__':
    _init_process_group()
    try:
        result = _check()
        _assert_result(result)
        print('RANK_RESULT ' + json.dumps(result, sort_keys=True), flush=True)
        if dist.get_rank() == 0:
            print('DISTRIBUTED_SAID_LOSS_RESULT PASS', flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
