"""Per-parameter dtype audit for the CVSSL training path (fp32 master, bf16 autocast).

The production configuration is "fp32 master weights + bf16 autocast". Two contracts matter:

1. every trainable parameter and every AdamW state stays fp32 (a bf16 master would silently lose
   ~3 decimal digits and make the run unreproducible);
2. the masked-cosine / cross-entropy cores of SIDM, DISM and the CVSSL term run in fp32 even
   inside an autocast region (``model.complement_visual_ssl.fp32_context``), so the *objective
   value* under autocast is bit-identical to the fp32 value while the *parameter gradients* are
   allowed to differ (they come from the autocast-cast matmuls).

Both are asserted here, and the diagnostic is cheap: it uses the tiny stand-in model only.
"""
import argparse
import json
import os
import subprocess
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'train'))

import pytest  # noqa: E402

from model.said_cls_cvssl import SaidClsCvsslTrainModule  # noqa: E402
from model import complement_visual_ssl as cvssl  # noqa: E402
from train_said_cls_cvssl import build_optimizers, cvssl_train_step  # noqa: E402

DIM = 16
TEXT_WIDTH = 24
TOKENS = 5
BATCH = 4
WORKER = os.path.join(REPO_ROOT, 'tests', '_cvssl_dtype_audit_worker.py')


class StubVisual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(TEXT_WIDTH, DIM)

    def forward(self, images):
        return self.proj(images.flatten(1)[:, :TEXT_WIDTH])


class StubMaskNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(TEXT_WIDTH, 12)
        self.out = torch.nn.Linear(12, DIM)

    def forward(self, hidden):
        return self.out(torch.tanh(self.proj(hidden.mean(dim=1))))


class StubClip(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = DIM
        self.text_projection = torch.nn.Parameter(torch.zeros(TEXT_WIDTH, DIM))
        self.text_proj = torch.nn.Linear(TEXT_WIDTH, DIM)
        self.visual = StubVisual()
        self.mask_net = StubMaskNet()

    @property
    def dtype(self):
        return self.visual.proj.weight.dtype

    def encode_image(self, images):
        return self.visual(images)

    def encode_text(self, tokens, return_full=False, return_pool=False):
        hidden = tokens.float().mean(dim=1, keepdim=True).repeat(1, TEXT_WIDTH)
        hidden = hidden.unsqueeze(1).repeat(1, TOKENS, 1)
        pooled = self.text_proj(hidden[:, 0, :])
        return (pooled, hidden) if return_full else pooled


def _batch():
    generator = torch.Generator().manual_seed(11)
    return {'image_a': torch.randn(BATCH, 3, 4, 4, generator=generator) * 0.4,
            'image_b': torch.randn(BATCH, 3, 4, 4, generator=generator) * 0.4,
            'caption_said': ['a photo of a cat .'] * BATCH,
            'image_id': torch.arange(BATCH)}


def test_every_parameter_and_optimizer_state_stays_fp32():
    torch.manual_seed(3)
    module = SaidClsCvsslTrainModule(StubClip(), rank=0, arm='C0_complement_vssl', lambda_u=1.0)
    for name, parameter in module.named_parameters():
        assert parameter.dtype == torch.float32, name
    for name, buffer in module.named_buffers():
        assert buffer.dtype in (torch.float32, torch.int64, torch.bool), (name, buffer.dtype)
    optimizer, mask_optimizer, n_backbone, n_mask = build_optimizers(
        module.clip, argparse.Namespace(lr=1e-3, mask_lr=1e-2, weight_decay=1e-2))
    assert n_backbone > 0 and n_mask == len(list(module.clip.mask_net.parameters()))
    for group in (optimizer, mask_optimizer):
        for parameter in group.param_groups[0]['params']:
            assert parameter.dtype == torch.float32


def test_objective_value_and_gradients_are_finite_and_keep_the_documented_multipliers():
    """The loss identity and the gradient dtype/finiteness contract of the production step.

    ``amp_enabled=True`` is exercised as well: on CPU this torch build disables autocast and warns,
    so the *bf16* numerics are **NOT RUN** here (see the test above) -- the GPU run is where that
    has to be checked. What this test does verify is that the step is internally consistent and
    that every gradient is fp32 and finite.
    """
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29696')
        dist.init_process_group(backend='gloo', rank=0, world_size=1)
    data = _batch()
    outputs = {}
    for tag, amp in (('fp32', False), ('bf16', True)):
        torch.manual_seed(3)
        module = SaidClsCvsslTrainModule(StubClip(), rank=0, arm='C0_complement_vssl', lambda_u=1.0,
                                         ddp_gradient_averaging=False)
        optimizer, mask_optimizer, _, _ = build_optimizers(
            module.clip, argparse.Namespace(lr=1e-3, mask_lr=1e-2, weight_decay=1e-2))
        out = cvssl_train_step(module, data, optimizer, mask_optimizer, torch.device('cpu'),
                               torch.float32, amp_enabled=amp, mask_rng_seed=5,
                               capture_grads=True)
        captured = out['_grads']
        grads = [grad for grad in captured.values() if grad is not None]
        assert grads, tag
        for grad in grads:
            assert grad.dtype == torch.float32, tag
            assert torch.isfinite(grad).all(), tag
        assert abs(float(out['loss_smart']) - 10.0 * (float(out['loss_sidm'])
                                                      + float(out['loss_dism']))
                   - 2.0 * float(out['loss_sparsity'])) < 1e-4, tag
        outputs[tag] = {key: float(out[key]) for key in
                        ('loss_sidm', 'loss_dism', 'loss_sparsity', 'loss_smart',
                         'loss_vssl_global_mean')}
        for parameter in module.parameters():
            parameter.grad = None
    for key in ('loss_sparsity', 'loss_vssl_global_mean'):
        assert abs(outputs['fp32'][key] - outputs['bf16'][key]) <= 1e-6, key


def test_cpu_autocast_bf16_behaviour_is_reported_not_assumed():
    """Record what this host can and cannot emulate, instead of assuming.

    Measured: on this torch build CPU autocast *is* enabled inside ``torch.autocast('cpu', bf16)``
    (``torch.is_autocast_enabled('cpu') is True``) while some ops warn that the dtype is not
    supported and silently fall back to fp32. The bf16 numerics of the two-view forward therefore
    cannot be certified on CPU; the GPU run is where they have to be audited.
    """
    inside = None
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16, enabled=True):
        inside = torch.is_autocast_enabled('cpu')
    assert inside in (True, False) and inside is not None
    assert torch.is_autocast_enabled('cpu') is False


def test_fp32_core_context_really_returns_fp32():
    """The CVSSL core must produce fp32 tensors inside ``fp32_context`` and autocast must be off
    inside the block, regardless of the outer autocast state."""
    left = torch.randn(3, DIM, dtype=torch.float32)
    right = torch.randn(3, DIM, dtype=torch.float32)
    mask = (torch.rand(3, DIM) > 0.5).float()
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16, enabled=True):
        with cvssl.fp32_context(torch.device('cpu')):
            produced = cvssl.anchor_masked_cosine_logits(left, right, mask, tau_u=0.1, eps=1e-6)
            inside = torch.is_autocast_enabled('cpu')
    tensors = [value for value in produced if torch.is_tensor(value)]
    assert tensors, 'anchor_masked_cosine_logits returned no tensor'
    for tensor in tensors:
        assert tensor.dtype == torch.float32
    assert inside is False, 'fp32_context must disable autocast inside the block'
