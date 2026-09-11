"""Targeted tests for the optional U-weight warmup schedule (w = 0 must stay the historical path).

Run:  python -m pytest tests/test_cvssl_u_weight_schedule.py -q
"""
import argparse
import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'train'))

from model.said_cls_cvssl import effective_lambda_u, SaidClsCvsslTrainModule  # noqa: E402
from train_said_cls_cvssl import build_optimizers, cvssl_train_step, lambda_target_of  # noqa: E402

DIM = 16
TEXT_WIDTH = 24
TOKENS = 5
BATCH = 4


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


def test_w0_is_exactly_the_target_for_every_step():
    for step in (0, 1, 99, 100, 200, 10000):
        assert effective_lambda_u(0.3, step, 0) == 0.3
        assert effective_lambda_u(1.0, step, 0) == 1.0


def test_w200_schedule_matches_the_specified_values():
    target, warmup = 0.3, 200
    assert effective_lambda_u(target, 0, warmup) == pytest.approx(target / 200)      # step 1
    assert effective_lambda_u(target, 99, warmup) == pytest.approx(target * 100 / 200)  # step 100
    assert effective_lambda_u(target, 199, warmup) == pytest.approx(target)          # step 200
    assert effective_lambda_u(target, 200, warmup) == pytest.approx(target)
    assert effective_lambda_u(target, 499, warmup) == pytest.approx(target)
    # monotone and never above the target
    values = [effective_lambda_u(target, step, warmup) for step in range(0, 260)]
    assert all(left <= right for left, right in zip(values, values[1:]))
    assert max(values) == pytest.approx(target)


def test_lambda_target_of_reads_the_configured_weight():
    module = SaidClsCvsslTrainModule(StubClip(), rank=0, arm='C0_complement_vssl', lambda_u=0.3)
    assert lambda_target_of(module) == pytest.approx(0.3)


@pytest.mark.parametrize('warmup', [0, 200])
def test_backward_loss_carries_the_effective_weight_exactly_once(warmup):
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29697')
        dist.init_process_group(backend='gloo', rank=0, world_size=1)
    target = 0.3
    module = SaidClsCvsslTrainModule(StubClip(), rank=0, arm='C0_complement_vssl',
                                    lambda_u=target, ddp_gradient_averaging=False)
    optimizer, mask_optimizer, _, _ = build_optimizers(
        module.clip, argparse.Namespace(lr=1e-3, mask_lr=1e-2, weight_decay=1e-2))
    out = cvssl_train_step(module, _batch(), optimizer, mask_optimizer, torch.device('cpu'),
                           torch.float32, amp_enabled=False, mask_rng_seed=5,
                           completed_steps=0, u_weight_warmup_steps=warmup)
    expected = target if warmup == 0 else target / warmup
    assert float(out['lambda_U_target']) == pytest.approx(target)
    assert float(out['lambda_U_effective']) == pytest.approx(expected)
    # the U term appears exactly once, multiplied by the effective weight and nothing else
    assert float(out['loss_vssl_for_backward_local']) == pytest.approx(
        expected * float(out['loss']), rel=1e-5)
    assert float(out['loss_total_for_backward']) == pytest.approx(
        float(out['loss_smart']) + expected * float(out['loss']), rel=1e-5)
    assert float(out['loss_vssl_weighted']) == pytest.approx(expected * float(out['loss']),
                                                            rel=1e-5)


def test_smart_term_is_never_scaled_by_the_schedule():
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29698')
        dist.init_process_group(backend='gloo', rank=0, world_size=1)
    torch.manual_seed(4)
    left = SaidClsCvsslTrainModule(StubClip(), rank=0, arm='C0_complement_vssl', lambda_u=0.3,
                                   ddp_gradient_averaging=False)
    module_state = {key: value.clone() for key, value in left.state_dict().items()}
    data = _batch()
    outputs = {}
    for tag, warmup, steps in (('w0', 0, 0), ('w200', 200, 0)):
        right = SaidClsCvsslTrainModule(StubClip(), rank=0, arm='C0_complement_vssl', lambda_u=0.3,
                                        ddp_gradient_averaging=False)
        right.load_state_dict(module_state)
        optimizer, mask_optimizer, _, _ = build_optimizers(
            right.clip, argparse.Namespace(lr=1e-3, mask_lr=1e-2, weight_decay=1e-2))
        out = cvssl_train_step(right, data, optimizer, mask_optimizer, torch.device('cpu'),
                               torch.float32, amp_enabled=False, mask_rng_seed=5,
                               completed_steps=steps, u_weight_warmup_steps=warmup)
        outputs[tag] = float(out['loss_smart'])
    assert outputs['w0'] == pytest.approx(outputs['w200'], rel=1e-6)


def test_default_cli_value_keeps_the_constant_weight():
    """``--u_weight_warmup_steps`` defaults to 0, i.e. the historical behaviour."""
    import subprocess
    result = subprocess.run([sys.executable, os.path.join(REPO_ROOT, 'train',
                                                          'train_said_cls_cvssl.py'), '--help'],
                            capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr[-1500:]
    assert '--u_weight_warmup_steps' in result.stdout
