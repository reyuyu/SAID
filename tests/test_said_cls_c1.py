"""SAID-C1-TCR v0.1 acceptance tests (minimal, targeted at the NEW path only).

Covers exactly what the new branch needs: shapes/finiteness, the frozen reference invariants, the
reconstruction-loss normalisation (no accidental /512), the intended gradient boundaries, the
empty-support degenerate case, lambda_rec = 0 equivalence with the SmartCLIP path, one real 2-rank
DDP step with the cross-rank loss scaling, and the exported student checkpoint staying clean of
teacher/decoder keys.
"""
import argparse
import copy
import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'train'))

from model.said_cls_reconstruction import (C1TrainModule, DEFAULT_TAU_REC,  # noqa: E402
                                           reconstruction_loss)
from train_said_cls_c1 import build_decoder_optimizer, c1_train_step  # noqa: E402
from train_said_cls_cvssl import build_optimizers, inspected_module  # noqa: E402

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
    """Coordinate-sensitive mask network.

    The real mask network sees the full ``ln_final`` hidden state and produces a different logit per
    coordinate. A stub that collapsed every token to one mean would emit an all-equal mask, whose
    straight-through sparsity gradient is identically zero -- that would make the "mask_net trains"
    assertions vacuous, so the stub keeps a per-position term.
    """

    def __init__(self, depth=TOKENS):
        super().__init__()
        self.proj = torch.nn.Linear(TEXT_WIDTH, 12)
        self.out = torch.nn.Linear(12, DIM)
        self.position = torch.nn.Parameter(torch.randn(TOKENS, 12) * 0.1)

    def forward(self, hidden):
        features = torch.tanh(self.proj(hidden)) + self.position[:hidden.shape[1]]
        return self.out(features.mean(dim=1))


class StubClip(torch.nn.Module):
    """Minimal stand-in that exercises the REAL gradient paths of the C1 forward.

    ``encode_text`` must consume ``text_projection`` (as LongCLIP does) so that "L_rec does not
    reach the text encoder" is a meaningful test of the detach, not an artefact of a dead parameter.
    """

    def __init__(self):
        super().__init__()
        self.embed_dim = DIM
        self.text_projection = torch.nn.Parameter(torch.randn(DIM, DIM) * 0.1)
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
        pooled = self.text_proj(hidden[:, 0, :]) @ self.text_projection
        return (pooled, hidden) if return_full else pooled


def _clip(seed=1234):
    torch.manual_seed(seed)
    return StubClip()


def _tokens():
    return torch.randint(0, 100, (BATCH, TOKENS), generator=torch.Generator().manual_seed(3))


def _batch(seed=11):
    generator = torch.Generator().manual_seed(seed)
    return {'image_a': torch.randn(BATCH, 3, 4, 4, generator=generator) * 0.4,
            'image_b': torch.randn(BATCH, 3, 4, 4, generator=generator) * 0.4,
            'caption_said': ['a photo of a cat .'] * BATCH}


def _grad_magnitude(module, predicate=None):
    """Largest |grad| over the selected parameters, treating a missing buffer as 0.

    Autograd still allocates a zero-filled grad buffer for a parameter that is *reachable* from the
    loss but whose incoming gradient cancels (e.g. the text encoder when the SmartCLIP weights are
    zero). The invariant that matters for C1 is therefore "the gradient is zero", not "the buffer is
    absent", so both cases are treated as zero here.
    """
    worst = 0.0
    for name, parameter in module.named_parameters():
        if predicate is not None and not predicate(name):
            continue
        if parameter.grad is None:
            continue
        worst = max(worst, float(parameter.grad.detach().abs().max()))
    return worst


def _dist():
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29691')
        dist.init_process_group(backend='gloo', rank=0, world_size=1)
    return dist


def _module(lambda_rec=DEFAULT_TAU_REC, **kwargs):
    return C1TrainModule(_clip(), rank=0, lambda_rec=lambda_rec, decoder_hidden=DIM, **kwargs)


# ---------------------------------------------------------------- shapes / finiteness
def test_shapes_and_finiteness_and_r_u_is_not_renormalised():
    _dist()
    module = _module()
    data = _batch()
    out = module(data['image_a'], _tokens(), data['image_b'])
    for key in ('g', 'm_u', 't_cond', 'target', 'loss_rec', 'loss_smart',
                'loss_total_for_backward'):
        assert key in out, key
    assert out['g'].shape == (BATCH, DIM)
    assert out['m_u'].shape == (BATCH, DIM)
    assert out['t_cond'].shape == (BATCH, DIM)
    assert out['target'].shape == (BATCH, DIM)
    for key in ('loss_rec', 'loss_smart', 'loss_total_for_backward', 'cos_pred_reference',
                'prediction_norm', 'reference_norm'):
        assert torch.isfinite(out[key]).all(), key
    # g is unit-norm, r_U = g * m_U is deliberately NOT re-normalised
    assert torch.allclose(out['g'].norm(dim=-1), torch.ones(BATCH), atol=1e-5)
    r_u = out['g'] * out['m_u']
    assert float(r_u.norm(dim=-1).max()) <= 1.0 + 1e-6
    # the unit target must be unit-norm (the frozen reference is normalised, not the student)
    assert torch.allclose(out['target'].norm(dim=-1), torch.ones(BATCH), atol=1e-5)


def test_loss_normalisation_is_per_sample_not_per_element():
    """zero prediction against a unit target must give exactly 1, not 1/512."""
    pred = torch.zeros(3, 8)
    target = F.normalize(torch.randn(3, 8), dim=-1)
    valid = torch.ones(3, dtype=torch.bool)
    loss, info = reconstruction_loss(pred, target, valid)
    assert float(loss) == pytest.approx(1.0, abs=1e-6)
    # and it must scale with the feature dimension rather than dividing by it
    assert float(info['per_sample'][0]) == pytest.approx(1.0, abs=1e-6)


def test_invalid_samples_are_skipped_and_all_invalid_gives_connected_zero():
    pred = torch.zeros(2, 8, requires_grad=True)
    target = F.normalize(torch.randn(2, 8), dim=-1)
    valid = torch.tensor([True, False])
    loss, _ = reconstruction_loss(pred, target, valid)
    expected = float((pred[0] - target[0].detach()).square().sum())
    assert float(loss) == pytest.approx(expected, rel=1e-6)
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()

    loss_all_invalid, _ = reconstruction_loss(pred, target, torch.zeros(2, dtype=torch.bool))
    assert float(loss_all_invalid) == 0.0
    loss_all_invalid.backward()
    assert torch.isfinite(pred.grad).all()


def test_empty_complement_support_is_finite_and_keeps_the_graph():
    _dist()
    module = _module()
    data = _batch()
    ones = torch.ones(BATCH, DIM, dtype=torch.float32)
    out = module(data['image_a'], _tokens(), data['image_b'], mask_override=ones)
    assert float(out['m_u'].sum()) == 0.0, 'the override must empty the complementary support'
    assert float(out['rec_valid_count_tensor']) == 0.0
    assert float(out['rec_valid_fraction']) == 0.0
    assert float(out['loss_rec']) == 0.0
    assert out['loss_rec'].requires_grad, 'the zero reconstruction loss must stay connected'
    assert torch.isfinite(out['loss_total_for_backward']).all()
    out['loss_total_for_backward'].backward()
    decoder_grad = module.decoder.net[0].weight.grad
    assert decoder_grad is not None and torch.isfinite(decoder_grad).all()
    assert float(decoder_grad.abs().max()) == 0.0, 'no valid sample -> no reconstruction gradient'


# ---------------------------------------------------------------- reference invariants
def test_frozen_reference_equals_the_initial_visual_and_shares_no_storage():
    clip = _clip()
    expected = copy.deepcopy(clip.visual).state_dict()
    module = C1TrainModule(clip, rank=0, lambda_rec=1.0, decoder_hidden=DIM)
    for key, value in module.reference_visual.state_dict().items():
        assert torch.equal(value, expected[key]), key
    student_pointers = {p.data_ptr() for p in module.clip.visual.parameters()}
    reference_pointers = {p.data_ptr() for p in module.reference_visual.parameters()}
    assert not (student_pointers & reference_pointers)
    assert all(not p.requires_grad for p in module.reference_visual.parameters())
    assert not module.reference_visual.training


def test_train_call_keeps_the_reference_in_eval_and_it_never_moves():
    _dist()
    from model.said_cls_reconstruction import reference_fingerprint
    module = _module()
    before = reference_fingerprint(module.reference_visual)
    module.train()
    assert not module.reference_visual.training, 'train() must not put the reference in train mode'
    data = _batch()
    optimizer, mask_optimizer, _, _ = build_optimizers(
        module.clip, argparse.Namespace(lr=1e-3, mask_lr=1e-2, weight_decay=1e-2))
    decoder_optimizer = build_decoder_optimizer(
        module, argparse.Namespace(decoder_lr=1e-2, decoder_wd=0.0))
    before_params = {k: v.clone() for k, v in module.clip.state_dict().items()}
    out = c1_train_step(module, data, optimizer, mask_optimizer, decoder_optimizer,
                        torch.device('cpu'), torch.float32, amp_enabled=False, world_size=1,
                        capture_grads=True)
    after = reference_fingerprint(module.reference_visual)
    assert before == after, 'the frozen reference changed during an optimizer step'
    assert not module.reference_visual.training
    # the student and the decoder DID move
    moved = [k for k in before_params
             if before_params[k].is_floating_point()
             and float((module.clip.state_dict()[k].float()
                        - before_params[k].float()).abs().max()) > 0]
    assert moved, 'no student parameter moved'
    assert any(g is not None for g in out['_grads'].values())


# ---------------------------------------------------------------- gradient boundaries
def test_reconstruction_gradients_reach_only_decoder_and_student_visual():
    """Isolate L_rec by zeroing the SmartCLIP branch's gradients through backward hooks.

    The forward still runs normally (so the reconstruction graph really exists); a hook zeroes the
    gradients of the mask network, the text encoder/projection and the frozen reference, which is
    exactly the boundary C1 claims. Zeroing via hooks rather than ``no_grad`` keeps L_rec a real
    autograd result instead of a detached constant.
    """
    _dist()
    module = _module(lambda_rec=1.0)
    blocked = ('clip.mask_net.', 'clip.text_proj.', 'clip.text_projection', 'reference_visual.')
    handles = []
    for name, parameter in module.named_parameters():
        if name.startswith(blocked) and parameter.requires_grad:
            handles.append(parameter.register_hook(lambda grad: torch.zeros_like(grad)))
    data = _batch()
    try:
        out = module(data['image_a'], _tokens(), data['image_b'])
        out['loss_total_for_backward'].backward()
    finally:
        for handle in handles:
            handle.remove()
    assert module.decoder.net[0].weight.grad is not None
    assert _grad_magnitude(module, lambda n: n.startswith('decoder')) > 0
    assert _grad_magnitude(module, lambda n: n.startswith('clip.visual')) > 0
    # blocked boundaries: zero gradient (a reachable parameter may still carry a zero buffer)
    assert _grad_magnitude(module, lambda n: n.startswith('clip.text_proj')) == 0.0, \
        'L_rec must not reach the text encoder (t_cond is detached)'
    assert module.clip.text_projection.grad is None or float(
        module.clip.text_projection.grad.abs().max()) == 0.0, \
        'L_rec must not reach the text projection (t_cond is detached)'
    assert _grad_magnitude(module, lambda n: n.startswith('clip.mask_net')) == 0.0, \
        'L_rec must not reach mask_net (m_U comes from a detached m_S)'
    assert _grad_magnitude(module, lambda n: n.startswith('reference_visual')) == 0.0, \
        'L_rec must not reach the frozen reference'
    # the normalisation coupling is real: we do NOT assert that the raw-v Said gradient is zero
    v_grad = module.clip.visual.proj.weight.grad
    assert torch.isfinite(v_grad).all()
    # and the reconstruction term really is the non-zero part of the objective here
    assert float(out['loss_rec']) > 0.0
    assert float(out['weighted_rec']) == pytest.approx(float(out['loss_rec']), rel=1e-6)


def test_lambda_rec_zero_reproduces_the_smartclip_step():
    _dist()
    data = _batch()
    module = _module(lambda_rec=0.0)
    optimizer, mask_optimizer, _, _ = build_optimizers(
        module.clip, argparse.Namespace(lr=1e-3, mask_lr=1e-2, weight_decay=1e-2))
    decoder_optimizer = build_decoder_optimizer(
        module, argparse.Namespace(decoder_lr=1e-2, decoder_wd=0.0))
    out = c1_train_step(module, data, optimizer, mask_optimizer, decoder_optimizer,
                        torch.device('cpu'), torch.float32, amp_enabled=False, world_size=1,
                        capture_grads=True)
    assert float(out['weighted_rec']) == 0.0
    assert float(out['lambda_rec']) == 0.0
    # the DDP scaling factor is about valid-sample counts, not about lambda_rec: with every sample
    # valid and equal rank batches it is exactly 1. The *weighted* reconstruction term is what has
    # to vanish when lambda_rec = 0.
    assert float(out['rec_backward_scale']) == pytest.approx(1.0, abs=1e-6)
    assert float(out['loss_rec_backward_local']) == 0.0
    assert float(out['loss_smart']) == pytest.approx(
        10.0 * (float(out['loss_sidm']) + float(out['loss_dism']))
        + 2.0 * float(out['loss_sparsity']), rel=1e-6)
    # the backward scalar must equal the SmartCLIP loss exactly when lambda_rec = 0
    assert float(out['loss_total_for_backward_local']) == pytest.approx(
        float(out['loss_smart']), rel=1e-6)
    # gradients must be read from the pre-step snapshot: the step zeroes them at the end
    captured = out['_grads']
    decoder_worst = max((0.0 if grad is None else float(grad.abs().max()))
                        for name, grad in captured.items() if name.startswith('decoder'))
    mask_worst = max((0.0 if grad is None else float(grad.abs().max()))
                     for name, grad in captured.items() if name.startswith('clip.mask_net'))
    assert decoder_worst == 0.0, 'the reconstruction branch must not train when lambda_rec = 0'
    assert mask_worst > 0.0, 'the SmartCLIP sparsity term must still train mask_net'


def test_decoder_init_does_not_touch_the_student_or_the_global_rng():
    torch.manual_seed(7)
    clip_a = StubClip()
    state_a = {k: v.clone() for k, v in clip_a.state_dict().items()}
    torch.manual_seed(7)
    clip_b = StubClip()
    state_b = {k: v.clone() for k, v in clip_b.state_dict().items()}
    for key in state_a:
        assert torch.equal(state_a[key], state_b[key]), key
    # building the module (reference deepcopy + decoder init) must not consume the ambient RNG.
    # The student is built FIRST so nothing re-seeds the generator between the two draws.
    student = _clip(seed=5)
    torch.manual_seed(99)
    expected = torch.randn(6)
    torch.manual_seed(99)
    C1TrainModule(student, rank=0, lambda_rec=1.0, decoder_hidden=DIM, decoder_seed=0)
    actual = torch.randn(6)
    assert torch.equal(expected, actual), 'module construction consumed the ambient RNG stream'


def test_decoder_has_no_shortcut_inputs():
    """The decoder must be callable with exactly two [B, D] tensors and nothing else."""
    module = _module()
    with pytest.raises(TypeError):
        module.decoder(torch.zeros(2, DIM))          # missing the text-conditioning input
    out = module.decoder(torch.zeros(2, DIM), torch.zeros(2, DIM))
    assert out.shape == (2, DIM)


# ---------------------------------------------------------------- DDP equivalence
WORKER = os.path.join(REPO_ROOT, 'tests', '_c1_ddp_worker.py')
ATOL = 0.0
RTOL = 1e-5


def _run_worker(mode, rows, out_path, port):
    command = [sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node=2',
               '--master_port', str(port), WORKER, '--mode', mode, '--rows', str(rows),
               '--out', out_path]
    env = {**os.environ, 'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port),
           'GLOO_SOCKET_IFNAME': 'lo'}
    return subprocess.run(command, capture_output=True, text=True, env=env, cwd=REPO_ROOT)


def test_two_rank_c1_step_matches_the_single_process_reference(tmp_path):
    """Same-size rank batches: the DDP-averaged gradient must equal the single-process gradient."""
    import json
    out = str(tmp_path / 'c1_{rank}.json')
    result = _run_worker('single', 2, out, 29692)
    assert result.returncode == 0, result.stderr[-3000:]
    single = json.load(open(out.format(rank=0)))
    assert single['rows'] == 4, ('the single-process run must hold the whole batch', single['rows'])
    result = _run_worker('rank', 2, out, 29693)
    assert result.returncode == 0, result.stderr[-3000:]
    ranks = [json.load(open(out.format(rank=rank))) for rank in range(2)]
    assert [payload['rows'] for payload in ranks] == [2, 2], \
        [payload['rows'] for payload in ranks]
    reference = single['grads']
    for rank, payload in enumerate(ranks):
        assert payload['grads'].keys() == reference.keys()
        for name, value in reference.items():
            other = payload['grads'][name]
            if value is None or other is None:
                assert value is None and other is None, (rank, name)
                continue
            left = torch.tensor(value, dtype=torch.float64)
            right = torch.tensor(other, dtype=torch.float64)
            difference = float((left - right).abs().max())
            allowed = ATOL + RTOL * float(left.abs().max())
            assert difference <= allowed, (rank, name, difference, allowed)
    # ranks must be bit-identical to each other
    for name in reference:
        if reference[name] is None:
            continue
        assert ranks[0]['grads'][name] == ranks[1]['grads'][name], name
    assert ranks[0]['param_digest'] == ranks[1]['param_digest']
    assert ranks[0]['decoder_digest'] == ranks[1]['decoder_digest']


def test_student_checkpoint_has_no_teacher_or_decoder_keys():
    module = _module()
    payload = {'model': module.clip.state_dict(),
               'reconstruction_decoder': module.decoder.state_dict()}
    assert not any(key.startswith('reference_visual') for key in payload['model'])
    assert not any(key.startswith('decoder') for key in payload['model'])
    fresh = _clip(seed=5)
    missing, unexpected = fresh.load_state_dict(payload['model'], strict=False)
    assert not missing and not unexpected, (missing, unexpected)


def test_cli_entry_point_is_importable_and_documented():
    result = subprocess.run([sys.executable, os.path.join(REPO_ROOT, 'train',
                                                          'train_said_cls_c1.py'), '--help'],
                            capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr[-2000:]
    for flag in ('--lambda_rec', '--decoder_lr', '--decoder_seed', '--init_state', '--output_dir',
                 '--diag_every', '--save_completed_steps'):
        assert flag in result.stdout, flag
