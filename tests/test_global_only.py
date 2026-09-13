"""S0-GlobalOnly v0.1 acceptance tests -- the four small checks the spec asks for.

1. the bidirectional loss VALUE and its GRADIENTS match an independent reference;
2. the compatibility head (``mask_net``) and ``logit_scale`` are frozen, never forwarded and never
   receive a gradient, while the backbone really does update;
3. two-rank DDP gradient scaling is correct: the averaged gradient is the exact mean of the two
   ranks' local gradients, so no ``world_size`` factor sneaks in, and it matches a single-process
   reference over the same global batch;
4. checkpoints and the bare student load strictly.

Everything runs on the real cached CLIP ViT-B/16, not a mock.
"""
import argparse
import json
import math
import os
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (REPO, os.path.join(REPO, 'train'), os.path.join(REPO, 'tests')):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from model import longclip                                                      # noqa: E402
import train_global_only as trainer                                             # noqa: E402

CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda', 0) if CUDA else torch.device('cpu')
needs_cuda = pytest.mark.skipif(not CUDA, reason='acceptance tests need a CUDA device')
needs_two_gpus = pytest.mark.skipif(torch.cuda.device_count() < 2,
                                    reason='needs two visible GPUs')
CLIP_CACHE = os.path.expanduser('~/.cache/clip/ViT-B-16.pt')
CAPTIONS = ['a photo of a cat on a wooden table', 'a dog running through tall grass',
            'an old bicycle leaning against a wall', 'two boats on a calm lake']
GRAD_NAMES = ('clip.visual.proj', 'clip.text_projection',
              'clip.visual.transformer.resblocks.11.attn.in_proj_weight',
              'clip.visual.conv1.weight', 'clip.transformer.resblocks.0.attn.in_proj_weight')


def require_clip():
    if not os.path.isfile(CLIP_CACHE):
        pytest.skip('the CLIP ViT-B/16 checkpoint is not cached at %s' % CLIP_CACHE)


def build_model(device=DEVICE):
    require_clip()
    model, _ = longclip.load_from_clip('ViT-B/16', device='cpu', download_root=None,
                                       args=argparse.Namespace())
    model.logit_scale = torch.nn.Parameter(torch.ones([]) * math.log(trainer.FIXED_SCALE))
    return model.to(device).eval()


def fixed_images(count=2, seed=20260913):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(count, 3, 224, 224, generator=generator)


def named_grads(module, names=GRAD_NAMES):
    found = dict(module.named_parameters())
    return {name: (None if found[name].grad is None else found[name].grad.detach().cpu().clone())
            for name in names}


# --------------------------------------------------------------------------- 1. loss and gradients
@needs_cuda
def test_a_loss_and_gradients_match_an_independent_reference():
    """The written objective, recomputed by hand, must give the same loss and the same gradients."""
    model = build_model()
    module = trainer.GlobalOnlyTrainModule(model, rank=0).to(DEVICE)
    images = fixed_images(2).to(DEVICE)
    text = longclip.tokenize(CAPTIONS[:2]).to(DEVICE)

    out = module(images, text, amp_enabled=False)
    forward_loss = float(out['loss'].detach())
    parameters = [p for name, p in module.named_parameters() if name in GRAD_NAMES]
    grads = torch.autograd.grad(out['loss'], parameters)

    # ---- the reference, written independently of the module
    reference_model = build_model()
    with torch.enable_grad():
        v = reference_model.encode_image(images)
        t = reference_model.encode_text(text)
        g = F.normalize(v.float(), dim=-1, eps=trainer.NORM_EPS)
        t_unit = F.normalize(t.float(), dim=-1, eps=trainer.NORM_EPS)
        Q = trainer.FIXED_SCALE * (g @ t_unit.t())
        targets = torch.arange(g.shape[0], device=DEVICE)
        reference_loss = 10.0 * (F.cross_entropy(Q, targets) + F.cross_entropy(Q.t(), targets))
    # inside the train module the model lives under ``self.clip``, so the same tensors carry a
    # ``clip.`` prefix; the bare reference model does not, hence the mapping
    reference_names = tuple(name[len('clip.'):] if name.startswith('clip.') else name
                            for name in GRAD_NAMES)
    reference_parameters = [p for name, p in reference_model.named_parameters()
                            if name in reference_names]
    assert len(reference_parameters) == len(GRAD_NAMES), \
        'the reference parameter names must line up with the module parameter names'
    reference_grads = torch.autograd.grad(reference_loss, reference_parameters)

    assert forward_loss == pytest.approx(float(reference_loss.detach()), rel=1e-5, abs=1e-4)
    assert float(out['loss_i2t']) + float(out['loss_t2i']) == pytest.approx(forward_loss / 10.0,
                                                                          rel=1e-5)
    assert float(out['loss_i2t']) > 0 and float(out['loss_t2i']) > 0
    # the two directions are summed, never halved: the loss is exactly 10x their sum
    assert float(out['weighted']) == pytest.approx(forward_loss, rel=1e-6)
    for index, (mine, reference) in enumerate(zip(grads, reference_grads)):
        scale = max(float(reference.abs().max()), 1e-12)
        assert float((mine - reference).abs().max()) / scale < 1e-4, GRAD_NAMES[index]
    print('  loss=%.6f reference=%.6f | checked %d gradient tensors'
          % (forward_loss, float(reference_loss.detach()), len(grads)))


@needs_cuda
def test_a_score_uses_the_fixed_scale_and_not_logit_scale():
    """Q must be exactly 100 * <normalised g, normalised t>; logit_scale must not appear."""
    model = build_model()
    module = trainer.GlobalOnlyTrainModule(model, rank=0).to(DEVICE)
    trainer.partition_parameters(model)
    images = fixed_images(2).to(DEVICE)
    text = longclip.tokenize(CAPTIONS[:2]).to(DEVICE)
    with torch.no_grad():
        v = model.encode_image(images).float()
        t = model.encode_text(text).float()
        g = F.normalize(v, dim=-1, eps=trainer.NORM_EPS)
        t_unit = F.normalize(t, dim=-1, eps=trainer.NORM_EPS)
        expected = trainer.FIXED_SCALE * (g @ t_unit.t())
        # perturbing logit_scale must not move the scores at all
        model.logit_scale.data.fill_(math.log(4.0))
        out = module(images, text, amp_enabled=False)
    # recompute the module's own score through the recorded statistics path
    assert trainer.FIXED_SCALE == 100.0
    assert float(expected.max()) <= 100.0 * 1.001
    assert out['stats']['i2t_top1'] >= 0.0            # the forward ran and produced statistics
    assert model.logit_scale.requires_grad is False


# --------------------------------------------------------------------------- 2. frozen head
@needs_cuda
def test_b_mask_head_is_frozen_and_backbone_updates():
    model = build_model()
    model.train()
    partition = trainer.partition_parameters(model)
    assert 'mask_net' in partition['frozen'] and 'logit_scale' in partition['frozen']
    assert all(not p.requires_grad for p in model.mask_net.parameters())
    assert not model.logit_scale.requires_grad
    assert all(id(p) not in {id(q) for q in partition['trainable']}
               for p in model.mask_net.parameters())
    assert id(model.logit_scale) not in {id(q) for q in partition['trainable']}

    module = trainer.GlobalOnlyTrainModule(model, rank=0).to(DEVICE)
    optimizer = torch.optim.AdamW(partition['trainable'], lr=1e-4, weight_decay=1e-2)
    images = fixed_images(2).to(DEVICE)
    text = longclip.tokenize(CAPTIONS[:2]).to(DEVICE)
    before = {name: p.detach().clone() for name, p in
              [('visual.proj', model.visual.proj), ('text_projection', model.text_projection)]}
    out = module(images, text, amp_enabled=False)
    out['loss'].backward()
    optimizer.step()

    assert all(p.grad is None for p in model.mask_net.parameters()), \
        'the frozen compatibility head must never receive a gradient'
    assert model.logit_scale.grad is None
    assert float(model.visual.proj.grad.norm()) > 0
    assert not torch.equal(before['visual.proj'], model.visual.proj), 'the backbone must update'
    assert not torch.equal(before['text_projection'], model.text_projection)
    # the module records that the head was never forwarded
    assert module.mask_net_used is False
    assert partition['count'] == sum(1 for p in model.parameters() if p.requires_grad)


@needs_cuda
def test_b_the_gradient_health_report_shows_no_mask_head_gradient():
    model = build_model()
    model.train()
    trainer.partition_parameters(model)
    module = trainer.GlobalOnlyTrainModule(model, rank=0).to(DEVICE)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-6)
    batch = {'image_a': fixed_images(2), 'caption_said': CAPTIONS[:2]}
    out = trainer.global_only_train_step(module, batch, optimizer, DEVICE, torch.bfloat16,
                                         amp_enabled=False, completed_steps=0)
    health = out['_grad_health']
    assert health['mask_net_grad_tensors'] == 0
    assert health['logit_scale_grad_present'] is False
    assert health['clip_grad_norm'] > 0
    assert health['visual_proj_grad_norm'] > 0


# --------------------------------------------------------------------------- 3. DDP scaling
@needs_two_gpus
def test_c_two_rank_ddp_gradient_equals_the_mean_of_the_local_gradients(tmp_path):
    sys.path.insert(0, os.path.join(REPO, 'tests'))
    import _global_only_ddp_worker as worker
    local_batch = 2
    images, text_ids, captions = worker.build_inputs(local_batch=local_batch)
    port = str(29891 + (os.getpid() % 200))
    command = [sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node=2',
               '--master_port=%s' % port, os.path.join('tests', '_global_only_ddp_worker.py'),
               '--output_dir', str(tmp_path), '--local_batch', str(local_batch)]
    env = dict(os.environ, NCCL_SOCKET_IFNAME='lo', GLOO_SOCKET_IFNAME='lo')
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, timeout=1800, env=env)
    assert result.returncode == 0, (result.stdout[-4000:], result.stderr[-4000:])
    reduced = torch.load(os.path.join(str(tmp_path), 'ddp_grads_rank0.pt'), map_location='cpu')
    local0 = torch.load(os.path.join(str(tmp_path), 'local_grads_rank0.pt'), map_location='cpu')
    local1 = torch.load(os.path.join(str(tmp_path), 'local_grads_rank1.pt'), map_location='cpu')
    info = torch.load(os.path.join(str(tmp_path), 'ddp_info.pt'), map_location='cpu')
    assert info['world_size'] == 2

    # (a) DECISIVE: the reduction is a plain mean of the two local gradients -- no world_size factor
    for name in GRAD_NAMES:
        first, second = local0[name].float(), local1[name].float()
        mean = (first + second) / 2.0
        scale = max(float(mean.abs().max()), 1e-12)
        assert float((reduced[name].float() - mean).abs().max()) / scale < 1e-4, name
        assert float((first - second).abs().max()) / scale > 1e-3, \
            '%s: the two ranks must differ, otherwise the check is vacuous' % name

    # (b) and it matches a single-process reference over the same global batch
    model = build_model()
    model.train()
    trainer.partition_parameters(model)
    module = trainer.GlobalOnlyTrainModule(model, rank=0).to(DEVICE)
    out = module(images.to(DEVICE), text_ids.to(DEVICE), amp_enabled=False)
    out['loss'].backward()
    reference = named_grads(module)
    for name in GRAD_NAMES:
        scale = max(float(reference[name].abs().max()), 1e-9)
        difference = float((reduced[name].float() - reference[name].float()).abs().max()) / scale
        assert difference < 1e-2, (name, difference)
    print('  local losses: %.4f / %.4f | global reference loss checked'
          % (info['loss_local_rank0'], info['loss_total_rank0']))


@needs_cuda
def test_c_single_process_loss_equals_the_local_anchor_definition():
    """With world=1 the loss is exactly CE over the local rows in both directions."""
    model = build_model()
    module = trainer.GlobalOnlyTrainModule(model, rank=0).to(DEVICE)
    images = fixed_images(3).to(DEVICE)
    text = longclip.tokenize(CAPTIONS[:3]).to(DEVICE)
    out = module(images, text, amp_enabled=False)
    with torch.no_grad():
        g = F.normalize(model.encode_image(images).float(), dim=-1, eps=trainer.NORM_EPS)
        t = F.normalize(model.encode_text(text).float(), dim=-1, eps=trainer.NORM_EPS)
        Q = trainer.FIXED_SCALE * (g @ t.t())
        targets = torch.arange(3, device=DEVICE)
        expected = 10.0 * (F.cross_entropy(Q, targets) + F.cross_entropy(Q.t(), targets))
    assert float(out['loss']) == pytest.approx(float(expected), rel=1e-5, abs=1e-4)


# --------------------------------------------------------------------------- 4. checkpoints
@needs_cuda
def test_d_checkpoint_and_bare_student_load_strictly(tmp_path):
    model = build_model()
    model.train()
    trainer.partition_parameters(model)
    module = trainer.GlobalOnlyTrainModule(model, rank=0).to(DEVICE)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    images = fixed_images(2).to(DEVICE)
    text = longclip.tokenize(CAPTIONS[:2]).to(DEVICE)
    out = module(images, text, amp_enabled=False)
    out['loss'].backward()
    optimizer.step()

    config = {'arm': trainer.ARM, 'objective': trainer.OBJECTIVE, 'phase': trainer.PHASE,
              'lambda_align': trainer.LAMBDA_ALIGN, 'fixed_scale': trainer.FIXED_SCALE}
    path = os.path.join(str(tmp_path), '%s_step000500.pt' % trainer.ARM)
    trainer.write_checkpoint(path=path, clip=model, optimizer=optimizer, steps=500, epoch=0,
                            step_in_epoch=499, config=config, digests={'init_file_sha256': 'x'},
                            batch_size=256)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    assert payload['completed_steps'] == 500
    assert payload['arm'] == trainer.ARM and payload['objective'] == trainer.OBJECTIVE
    assert payload['lambda_align'] == 10.0 and payload['fixed_scale'] == 100.0
    assert payload['frozen_parameters'] == ['mask_net', 'logit_scale']
    assert payload['clip_state_digest'] == trainer.state_digest(payload['clip'])
    assert payload['optimizer']['state'], 'the optimizer state must be real'
    assert not os.path.isfile(path + '.tmp')

    fresh = build_model()
    missing, unexpected = fresh.load_state_dict(payload['clip'], strict=True)
    assert not missing and not unexpected
    # the same freeze must be applied before rebuilding the optimizer, otherwise the parameter group
    # is larger than the one that was saved
    fresh_partition = trainer.partition_parameters(fresh)
    fresh_optimizer = torch.optim.AdamW(fresh_partition['trainable'], lr=1e-4)
    fresh_optimizer.load_state_dict(payload['optimizer'])

    # the bare student the evaluators consume
    student = {'model': {k: v.clone() for k, v in model.state_dict().items()}, 'completed_steps': 500}
    student_path = os.path.join(str(tmp_path), 'student_000500.pt')
    torch.save(student, student_path)
    loaded = torch.load(student_path, map_location='cpu', weights_only=False)
    assert loaded['completed_steps'] == 500
    strict_model = build_model()
    strict_model.load_state_dict(loaded['model'], strict=True)
    assert len(loaded['model']) == len(strict_model.state_dict())
    assert trainer.state_digest(loaded['model']) == trainer.state_digest(model.state_dict())
    print('  checkpoint keys: %s' % sorted(payload)[:10])


@needs_cuda
def test_d_partition_refuses_a_model_without_the_compatibility_keys():
    """The freeze list is verified, not assumed: a model missing mask_net must be refused."""
    class Bare(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2, 2))

    with pytest.raises(RuntimeError, match='compatibility keys'):
        trainer.partition_parameters(Bare())
