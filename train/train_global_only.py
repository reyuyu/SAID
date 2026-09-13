"""S0-GlobalOnly v0.1 -- native image-text global alignment, and nothing else.

This is the simplest SmartCLIP control: the S0 recipe with every mask and every sparse term removed,
so the only objective left is the bidirectional global alignment. It exists to answer one question:
how much of S0@500 is attributable to the masked/local machinery at all?

The objective, exactly as specified
----------------------------------
    g = normalize(clip.encode_image(image))          # native visual tower + native visual.proj
    t = normalize(clip.encode_text(prefix_caption))  # native text tower + native text_projection
    Q = 100 * g @ t.T
    L = 10 * (cross_entropy(Q, targets) + cross_entropy(Q.T, targets))

* both directions are SUMMED, never multiplied by 0.5;
* the score scale is the fixed 100, ``logit_scale`` is never read;
* there is no mask, no sparse/energy/reconstruction/consistency/suffix/auxiliary term of any kind,
  and no "all-ones mask" stand-in either: the global alignment is a plain matrix product;
* ``mask_net`` and ``logit_scale`` are kept only so the state keys stay compatible: they are frozen,
  never called in the forward, and never enter an optimizer;
* the visual tower, the text tower and both native projections are fine-tuned normally.

Distributed semantics (the S0 reference route)
---------------------------------------------
Per rank the anchors are LOCAL and the candidates are the GLOBAL batch, gathered with the
autograd-aware ``torch.distributed.nn.all_gather`` (never detached), with the target index
``rank * local_batch + local_index``. I2T takes the local image rows, T2I takes the local text rows
against the gathered global images; both use the same target vector. DDP then averages parameter
gradients in the standard way -- there is no extra ``world_size`` factor anywhere.

How this differs from the S0 arm (stated, not hidden)
----------------------------------------------------
``model/said_cls_cvssl.py`` scores ``100 * cos(normalize(v_i * m_j), t_j)`` and adds
``lambda_align * (SIDM + DISM) + lambda_sparse * mean(|m_s|)`` with ``lambda_align = 10.0`` and
``lambda_sparse = 2.0`` (``S0_smartclip`` uses ``arm_mask = 'none'`` and ``lambda_U = 0.0``, so no
complementary/U term runs). This arm removes ``m_j`` (equivalently sets it to the constant 1,
implemented as a direct matmul rather than a mask of ones) and removes the sparse term and the mask
network entirely; everything else -- initialisation, data stream, batch, optimiser, warmup,
scheduler horizon, precision and the 500-update budget -- is the S0 recipe.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn as nn_dist
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (os.path.dirname(os.path.abspath(__file__)), REPO):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from model import longclip                                                      # noqa: E402
from scheduler import cosine_lr                                                 # noqa: E402
from said_cvssl_data import Share4VCvsslDataset, cvssl_collate                  # noqa: E402

ARM = 'S0_GLOBAL_ONLY'
OBJECTIVE = 'global_alignment_only'
PHASE = 's0-global-only-v0.1'
LAMBDA_ALIGN = 10.0
FIXED_SCALE = 100.0
NORM_EPS = 1e-6
TOKENIZER_CONTEXT = 248
PRECISION_NOTE = ('fp32 master weights; bf16 autocast for the visual and text towers; the fixed '
                  '100x score, the normalisation and both cross entropies are computed in fp32')
STATS_SCOPE = 'rank0 local batch unless the field name says otherwise'
FROZEN_KEYS = ('mask_net', 'logit_scale')


# --------------------------------------------------------------------------- helpers
def seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def setup_distributed(backend='nccl'):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
    else:
        rank, world, local_rank = 0, 1, 0
    if world > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, rank=rank, world_size=world)
    return rank, local_rank, world


def gather_rows(tensor: torch.Tensor) -> torch.Tensor:
    """Autograd-aware row gather in rank order (identity when not distributed)."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    return torch.cat(nn_dist.all_gather(tensor), dim=0)


def global_targets(local_size: int, rank: int, device) -> torch.Tensor:
    """``rank * local_batch + local_index`` -- the S0 target convention."""
    return torch.linspace(rank * local_size, rank * local_size + local_size - 1,
                          local_size, dtype=torch.long).to(device)


def assert_equal_local_batch(local_size: int, group=None, device=None) -> None:
    if not dist.is_initialized() or dist.get_world_size(group) == 1:
        return
    world = dist.get_world_size(group)
    device = device or torch.device('cuda')
    sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world)]
    dist.all_gather(sizes, torch.tensor([int(local_size)], dtype=torch.long, device=device),
                    group=group)
    values = sorted({int(item.item()) for item in sizes})
    if len(values) != 1:
        raise RuntimeError('ragged local batch across ranks: %s' % values)


def tensor_digest(tensor) -> str:
    array = tensor.detach().to(torch.float32).cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def state_digest(state_dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key]
        if torch.is_tensor(value):
            digest.update(key.encode('utf-8'))
            digest.update(value.detach().to(torch.float32).cpu().numpy().tobytes())
    return digest.hexdigest()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(repo=None) -> str:
    import subprocess
    repo = repo or REPO
    try:
        return subprocess.run(['git', '-C', repo, 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, timeout=20).stdout.strip()
    except Exception:                                            # pragma: no cover
        return 'unknown'


def grad_norm(parameters, device) -> float:
    total = torch.zeros((), device=device, dtype=torch.float32)
    for parameter in parameters:
        if parameter.grad is not None:
            total = total + parameter.grad.detach().float().pow(2).sum()
    return float(total.sqrt())


def grads_finite(parameters) -> bool:
    for parameter in parameters:
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            return False
    return True


def partition_parameters(clip: torch.nn.Module) -> dict:
    """One optimizer group over every trainable parameter; the compatibility keys stay frozen."""
    frozen = {}
    trainable = []
    for name, parameter in clip.named_parameters():
        if name.startswith('mask_net.') or name == 'logit_scale':
            parameter.requires_grad_(False)
            frozen[name.split('.')[0]] = frozen.get(name.split('.')[0], 0) + 1
            continue
        if parameter.requires_grad:
            trainable.append(parameter)
    expected = {id(p) for p in clip.parameters() if p.requires_grad}
    if {id(p) for p in trainable} != expected:
        raise RuntimeError('the optimizer group does not cover exactly the trainable parameters')
    if not frozen:
        raise RuntimeError('the compatibility keys (mask_net / logit_scale) were not found to freeze')
    return {'trainable': trainable, 'frozen': frozen, 'count': len(trainable)}


# --------------------------------------------------------------------------- the objective
class GlobalOnlyTrainModule(torch.nn.Module):
    """The whole arm: one CLIP model, one global alignment loss, no mask anywhere."""

    def __init__(self, clip: torch.nn.Module, rank: int = 0):
        super().__init__()
        self.clip = clip
        self.rank = int(rank)
        self.objective = OBJECTIVE
        # a hard guarantee that the compatibility head can never run: it is not referenced in
        # forward() and this flag is recorded, so a stray call would show up as a missing marker
        self.mask_net_used = False

    def forward(self, images: torch.Tensor, text: torch.Tensor, amp_dtype=torch.bfloat16,
                amp_enabled: bool = True):
        device_type = 'cuda' if images.is_cuda else 'cpu'
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
            v_raw = self.clip.encode_image(images)
            t_raw = self.clip.encode_text(text)
        with torch.autocast(device_type=device_type, enabled=False):
            g = F.normalize(v_raw.float(), dim=-1, eps=NORM_EPS)          # local images
            t_unit = F.normalize(t_raw.float(), dim=-1, eps=NORM_EPS)     # local texts
            local = int(g.shape[0])
            assert_equal_local_batch(local)
            t_all = gather_rows(t_unit)                                   # autograd-aware, live
            g_all = gather_rows(g)
            targets = global_targets(local, self.rank, g.device)
            score_i2t = FIXED_SCALE * (g @ t_all.t())                     # rows: local images
            score_t2i = FIXED_SCALE * (t_unit @ g_all.t())                # rows: local texts
            loss_i2t = F.cross_entropy(score_i2t, targets)
            loss_t2i = F.cross_entropy(score_t2i, targets)
            loss = LAMBDA_ALIGN * (loss_i2t + loss_t2i)
        with torch.no_grad():
            stats = {
                'i2t_top1': float((score_i2t.argmax(dim=1) == targets).float().mean()),
                't2i_top1': float((score_t2i.argmax(dim=1) == targets).float().mean()),
                'i2t_positive_mean': float(score_i2t.gather(1, targets.view(-1, 1)).mean()),
                't2i_positive_mean': float(score_t2i.gather(1, targets.view(-1, 1)).mean()),
                'local_batch': local, 'global_batch': int(t_all.shape[0]),
                'text_norm_before_normalise': float(t_raw.float().norm(dim=-1).mean()),
                'image_norm_before_normalise': float(v_raw.float().norm(dim=-1).mean()),
            }
        return {'loss': loss, 'loss_i2t': loss_i2t, 'loss_t2i': loss_t2i,
                'weighted': LAMBDA_ALIGN * (loss_i2t + loss_t2i), 'stats': stats}


def global_only_train_step(ddp_model, batch, optimizer, device, amp_dtype, amp_enabled=True,
                           completed_steps=0, text_ids=None):
    """DDP forward -> backward -> collective gradient-health check -> one optimizer step."""
    images = batch['image_a'].to(device, non_blocking=True)
    text = (text_ids if text_ids is not None else longclip.tokenize(batch['caption_said'],
                                                                   truncate=True)).to(device)
    out = ddp_model(images, text, amp_dtype=amp_dtype, amp_enabled=amp_enabled)
    out['loss'].backward()
    module = getattr(ddp_model, 'module', ddp_model)
    parameters = [p for p in module.clip.parameters() if p.requires_grad]
    health = {'clip_grad_norm': grad_norm(parameters, device),
              'visual_proj_grad_norm': float(module.clip.visual.proj.grad.detach().norm())
              if module.clip.visual.proj.grad is not None else 0.0,
              'text_projection_grad_norm': float(module.clip.text_projection.grad.detach().norm())
              if module.clip.text_projection.grad is not None else 0.0,
              'mask_net_grad_tensors': sum(1 for p in module.clip.mask_net.parameters()
                                           if p.grad is not None)
              if hasattr(module.clip, 'mask_net') else 0,
              'logit_scale_grad_present': bool(getattr(module.clip, 'logit_scale', None) is not None
                                               and module.clip.logit_scale.grad is not None)}
    local_ok = grads_finite(parameters)
    flag = torch.tensor([0.0 if local_ok else 1.0], device=device)
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if float(flag.item()) > 0:
        raise SystemExit('REFUSING the optimizer step at step %d: a non-finite gradient was found '
                         'on rank %d (no clipping, no nan_to_num, no skipping)'
                         % (completed_steps, module.rank))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    out['_grad_health'] = health
    return out


# --------------------------------------------------------------------------- checkpoint writer
def write_checkpoint(path, clip, optimizer, steps, epoch, step_in_epoch, config, digests,
                     batch_size, precision=PRECISION_NOTE, clip_digest=None, rng=None,
                     rank=0, world=1):
    """Self-describing, atomic, with the frozen compatibility keys recorded as frozen."""
    if clip_digest is None:
        clip_digest = state_digest(clip.state_dict())
    payload = {
        'model': clip.state_dict(),                 # the frozen arms' checkpoint layout
        'clip': clip.state_dict(),                  # the same tensors under the lineage's key
        'optimizer': optimizer.state_dict(),
        'epoch': int(epoch), 'step_in_epoch': int(step_in_epoch),
        'rng': dict(rng or {}),
        'arm': ARM, 'objective': OBJECTIVE, 'phase': PHASE,
        'lambda_align': LAMBDA_ALIGN, 'fixed_scale': FIXED_SCALE, 'norm_eps': NORM_EPS,
        'loss_definition': 'L = 10 * (CE(100*g@t.T, y) + CE(100*t@g.T, y)), no mask, no sparse term',
        'completed_steps': int(steps), 'rank': int(rank), 'world_size': int(world),
        'batch_size_per_gpu': int(batch_size), 'global_batch': int(batch_size) * int(world),
        'clip_state_digest': clip_digest, 'precision': precision,
        'frozen_parameters': ['mask_net', 'logit_scale'],
        'frozen_note': ('kept only for state-key compatibility: frozen, never called in the forward, '
                        'never in an optimizer'),
        'config': dict(config), 'digests': dict(digests),
    }
    if payload['clip_state_digest'] != state_digest(payload['clip']):
        raise RuntimeError('clip_state_digest does not describe the written state')
    temporary = path + '.tmp'
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return payload


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description='S0-GlobalOnly v0.1 (%s)' % OBJECTIVE)
    parser.add_argument('--base_model', default='B16')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--betas', default='0.9,0.999')
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--warmup_length', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--init_state', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--save_completed_steps', default='0,100,250,500')
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--amp_dtype', default='bf16', choices=['bf16', 'fp32'])
    parser.add_argument('--augment_view_b', type=int, default=0)
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()
    if args.base_model == 'B16':
        args.base_model = 'ViT-B/16'

    seed_everything(args.seed)
    rank, local_rank, world = setup_distributed()
    device = torch.device('cuda', local_rank)
    save_completed = sorted({int(s) for s in args.save_completed_steps.split(',') if s.strip()})
    betas = tuple(float(x) for x in args.betas.split(','))

    model, _preprocess = longclip.load_from_clip(args.base_model, device='cpu', download_root=None,
                                                 args=args)
    model.train()
    # compatibility key only: this arm uses the fixed 100x scale and never reads logit_scale
    model.logit_scale = torch.nn.Parameter(torch.ones([]) * math.log(FIXED_SCALE))
    model = model.to(device)
    if not os.path.isfile(args.init_state):
        raise SystemExit('the shared initialisation is missing: %s' % args.init_state)
    init_file_sha = file_sha256(args.init_state)
    payload = torch.load(args.init_state, map_location='cpu', weights_only=False)
    if isinstance(payload, dict) and 'model' in payload:
        payload = payload['model']
    target = set(model.state_dict())
    mapped, ignored = {}, []
    for key, value in payload.items():
        if key in target:
            mapped[key] = value
        elif key.startswith('clip.') and key[len('clip.'):] in target:
            mapped[key[len('clip.'):]] = value
        elif ('clip.' + key) in target:
            mapped['clip.' + key] = value
        else:
            ignored.append(key)
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    bad_missing = [k for k in missing if not k.startswith('mask_net')]
    if bad_missing or list(unexpected):
        raise RuntimeError('init_state did not load cleanly: missing %r unexpected %r'
                           % (bad_missing, list(unexpected)))
    initial_state_digest = state_digest(model.state_dict())
    if rank == 0:
        print('INIT_STATE_LOADED %s tensors=%d ignored=%r digest=%s'
              % (args.init_state, len(mapped), ignored[:5], initial_state_digest), flush=True)

    partition = partition_parameters(model)
    if rank == 0:
        print('PARAM_PARTITION trainable=%d frozen=%s' % (partition['count'], partition['frozen']),
              flush=True)
    module = GlobalOnlyTrainModule(model, rank=rank).to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(module, device_ids=[local_rank],
                                                          output_device=local_rank,
                                                          find_unused_parameters=True)
    ddp_model._set_static_graph()
    clip_handle = module.clip
    optimizer = torch.optim.AdamW(partition['trainable'], lr=args.lr, weight_decay=args.weight_decay,
                                  betas=betas, eps=args.eps)

    dataset = Share4VCvsslDataset(seed=args.seed, augment_view_b=bool(args.augment_view_b),
                                 strict_manifest=os.environ.get('SHARE4V_FULL_AUDIT'))
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, seed=args.seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                                        num_workers=args.num_workers, pin_memory=True,
                                        collate_fn=cvssl_collate, drop_last=False)
    steps_per_epoch = len(loader)
    total_steps = args.epochs * steps_per_epoch
    scheduler = cosine_lr(optimizer, base_lr=args.lr, warmup_length=args.warmup_length,
                          steps=total_steps)

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, 'salu_log.jsonl')
    config = {
        'arm': ARM, 'objective': OBJECTIVE, 'phase': PHASE,
        'loss': 'L = 10 * (CE(100 * g @ t.T, y) + CE(100 * t @ g.T, y))',
        'lambda_align': LAMBDA_ALIGN, 'fixed_scale': FIXED_SCALE, 'no_half_factor': True,
        'mask': 'none anywhere: no mask_net forward, no mask tensor, no all-ones mask stand-in',
        'sparse_term': 'removed (S0 uses lambda_sparse = 2.0 * mean(|m_s|))',
        'other_auxiliary_terms': 'none (no reconstruction, consistency, suffix, or U term)',
        's0_reference_for_the_recipe': {
            'source': '/root/SAID-s0-continue-v01/runs_salu/said_cls_cvssl/s0_continue_1000/config.json',
            'arm': 'S0_smartclip', 'objective': 'said_cls_cvssl', 'lambda_align': 10.0,
            'lambda_sparse': 2.0, 'lambda_U': 0.0, 'arm_mask': 'none',
            'note': ('S0 scores 100 * cos(normalize(v * m_j), t_j) plus 2.0 * mean(|m_s|); this arm '
                     'drops m_j (constant 1, computed as a plain matmul) and the sparse term'),
        },
        'init_state': args.init_state, 'init_file_sha256': init_file_sha,
        'initial_state_digest': initial_state_digest,
        'frozen_parameters': ['mask_net', 'logit_scale'],
        'lr': args.lr, 'weight_decay': args.weight_decay, 'betas': list(betas), 'eps': args.eps,
        'warmup_length': args.warmup_length, 'epochs': args.epochs,
        'batch_size_per_gpu': args.batch_size, 'world_size': world,
        'global_batch': args.batch_size * world, 'grad_accumulation': 1,
        'loader_batches': steps_per_epoch, 'lr_horizon_steps': total_steps,
        'max_steps': args.max_steps, 'seed': args.seed,
        'sampling': 'DistributedSampler(shuffle=True), no drop_last, steps_per_epoch=len(loader)=%d'
                    % steps_per_epoch,
        'view_a': 'reference openai-clip _transform(224)',
        'view_b': 'not generated (augment_view_b=%s); this arm is single-view' % bool(args.augment_view_b),
        'caption_stream': "caption.replace('\\n', ' ').split('. '), K ~ randint(1, len(sentences)), "
                          "prefix = '. '.join(sentences[:K])",
        'precision': PRECISION_NOTE, 'amp_dtype': args.amp_dtype,
        'ddp_route': 'per-rank anchors + autograd-aware all_gather (candidates live) + standard DDP '
                     'averaging; no world_size factor',
        'candidate_scope': 'global batch, candidates NOT detached',
        'targets': 'rank * local_batch + local_index',
        'git_head': git_head(REPO), 'tokenizer_context': TOKENIZER_CONTEXT,
        'statistics_scope': STATS_SCOPE,
    }
    if rank == 0:
        print('CONFIG ' + json.dumps(config, sort_keys=True), flush=True)
        print('DATASET_SIZE %d STEPS_PER_EPOCH %d LR_HORIZON %d TRAINABLE %d'
              % (len(dataset), steps_per_epoch, total_steps, partition['count']), flush=True)
        with open(os.path.join(args.output_dir, 'config.json'), 'w') as handle:
            json.dump(config, handle, indent=2, sort_keys=True)

    if args.resume:
        payload = torch.load(args.resume, map_location='cpu', weights_only=False)
        clip_handle.load_state_dict(payload['clip'])
        optimizer.load_state_dict(payload['optimizer'])
        start_step = int(payload['completed_steps'])
        if rank == 0:
            print('RESUMED at completed step %d' % start_step, flush=True)
    else:
        start_step = 0

    caption_digest = hashlib.sha256()
    sample_digest = hashlib.sha256()
    image_digest = hashlib.sha256()
    prefix_digest = hashlib.sha256()

    def save_checkpoint(steps, epoch, step_in_epoch, health=None):
        digests = {'init_file_sha256': init_file_sha, 'initial_state_digest': initial_state_digest,
                   'caption_stream_sha256': caption_digest.hexdigest(),
                   'sample_stream_sha256': sample_digest.hexdigest(),
                   'image_id_stream_sha256': image_digest.hexdigest(),
                   'prefix_k_stream_sha256': prefix_digest.hexdigest()}
        path = os.path.join(args.output_dir, '%s_step%06d.pt' % (ARM, steps))
        write_checkpoint(path=path, clip=clip_handle, optimizer=optimizer, steps=steps, epoch=epoch,
                         step_in_epoch=step_in_epoch, config=config, digests=digests,
                         batch_size=args.batch_size, rng={'torch': torch.get_rng_state()},
                         rank=rank, world=world)
        print('SAVED ' + path, flush=True)

    if rank == 0 and start_step == 0 and 0 in save_completed:
        save_checkpoint(0, 0, -1)
    in_loop_saves = sorted(s for s in save_completed if s > start_step and s != 0)

    completed = start_step
    t_start = time.time()
    compute_times = []
    presentations = 0
    stopped = False
    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        for i, batch in enumerate(loader):
            if args.max_steps is not None and completed >= args.max_steps:
                stopped = True
                break
            t0 = time.time()
            scheduler(completed)
            out = global_only_train_step(ddp_model, batch, optimizer, device,
                                         torch.bfloat16, amp_enabled=args.amp_dtype == 'bf16',
                                         completed_steps=completed)
            completed += 1
            compute_times.append(time.time() - t0)
            presentations += args.batch_size * world
            caption_digest.update(('\n'.join(batch['caption_said'])).encode('utf-8'))
            sample_digest.update(batch['sample_id'].numpy().tobytes())
            image_digest.update(batch['image_id'].numpy().tobytes())
            prefix_digest.update(batch['prefix_k'].numpy().tobytes())
            if rank == 0 and (completed % args.log_every == 0 or completed == 1
                              or completed in save_completed):
                stats = out['stats']
                record = {
                    'completed_steps': completed, 'epoch': epoch, 'step_in_epoch': i,
                    'arm': ARM, 'objective': OBJECTIVE, 'world_size': world,
                    'batch_size_local': stats['local_batch'], 'global_pairs': stats['global_batch'],
                    'lr': optimizer.param_groups[0]['lr'], 'grad_accumulation': 1,
                    'loss': float(out['loss']), 'loss_i2t': float(out['loss_i2t']),
                    'loss_t2i': float(out['loss_t2i']), 'lambda_align': LAMBDA_ALIGN,
                    'weighted_loss': float(out['weighted']),
                    'i2t_top1': stats['i2t_top1'], 't2i_top1': stats['t2i_top1'],
                    'i2t_positive_mean': stats['i2t_positive_mean'],
                    't2i_positive_mean': stats['t2i_positive_mean'],
                    'sec_per_step': compute_times[-1],
                    'samples_per_sec': float(stats['local_batch']) * world
                    / max(compute_times[-1], 1e-9),
                    'peak_memory_gb': torch.cuda.max_memory_allocated(device) / (1024 ** 3),
                    'memory_unit': 'GiB (1024^3 bytes)',
                    'rank_param_digest': state_digest(clip_handle.state_dict())[:16],
                    'batch_caption_sha256': hashlib.sha256(
                        ('\n'.join(batch['caption_said'])).encode('utf-8')).hexdigest()[:16],
                    'batch_image_id_sha256': tensor_digest(batch['image_id']),
                    'batch_sample_id_sha256': tensor_digest(batch['sample_id']),
                    'batch_prefix_k_sha256': tensor_digest(batch['prefix_k']),
                    'prefix_k_values': [int(v) for v in batch['prefix_k'][:8]],
                    'image_id_values': [int(v) for v in batch['image_id'][:8]],
                    'rank_local_stream_digests': {
                        'caption': caption_digest.hexdigest(), 'sample': sample_digest.hexdigest(),
                        'image_id': image_digest.hexdigest(), 'prefix_k': prefix_digest.hexdigest()},
                    'synchronized_pair_presentations': presentations,
                    'mask_net_forwarded': False, 'mask_tensors_in_loss': 0,
                    'statistics_scope': 'rank0_local_batch',
                }
                record.update(out['_grad_health'])
                with open(log_path, 'a') as handle:
                    handle.write(json.dumps(record, sort_keys=True) + '\n')
                print('LOG ' + json.dumps(record, sort_keys=True), flush=True)
            if rank == 0 and completed in in_loop_saves:
                save_checkpoint(completed, epoch, i, health=out['_grad_health'])
        if stopped:
            break

    if rank == 0:
        summary = {
            'arm': ARM, 'objective': OBJECTIVE, 'completed_steps': completed,
            'max_steps': args.max_steps, 'epochs': args.epochs,
            'wall_sec': time.time() - t_start,
            'mean_sec_per_step': sum(compute_times) / max(len(compute_times), 1),
            'steps_per_epoch': steps_per_epoch, 'lr_horizon_steps': total_steps,
            'world_size': world, 'batch_size_per_gpu': args.batch_size,
            'global_batch': args.batch_size * world,
            'grad_accumulation': 1,
            'synchronized_pair_presentations': presentations,
            'expected_at_500': 500 * args.batch_size * world,
            'peak_memory_gb': torch.cuda.max_memory_allocated(device) / (1024 ** 3),
            'memory_unit': 'GiB (1024^3 bytes)',
            'init_file_sha256': init_file_sha, 'initial_state_digest': initial_state_digest,
            'final_state_digest': state_digest(clip_handle.state_dict()),
            'stream_digests': {'caption': caption_digest.hexdigest(),
                               'sample': sample_digest.hexdigest(),
                               'image_id': image_digest.hexdigest(),
                               'prefix_k': prefix_digest.hexdigest()},
            'precision': PRECISION_NOTE, 'statistics_scope': STATS_SCOPE,
            'frozen_parameters': ['mask_net', 'logit_scale'],
        }
        with open(os.path.join(args.output_dir, 'run_summary.json'), 'w') as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print('RUN_SUMMARY ' + json.dumps(summary, sort_keys=True), flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
