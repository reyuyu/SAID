"""SAID-CLS-CVSSL v0.1 trainer.

    L = L_smart + lambda_U * L_U(visual, complementary coordinates)

Faithful to the SmartCLIP reference (``train/train.py``) wherever the reference defines the
protocol:

* two AdamW optimizers -- ``lr=1e-6, weight_decay=1e-2`` for everything except ``mask_net`` and
  ``mask_lr=1e-3, weight_decay=0`` for ``mask_net``;
* ``cosine_lr`` with ``warmup_length=200`` for the main optimizer and ``warmup_length=0`` for the
  mask optimizer, over the **full 3-epoch horizon** ``epochs * len(loader)`` (``--max_steps`` only
  truncates a run, it never compresses the schedule);
* the same dataset, caption stream, ``DistributedSampler(shuffle=True)``, ``num_workers=8``,
  ``pin_memory=True`` and **no** ``drop_last`` (so ``len(loader)`` is whatever the real DataLoader
  reports; the ragged last batch is handled, not dropped);
* ``logit_scale`` is replaced by ``ones * log_scale`` exactly as the reference does. It is unused by
  the objective (the reference's SmartCLIP scale is the fixed 100), which is why the trainer runs
  with plain per-rank gradients and no DDP wrapper -- the reference trainer does the same and
  relies on the autograd-aware all-gather inside the loss.

Deliberate differences (all documented in the report):

* precision: FP32 master weights + **bf16** autocast (the reference uses fp16 autocast with a
  GradScaler). The S0 equivalence gate compares the two objectives on the *same* weights and dtype,
  so this is a training-precision choice, not a loss change;
* the model is not DDP-wrapped (see above), so the CVSSL term scales by ``1 / V_global`` with
  ``--ddp_gradient_averaging 0``. Setting the flag to 1 multiplies by the world size for callers
  that do wrap the model in DDP.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import complement_visual_ssl as cvssl  # noqa: E402
from model import longclip  # noqa: E402
from model.said_cls_cvssl import (ARMS, ARM_LAMBDA_U, ARM_MASK, SaidClsCvsslObjective,  # noqa: E402
                                  SaidClsCvsslTrainModule, LAMBDA_ALIGN, LAMBDA_SPARSE)
from said_cvssl_data import Share4VCvsslDataset, cvssl_collate, stateless_seed  # noqa: E402
from scheduler import cosine_lr  # noqa: E402

TOKENIZER_CONTEXT = 248


def seed_everything(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def state_digest(state_dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key]
        digest.update(key.encode('utf-8'))
        if torch.is_tensor(tensor):
            digest.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
        else:
            digest.update(repr(tensor).encode('utf-8'))
    return digest.hexdigest()


def tensor_digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()
                          ).hexdigest()[:16]


def setup_distributed(backend='nccl'):
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ.get('LOCAL_RANK', rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, world_size=world, rank=rank)
    return rank, local_rank, world


def git_head() -> str:
    import subprocess
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:                                        # pragma: no cover
        return 'unknown'


def load_init_state(model, path: str, rank: int):
    """Load the *complete* frozen initial state dict, or fail loudly.

    All four arms must start from this exact state (clip + mask_net). A ``clip.``-prefixed SALU
    layout and a bare CLIP layout are both accepted; a bare CLIP checkpoint legitimately has no
    ``mask_net``, in which case the (deterministically seeded) freshly built mask_net is kept and
    the digest of the *resulting* full state is what every arm shares.
    """
    payload = torch.load(path, map_location='cpu', weights_only=False)
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
    if not mapped:
        raise ValueError('--init_state %s matches no model tensor' % path)
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    bad_missing = [key for key in missing if not key.startswith('mask_net')]
    if bad_missing or list(unexpected):
        raise RuntimeError('--init_state did not load cleanly: missing %r unexpected %r'
                           % (bad_missing, list(unexpected)))
    if rank == 0:
        print('INIT_STATE_LOADED %s tensors=%d mask_net_absent=%d ignored=%r'
              % (path, len(mapped), len([k for k in missing if k.startswith('mask_net')]),
                 ignored[:5]), flush=True)


def build_optimizers(model, args):
    mask_net_ids = {id(parameter) for parameter in model.mask_net.parameters()}
    backbone, mask_net = [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (mask_net if id(parameter) in mask_net_ids else backbone).append(parameter)
    seen = {id(p) for p in backbone} | {id(p) for p in mask_net}
    assert len(seen) == len(backbone) + len(mask_net), 'a parameter was registered twice'
    optimizer = torch.optim.AdamW(backbone, lr=args.lr, weight_decay=args.weight_decay)
    mask_optimizer = torch.optim.AdamW(mask_net, lr=args.mask_lr, weight_decay=0)
    return optimizer, mask_optimizer, len(backbone), len(mask_net)


def grad_probe(objective, batch, device, args):
    """Separate-graph diagnostics: gradient norms per source and the smart/U direction cosine."""
    image_a = batch['image_a'].to(device, non_blocking=True)
    image_b = batch['image_b'].to(device, non_blocking=True)
    text = longclip.tokenize(batch['caption_said'], truncate=True).to(device)
    image_ids = batch['image_id'].to(device)
    generator = torch.Generator().manual_seed(stateless_seed(args.seed, 'mask', 0, args.arm))
    with torch.enable_grad():
        out = objective(image_a, image_b, text, image_ids, random_generator=generator)
        parameters = [p for p in objective.clip.parameters() if p.requires_grad]
        visual = [objective.clip.visual.conv1.weight]
        text_proj = [objective.clip.text_projection]
        mask_params = list(objective.clip.mask_net.parameters())

        def norms(term):
            result = {}
            for name, group in (('visual', visual), ('text', text_proj), ('mask_net', mask_params)):
                grads = torch.autograd.grad(term, group, retain_graph=True, allow_unused=True)
                total = 0.0
                for grad in grads:
                    if grad is not None:
                        total += float(grad.detach().float().pow(2).sum())
                result[name] = math.sqrt(total)
            return result

        smart_norms = norms(out['loss_smart'])
        u_norms = norms(args.lambda_U * out['loss'])
        smart_grad = torch.autograd.grad(out['loss_smart'], visual, retain_graph=True,
                                         allow_unused=True)[0]
        u_grad = torch.autograd.grad(args.lambda_U * out['loss'], visual, retain_graph=False,
                                     allow_unused=True)[0]
        cosine = (float(torch.nn.functional.cosine_similarity(
            smart_grad.flatten(), u_grad.flatten(), dim=0))
            if smart_grad is not None and u_grad is not None else float('nan'))
        ratio = (u_norms['visual'] / max(smart_norms['visual'], 1e-12))
    return {
        'grad_norm_visual_smart': smart_norms['visual'],
        'grad_norm_visual_vssl': u_norms['visual'],
        'grad_norm_mask_net_smart': smart_norms['mask_net'],
        'grad_norm_mask_net_vssl': u_norms['mask_net'],
        'grad_norm_text_smart': smart_norms['text'],
        'grad_norm_text_vssl': u_norms['text'],
        'weighted_vssl_to_smart_grad_ratio': ratio,
        'cos_grad_visual_smart_vssl': cosine,
    }


def cvssl_train_step(ddp_model, batch, optimizer, mask_optimizer, device, amp_dtype,
                     amp_enabled: bool = True, mask_rng_seed: int = 0,
                     capture_grads: bool = False, mask_override=None):
    """One real training step: DDP forward -> backward -> both optimizer steps.

    This is the *production* step function; the distributed acceptance tests call it directly so
    that what is verified is what is run. ``mask_rng_seed`` must be identical on every rank (the
    random-mask permutation is part of the objective, not of the data stream).

    Returns the (rank-local) output dict. With ``capture_grads`` the post-backward,
    pre-optimizer gradients of every trainable parameter are attached under ``'_grads'`` --
    a read-only test hook that does not change the update.
    """
    image_a = batch['image_a'].to(device, non_blocking=True)
    image_b = batch['image_b'].to(device, non_blocking=True)
    image_ids = batch['image_id'].to(device)
    # the ragged-batch guard must run BEFORE the first cross-rank gather: otherwise gloo aborts
    # with a length mismatch instead of reporting the real problem
    cvssl.assert_equal_local_batch(int(image_a.shape[0]))
    with torch.no_grad():
        text = longclip.tokenize(batch['caption_said'], truncate=True).to(device)
    generator = torch.Generator().manual_seed(int(mask_rng_seed))
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
        out = ddp_model(image_a, image_b, text, image_ids, generator,
                        mask_seed=int(mask_rng_seed), mask_override=mask_override)
        loss = out['loss_total_for_backward']
    loss.backward()
    if capture_grads:
        # ``ddp_model`` is the DDP wrapper in training and the bare module in single-process runs;
        # ``.module`` is read-only diagnostics, never used for the training forward.
        inspected = getattr(ddp_model, 'module', ddp_model)
        out['_grads'] = {name: (None if parameter.grad is None else parameter.grad.detach().clone())
                         for name, parameter in inspected.named_parameters()}
    optimizer.step()
    mask_optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    mask_optimizer.zero_grad(set_to_none=True)
    return out


def main():
    parser = argparse.ArgumentParser(description='SAID-CLS-CVSSL v0.1')
    parser.add_argument('--arm', default='C0_complement_vssl', choices=list(ARMS))
    parser.add_argument('--lambda_U', type=float, default=None,
                        help='default: the pre-registered value of --arm (0 for S0, else 1)')
    parser.add_argument('--tau_U', type=float, default=0.1)
    parser.add_argument('--rho', type=float, default=0.0)
    parser.add_argument('--soft_mask', type=float, default=0.0)
    parser.add_argument('--lambda_align', type=float, default=LAMBDA_ALIGN)
    parser.add_argument('--lambda_sparse', type=float, default=LAMBDA_SPARSE)
    parser.add_argument('--ddp_gradient_averaging', type=int, default=1,
                        help='1 (default): the CVSSL term is scaled by the world size and the '
                             'module is DDP-wrapped, so its backward value is the global mean')
    parser.add_argument('--duplicate_policy', default='exclude', choices=['exclude', 'none'])
    parser.add_argument('--base_model', default='B16')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--mask_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--log_scale', type=float, default=4.6052)
    parser.add_argument('--warmup_length', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--init_state', default=None)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--save_completed_steps', default='0')
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--grad_probe_steps', default='')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--amp_dtype', default='bf16', choices=['bf16', 'fp32'])
    parser.add_argument('--grad_checkpoint_views', type=int, default=1,
                        help='run both views through the transformer with activation '
                             'checkpointing (memory only; the objective math is unchanged, and '
                             'the setting is identical for all four arms)')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--view_b_off', type=int, default=0,
                        help='diagnostic only: 1 disables the view-b augmentation')
    args = parser.parse_args()
    if args.lambda_U is None:
        args.lambda_U = ARM_LAMBDA_U[args.arm]
    if args.base_model == 'B16':
        args.base_model = 'ViT-B/16'
    elif args.base_model == 'L14':
        args.base_model = 'ViT-L/14'

    seed_everything(args.seed)
    rank, local_rank, world = setup_distributed()
    device = torch.device('cuda', local_rank)
    save_completed = sorted({int(s) for s in args.save_completed_steps.split(',') if s.strip()})
    grad_probe_steps = {int(s) for s in args.grad_probe_steps.split(',') if s.strip()}

    model, preprocess = longclip.load_from_clip(args.base_model, device='cpu',
                                               download_root=None, args=args)
    model.train()
    model.logit_scale = torch.nn.Parameter(torch.ones([]) * args.log_scale)
    model = model.to(device)
    if args.init_state:
        load_init_state(model, args.init_state, rank)
    initial_digest = state_digest(model.state_dict())

    train_module = SaidClsCvsslTrainModule(
        model, rank=rank, arm=args.arm, lambda_u=args.lambda_U, tau_u=args.tau_U, rho=args.rho,
        soft_mask=bool(args.soft_mask), lambda_align=args.lambda_align,
        lambda_sparse=args.lambda_sparse, duplicate_policy=args.duplicate_policy,
        ddp_gradient_averaging=bool(args.ddp_gradient_averaging),
        grad_checkpoint_views=bool(args.grad_checkpoint_views)).to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        train_module, device_ids=[local_rank], output_device=local_rank,
        find_unused_parameters=True)
    ddp_model._set_static_graph()
    model = train_module.clip                       # read-only handle (checkpoints, logging)

    optimizer, mask_optimizer, n_backbone, n_mask = build_optimizers(model, args)
    use_amp = args.amp_dtype == 'bf16'
    amp_dtype = torch.bfloat16

    # real dtype audit, not config metadata: every trainable parameter and every AdamW group
    dtype_counts = {}
    for parameter in model.parameters():
        key = str(parameter.dtype)
        dtype_counts[key] = dtype_counts.get(key, 0) + 1
    group_dtypes = {}
    for name, group in (('backbone', optimizer), ('mask_net', mask_optimizer)):
        group_dtypes[name] = sorted({str(p.dtype) for p in group.param_groups[0]['params']})
    bf16_count = sum(1 for value in model.state_dict().values()
                     if torch.is_tensor(value) and value.dtype == torch.bfloat16)
    print('DTYPE_AUDIT rank=%d parameter_dtypes=%s backbone_group=%s mask_group=%s '
          'bf16_tensors_in_model=%d amp_enabled=%s amp_dtype=%s'
          % (rank, sorted(dtype_counts.items()), group_dtypes['backbone'],
             group_dtypes['mask_net'], bf16_count, use_amp, args.amp_dtype), flush=True)
    if dtype_counts != {'torch.float32': len(list(model.parameters()))}:
        raise RuntimeError('fp32 master weights expected, got %r' % (dtype_counts,))
    if group_dtypes != {'backbone': ['torch.float32'], 'mask_net': ['torch.float32']}:
        raise RuntimeError('optimizer groups must be fp32, got %r' % (group_dtypes,))


    dataset = Share4VCvsslDataset(seed=args.seed, augment_view_b=not bool(args.view_b_off),
                                 strict_manifest=os.environ.get('SHARE4V_FULL_AUDIT'))
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True,
                                                             seed=args.seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                                         num_workers=args.num_workers, pin_memory=True,
                                         collate_fn=cvssl_collate, drop_last=False)
    steps_per_epoch = len(loader)
    total_steps = args.epochs * steps_per_epoch
    scheduler = cosine_lr(optimizer, base_lr=args.lr, warmup_length=args.warmup_length,
                          steps=total_steps)
    mask_scheduler = cosine_lr(mask_optimizer, base_lr=args.mask_lr, warmup_length=0,
                               steps=total_steps)

    objective = ddp_model          # training ALWAYS goes through the DDP forward
    objective_module = train_module

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, 'salu_log.jsonl')
    config = {
        'objective': 'said_cls_cvssl',
        'arm': args.arm,
        'arm_mask': ARM_MASK[args.arm],
        'lambda_U': args.lambda_U,
        'tau_U': args.tau_U,
        'rho': args.rho,
        'mask_source': 'hard straight-through mask_net(C_S hidden) as used by the Said forward'
        if not args.soft_mask else 'soft mask_net(C_S hidden)',
        'candidate_scope': 'global batch, candidates detached',
        'stopgrad_rule': 'candidates detached; anchor live; m_U detached; both directions summed',
        'view_a': 'reference openai-clip _transform(224)',
        'view_b': 'shared_content_weak_v1: same window, resample in [192,224] bilinear, '
                  'gaussian blur k=3 sigma~U[0.1,1.0]',
        'sampling': 'DistributedSampler(shuffle=True), no drop_last, steps_per_epoch=len(loader)',
        'precision': 'fp32 master + %s autocast' % args.amp_dtype,
        'lambda_align': args.lambda_align,
        'lambda_sparse': args.lambda_sparse,
        'lr': args.lr, 'mask_lr': args.mask_lr, 'weight_decay': args.weight_decay,
        'warmup_length': args.warmup_length, 'epochs': args.epochs,
        'batch_size_per_gpu': args.batch_size, 'world_size': world,
        'loader_batches': steps_per_epoch, 'lr_horizon_steps': total_steps,
        'seed': args.seed, 'init_state': args.init_state,
        'initial_state_sha256': initial_digest, 'git_head': git_head(),
        'tokenizer_context': TOKENIZER_CONTEXT,
        'grad_checkpoint_views': bool(args.grad_checkpoint_views),
    }
    if rank == 0:
        print('CONFIG ' + json.dumps(config, sort_keys=True), flush=True)
        print('DATASET_SIZE %d STEPS_PER_EPOCH %d LR_HORIZON %d BACKBONE_TENSORS %d '
              'MASK_TENSORS %d' % (len(dataset), steps_per_epoch, total_steps, n_backbone, n_mask),
              flush=True)
        with open(os.path.join(args.output_dir, 'config.json'), 'w') as handle:
            json.dump(config, handle, indent=2, sort_keys=True)

    if args.resume:
        payload = torch.load(args.resume, map_location='cpu', weights_only=False)
        if payload.get('config', {}).get('arm') != args.arm:
            raise ValueError('resume arm mismatch: %r vs %r'
                             % (payload.get('config', {}).get('arm'), args.arm))
        model.load_state_dict(payload['model'])
        optimizer.load_state_dict(payload['optimizer'])
        mask_optimizer.load_state_dict(payload['mask_optimizer'])
        start_step = int(payload['completed_steps'])
        if rank == 0:
            print('RESUMED from %s at completed step %d' % (args.resume, start_step), flush=True)
    else:
        start_step = 0

    caption_digest = hashlib.sha256()
    sample_digest = hashlib.sha256()
    view_digest = hashlib.sha256()
    image_id_digest = hashlib.sha256()
    view_pixels_sha = None
    completed = start_step
    t_start = time.time()
    compute_times = []
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
            mask_scheduler(completed)
            out = cvssl_train_step(
                ddp_model, batch, optimizer, mask_optimizer, device, amp_dtype,
                amp_enabled=use_amp,
                mask_rng_seed=stateless_seed(args.seed, 'mask_rng', completed, args.arm))
            image_a = batch['image_a']
            image_ids = batch['image_id'].to(device)
            completed += 1
            compute_times.append(time.time() - t0)

            caption_digest.update(('\n'.join(batch['caption_said'])).encode('utf-8'))
            sample_digest.update(batch['sample_id'].numpy().tobytes())
            image_id_digest.update(batch['image_id'].numpy().tobytes())
            view_digest.update(batch['view_b_resample_size'].numpy().tobytes())
            view_digest.update(batch['view_b_blur_sigma'].numpy().tobytes())
            if view_pixels_sha is None:
                # ``image_b`` only exists inside the step/probe helpers; the loop owns ``batch``.
                view_pixels_sha = tensor_digest(batch['image_b'][0])

            # The gradient probe re-runs the objective, whose loss contains distributed collectives:
            # it must therefore run on EVERY rank at the same step (a rank-0-only probe would hang
            # the others). Only rank 0 logs it.
            probe = grad_probe(objective_module, batch, device, args) \
                if completed in grad_probe_steps else None
            is_last = args.max_steps is not None and completed >= args.max_steps
            if rank == 0 and (completed % args.log_every == 0 or completed == 1 or is_last
                              or completed in save_completed):
                record = {
                    'completed_steps': completed,
                    'epoch': epoch,
                    'step_in_epoch': i,
                    'arm': args.arm,
                    'lr': optimizer.param_groups[0]['lr'],
                    'mask_lr': mask_optimizer.param_groups[0]['lr'],
                    'batch_size_local': int(image_a.shape[0]),
                    'global_pairs': int(image_a.shape[0]) * world,
                    'global_image_views': int(image_a.shape[0]) * world * 2,
                    'world_size': world,
                    'lambda_U': args.lambda_U,
                    'ddp_gradient_averaging': bool(args.ddp_gradient_averaging),
                    'grad_checkpoint_views': bool(args.grad_checkpoint_views),
                    'global_valid_ab': float(out.get('vssl_valid_ab', 0.0)),
                    'global_valid_ba': float(out.get('vssl_valid_ba', 0.0)),
                    'rank_param_digest': state_digest(model.state_dict())[:16],
                    'sec_per_step': compute_times[-1],
                    'samples_per_sec': float(image_a.shape[0]) * world / max(compute_times[-1], 1e-9),
                    'peak_memory_gb': torch.cuda.max_memory_allocated(device) / (1024 ** 3),
                    'batch_caption_sha256': hashlib.sha256(
                        ('\n'.join(batch['caption_said'])).encode('utf-8')).hexdigest()[:16],
                    'batch_image_id_sha256': tensor_digest(batch['image_id']),
                    'view_b_resample_mean': float(batch['view_b_resample_size'].float().mean()),
                    'view_b_sigma_mean': float(batch['view_b_blur_sigma'].mean()),
                }
                for key, value in out.items():
                    if torch.is_tensor(value) and value.numel() == 1:
                        record[key] = float(value.detach())
                    elif torch.is_tensor(value):
                        continue
                    else:
                        record[key] = value
                if probe:
                    record.update(probe)
                with open(log_path, 'a') as handle:
                    handle.write(json.dumps(record, sort_keys=True) + '\n')
                print('LOG ' + json.dumps(record, sort_keys=True), flush=True)

            if rank == 0 and completed in save_completed:
                payload = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'mask_optimizer': mask_optimizer.state_dict(),
                    'completed_steps': completed,
                    'epoch': epoch,
                    'step_in_epoch': i,
                    'phase': 'said-cls-cvssl-v0.1',
                    'objective': 'said_cls_cvssl',
                    'config': config,
                    'precision': 'fp32 master + %s autocast' % args.amp_dtype,
                    'ddp': {'wrapped': True, 'find_unused_parameters': True, 'static_graph': True,
                            'world_size': world, 'gradient_averaging_scaled_u': bool(
                                args.ddp_gradient_averaging)},
                    'lr_horizon_steps': total_steps,
                    'git_head': config['git_head'],
                    'digests': {
                        'initial_state_sha256': initial_digest,
                        'caption_stream_sha256': caption_digest.hexdigest(),
                        'sample_stream_sha256': sample_digest.hexdigest(),
                        'image_id_stream_sha256': image_id_digest.hexdigest(),
                        'view_b_param_stream_sha256': view_digest.hexdigest(),
                        'view_b_pixels_sha256_step0': view_pixels_sha,
                    },
                    'rng': {'torch': torch.get_rng_state(),
                            'cuda': torch.cuda.get_rng_state(device)
                            if torch.cuda.is_available() else None},
                }
                path = os.path.join(args.output_dir, 'cvssl_%s_step%06d.pt' % (args.arm,
                                                                               completed))
                torch.save(payload, path)
                print('SAVED ' + path, flush=True)
        if stopped:
            break

    if rank == 0:
        summary = {
            'arm': args.arm, 'completed_steps': completed, 'epochs': args.epochs,
            'wall_sec': time.time() - t_start,
            'mean_sec_per_step': sum(compute_times) / max(len(compute_times), 1),
            'steps_per_epoch': steps_per_epoch, 'lr_horizon_steps': total_steps,
            'initial_state_sha256': initial_digest,
            'caption_stream_sha256': caption_digest.hexdigest(),
            'sample_stream_sha256': sample_digest.hexdigest(),
            'view_b_param_stream_sha256': view_digest.hexdigest(),
            'view_b_pixels_sha256_step0': view_pixels_sha,
            'peak_memory_gb': torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        }
        with open(os.path.join(args.output_dir, 'run_summary.json'), 'w') as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print('RUN_SUMMARY ' + json.dumps(summary, sort_keys=True), flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
