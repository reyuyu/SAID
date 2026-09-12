"""SAID-C1-TCR v0.1 trainer -- ``objective = said_cls_tcr``.

Reuses, verbatim, from the existing CVSSL trainer: ``seed_everything``, ``state_digest``,
``tensor_digest``, ``setup_distributed``, ``git_head``, ``load_init_state``, ``build_optimizers``
(the same two AdamW groups for the student backbone / ``mask_net``) and the cosine LR scheduler.
Only the objective, the third optimizer group (the decoder) and the checkpoint payload are new.

Budget note, recorded rather than hidden: per sample C1 encodes the student image once (with graph),
the student text once, the frozen reference image once (no grad) and runs one small MLP. That is NOT
the same FLOPs as the C0 two-view step, which encodes two student views and runs the U contrastive
term; the two are only comparable because the step budget, data and init are matched.
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
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import longclip  # noqa: E402
from model.said_cls_reconstruction import (ARM, C1TrainModule, DEFAULT_TAU_REC,  # noqa: E402
                                           reference_fingerprint)
from said_cvssl_data import Share4VCvsslDataset, cvssl_collate, stateless_seed  # noqa: E402
from scheduler import cosine_lr  # noqa: E402
from train_said_cls_cvssl import (build_optimizers, git_head, load_init_state,  # noqa: E402
                                  seed_everything, setup_distributed, state_digest, tensor_digest)

TOKENIZER_CONTEXT = 248


def inspected_module(model):
    """The underlying module: works for a DDP wrapper and for the bare module."""
    return getattr(model, 'module', model)


def build_decoder_optimizer(module, args):
    parameters = [p for p in module.decoder.parameters() if p.requires_grad]
    assert parameters, 'the decoder has no trainable parameter'
    return torch.optim.AdamW(parameters, lr=args.decoder_lr, weight_decay=args.decoder_wd)


def c1_train_step(ddp_model, batch, optimizer, mask_optimizer, decoder_optimizer, device,
                  amp_dtype, amp_enabled=True, world_size=1, capture_grads=False,
                  completed_steps=0):
    """One real training step: DDP forward -> backward -> all three optimizer steps.

    Reconstruction scaling: the objective returns the local *mean* over this rank's valid samples.
    Under standard DDP averaging the backward value must be ``W * S_r / max(V, 1)`` so that the
    averaged parameter gradient equals the gradient of the global mean. Since ``V = sum_r v_r`` and
    the mean is ``S_r / v_r`` when the rank is the only contributor, the local mean has to be
    rescaled by ``W * v_r / V``; with all samples valid and equal rank batches that factor is exactly
    1, i.e. a plain local mean (never multiplied by W a second time).
    """
    image_a = batch['image_a'].to(device, non_blocking=True)
    image_b = batch['image_b'].to(device, non_blocking=True)
    with torch.no_grad():
        text = longclip.tokenize(batch['caption_said'], truncate=True).to(device)

    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
        out = ddp_model(image_a, text, image_b, completed_steps=completed_steps)

    rec_local = out['loss_rec']                       # mean over THIS rank's valid samples
    valid_local = out['rec_valid_count_tensor']       # tensor, so autograd/dtype stay intact
    local_batch = int(out['m_u'].shape[0])
    distributed = bool(world_size > 1 and dist.is_initialized())

    if distributed:
        totals = torch.stack([valid_local.detach(), out['loss_rec_sum'].detach()])
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        valid_global, global_sum = totals[0], totals[1]
    else:
        valid_global, global_sum = valid_local.detach(), out['loss_rec_sum'].detach()

    # DDP averages the parameter gradients over ranks, so the local backward value has to be
    # W * S_r / max(V, 1), weighted by lambda_rec. With equal rank batches and every sample valid the
    # scale factor is exactly 1, i.e. a plain local mean (never multiplied by W a second time).
    # Taking lambda_rec into account here is essential: the module hands back the *unweighted* local
    # mean, and dropping the weight would silently train the decoder at weight 1 regardless of the
    # configured lambda_rec.
    scale = float(world_size) * valid_local.detach() / valid_global.clamp_min(1.0)
    rec_backward = out['lambda_rec'] * rec_local * scale

    loss = out['loss_smart'] + rec_backward
    loss.backward()

    if capture_grads:
        module = inspected_module(ddp_model)
        out['_grads'] = {name: (None if p.grad is None else p.grad.detach().clone())
                         for name, p in module.named_parameters()}

    optimizer.step()
    mask_optimizer.step()
    decoder_optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    mask_optimizer.zero_grad(set_to_none=True)
    decoder_optimizer.zero_grad(set_to_none=True)

    rec_global = global_sum / valid_global.clamp_min(1.0) if float(valid_global) > 0 \
        else global_sum * 0.0

    out['loss_rec_global_mean'] = rec_global.detach()
    out['loss_rec_backward_local'] = rec_backward.detach()
    out['rec_valid_count'] = float(valid_local)
    out['rec_valid_global'] = float(valid_global)
    out['rec_valid_fraction'] = float(valid_local) / max(local_batch, 1)
    out['rec_backward_scale'] = torch.as_tensor(scale).detach()
    out['loss_total_for_backward_local'] = loss.detach()
    return out


def main():
    parser = argparse.ArgumentParser(description='SAID-C1-TCR v0.1')
    parser.add_argument('--arm', default=ARM, choices=[ARM])
    parser.add_argument('--objective', default='said_cls_tcr', choices=['said_cls_tcr'])
    parser.add_argument('--lambda_rec', type=float, default=DEFAULT_TAU_REC,
                        help='fixed constant in this version; no warmup, no dynamic balancing')
    parser.add_argument('--decoder_hidden', type=int, default=512)
    parser.add_argument('--decoder_lr', type=float, default=1e-4)
    parser.add_argument('--decoder_wd', type=float, default=0.0)
    parser.add_argument('--decoder_seed', type=int, default=0)
    parser.add_argument('--lambda_align', type=float, default=10.0)
    parser.add_argument('--lambda_sparse', type=float, default=2.0)
    parser.add_argument('--duplicate_policy', default='exclude', choices=['exclude', 'none'])
    parser.add_argument('--base_model', default='B16')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--mask_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--warmup_length', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--init_state', required=True,
                        help='the shared student init; the frozen reference is copied from the '
                             'student *after* this load, never from a trained checkpoint')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--save_completed_steps', default='0')
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--diag_every', type=int, default=0,
                        help='>0: run the input-dependency diagnostics every N steps (rank 0 only)')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--amp_dtype', default='bf16', choices=['bf16', 'fp32'])
    parser.add_argument('--grad_checkpoint_views', type=int, default=1)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--variant', default='C1', choices=['C1', 'C1-UN', 'C1-VWarm'])
    parser.add_argument('--reference_state_out', default=None,
                        help='optional path for the frozen reference state copy')
    args = parser.parse_args()
    if args.base_model == 'B16':
        args.base_model = 'ViT-B/16'
    elif args.base_model == 'L14':
        args.base_model = 'ViT-L/14'

    seed_everything(args.seed)
    rank, local_rank, world = setup_distributed()
    device = torch.device('cuda', local_rank)
    save_completed = sorted({int(s) for s in args.save_completed_steps.split(',') if s.strip()})

    model, preprocess = longclip.load_from_clip(args.base_model, device='cpu',
                                               download_root=None, args=args)
    model.train()
    model.logit_scale = torch.nn.Parameter(torch.ones([]) * 4.6052)
    model = model.to(device)
    # 1. build and STRICTLY load the shared student init, in fp32 master precision
    load_init_state(model, args.init_state, rank)
    initial_digest = state_digest(model.state_dict())

    train_module = C1TrainModule(model, rank=rank, lambda_rec=args.lambda_rec,
                                lambda_align=args.lambda_align,
                                lambda_sparse=args.lambda_sparse,
                                decoder_hidden=args.decoder_hidden,
                                decoder_seed=args.decoder_seed,
                                duplicate_policy=args.duplicate_policy,
                                ddp_gradient_averaging=True,
                                grad_checkpoint_views=bool(args.grad_checkpoint_views),
                                reconstruction_variant=args.variant).to(device)
    # 2. the reference was deep-copied from the student *after* the init load
    reference_digest = reference_fingerprint(train_module.reference_visual)
    reference_zero_storage_overlap = not any(
        a.data_ptr() == b.data_ptr()
        for a in train_module.clip.visual.parameters()
        for b in train_module.reference_visual.parameters())
    if not reference_zero_storage_overlap:
        raise RuntimeError('the frozen reference shares storage with the student visual tower')

    ddp_model = torch.nn.parallel.DistributedDataParallel(
        train_module, device_ids=[local_rank], output_device=local_rank,
        find_unused_parameters=True)
    ddp_model._set_static_graph()
    clip_model = train_module.clip
    decoder = train_module.decoder

    optimizer, mask_optimizer, n_backbone, n_mask = build_optimizers(clip_model, args)
    decoder_optimizer = build_decoder_optimizer(train_module, args)
    n_decoder = sum(p.numel() for p in decoder.parameters())
    use_amp = args.amp_dtype == 'bf16'
    amp_dtype = torch.bfloat16

    dtype_counts = {}
    for parameter in clip_model.parameters():
        key = str(parameter.dtype)
        dtype_counts[key] = dtype_counts.get(key, 0) + 1
    for parameter in decoder.parameters():
        key = 'decoder:' + str(parameter.dtype)
        dtype_counts[key] = dtype_counts.get(key, 0) + 1
    reference_dtypes = sorted({str(p.dtype) for p in train_module.reference_visual.parameters()})
    reference_grads = sorted({bool(p.requires_grad) for p in train_module.reference_visual.parameters()})
    print('DTYPE_AUDIT rank=%d parameter_dtypes=%s reference_dtypes=%s reference_requires_grad=%s '
          'decoder_params=%d amp_enabled=%s amp_dtype=%s reference_eval=%s'
          % (rank, sorted(dtype_counts.items()), reference_dtypes, reference_grads, n_decoder,
             use_amp, args.amp_dtype, not train_module.reference_visual.training), flush=True)

    dataset = Share4VCvsslDataset(seed=args.seed, augment_view_b=True,
                                  strict_manifest=os.environ.get('SHARE4V_FULL_AUDIT'))
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, seed=args.seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                                        num_workers=args.num_workers, pin_memory=True,
                                        collate_fn=cvssl_collate, drop_last=False)
    steps_per_epoch = len(loader)
    total_steps = args.epochs * steps_per_epoch
    scheduler = cosine_lr(optimizer, base_lr=args.lr, warmup_length=args.warmup_length,
                          steps=total_steps)
    mask_scheduler = cosine_lr(mask_optimizer, base_lr=args.mask_lr, warmup_length=0,
                               steps=total_steps)
    decoder_scheduler = cosine_lr(decoder_optimizer, base_lr=args.decoder_lr, warmup_length=0,
                                  steps=total_steps)

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, 'salu_log.jsonl')
    reference_path = args.reference_state_out or os.path.join(args.output_dir,
                                                             'reference_visual_state.pt')
    config = {
        'objective': 'said_cls_tcr',
        'variant': args.variant,
        'arm': ARM,
        'lambda_rec': args.lambda_rec,
        'lambda_align': args.lambda_align,
        'lambda_sparse': args.lambda_sparse,
        'decoder': 'Linear(2*512->512) + GELU + Linear(512->512), no dropout/LN/residual/out-norm',
        'decoder_hidden': args.decoder_hidden,
        'decoder_lr': args.decoder_lr,
        'decoder_wd': args.decoder_wd,
        'decoder_seed': args.decoder_seed,
        'reconstruction_input': 'r_U = Norm(v) * m_U  (normalise the full vector first, no re-norm); '
                                't_cond = sg(Norm(t_raw)); pred = D([r_U; t_cond])',
        'reconstruction_target': 'g0 = Norm(frozen_initial_visual(I_a)) in fp32, no_grad',
        'reference_source': 'deep copy of the student visual AFTER loading --init_state %s'
                            % args.init_state,
        'mask_source': 'hard straight-through mask_net(C_S hidden), rho = 0',
        'duplicate_policy': args.duplicate_policy,
        'view_a': 'reference openai-clip _transform(224)',
        'view_b': 'produced by the loader but NOT used by C1 (no second view, no CVSSL term)',
        'sampling': 'DistributedSampler(shuffle=True), no drop_last, steps_per_epoch=len(loader)',
        'precision': 'fp32 master + %s autocast for the student; fp32 frozen reference and fp32 '
                     'reconstruction core' % args.amp_dtype,
        'batch_size_per_gpu': args.batch_size,
        'global_pairs': args.batch_size * world,
        'world_size': world,
        'epochs': args.epochs,
        'loader_batches': steps_per_epoch,
        'lr_horizon_steps': total_steps,
        'warmup_length': args.warmup_length,
        'grad_checkpoint_views': bool(args.grad_checkpoint_views),
        'init_state': args.init_state,
        'initial_state_sha256': initial_digest,
        'reference_visual_state_sha256': reference_digest,
        'reference_state_path': reference_path,
        'git_head': git_head(),
        'tokenizer_context': TOKENIZER_CONTEXT,
    }
    if rank == 0:
        print('CONFIG ' + json.dumps(config, sort_keys=True), flush=True)
        print('DATASET_SIZE %d STEPS_PER_EPOCH %d LR_HORIZON %d BACKBONE_TENSORS %d MASK_TENSORS %d '
              'DECODER_PARAMS %d' % (len(dataset), steps_per_epoch, total_steps, n_backbone, n_mask,
                                     n_decoder), flush=True)
        with open(os.path.join(args.output_dir, 'config.json'), 'w') as handle:
            json.dump(config, handle, indent=2, sort_keys=True)
        torch.save({'reference_visual': train_module.reference_visual.state_dict(),
                    'sha256': reference_digest, 'source': 'initial visual tower',
                    'init_state': args.init_state, 'initial_state_sha256': initial_digest},
                   reference_path)
        print('SAVED_REFERENCE ' + reference_path, flush=True)

    if args.resume:
        payload = torch.load(args.resume, map_location='cpu', weights_only=False)
        if payload.get('config', {}).get('arm') != ARM:
            raise ValueError('resume arm mismatch')
        clip_model.load_state_dict(payload['model'])
        decoder.load_state_dict(payload['reconstruction_decoder'])
        optimizer.load_state_dict(payload['optimizer'])
        mask_optimizer.load_state_dict(payload['mask_optimizer'])
        decoder_optimizer.load_state_dict(payload['decoder_optimizer'])
        start_step = int(payload['completed_steps'])
        start_epoch = int(payload.get('epoch', 0))
        start_batch = int(payload.get('next_batch_index', int(payload['step_in_epoch']) + 1))
        if start_step != start_epoch * steps_per_epoch + start_batch:
            raise ValueError('resume cursor disagrees with completed steps')
        if payload['lr_horizon_steps'] != total_steps:
            raise ValueError('resume must preserve LR horizon')
        reference_saved = torch.load(payload['reference_state_path'], map_location='cpu', weights_only=False)
        train_module.reference_visual.load_state_dict(reference_saved['reference_visual'], strict=True)
        if reference_fingerprint(train_module.reference_visual) != reference_digest or reference_digest != payload['reference_visual_state_sha256']:
            raise ValueError('resume frozen reference fingerprint mismatch')
        if payload.get('config', {}).get('variant', args.variant) != args.variant:
            raise ValueError('resume variant mismatch')
        if rank == 0:
            print('RESUMED from %s at %d' % (args.resume, start_step), flush=True)
    else:
        start_step = 0
        start_epoch = 0
        start_batch = 0

    caption_digest = hashlib.sha256()
    sample_digest = hashlib.sha256()
    view_digest = hashlib.sha256()
    view_a_pixels_sha = None
    completed = start_step
    t_start = time.time()
    compute_times = []
    stopped = False
    replay_caption = hashlib.sha256()
    replay_sample = hashlib.sha256()
    replay_verified = False

    for epoch in range(args.epochs):
        if epoch < start_epoch:
            continue
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        for i, batch in enumerate(loader):
            if epoch == start_epoch and i < start_batch:
                replay_caption.update(('\n'.join(batch['caption_said'])).encode('utf-8'))
                replay_sample.update(batch['sample_id'].numpy().tobytes())
                continue
            if args.resume and not replay_verified:
                if rank == 0:
                    expected = payload['digests']
                    if replay_sample.hexdigest() != expected['sample_stream_sha256']:
                        raise RuntimeError('resume sample replay mismatch')
                    if replay_caption.hexdigest() != expected['caption_stream_sha256']:
                        raise RuntimeError('resume caption replay mismatch')
                    print('REPLAY_VERIFIED epoch=%d next_batch=%d next_update=%d caption_sha=%s sample_sha=%s; model RNG at checkpoint not saved, bitwise full resume NOT CLAIMED' % (epoch, i, completed + 1, replay_caption.hexdigest(), replay_sample.hexdigest()), flush=True)
                replay_verified = True
            if args.max_steps is not None and completed >= args.max_steps:
                stopped = True
                break
            t0 = time.time()
            scheduler(completed)
            mask_scheduler(completed)
            decoder_scheduler(completed)
            out = c1_train_step(ddp_model, batch, optimizer, mask_optimizer, decoder_optimizer,
                                device, amp_dtype, amp_enabled=use_amp, world_size=world,
                                completed_steps=completed)
            image_a = batch['image_a']
            completed += 1
            compute_times.append(time.time() - t0)

            caption_digest.update(('\n'.join(batch['caption_said'])).encode('utf-8'))
            sample_digest.update(batch['sample_id'].numpy().tobytes())
            view_digest.update(batch['view_b_resample_size'].numpy().tobytes())
            view_digest.update(batch['view_b_blur_sigma'].numpy().tobytes())
            if view_a_pixels_sha is None:
                view_a_pixels_sha = tensor_digest(batch['image_a'][0])

            is_last = args.max_steps is not None and completed >= args.max_steps
            if rank == 0 and (completed % args.log_every == 0 or completed == 1 or is_last
                              or completed in save_completed):
                record = {
                    'completed_steps': completed,
                    'epoch': epoch, 'next_batch_index': i + 1,
                    'epoch': epoch,
                    'next_batch_index': i + 1,
                    'step_in_epoch': i,
                    'arm': ARM,
                    'objective': 'said_cls_tcr',
                    'lr': optimizer.param_groups[0]['lr'],
                    'mask_lr': mask_optimizer.param_groups[0]['lr'],
                    'decoder_lr': decoder_optimizer.param_groups[0]['lr'],
                    'lambda_rec': args.lambda_rec,
                    'batch_size_local': int(image_a.shape[0]),
                    'global_pairs': int(image_a.shape[0]) * world,
                    'world_size': world,
                    'reference_visual_state_sha256': reference_digest,
                    'rank_param_digest': state_digest(clip_model.state_dict())[:16],
                    'rank_decoder_digest': state_digest(decoder.state_dict())[:16],
                    'sec_per_step': compute_times[-1],
                    'samples_per_sec': float(image_a.shape[0]) * world / max(compute_times[-1],
                                                                            1e-9),
                    'peak_memory_gb': torch.cuda.max_memory_allocated(device) / (1024 ** 3),
                    'batch_caption_sha256': hashlib.sha256(
                        ('\n'.join(batch['caption_said'])).encode('utf-8')).hexdigest()[:16],
                }
                for key, value in out.items():
                    if torch.is_tensor(value) and value.numel() == 1:
                        record[key] = float(value.detach())
                if args.diag_every > 0 and (completed == 1 or completed in save_completed):
                    record.update(input_dependency_diagnostics(ddp_model, batch, device))
                with open(log_path, 'a') as handle:
                    handle.write('LOG ' + json.dumps(record, sort_keys=True) + '\n')
                print('LOG ' + json.dumps(record, sort_keys=True), flush=True)

            if rank == 0 and completed in save_completed:
                payload = {
                    'model': clip_model.state_dict(),
                    'reconstruction_decoder': decoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'mask_optimizer': mask_optimizer.state_dict(),
                    'decoder_optimizer': decoder_optimizer.state_dict(),
                    'completed_steps': completed,
                    'epoch': epoch,
                    'step_in_epoch': i,
                    'next_batch_index': i + 1,
                    'resume_parent': args.resume,
                    'phase': 'said-c1-tcr-v0.1',
                    'objective': 'said_cls_tcr',
                    'config': config,
                    'precision': 'fp32 master + %s autocast' % args.amp_dtype,
                    'lr_horizon_steps': total_steps,
                    'git_head': config['git_head'],
                    'reference_visual_state_sha256': reference_digest,
                    'reference_state_path': reference_path,
                    'digests': {
                        'initial_state_sha256': initial_digest,
                        'caption_stream_sha256': caption_digest.hexdigest(),
                        'sample_stream_sha256': sample_digest.hexdigest(),
                        'view_b_param_stream_sha256': view_digest.hexdigest(),
                        'view_a_pixels_sha256_step0': view_a_pixels_sha,
                    },
                }
                path = os.path.join(args.output_dir, 'c1_%s_step%06d.pt' % (ARM, completed))
                torch.save(payload, path)
                print('SAVED ' + path, flush=True)
        if stopped:
            break

    if rank == 0:
        final_reference_digest = reference_fingerprint(train_module.reference_visual)
        summary = {
            'arm': ARM, 'objective': 'said_cls_tcr', 'completed_steps': completed,
            'epochs': args.epochs, 'wall_sec': time.time() - t_start,
            'mean_sec_per_step': sum(compute_times) / max(len(compute_times), 1),
            'steps_per_epoch': steps_per_epoch, 'lr_horizon_steps': total_steps,
            'initial_state_sha256': initial_digest,
            'reference_visual_state_sha256': reference_digest,
            'reference_visual_state_sha256_final': final_reference_digest,
            'reference_unchanged': reference_digest == final_reference_digest,
            'caption_stream_sha256': caption_digest.hexdigest(),
            'sample_stream_sha256': sample_digest.hexdigest(),
            'view_a_pixels_sha256_step0': view_a_pixels_sha,
            'peak_memory_gb': torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        }
        with open(os.path.join(args.output_dir, 'run_summary.json'), 'w') as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print('RUN_SUMMARY ' + json.dumps(summary, sort_keys=True), flush=True)
        if not summary['reference_unchanged']:
            print('REFERENCE_CHANGED (this is a hard error)', flush=True)
            sys.exit(3)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def input_dependency_diagnostics(ddp_model, batch, device, n_images=64):
    """Cheap no_grad input-intervention table for the trained decoder (rank 0, evaluation only).

    The mask is computed ONCE and reused for every variant, so the interventions change only the
    decoder inputs and never the mask. Permutations are fixed derangements (no self-match).
    """
    module = inspected_module(ddp_model)
    clip = module.clip
    image_a = batch['image_a'][:n_images].to(device)
    text = longclip.tokenize(list(batch['caption_said'])[:n_images], truncate=True).to(device)
    with torch.no_grad():
        with torch.autocast(device_type=device.type, enabled=False):
            v = clip.encode_image(image_a.float()).float()
            g = F.normalize(v, dim=-1, eps=1e-6)
            t_raw, hidden = clip.encode_text(text, return_full=True)
            from model.said_cls_cvssl import said_mask_from_hidden
            m_s, _, _ = said_mask_from_hidden(clip.mask_net, hidden, soft_mask=False)
            m_u = 1.0 - m_s.detach().float()
            r_u = g * m_u
            t_cond = F.normalize(t_raw.float(), dim=-1, eps=1e-6)
            ref_raw = module.reference_visual(image_a.float())
            target = F.normalize(ref_raw.float(), dim=-1, eps=1e-6)
            rows = len(image_a)
            valid = m_u.sum(dim=-1) > 0
            perm = torch.arange(rows, device=device).roll(1)
            variants = {
                'D(r_U,t)': (r_u, t_cond),
                'D(r_U,0)': (r_u, torch.zeros_like(t_cond)),
                'D(0,t)': (torch.zeros_like(r_u), t_cond),
                'D(r_U,t_perm)': (r_u, t_cond[perm]),
                'D(r_U_perm,t)': (r_u[perm], t_cond),
            }
            errors = {}
            for name, (left, right) in variants.items():
                pred = module.decoder(left, right)
                per_sample = (pred - target).square().sum(dim=-1)
                errors[name] = float(per_sample[valid].mean()) if bool(valid.any()) else float('nan')
            mean_target = target[valid].mean(dim=0) if bool(valid.any()) else target.mean(dim=0)
            denominator = float((target - mean_target).square().sum(dim=-1)[valid].sum())
            pred = module.decoder(r_u, t_cond)
            numerator = float((pred - target).square().sum(dim=-1)[valid].sum())
            r2 = (1.0 - numerator / denominator) if denominator > 1e-12 else float('nan')
            pred_variance = float(pred[valid].var(dim=0).mean()) if bool(valid.any()) else float('nan')
    base = errors['D(r_U,t)']
    return {
        'diag_n_images': rows,
        'diag_rec_error_normal': base,
        'diag_rec_error_zero_text': errors['D(r_U,0)'],
        'diag_rec_error_zero_u': errors['D(0,t)'],
        'diag_rec_error_permuted_text': errors['D(r_U,t_perm)'],
        'diag_rec_error_permuted_u': errors['D(r_U_perm,t)'],
        'diag_delta_zero_text': errors['D(r_U,0)'] - base,
        'diag_delta_zero_u': errors['D(0,t)'] - base,
        'diag_delta_permuted_text': errors['D(r_U,t_perm)'] - base,
        'diag_delta_permuted_u': errors['D(r_U_perm,t)'] - base,
        'diag_constant_baseline_error': float(
            (target - mean_target).square().sum(dim=-1)[valid].mean()) if bool(valid.any()) else float('nan'),
        'diag_feature_r2': r2,
        'diag_pred_variance': pred_variance,
    }


if __name__ == '__main__':
    main()
