"""Phase 2 Said-only training script: L = lambda_global * L_global + lambda_said * L_said.

This script is independent from ``train/train.py`` (the official SmartCLIP
baseline), which stays untouched and fully runnable.

It trains exactly two objectives:
  1. global CLIP symmetric InfoNCE on (z_global, t)
  2. Said symmetric InfoNCE on (z_s, t), where z_s is the caption-conditioned
     Said visual representation produced by the Said router.

Nothing else is trained: no prototype bank, no unsaid explorer, no sparsity /
entropy / overlap / reconstruction loss. SmartCLIP's ``mask_net`` is kept for
checkpoint compatibility but frozen and excluded from the optimizer.

Example (4 GPUs, 676K no-SAM subset)::

    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
    SHARE4V_DATA_ROOT=/root/datasets/ShareGPT4V \
    SHARE4V_JSON=debug/share4v_smoke_nosam.json \
    COCO_DATA_ROOT=/root/datasets/coco \
    torchrun --nproc_per_node=4 --master_port=25961 train/train_salu.py \
        --base_model B16 --batch_size 256 --max_steps 100 \
        --backbone_lr 1e-6 --head_lr 1e-4 \
        --lambda_global 1.0 --lambda_said 1.0 --tau_said 0.07
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TRAIN_DIR)
for _p in (REPO_ROOT, TRAIN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import longclip  # noqa: E402
from model.salu_model import SALUModel  # noqa: E402
from sharegpt4v import share4v_train_dataset  # noqa: E402
from train_utils import eval_coco  # noqa: E402  (standard CLIP retrieval evaluation)


def _load_baseline_setup_distributed():
    """Reuse ``setup_distributed`` from ``train/train.py``.

    ``import train`` resolves to the ``train/`` package, so the baseline training
    module is loaded explicitly from its file path instead of shadowing it.
    """
    import importlib.util

    baseline_path = os.path.join(TRAIN_DIR, 'train.py')
    spec = importlib.util.spec_from_file_location('smartclip_baseline_train', baseline_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.setup_distributed


setup_distributed = _load_baseline_setup_distributed()


def lr_scale(step: int, warmup_length: int, total_steps: int) -> float:
    """Cosine decay with linear warmup, expressed as a multiplier."""
    if warmup_length > 0 and step < warmup_length:
        return float(step + 1) / float(warmup_length)
    e = max(0, step - warmup_length)
    es = max(1, total_steps - warmup_length)
    return 0.5 * (1.0 + math.cos(math.pi * e / es))


def set_lrs(optimizer, base_lrs, step, warmup_length, total_steps) -> float:
    scale = lr_scale(step, warmup_length, total_steps)
    for group, base in zip(optimizer.param_groups, base_lrs):
        group['lr'] = base * scale
    return scale


def build_optimizer(model: SALUModel, backbone_lr, head_lr, weight_decay):
    backbone = model.backbone_parameters()
    head = model.said_head_parameters()
    if not head:
        raise RuntimeError('Said router has no trainable parameters')
    optimizer = torch.optim.AdamW(
        [
            {'params': backbone, 'lr': backbone_lr, 'weight_decay': weight_decay, 'name': 'backbone'},
            {'params': head, 'lr': head_lr, 'weight_decay': weight_decay, 'name': 'said_router'},
        ]
    )
    return optimizer, len(backbone), len(head)


def save_checkpoint(path, ddp_model, optimizer, scaler, step, epoch, args, base_lrs, total_steps):
    torch.save(
        {
            'model': ddp_model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'scheduler': {
                'type': 'cosine_with_warmup',
                'base_lrs': list(base_lrs),
                'warmup_length': args.warmup_length,
                'total_steps': total_steps,
                'step': step,
            },
            'step': step,
            'epoch': epoch,
            'args': vars(args),
            'phase': 'phase2-said-only',
        },
        path,
    )
    # a plain CLIP-only state dict so standard tooling (eval/retrieval/coco.py)
    # can load the backbone without knowing about SALU
    clip_path = os.path.join(os.path.dirname(path), 'clip_only_' + os.path.basename(path))
    torch.save(ddp_model.module.clip.state_dict(), clip_path)


def main():
    parser = argparse.ArgumentParser(description='Phase 2 Said-only training')
    parser.add_argument('--base_model', default='B16', help='B16 or L14')
    parser.add_argument('--batch_size', type=int, default=256, help='batch per GPU')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max_steps', type=int, default=None,
                        help='debug early stop; None = full run')
    parser.add_argument('--backbone_lr', type=float, default=1e-6)
    parser.add_argument('--head_lr', type=float, default=1e-4)
    parser.add_argument('--lambda_global', type=float, default=1.0)
    parser.add_argument('--lambda_said', type=float, default=1.0)
    parser.add_argument('--tau_said', type=float, default=0.07)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--warmup_length', type=int, default=200)
    parser.add_argument('--output_dir', default='runs_salu')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='save a checkpoint every N steps (0 = only at the end)')
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--download_root', default=None)
    parser.add_argument('--amp_dtype', default='bf16', choices=['bf16', 'fp16', 'fp32'],
                        help='autocast dtype; bf16 needs no loss scaling (default)')
    parser.add_argument('--scaler_init_scale', type=float, default=1024.0)
    parser.add_argument('--resume', default=None, help='SALU checkpoint to resume from')
    parser.add_argument('--eval_coco', action='store_true',
                        help='run standard CLIP COCO retrieval evaluation at the end (rank 0)')
    args = parser.parse_args()

    if args.base_model == 'B16':
        args.base_model = 'ViT-B/16'
    elif args.base_model == 'L14':
        args.base_model = 'ViT-L/14'

    rank, local_rank = setup_distributed()
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    world_size = dist.get_world_size()

    clip_model, preprocess = longclip.load_from_clip(
        args.base_model, device='cpu', download_root=args.download_root, args=args
    )
    salu = SALUModel(clip_model, tau_said=args.tau_said)
    salu = salu.to(device)
    ddp_model = DDP(salu, device_ids=[local_rank], find_unused_parameters=True)

    optimizer, n_backbone, n_head = build_optimizer(
        salu, args.backbone_lr, args.head_lr, args.weight_decay
    )
    base_lrs = [args.backbone_lr, args.head_lr]
    use_amp = args.amp_dtype != 'fp32'
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(args.amp_dtype == 'fp16'),
                                  init_scale=args.scaler_init_scale)

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print('args', vars(args), flush=True)
        print('world_size', world_size, 'backbone_tensors', n_backbone, 'said_head_tensors', n_head, flush=True)
        print('mask_net_trainable',
              any(p.requires_grad for p in salu.clip.mask_net.parameters()) if hasattr(salu.clip, 'mask_net') else 'n/a',
              flush=True)

    train_set = share4v_train_dataset()
    sampler = DistributedSampler(train_set, shuffle=True)
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    steps_per_epoch = len(loader)
    total_steps = args.max_steps if args.max_steps is not None else args.epochs * steps_per_epoch

    start_step, start_epoch = 0, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        ddp_model.module.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scaler' in ckpt and ckpt['scaler']:
            scaler.load_state_dict(ckpt['scaler'])
        start_step = int(ckpt.get('step', 0))
        start_epoch = int(ckpt.get('epoch', 0))
        if rank == 0:
            print('resumed from %s at step %d epoch %d' % (args.resume, start_step, start_epoch), flush=True)

    log_path = os.path.join(args.output_dir, 'salu_log.jsonl')
    ddp_model.train()
    step = start_step
    stopped = False
    t_start = time.time()
    step_times = []

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for images, texts in loader:
            if args.max_steps is not None and step >= args.max_steps:
                stopped = True
                break

            images = images.to(device, non_blocking=True)
            text_tokens = longclip.tokenize(texts, truncate=True).to(device)

            scale_factor = set_lrs(optimizer, base_lrs, step, args.warmup_length, total_steps)
            t0 = time.time()
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                out = ddp_model(images, text_tokens, args.lambda_global, args.lambda_said)
                loss = out['loss_total']
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step_times.append(time.time() - t0)

            is_last_step = args.max_steps is not None and step == args.max_steps - 1
            if rank == 0 and (step % args.log_every == 0 or step == 0 or is_last_step):
                record = {
                    'step': step,
                    'epoch': epoch,
                    'lr_scale': scale_factor,
                    'backbone_lr': optimizer.param_groups[0]['lr'],
                    'head_lr': optimizer.param_groups[1]['lr'],
                    'loss_global': float(out['loss_global'].detach()),
                    'loss_said': float(out['loss_said'].detach()),
                    'loss_total': float(out['loss_total'].detach()),
                    'said_attention_entropy': float(out['said_attention_entropy']),
                    'said_effective_patch_count': float(out['said_effective_patch_count']),
                    'said_attention_max': float(out['said_attention_max']),
                    'said_attention_min': float(out['said_attention_min']),
                    'said_feature_norm': float(out['said_feature_norm']),
                    'global_feature_norm': float(out['global_feature_norm']),
                    'sec_per_step_avg': sum(step_times) / len(step_times),
                }
                with open(log_path, 'a') as fp:
                    fp.write(json.dumps(record, sort_keys=True) + '\n')
                print('LOG ' + json.dumps(record, sort_keys=True), flush=True)

            step += 1
            if rank == 0 and args.save_every and step % args.save_every == 0:
                ckpt_path = os.path.join(args.output_dir, 'salu_said_only_step%06d.pt' % step)
                save_checkpoint(ckpt_path, ddp_model, optimizer, scaler, step, epoch, args, base_lrs, total_steps)
                print('SAVED ' + ckpt_path, flush=True)

        if stopped:
            break

    if rank == 0:
        final_path = os.path.join(args.output_dir, 'salu_said_only_last.pt')
        save_checkpoint(final_path, ddp_model, optimizer, scaler, step, epoch, args, base_lrs, total_steps)
        elapsed = time.time() - t_start
        steps_done = max(1, len(step_times))
        summary = {
            'steps': step - start_step,
            'wall_sec': elapsed,
            'sec_per_step': elapsed / steps_done,
            'samples_per_sec': args.batch_size * world_size / max(1e-9, sum(step_times) / len(step_times)),
            'final_checkpoint': final_path,
        }
        print('SUMMARY ' + json.dumps(summary, sort_keys=True), flush=True)
        with open(os.path.join(args.output_dir, 'salu_summary.json'), 'w') as fp:
            json.dump(summary, fp, indent=2, sort_keys=True)

        if args.eval_coco:
            ddp_model.eval()
            result = eval_coco(ddp_model, preprocess)
            print('COCO_RETRIEVAL ' + json.dumps(result, sort_keys=True), flush=True)
            with open(os.path.join(args.output_dir, 'coco_retrieval.json'), 'w') as fp:
                json.dump(result, fp, indent=2, sort_keys=True)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
