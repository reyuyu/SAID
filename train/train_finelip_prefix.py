"""Independent FP0 trainer; all clocks are explicit, no upstream runtime dependency."""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'train'))
from model import longclip
from model.finelip_prefix import FineLIPPrefix, eos_positions
from finelip_prefix_data import PrefixDataset
from finelip_prefix_state import group_size, lr_factor, validate_resume
from train_said_cls_cvssl import load_init_state, state_digest, seed_everything

UPSTREAM = '2118312c9d640c71904379e90129649a46e6f2dd'


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--init_state', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--resume')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--schedule_epochs', type=int, default=6)
    p.add_argument('--run_epochs', type=int, default=2)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    rank, world, local_rank = (int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    if (world, args.batch_size, args.accumulation, args.schedule_epochs, args.run_epochs) != (4, 32, 4, 6, 2):
        raise ValueError('FP0 formal configuration must be 4 x 32, accumulation 4, schedule 6, run 2')
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    device = torch.device('cuda', local_rank)
    seed_everything(args.seed)
    output = Path(args.output_dir)
    if rank == 0:
        if output.exists():
            raise FileExistsError('select a fresh output directory; resume also writes a new directory')
        output.mkdir(parents=True)
    dist.barrier()
    args.base_model = 'ViT-B/16'
    clip, _ = longclip.load_from_clip(args.base_model, device='cpu', args=args)
    load_init_state(clip, args.init_state, rank)
    initial_digest = state_digest(clip.state_dict())
    module = FineLIPPrefix(clip.float(), seed=args.seed).to(device)
    backbone = [x for x in module.clip.parameters() if x.requires_grad]
    extra = list(module.image_aggregator.parameters()) + list(module.text_aggregator.parameters())
    optimizer = torch.optim.AdamW([{'params': backbone, 'lr': 1e-6, 'base_lr': 1e-6},
                                  {'params': extra, 'lr': 2e-4, 'base_lr': 2e-4}],
                                 weight_decay=.01, betas=(.9, .999), eps=1e-8)
    ddp = DDP(module, device_ids=[local_rank], find_unused_parameters=False)
    dataset = PrefixDataset(seed=args.seed, augment_view_b=False,
                            strict_manifest=os.environ['SHARE4V_FULL_AUDIT'])
    sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=world, rank=rank,
                                                   shuffle=True, seed=args.seed, drop_last=False)
    length = math.ceil(len(sampler) / args.batch_size)
    per_epoch = math.ceil(length / args.accumulation)
    horizon = args.schedule_epochs * per_epoch
    manifest = Path(os.environ['SHARE4V_DATA_ROOT']) / os.environ['SHARE4V_JSON']
    config = {**vars(args), 'arm': 'FP0', 'objective': 'finelip_prefix', 'world_size': world,
              'dataset_size': len(dataset), 'loader_batches': length, 'updates_per_epoch': per_epoch,
              'lr_horizon_updates': horizon, 'warmup_updates': 200,
              'matching_candidates': args.batch_size * world, 'slots': 39,
              'backbone_lr': 1e-6, 'aggregation_lr': 2e-4, 'weight_decay': .01,
              'betas': [.9, .999], 'eps': 1e-8, 'margin': .2, 'reduction': 'SUM',
              'init_sha256': sha_file(args.init_state), 'initial_state_digest': initial_digest,
              'manifest': str(manifest), 'manifest_sha256': sha_file(manifest),
              'upstream_sha': UPSTREAM,
              'git_sha': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
              'implementation': 'independent mathematical implementation',
              'UPSTREAM_RUNTIME_PARITY': 'NOT RUN', 'LICENSE_NOT_IDENTIFIED': True,
              'REUSE_PERMISSION_UNCONFIRMED': True,
              'precision': 'FP32 master; bf16 backbone; FP32 aggregation/scoring/ranking',
              'worker_rng': 'fresh workers per epoch; explicit rank/epoch generator; replay consumed batches on resume',
              'caption_stream_matches_legacy': 'NOT CLAIMED'}
    update, micro, start_epoch, start_index, presentations = 0, 0, 0, 0, 0
    if args.resume:
        parent = torch.load(args.resume, map_location='cpu', weights_only=False)
        start_epoch, start_index = validate_resume(parent, config, length)
        module.load_state_dict(parent['module'], strict=True)
        optimizer.load_state_dict(parent['optimizer'])
        update, micro, presentations = parent['optimizer_step'], parent['micro_step'], parent['sample_presentations']
        config['parent_checkpoint_sha256'] = sha_file(args.resume)
        config['parent_checkpoint'] = str(Path(args.resume).resolve())
        config['resume_cursor'] = [start_epoch, start_index]
        config['resume_worker_rng'] = 'REPLAY_TO_CURSOR; caption hashes checked against parent stream'
        r = parent['rng_by_rank'][rank]
        random.setstate(r['python']); np.random.set_state(r['numpy']); torch.set_rng_state(r['torch'])
        torch.cuda.set_rng_state(r['cuda'])
    if rank == 0:
        (output / 'config.json').write_text(json.dumps(config, indent=2))
        print('CONFIG ' + json.dumps(config), flush=True)
    optimizer.zero_grad(set_to_none=True)

    def save(epoch, next_index):
        states = [None] * world
        dist.all_gather_object(states, rng_state())
        if rank == 0:
            payload = {'arm': 'FP0', 'objective': 'finelip_prefix', 'module': module.state_dict(),
                       'model': module.clip.state_dict(), 'optimizer': optimizer.state_dict(),
                       'optimizer_step': update, 'completed_steps': update, 'micro_step': micro,
                       'epoch': epoch, 'next_batch_index': next_index, 'accumulation_pending': 0,
                       'sample_presentations': presentations, 'rng_by_rank': states, 'config': config,
                       'git_sha': config['git_sha'], 'upstream_sha': UPSTREAM}
            path = output / ('FP0_update%06d.pt' % update)
            temporary = path.with_suffix('.tmp')
            torch.save(payload, temporary); temporary.replace(path)
            print('SAVED ' + str(path), flush=True)
        dist.barrier()

    if not args.resume:
        save(0, 0)
    wall_start = time.time()
    compute_seconds = 0.
    last_update_time = time.time()
    log = open(output / ('stream_rank%d.jsonl' % rank), 'a', buffering=1)
    for epoch in range(start_epoch, args.run_epochs):
        dataset.set_epoch(epoch); sampler.set_epoch(epoch)
        generator = torch.Generator().manual_seed(args.seed + epoch * world + rank + 70001)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                                            num_workers=args.workers, pin_memory=True, drop_last=False,
                                            generator=generator, persistent_workers=False)
        replay = None
        if args.resume and epoch == start_epoch and start_index:
            replay = {}
            source = Path(args.resume).parent / ('stream_rank%d.jsonl' % rank)
            if not source.exists():
                raise FileNotFoundError('resume stream audit required: ' + str(source))
            for line in source.open():
                row = json.loads(line)
                if row['epoch'] == epoch:
                    replay[row['batch_index']] = row['caption_hash']
        group_stats = torch.zeros(8, device=device)
        group_objective = 0.
        group_samples = 0
        for i, batch in enumerate(loader):
            caption_hash = hashlib.sha256('\n'.join(batch['caption_said']).encode()).hexdigest()
            if epoch == start_epoch and i < start_index:
                if replay is not None and i in replay and replay[i] != caption_hash:
                    raise RuntimeError('caption replay hash mismatch at batch %d' % i)
                continue
            tokens = longclip.tokenize(batch['caption_said'], context_length=248, truncate=True).to(device)
            token_lengths = eos_positions(tokens) + 1
            log.write(json.dumps({'epoch': epoch, 'batch_index': i, 'sample_id': batch['sample_id'].tolist(),
                                  'image_id': batch['image_id'].tolist(), 'prefix_k': batch['prefix_k'].tolist(),
                                  'token_length': token_lengths.tolist(), 'caption_hash': caption_hash}) + '\n')
            size = group_size(i, length, args.accumulation)
            boundary = (i + 1) % args.accumulation == 0 or i + 1 == length
            alpha = i / length if epoch == 0 else 1.
            for group in optimizer.param_groups:
                group['lr'] = group['base_lr'] * lr_factor(update, horizon)
            torch.cuda.synchronize(); t0 = time.time()
            with contextlib.nullcontext() if boundary else ddp.no_sync():
                loss, stats, extra = ddp(batch['image_a'].to(device, non_blocking=True), tokens,
                                         batch['image_id'].to(device), diagnostics=boundary and (update < 20 or (update + 1) % 100 == 0))
                finite = torch.isfinite(loss).to(torch.int32)
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if finite.item() == 0:
                    raise FloatingPointError('nonfinite FP0 objective')
                (loss * alpha / size).backward()
            micro += 1
            n = len(tokens) * world
            presentations += n; group_samples += n
            group_stats += stats
            group_objective += alpha * float(stats[0] + stats[1]) / size
            if boundary:
                gradients_finite = torch.stack([torch.isfinite(q.grad).all() for q in module.parameters() if q.grad is not None]).all().to(torch.int32)
                dist.all_reduce(gradients_finite, op=dist.ReduceOp.MIN)
                if gradients_finite.item() == 0:
                    raise FloatingPointError('nonfinite FP0 gradient')
                optimizer.step(); optimizer.zero_grad(set_to_none=True); update += 1
            torch.cuda.synchronize(); elapsed = time.time() - t0; compute_seconds += elapsed
            if boundary:
                if rank == 0 and (update <= 20 or update % 25 == 0 or i + 1 == length):
                    a, b, pairs, ai, at, pos, npos, neg = group_stats.tolist()
                    record = {'optimizer_step': update, 'micro_step': micro, 'epoch': epoch, 'next_batch_index': i + 1,
                              'actual_group_size': size, 'group_samples': group_samples, 'sample_presentations': presentations,
                              'micro_sum_group_average': (a + b) / size, 'group_objective': group_objective,
                              'pair_mean_display_only': (a + b) / max(pairs, 1), 'I2T_sum': a / size, 'T2I_sum': b / size,
                              'I2T_active': ai / max(pairs, 1), 'T2I_active': at / max(pairs, 1),
                              'positive_score': pos / npos, 'negative_score': neg / max(pairs, 1),
                              'alpha_micro_last': alpha, 'backbone_lr': optimizer.param_groups[0]['lr'],
                              'aggregation_lr': optimizer.param_groups[1]['lr'],
                              'image_scale': module.image_aggregator.scale.item(), 'text_scale': module.text_aggregator.scale.item(),
                              'last_micro_seconds': elapsed, 'update_wall_seconds': time.time() - last_update_time,
                              'peak_memory_gib': torch.cuda.max_memory_allocated() / 2**30,
                              'core_dtype': str(module.image_aggregator.scale.dtype), 'backbone_autocast': 'bfloat16',
                              **{k: float(v) for k, v in extra.items()}}
                    print('LOG ' + json.dumps(record), flush=True)
                    with (output / 'metrics.jsonl').open('a') as f: f.write(json.dumps(record) + '\n')
                if update == 1000 or i + 1 == length:
                    save(epoch, i + 1)
                group_stats.zero_(); group_samples = 0; group_objective = 0.
                last_update_time = time.time()
    log.close()
    if rank == 0:
        summary = {'optimizer_step': update, 'micro_step': micro, 'sample_presentations': presentations,
                   'wall_seconds': time.time() - wall_start, 'compute_seconds': compute_seconds,
                   'gpu_compute_hours': compute_seconds * world / 3600, 'config': config}
        (output / 'run_summary.json').write_text(json.dumps(summary, indent=2))
        print('FP0_TRAIN_COMPLETE ' + json.dumps(summary), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
