"""Four-A100 direct JointInput pair microbenchmark and full-step lower-bound projection."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from model import longclip
from model.nested_joint_input import VisualInputProjection, hard_st, joint_logits
from train.nested_semantic_data import file_sha

PAIRS_PER_FULL_STEP_PER_RANK = 3 * 2 * 256 * 1024


class PairWork(nn.Module):
    def __init__(self, mask_net, visual_mode, readout_mode, checkpoint_block):
        super().__init__()
        self.mask_net = mask_net
        self.visual_mode = visual_mode
        self.readout_mode = readout_mode
        self.checkpoint_block = checkpoint_block
        self.visual_projection = VisualInputProjection(768, 512)

    def forward(self, visual_raw, text_hidden, z, text):
        visual = self.visual_projection(visual_raw)
        logits = joint_logits(self.mask_net, visual, text_hidden.detach(), self.readout_mode,
                              self.checkpoint_block)
        probabilities = torch.sigmoid(logits)
        masks = hard_st(probabilities)
        scores = 100 * (torch.nn.functional.normalize(z * masks, dim=-1) *
                        torch.nn.functional.normalize(text, dim=-1)).sum(-1)
        return scores.square().mean() + probabilities.mean()


def state_sha(module):
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def percentile(values, q):
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q / 100
    low, high = int(position), min(int(position) + 1, len(values) - 1)
    fraction = position - low
    return values[low] * (1 - fraction) + values[high] * fraction


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--visual-mode', choices=['cls', 'patch'], required=True)
    p.add_argument('--readout-mode', choices=['all', 'text'], required=True)
    p.add_argument('--pair-batch', type=int, required=True)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--steps', type=int, default=5)
    p.add_argument('--checkpoint-block', type=int, choices=[0, 1], default=1)
    p.add_argument('--init-state', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    rank, local, world = (int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    assert world == 4
    torch.cuda.set_device(local)
    dist.init_process_group('nccl')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)

    clip, _ = longclip.load_from_clip('ViT-B/16', device='cpu', args=argparse.Namespace())
    initial = torch.load(args.init_state, map_location='cpu', weights_only=False)
    clip.load_state_dict(initial['model'], strict=True)
    cpu_rng = torch.get_rng_state().clone()
    work = PairWork(clip.mask_net, args.visual_mode, args.readout_mode,
                    bool(args.checkpoint_block))
    projection_sha256 = state_sha(work.visual_projection)
    torch.set_rng_state(cpu_rng)
    del clip, initial
    work = work.cuda().train()
    ddp = DDP(work, device_ids=[local], output_device=local)
    optimizer = torch.optim.AdamW([
        {'params': work.mask_net.parameters(), 'lr': 1e-3, 'weight_decay': 0.},
        {'params': work.visual_projection.parameters(), 'lr': 1e-4, 'weight_decay': 0.},
    ], betas=(.9, .999), eps=1e-8)

    visual_count = 1 if args.visual_mode == 'cls' else 196
    generator = torch.Generator(device='cuda').manual_seed(100 + rank)
    visual_raw = torch.randn(args.pair_batch, visual_count, 768, device='cuda', generator=generator)
    text_hidden = torch.randn(args.pair_batch, 248, 512, device='cuda', generator=generator)
    z = torch.randn(args.pair_batch, 512, device='cuda', generator=generator)
    text = torch.randn(args.pair_batch, 512, device='cuda', generator=generator)
    records = []
    torch.cuda.reset_peak_memory_stats()
    for iteration in range(args.warmup + args.steps):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); started = time.perf_counter()
        loss = ddp(visual_raw, text_hidden, z, text)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(); elapsed = time.perf_counter() - started
        finite = torch.isfinite(loss).int(); dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        assert finite.item()
        if iteration >= args.warmup:
            records.append(elapsed)
    local_result = {
        'rank': rank, 'pid': os.getpid(), 'hostname': socket.gethostname(),
        'uuid': str(torch.cuda.get_device_properties(local).uuid),
        'seconds': records,
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
        'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
    }
    gathered = [None] * world
    dist.all_gather_object(gathered, local_result)
    if rank == 0:
        max_steps = [max(item['seconds'][i] for item in gathered) for i in range(args.steps)]
        median = statistics.median(max_steps)
        p90 = percentile(max_steps, 90)
        throughput = args.pair_batch / median
        length = visual_count + 248
        forward_flops_pair = 24 * length * 512**2 + 4 * length**2 * 512
        payload = {
            'passed': True, 'args': vars(args), 'world_size': world,
            'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'init_sha256': file_sha(args.init_state),
            'sequence_length': length, 'visual_token_count': visual_count,
            'projection_sha256': projection_sha256,
            'pairs_per_full_step_per_rank': PAIRS_PER_FULL_STEP_PER_RANK,
            'max_rank_step_seconds': max_steps,
            'median_seconds': median, 'p90_seconds': p90,
            'measured_pairs_per_second_per_rank': throughput,
            'projected_pair_compute_seconds_per_full_step_per_rank':
                PAIRS_PER_FULL_STEP_PER_RANK / throughput,
            'forward_flops_per_pair': forward_flops_pair,
            'forward_flops_per_full_step_per_rank':
                forward_flops_pair * PAIRS_PER_FULL_STEP_PER_RANK,
            'ranks': gathered,
            'resource_gate': {'step_median_seconds_max': 30.,
                              'peak_allocated_gib_max': 65.},
        }
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        (output / 'result.json').write_text(json.dumps(payload, indent=2) + '\n')
        print(json.dumps({
            'output': str(output), 'median_seconds': median, 'p90_seconds': p90,
            'pairs_per_second': throughput,
            'projected_full_step_pair_seconds': payload['projected_pair_compute_seconds_per_full_step_per_rank'],
            'peak_allocated_gib': max(x['peak_allocated_gib'] for x in gathered),
        }))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
