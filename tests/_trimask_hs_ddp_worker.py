"""Worker for the S0-TriMask-HS two-rank DDP equivalence test (gloo/CPU, real DDP, real step fn).

    --mode single --rows 2     one process holds the whole global batch, plain module
    --mode rank   --rows 2     two ranks, DistributedDataParallel, 2 rows each

The gate is deliberately opened so the hard mask is a genuine mixture of 0s and 1s: the equivalence
is checked with the text sparsity term active on a non-trivial mask, not on the all-open
initialisation (where several gradient paths vanish for structural reasons).
"""
import argparse
import hashlib
import json
import os
import sys

import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'train'))

DIM = 32
TEXT_WIDTH = 32
TOKENS = 5


class StubVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(TEXT_WIDTH, DIM)

    def forward(self, images):
        return self.proj(images.flatten(1)[:, :TEXT_WIDTH])


class StubMaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(TEXT_WIDTH, 12)
        self.out = nn.Linear(12, DIM)
        self.position = nn.Parameter(torch.randn(TOKENS, 12) * 0.1)

    def forward(self, hidden):
        features = torch.tanh(self.proj(hidden)) + self.position[:hidden.shape[1]]
        return self.out(features.mean(dim=1))


class StubClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = DIM
        self.text_projection = nn.Parameter(torch.randn(TEXT_WIDTH, DIM) * 0.1)
        self.text_offset = nn.Parameter(torch.randn(TOKENS, TEXT_WIDTH) * 0.1)
        self.logit_scale = nn.Parameter(torch.ones([]) * 4.6052)
        self.visual = StubVisual()
        self.mask_net = StubMaskNet()

    @property
    def dtype(self):
        return self.visual.proj.weight.dtype

    def encode_image(self, images):
        return self.visual(images)

    def encode_text(self, tokens, return_full=False, return_pool=False):
        base = tokens.float().mean(dim=1, keepdim=True)
        ramp = torch.arange(TOKENS, dtype=torch.float32).view(1, TOKENS)
        hidden = (base + ramp).unsqueeze(-1).repeat(1, 1, TEXT_WIDTH) * 0.05
        hidden = hidden + self.text_offset
        pooled = hidden.mean(dim=1) @ self.text_projection
        return (pooled, hidden) if return_full else pooled


def digest(state):
    hasher = hashlib.sha256()
    for key in sorted(state):
        hasher.update(key.encode())
        hasher.update(state[key].detach().float().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['single', 'rank'], required=True)
    parser.add_argument('--rows', type=int, required=True)
    parser.add_argument('--out', required=True)
    parsed = parser.parse_args()

    import torch.distributed as dist

    from model.said_trimask import HARD_GATE, LAMBDA_SPARSE_T_HS, TriMaskTrainModule
    from train_said_trimask import build_trimask_optimizers, trimask_train_step

    if parsed.mode == 'rank':
        rank = int(os.environ['RANK'])
        world = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='gloo', rank=rank, world_size=world)
    else:
        rank, world = 0, 1
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29707')
        dist.init_process_group(backend='gloo', rank=0, world_size=1)

    torch.manual_seed(1234)                      # identical student on every rank
    clip = StubClip()
    module = TriMaskTrainModule(clip, rank=rank, text_mask_width=TEXT_WIDTH, text_mask_layers=1,
                                text_mask_heads=8, text_mask_seed=0,
                                lambda_sparse_t=LAMBDA_SPARSE_T_HS,
                                text_gate_mode=HARD_GATE)
    # open the gate: the hard mask must contain both 0 and 1 coordinates
    torch.manual_seed(21)
    nn.init.normal_(module.text_mask_net.text_gate_projection.weight, std=1.0)
    clip.logit_scale.requires_grad_(False)

    if parsed.mode == 'rank':
        ddp_model = torch.nn.parallel.DistributedDataParallel(module, find_unused_parameters=True)
    else:
        ddp_model = module

    optimizer, mask_optimizer, _, _ = build_trimask_optimizers(
        clip, module.text_mask_net, argparse.Namespace(lr=1e-3, mask_lr=1e-2,
                                                       weight_decay=1e-2))

    generator = torch.Generator().manual_seed(7)
    total = 2 * parsed.rows
    images = torch.randn(total, 3, 4, 4, generator=generator) * 0.4
    captions = ['a photo of a cat .', 'a dog on the grass .', 'a red bus downtown .',
                'pasta on a plate .', 'a mountain range .', 'a small boat .',
                'a train at a station .', 'a horse in a field .'][:total]
    start = 0 if parsed.mode == 'single' else rank * parsed.rows
    stop = total if parsed.mode == 'single' else (rank + 1) * parsed.rows
    batch = {'image_a': images[start:stop], 'caption_said': captions[start:stop]}

    out = trimask_train_step(ddp_model, batch, optimizer, mask_optimizer, torch.device('cpu'),
                             torch.float32, amp_enabled=False, capture_grads=True, world_size=world,
                             heavy_diagnostics=False)
    sample_names = ('clip.mask_net.out.weight', 'clip.mask_net.proj.weight',
                    'clip.visual.proj.weight', 'clip.text_projection',
                    'text_mask_net.text_gate_projection.weight',
                    'text_mask_net.text_gate_projection.bias',
                    'text_mask_net.text_stem.attn_pool.attention.weight')
    named = dict(module.named_parameters())
    payload = {
        'mode': parsed.mode, 'rank': rank, 'world': world, 'rows': int(batch['image_a'].shape[0]),
        'loss_total': float(out['loss_total']),
        'loss_1': float(out['loss_1']),
        'loss_2': float(out['loss_2']),
        'loss_3': float(out['loss_3']),
        'loss_sparse_i': float(out['loss_sparse_i']),
        'loss_sparse_t': float(out['loss_sparse_t']),
        'text_keep_ratio': float(out['m_t'].detach().mean()),
        'text_zero_fraction': float((out['m_t'].detach() <= 0).float().mean()),
        'text_is_binary': bool(((out['m_t'].detach() <= 0)
                                | (out['m_t'].detach() >= 1)).all()),
        'grads': {name: (None if grad is None else grad.tolist())
                  for name, grad in out['_grads'].items()},
        'param_sample': {name: named[name].detach().tolist() for name in sample_names},
        'param_digest': digest(clip.state_dict()),
        'text_branch_digest': digest(module.text_mask_net.state_dict()),
    }
    with open(parsed.out.format(rank=rank), 'w') as handle:
        json.dump(payload, handle)
    if parsed.mode == 'rank':
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
