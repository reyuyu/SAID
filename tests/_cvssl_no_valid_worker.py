"""Worker: one rank has NO valid anchor at all. Every collective must still be executed."""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import complement_visual_ssl as cvssl  # noqa: E402

DIM = 8
LOCAL = 4

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    import torch.distributed as dist
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    dist.init_process_group(backend='gloo', rank=rank, world_size=world)

    torch.manual_seed(3)
    anchors = torch.randn(LOCAL, DIM, requires_grad=True)
    candidates = torch.randn(LOCAL, DIM)
    mask = torch.ones(LOCAL, DIM)
    if rank == 1:
        mask = torch.zeros(LOCAL, DIM)                 # empty complement on this rank only
    image_ids = torch.arange(LOCAL) + rank * LOCAL

    result = cvssl.complement_visual_contrastive_loss(anchors, candidates, mask, image_ids,
                                                     tau_u=0.1, rank=rank)
    result['loss'].backward()
    payload = {
        'rank': rank,
        'finite': bool(torch.isfinite(result['loss']).all()),
        'loss': float(result['loss']),
        'valid_local': float(result['valid_ab_local'] + result['valid_ba_local']),
        'grad_norm': float(anchors.grad.abs().sum()) if anchors.grad is not None else 0.0,
    }
    with open(args.out.format(rank=rank), 'w') as handle:
        json.dump(payload, handle)
    dist.barrier()
    dist.destroy_process_group()
