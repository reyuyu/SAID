"""Worker for the single-process vs 2-rank CVSSL gradient equivalence test.

Run either as a plain process (``--mode single``: one global batch of 2b samples) or under
``torchrun --nproc_per_node=2`` (``--mode rank``: each rank holds b anchors, candidates gathered).

The features are an affine function of one shared parameter vector, so a gradient comparison is a
statement about the objective's cross-rank scaling, not about the encoder.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import complement_visual_ssl as cvssl  # noqa: E402

DIM = 8
LOCAL = 5


def build_problem(uneven: bool):
    torch.manual_seed(11)
    base = torch.randn(2 * LOCAL, DIM)
    other_view = torch.randn(2 * LOCAL, DIM)
    mask = torch.ones(2 * LOCAL, DIM)
    mask[:, DIM // 2:] = 0.5
    image_ids = torch.arange(2 * LOCAL) % 3                        # some shared images
    if uneven:
        mask[LOCAL + 1] = 0.0                                      # rank 1 anchor 1: empty mask
        other_view[LOCAL + 2] = 0.0                                # degenerate candidate
    return base, other_view, mask, image_ids


def run(mode: str, uneven: bool, out_path: str, ddp_scaling: bool):
    base, other_view, mask, image_ids = build_problem(uneven)
    if mode == 'rank':
        import torch.distributed as dist
        rank = int(os.environ['RANK'])
        world = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='gloo', rank=rank, world_size=world)
        sl = slice(rank * LOCAL, (rank + 1) * LOCAL)
    else:
        rank, world, sl = 0, 1, slice(0, 2 * LOCAL)

    theta = nn.Parameter(torch.zeros(DIM))
    anchors = base[sl] + theta                                     # live anchors
    candidates_all = other_view[sl]                                # opposite-view features
    mask_local = mask[sl]
    ids_local = image_ids[sl]

    if mode == 'rank':
        # each rank contributes its own local anchors; the candidate bank is gathered inside
        result = cvssl.complement_visual_contrastive_loss(
            anchors, candidates_all, mask_local, ids_local, tau_u=0.1, rank=rank,
            ddp_gradient_averaging=ddp_scaling, duplicate_policy='exclude')
    else:
        result = cvssl.complement_visual_contrastive_loss(
            anchors, candidates_all, mask_local, ids_local, tau_u=0.1, rank=0,
            ddp_gradient_averaging=False, duplicate_policy='exclude')
    result['loss'].backward()

    payload = {
        'mode': mode, 'rank': rank, 'world': world,
        'loss': float(result['loss'].detach()),
        'global_mean_loss': float(result['global_mean_loss']),
        'valid_ab': float(result['valid_ab']), 'valid_ba': float(result['valid_ba']),
        'valid_local_ab': float(result['valid_ab_local']),
        'valid_local_ba': float(result['valid_ba_local']),
        'loss_scale': float(result['loss_scale']),
        'grad_theta': theta.grad.tolist(),
    }
    with open(out_path.format(rank=rank), 'w') as handle:
        json.dump(payload, handle)
    if mode == 'rank':
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['single', 'rank'], required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--uneven', type=int, default=0)
    parser.add_argument('--ddp_scaling', type=int, default=0)
    args = parser.parse_args()
    run(args.mode, bool(args.uneven), args.out, bool(args.ddp_scaling))
