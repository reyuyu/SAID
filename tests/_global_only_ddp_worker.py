"""Two-rank worker for the S0-GlobalOnly DDP test.

Run through ``torch.distributed.run --nproc_per_node=2``. Both ranks build the SAME frozen CLIP
(cached checkpoint) and the SAME global batch of images/captions; rank ``r`` takes the slice
``[r * local_batch, (r + 1) * local_batch)``. Each rank saves:

* ``ddp_grads_rank{r}.pt``   -- the DDP-averaged gradient (identical on both ranks)
* ``local_grads_rank{r}.pt`` -- the rank's own contribution before any reduction
* ``ddp_info.pt``            -- world size, losses, digests (rank 0)

The test then requires the DDP gradient to be the exact mean of the two local gradients (standard
averaging, no ``world_size`` factor) and to agree with a single-process reference over the same
global batch.
"""
import argparse
import math
import os
import sys

import torch
import torch.distributed as dist

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (REPO, os.path.join(REPO, 'train')):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from model import longclip                                                      # noqa: E402
import train_global_only as trainer                                             # noqa: E402

CAPTIONS = ['a photo of a cat on a wooden table', 'a dog running through tall grass',
            'an old bicycle leaning against a wall', 'two boats on a calm lake']
GRAD_NAMES = ('clip.visual.proj', 'clip.text_projection',
              'clip.visual.transformer.resblocks.11.attn.in_proj_weight',
              'clip.visual.conv1.weight', 'clip.transformer.resblocks.0.attn.in_proj_weight')


def build_inputs(local_batch=2, seed=20260913):
    world = 2
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(world * local_batch, 3, 224, 224, generator=generator)
    captions = [CAPTIONS[index % len(CAPTIONS)] for index in range(world * local_batch)]
    text_ids = longclip.tokenize(captions, truncate=True)
    return images, text_ids, captions


def main():
    parser = argparse.ArgumentParser(description='two-rank S0-GlobalOnly gradient worker')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--local_batch', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20260913)
    args = parser.parse_args()

    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world)
    device = torch.device('cuda', local_rank)

    model, _ = longclip.load_from_clip('ViT-B/16', device='cpu', download_root=None,
                                       args=argparse.Namespace())
    model.logit_scale = torch.nn.Parameter(torch.ones([]) * math.log(trainer.FIXED_SCALE))
    model = model.to(device)
    model.train()
    trainer.partition_parameters(model)          # freeze the compatibility keys like the trainer does
    module = trainer.GlobalOnlyTrainModule(model, rank=rank).to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(module, device_ids=[local_rank],
                                                          output_device=local_rank,
                                                          find_unused_parameters=True)
    ddp_model._set_static_graph()

    images, text_ids, captions = build_inputs(local_batch=args.local_batch, seed=args.seed)
    start = rank * args.local_batch
    local_images = images[start:start + args.local_batch].to(device)
    local_text = text_ids[start:start + args.local_batch].to(device)

    out = ddp_model(local_images, local_text, amp_dtype=torch.bfloat16, amp_enabled=False)
    out['loss'].backward()
    reduced_named = dict(ddp_model.module.named_parameters())
    reduced = {name: (None if reduced_named[name].grad is None
                      else reduced_named[name].grad.detach().cpu().clone()) for name in GRAD_NAMES}

    ddp_model.zero_grad(set_to_none=True)
    local_out = module(local_images, local_text, amp_dtype=torch.bfloat16, amp_enabled=False)
    local_out['loss'].backward()
    local_named = dict(module.named_parameters())
    local = {name: (None if local_named[name].grad is None
                    else local_named[name].grad.detach().cpu().clone()) for name in GRAD_NAMES}
    torch.save(local, os.path.join(args.output_dir, 'local_grads_rank%d.pt' % rank))

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    dist.barrier()
    if rank == 0:
        torch.save(reduced, os.path.join(args.output_dir, 'ddp_grads_rank0.pt'))
    torch.save(local, os.path.join(args.output_dir, 'local_grads_rank%d.pt' % rank))
    dist.barrier()
    if rank == 0:
        torch.save({
            'world_size': world, 'local_batch': args.local_batch,
            'global_batch': world * args.local_batch,
            'loss_total_rank0': float(out['loss'].detach()),
            'loss_i2t_rank0': float(out['loss_i2t'].detach()),
            'loss_t2i_rank0': float(out['loss_t2i'].detach()),
            'loss_local_rank0': float(local_out['loss'].detach()),
            'i2t_top1_rank0': out['stats']['i2t_top1'], 't2i_top1_rank0': out['stats']['t2i_top1'],
            'state_digest': trainer.state_digest(model.state_dict()),
            'captions': captions, 'amp': 'fp32 (autocast disabled for exactness)',
        }, os.path.join(args.output_dir, 'ddp_info.pt'))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
