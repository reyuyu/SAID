"""Two-rank CG-CLIP worker: one real optimizer step's gradients, saved for the pytest comparison.

Run through ``torch.distributed.run --nproc_per_node=2``. Each rank builds the SAME frozen CLIP
(cached checkpoint), the SAME deterministic mixed caption gate and the SAME global batch of images and
captions; rank ``r`` takes the slice ``[r * local_batch, (r + 1) * local_batch)``, exactly like the
trainer's DistributedSampler would. After one forward and one backward the DDP-averaged parameter
gradients are written to ``<output_dir>/ddp_grads.pt`` (rank 0 only) together with a small info
record. ``tests/test_cgclip.py`` recomputes the same step in a single process over the whole global
batch and requires the two to agree: the per-rank anchor means plus standard DDP averaging must equal
the single-process mean over all anchors, with no extra world-size factor anywhere.
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (REPO, os.path.join(REPO, 'train')):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from model.cgclip import CaptionGate, state_digest                              # noqa: E402
import train_cgclip as trainer                                                 # noqa: E402

GATE_SCALE = 2.5
GATE_BIAS = 0.0
CAPTIONS = ['a photo of a cat on a wooden table', 'a dog running through tall grass',
            'an old bicycle leaning against a wall', 'two boats on a calm lake']
GRAD_NAMES = ('gate.query.weight', 'gate.key.weight', 'gate.bias', 'clip.visual.proj',
              'clip.text_projection', 'clip.visual.transformer.resblocks.11.attn.in_proj_weight',
              'clip.visual.transformer.resblocks.11.mlp.c_fc.weight',
              'clip.visual.transformer.resblocks.0.ln_1.weight',
              'clip.transformer.resblocks.0.attn.in_proj_weight')


def build_inputs(local_batch=2, seed=20260913):
    """The deterministic global batch: ``2 * local_batch`` images, captions and token ids."""
    from model import longclip
    world = 2
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(world * local_batch, 3, 224, 224, generator=generator)
    captions = [CAPTIONS[index % len(CAPTIONS)] for index in range(world * local_batch)]
    text_ids = longclip.tokenize(captions, truncate=True)
    return images, text_ids, captions


def build_gate(device):
    gate = CaptionGate(device=device).to(device)
    generator = torch.Generator(device='cpu').manual_seed(7)
    with torch.no_grad():
        gate.query.weight.copy_(torch.randn(gate.query.weight.shape, generator=generator)
                                * GATE_SCALE)
        gate.bias.fill_(GATE_BIAS)
    return gate


def main():
    parser = argparse.ArgumentParser(description='two-rank CG-CLIP gradient worker')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--local_batch', type=int, default=2)
    parser.add_argument('--image_chunk', type=int, default=2)
    parser.add_argument('--text_chunk', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20260913)
    args = parser.parse_args()

    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world)
    device = torch.device('cuda', local_rank)

    model, _ = trainer.longclip.load_from_clip('ViT-B/16', device='cpu', download_root=None,
                                               args=argparse.Namespace())
    model.logit_scale = torch.nn.Parameter(torch.ones([]) * torch.log(torch.tensor(100.0)))
    model = model.to(device)
    model.train()
    gate = build_gate(device)
    module = trainer.CgClipTrainModule(model, gate, rank=rank, image_chunk=args.image_chunk,
                                       text_chunk=args.text_chunk,
                                       cond_checkpoint=False).to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(module, device_ids=[local_rank],
                                                          output_device=local_rank,
                                                          find_unused_parameters=True)
    ddp_model._set_static_graph()

    images, text_ids, captions = build_inputs(local_batch=args.local_batch, seed=args.seed)
    local_images = images[rank * args.local_batch:(rank + 1) * args.local_batch].to(device)
    local_text = text_ids[rank * args.local_batch:(rank + 1) * args.local_batch].to(device)

    out = ddp_model(local_images, local_text, amp_dtype=torch.bfloat16, amp_enabled=False)
    out['loss_total'].backward()

    # SNAPSHOT FIRST: the reduced (DDP-averaged) gradients must be read before anything else touches
    # ``.grad``, otherwise the later local backward would silently overwrite them
    reduced_named = dict(ddp_model.module.named_parameters())
    reduced_grads = {name: (None if reduced_named[name].grad is None
                            else reduced_named[name].grad.detach().cpu().clone())
                     for name in GRAD_NAMES}

    # then the SAME rank-local computation without the DDP wrapper: the rank's own contribution before
    # the allreduce, whose exact mean the reduced gradient must equal
    ddp_model.zero_grad(set_to_none=True)
    local_out = module(local_images, local_text, amp_dtype=torch.bfloat16, amp_enabled=False)
    local_out['loss_total'].backward()
    local_named = dict(module.named_parameters())
    local_grads = {name: (None if local_named[name].grad is None
                          else local_named[name].grad.detach().cpu().clone())
                   for name in GRAD_NAMES}
    torch.save(local_grads, os.path.join(args.output_dir, 'local_grads_rank%d.pt' % rank))

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    dist.barrier()
    if rank == 0:
        torch.save(reduced_grads, os.path.join(args.output_dir, 'ddp_grads.pt'))
        torch.save({
            'world_size': world, 'local_batch': args.local_batch,
            'global_batch': world * args.local_batch,
            'all_gates_open': bool(out['stats']['all_gates_open']),
            'gate_closed_fraction': float((out['positive_mask'].detach() < 0.5).float().mean()),
            'loss_total_rank0': float(out['loss_total']),
            'loss_global_rank0': float(out['loss_global']),
            'loss_attention_rank0': float(out['loss_attention']),
            'loss_sparse_rank0': float(out['loss_sparse']),
            'loss_total_local_rank0': float(local_out['loss_total']),
            'gate_state_digest': state_digest(gate.state_dict()),
            'clip_state_digest': state_digest(model.state_dict()),
            'image_chunk': args.image_chunk, 'text_chunk': args.text_chunk,
            'cond_checkpoint': False, 'amp': 'fp32 (autocast disabled on purpose for exactness)',
            'captions': captions, 'rank': rank,
        }, os.path.join(args.output_dir, 'ddp_info.pt'))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
