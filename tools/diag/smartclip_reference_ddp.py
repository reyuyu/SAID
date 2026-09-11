"""Committed copy of the SAID-CLS-CVSSL v0.1 distributed-correctness diagnosis harness.

The measured results it produced are in docs/said_cls_cvssl/v01_ddp_debug_matrix.md;
the conclusions are in docs/said_cls_cvssl/v01_distributed_correctness_fix_report.md.
Collect everything with tools/diag/run_ddp_diagnosis.sh, which keeps the locked order
S0 -> per-term -> golden reference -> U-only -> R0.

Golden reference: the REAL ``model.model_longclip.CLIP.forward`` under real DDP.
"""

import argparse
import hashlib
import json
import os
import sys

import torch
import torch.nn.functional as F

REPO = '/root/SAID-gap-completion'
sys.path.insert(0, REPO)

GLOBAL_BATCH = 6
LAMBDA_ALIGN = 10.0
LAMBDA_SPARSE = 2.0
CAPTIONS = [
    'a photo of a cat sitting on a wooden chair .',
    'a photo of a dog running on green grass .',
    'a photo of a red bus on a city street .',
    'a photo of a plate of pasta on a table .',
    'a photo of a mountain range under blue sky .',
    'a photo of a small boat on calm water .',
]


def build():
    from model.model_longclip import CLIP
    torch.manual_seed(1234)
    # vision_heads = vision_width // 64 must be > 0 -> vision_width = 128 gives 2 heads; the
    # transformer width must equal embed_dim because mask_net outputs embed_dim coordinates.
    model = CLIP(embed_dim=16, image_resolution=16, vision_layers=1, vision_width=128,
                 vision_patch_size=8, context_length=8, vocab_size=50000, transformer_width=16,
                 transformer_heads=4, transformer_layers=1, load_from_clip=False)
    model.float()
    return model


def batch():
    generator = torch.Generator().manual_seed(7)
    images = torch.randn(GLOBAL_BATCH, 3, 16, 16, generator=generator) * 0.4
    return images, CAPTIONS


def digest(module):
    hasher = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        hasher.update(name.encode())
        hasher.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['single', 'rank'], required=True)
    parser.add_argument('--ddp_mode', default='ddp', choices=['ddp', 'raw'])
    parser.add_argument('--out', required=True)
    parsed = parser.parse_args()

    import torch.distributed as dist
    from model import longclip
    if parsed.mode == 'rank':
        rank = int(os.environ['RANK'])
        world = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='gloo', rank=rank, world_size=world)
    else:
        rank, world = 0, 1
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29799')
        dist.init_process_group(backend='gloo', rank=0, world_size=1)

    model = build()
    payload = {'mode': parsed.mode, 'rank': rank, 'world': world, 'ddp_mode': parsed.ddp_mode,
               'init_digest': digest(model)}
    if parsed.mode == 'rank' and parsed.ddp_mode == 'ddp':
        model_ddp = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif parsed.mode == 'rank':
        model_ddp = model          # the reference trainer's *own* wrapping decision (no DDP)
    else:
        model_ddp = model

    local_batch = GLOBAL_BATCH // max(world, 1)
    images, captions = batch()
    start, stop = rank * local_batch, (rank + 1) * local_batch
    if parsed.mode == 'rank':
        images = images[start:stop]
        captions = captions[start:stop]
    text = longclip.tokenize(captions, truncate=True)
    if parsed.mode == 'rank':
        payload['used_rank'] = int(os.environ.get('RANK'))
        out = model_ddp(images, text, rank)
    else:
        out = model_ddp(images, text, 0)
    loss = LAMBDA_ALIGN * (out['loss_sidm'] + out['loss_dism']) + LAMBDA_SPARSE * out['loss_sparsity']
    loss.backward()
    payload['loss_sidm'] = float(out['loss_sidm'])
    payload['loss_dism'] = float(out['loss_dism'])
    payload['loss_sparsity'] = float(out['loss_sparsity'])
    payload['loss_total'] = float(loss)
    payload['grads'] = {name: (None if parameter.grad is None
                               else parameter.grad.detach().clone().tolist())
                        for name, parameter in model.named_parameters()}
    with open(parsed.out.format(rank=rank), 'w') as handle:
        json.dump(payload, handle)
    if parsed.mode == 'rank':
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
