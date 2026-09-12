"""Worker for the C1 2-rank DDP equivalence test (gloo/CPU, real DDP, production step function).

    --mode single --rows 4     one process holds the whole batch, plain module
    --mode rank   --rows 2     two ranks, DistributedDataParallel, 2 rows each

Writes the post-backward parameter gradients plus the parameter/decoder digests, so the test can
check both the DDP-vs-single equivalence and the cross-rank bit-identity.
"""
import argparse
import hashlib
import json
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'train'))

DIM = 16
TEXT_WIDTH = 24
TOKENS = 5


class StubVisual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(TEXT_WIDTH, DIM)

    def forward(self, images):
        return self.proj(images.flatten(1)[:, :TEXT_WIDTH])


class StubMaskNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(TEXT_WIDTH, 12)
        self.out = torch.nn.Linear(12, DIM)
        self.position = torch.nn.Parameter(torch.randn(TOKENS, 12) * 0.1)

    def forward(self, hidden):
        features = torch.tanh(self.proj(hidden)) + self.position[:hidden.shape[1]]
        return self.out(features.mean(dim=1))


class StubClip(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = DIM
        self.text_projection = torch.nn.Parameter(torch.randn(DIM, DIM) * 0.1)
        self.text_proj = torch.nn.Linear(TEXT_WIDTH, DIM)
        self.visual = StubVisual()
        self.mask_net = StubMaskNet()

    @property
    def dtype(self):
        return self.visual.proj.weight.dtype

    def encode_image(self, images):
        return self.visual(images)

    def encode_text(self, tokens, return_full=False, return_pool=False):
        hidden = tokens.float().mean(dim=1, keepdim=True).repeat(1, TEXT_WIDTH)
        hidden = hidden.unsqueeze(1).repeat(1, TOKENS, 1)
        pooled = self.text_proj(hidden[:, 0, :]) @ self.text_projection
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

    from model.said_cls_reconstruction import C1TrainModule
    from train_said_cls_c1 import build_decoder_optimizer, c1_train_step
    from train_said_cls_cvssl import build_optimizers

    if parsed.mode == 'rank':
        import torch.distributed as dist
        rank = int(os.environ['RANK'])
        world = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='gloo', rank=rank, world_size=world)
    else:
        import torch.distributed as dist
        rank, world = 0, 1
        # the SmartCLIP gather (torch.distributed.nn.all_gather) needs a process group even for the
        # single-process reference run
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29694')
        dist.init_process_group(backend='gloo', rank=0, world_size=1)

    torch.manual_seed(1234)                      # identical student on every rank
    clip = StubClip()
    module = C1TrainModule(clip, rank=rank, lambda_rec=1.0, decoder_hidden=DIM, decoder_seed=0)
    if parsed.mode == 'rank':
        ddp_model = torch.nn.parallel.DistributedDataParallel(module, find_unused_parameters=True)
    else:
        ddp_model = module

    optimizer, mask_optimizer, _, _ = build_optimizers(
        module.clip, argparse.Namespace(lr=1e-3, mask_lr=1e-2, weight_decay=1e-2))
    decoder_optimizer = build_decoder_optimizer(
        module, argparse.Namespace(decoder_lr=1e-2, decoder_wd=0.0))

    # one deterministic global batch of ``2 * rows`` samples, sliced per rank from the SAME tensors
    generator = torch.Generator().manual_seed(7)
    total = 2 * parsed.rows
    images = torch.randn(total, 3, 4, 4, generator=generator) * 0.4
    captions = ['a photo of a cat .', 'a dog on the grass .', 'a red bus downtown .',
                'pasta on a plate .', 'a mountain range .', 'a small boat .',
                'a train at a station .', 'a horse in a field .'][:total]
    start = 0 if parsed.mode == 'single' else rank * parsed.rows
    stop = total if parsed.mode == 'single' else (rank + 1) * parsed.rows
    batch = {'image_a': images[start:stop], 'image_b': images[start:stop],
             'caption_said': captions[start:stop], 'image_id': torch.arange(start, stop)}

    out = c1_train_step(ddp_model, batch, optimizer, mask_optimizer, decoder_optimizer,
                        torch.device('cpu'), torch.float32, amp_enabled=False, world_size=world,
                        capture_grads=True)
    payload = {
        'mode': parsed.mode, 'rank': rank, 'world': world, 'rows': batch['image_a'].shape[0],
        'loss_rec': float(out['loss_rec']),
        'loss_rec_backward': float(out['loss_rec_backward_local']),
        'loss_smart': float(out['loss_smart']),
        'rec_backward_scale': float(out['rec_backward_scale']),
        'grads': {name: (None if grad is None else grad.tolist())
                  for name, grad in out['_grads'].items()},
        'param_digest': digest(module.clip.state_dict()),
        'decoder_digest': digest(module.decoder.state_dict()),
    }
    with open(parsed.out.format(rank=rank), 'w') as handle:
        json.dump(payload, handle)
    if parsed.mode == 'rank':
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
