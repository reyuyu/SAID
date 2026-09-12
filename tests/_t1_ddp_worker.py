"""Worker for the SAID-Token v1 2-rank DDP equivalence test.

    --mode single --rows 2     one process holds the whole batch, plain module
    --mode rank   --rows 2     two ranks, DistributedDataParallel, 2 rows each

The stub CLIP exposes exactly the interface the module uses (last-layer image tokens, text hidden
state, ``text_projection``, a ``visual`` tower to deep-copy as the frozen reference, plus a historic
``mask_net`` and ``logit_scale`` that must stay frozen). It is deliberately tiny: the point of this
test is the distributed *arithmetic*, not the image model. Writes the post-backward gradients, the
global means of both loss terms and parameter digests, so the test can check the DDP-vs-single
equivalence, the cross-rank bit-identity, and the exactness of the reported global means.
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
PATCHES = 6


class StubVisual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(TEXT_WIDTH, DIM)

    def forward(self, images):
        return self.proj(images.flatten(1)[:, :TEXT_WIDTH])


class StubMaskNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.out = torch.nn.Linear(TEXT_WIDTH, DIM)

    def forward(self, hidden):
        return self.out(hidden.mean(dim=1))


class StubClip(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = DIM
        self.text_projection = torch.nn.Parameter(torch.randn(TEXT_WIDTH, DIM) * 0.1)
        self.visual = StubVisual()
        self.mask_net = StubMaskNet()
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * 4.6052)
        self.embedding = torch.nn.Embedding(49408, TEXT_WIDTH)
        self.patch_proj = torch.nn.Linear(TEXT_WIDTH, DIM)

    @property
    def dtype(self):
        return self.visual.proj.weight.dtype

    def encode_image_with_patches(self, images, use_checkpoint=False):
        flattened = images.flatten(1)
        base = flattened[:, :TEXT_WIDTH]
        patches = self.patch_proj(base[:, None, :].expand(-1, PATCHES, -1))
        patches = patches * (base[:, :1] != 0)[:, :, None]
        return self.visual(images), patches

    def encode_image(self, images):
        return self.visual(images)

    def encode_text(self, text, return_full=False):
        hidden = self.embedding(text)
        pooled = hidden[torch.arange(text.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        if return_full:
            return pooled, hidden
        return pooled


def digest(state_dict):
    hasher = hashlib.sha256()
    for key in sorted(state_dict):
        hasher.update(key.encode('utf-8'))
        hasher.update(state_dict[key].detach().float().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['single', 'rank', 'ragged'], required=True)
    parser.add_argument('--optimizer', default='adamw', choices=['adamw','sgd'])
    parser.add_argument('--scenario', default='normal', choices=['normal','invalid_rank','all_invalid','duplicates'])
    parser.add_argument('--rows', type=int, required=True)
    parser.add_argument('--out', required=True)
    parsed = parser.parse_args()

    from model.said_token_v1 import SaidTokenV1Module  # noqa: E402
    from train_said_token_v1 import check_equal_local_batch, build_optimizers, t1_train_step  # noqa: E402

    if parsed.mode == 'rank':
        import torch.distributed as dist
        rank = int(os.environ['RANK'])
        world = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='gloo', rank=rank, world_size=world)
    else:
        import torch.distributed as dist
        rank, world = 0, 1
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29699')
        if parsed.mode == 'ragged':
            # two ranks, deliberately unequal local batch sizes
            rank = int(os.environ['RANK'])
            world = int(os.environ['WORLD_SIZE'])
            dist.init_process_group(backend='gloo', rank=rank, world_size=world)
        else:
            dist.init_process_group(backend='gloo', rank=0, world_size=1)

    if parsed.mode == 'ragged':
        sizes = [parsed.rows, parsed.rows + 1]
        check_equal_local_batch(sizes[rank], torch.device('cpu'), world)
        raise SystemExit('the ragged-batch guard did not fire')

    torch.manual_seed(1234)                      # identical student on every rank
    clip = StubClip()
    module = SaidTokenV1Module(clip, rank=rank, chunk_image=2, chunk_text=3, seed=0)
    ddp_model = (torch.nn.parallel.DistributedDataParallel(module, find_unused_parameters=True)
                 if parsed.mode == 'rank' else module)

    args = argparse.Namespace(lr=1e-3, weight_decay=1e-2, module_lr=2e-3, module_wd=1e-2,
                              decoder_lr=1e-2, decoder_wd=0.0)
    optimizer, module_optimizer, decoder_optimizer, _, _, _ = build_optimizers(module, args)

    if parsed.optimizer == 'sgd':
        # Linear update verifies the backward equivalence without Adam amplifying
        # rounding in mathematically zero softmax-bias gradients. Production stays AdamW.
        optimizer, module_optimizer, decoder_optimizer = [torch.optim.SGD(
            opt.param_groups[0]['params'], lr=opt.param_groups[0]['lr'],
            weight_decay=opt.param_groups[0]['weight_decay'])
            for opt in (optimizer, module_optimizer, decoder_optimizer)]

    generator = torch.Generator().manual_seed(7)
    total = 2 * parsed.rows
    images = torch.randn(total, 3, 4, 4, generator=generator) * 0.4
    if parsed.scenario == 'invalid_rank':
        images[:parsed.rows, 0, 0, 0] = 0
    elif parsed.scenario == 'all_invalid':
        images[:, 0, 0, 0] = 0
    captions = ['a photo of a cat .', 'a dog on the grass .', 'a red bus downtown .',
                'pasta on a plate .', 'a mountain range .', 'a small boat .',
                'a train at a station .', 'a horse in a field .'][:total]
    ids = torch.arange(total)
    if parsed.scenario == 'duplicates':
        captions[1] = captions[0]
        ids[-1] = ids[-2]
    start = 0 if parsed.mode == 'single' else rank * parsed.rows
    stop = total if parsed.mode == 'single' else (rank + 1) * parsed.rows
    batch = {'image_a': images[start:stop], 'caption_said': captions[start:stop],
             'image_id': ids[start:stop]}

    out = t1_train_step(ddp_model, batch, (optimizer, module_optimizer, decoder_optimizer),
                        torch.device('cpu'), torch.float32, amp_enabled=False, world_size=world,
                        capture_grads=True, diagnostics=True, health_check=True)
    payload = {
        'mode': parsed.mode, 'rank': rank, 'world': world, 'rows': batch['image_a'].shape[0],
        'loss_said_global_mean': float(out['loss_said_global_mean']),
        'loss_rec_global_mean': float(out['loss_rec_global_mean']),
        'loss_total_global_mean': float(out['loss_total_global_mean']),
        'loss_said_backward_value': float(out['loss_said_backward_value']),
        'loss_rec_backward_value': float(out['loss_rec_backward_value']),
        'said_scaling_abs_diff': float(out['said_scaling_abs_diff']),
        'rec_scaling_abs_diff': float(out['rec_scaling_abs_diff']),
        'positive_said_score': float(out['positive_said_score']),
        'loss_rec_valid_global': float(out['loss_rec_valid_global']),
        'updated': {n: p.detach().tolist() for n,p in module.named_parameters() if p.requires_grad and not n.startswith('clip.embedding')},
        'grads': {name: (None if grad is None else grad.tolist())
                  for name, grad in out['_grads'].items()},
        'param_digest': digest(module.clip.state_dict()),
        'module_digest': digest(module.image_aggregator.state_dict()),
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
