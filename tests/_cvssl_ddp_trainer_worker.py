"""Worker: real-trainer DDP acceptance run (single process OR 2 ranks, gloo/CPU).

Uses the production components only: ``SaidClsCvsslTrainModule``, ``DistributedDataParallel``,
``build_optimizers`` (the two AdamW groups) and ``cvssl_train_step``. A reduced model stand-in is
used for speed; the training path, the loss, the scaling and the optimizers are the real ones.

Two properties matter and are implemented explicitly here:

1. ``build_optimizers`` is called exactly the way the production trainer calls it
   (``build_optimizers(train_module.clip, args)``) -- the optimizer parameter groups must be the
   ones a real run builds.
2. Every rank slices the SAME deterministic global batch, so ``single global batch ==
   rank0 local batch + rank1 local batch`` element-wise (candidate order and positive indices
   depend on it). Nothing is regenerated per rank.
"""
import argparse
import hashlib
import json
import os
import sys

import torch
import torch.nn as nn

REPO = '/root/SAID-gap-completion'
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'train'))

from model.said_cls_cvssl import SaidClsCvsslTrainModule  # noqa: E402
from train_said_cls_cvssl import build_optimizers, cvssl_train_step  # noqa: E402

DIM = 16
TEXT_WIDTH = 24
LOCAL_BATCH = 3
TOKENS = 5
GLOBAL_BATCH = 2 * LOCAL_BATCH
CAPTIONS = [
    'a cat on a chair',
    'a dog on the grass',
    'a red bus downtown',
    'pasta on a white plate',
    'a mountain under the sky',
    'a boat on calm water',
]
# case -> (arm, lambda_u). A/C/D/E/F are the four pre-registered arms; case G is the ragged-batch
# failure-mode case and case E-F add a duplicate image id / a repeated caption.
CASES = {
    'S0': ('S0_smartclip', 0.0),
    'G0': ('G0_global_vssl', 1.0),
    'R0': ('R0_random_vssl', 1.0),
    'C0': ('C0_complement_vssl', 1.0),
    'E': ('C0_complement_vssl', 1.0),
    'F': ('R0_random_vssl', 1.0),
    'G': ('C0_complement_vssl', 1.0),
    # degenerate case: empty captions -> empty mask -> all anchors invalid -> connected zero U
    'D': ('C0_complement_vssl', 1.0),
}


class StubMaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(TEXT_WIDTH, 12)
        self.out = nn.Linear(12, DIM)

    def forward(self, hidden):
        return self.out(torch.tanh(self.proj(hidden.mean(dim=1))))


class StubVisual(nn.Module):
    """Accepts the 4D NCHW view tensor, like the real ``clip.visual``."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(TEXT_WIDTH, DIM)

    def forward(self, images):
        return self.proj(images.flatten(1)[:, :TEXT_WIDTH])


class StubClip(nn.Module):
    """Cheap stand-in with the same interface the train module and loss use."""

    def __init__(self):
        super().__init__()
        self.embed_dim = DIM
        self.text_projection = nn.Parameter(torch.zeros(TEXT_WIDTH, DIM))
        self.text_proj = nn.Linear(TEXT_WIDTH, DIM)
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
        pooled = self.text_proj(hidden[:, 0, :])
        return (pooled, hidden) if return_full else pooled


def global_batch(case: str):
    """The single canonical global batch (identical in every mode and on every rank)."""
    generator = torch.Generator().manual_seed(7)
    images = torch.randn(GLOBAL_BATCH, 3, 4, 4, generator=generator) * 0.4
    other = images + torch.randn(GLOBAL_BATCH, 3, 4, 4, generator=generator) * 0.05
    image_ids = torch.arange(GLOBAL_BATCH)
    captions = list(CAPTIONS)
    if case == 'E':
        image_ids[2] = 0                      # duplicate image inside the candidate bank
    if case == 'F':
        captions[3] = captions[0]             # repeated caption across ranks
    if case == 'D':
        captions = [''] * GLOBAL_BATCH        # empty text -> empty mask -> no valid anchor
    return {'image_a': images, 'image_b': other, 'caption_said': captions,
            'image_id': image_ids, 'sample_id': torch.arange(GLOBAL_BATCH)}


def slice_batch(batch, start, stop):
    return {'image_a': batch['image_a'][start:stop], 'image_b': batch['image_b'][start:stop],
            'caption_said': batch['caption_said'][start:stop],
            'image_id': batch['image_id'][start:stop], 'sample_id': batch['sample_id'][start:stop]}


def run(case: str, mode: str, out_path: str, steps: int, lr: float, mask_lr: float,
        weight_decay: float, static_graph: bool, checkpoint_views: bool):
    if mode == 'rank':
        import torch.distributed as dist
        rank = int(os.environ['RANK'])
        world = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='gloo', rank=rank, world_size=world)
    else:
        import torch.distributed as dist
        rank, world = 0, 1
        # the reference SmartCLIP gather is ``torch.distributed.nn.all_gather`` (autograd-aware),
        # which needs an initialised process group even for the single-process reference run
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29695')
        dist.init_process_group(backend='gloo', rank=0, world_size=1)

    torch.manual_seed(1234)
    clip = StubClip()
    arm, lambda_u = CASES[case]
    module = SaidClsCvsslTrainModule(
        clip, rank=rank, arm=arm, lambda_u=lambda_u, tau_u=0.1, rho=0.0,
        ddp_gradient_averaging=True, grad_checkpoint_views=checkpoint_views)
    if mode == 'rank':
        ddp_model = torch.nn.parallel.DistributedDataParallel(module,
                                                              find_unused_parameters=True)
        if static_graph:
            ddp_model._set_static_graph()
    else:
        ddp_model = module

    # exactly what the production trainer does: the CLIP model, not the train module
    args = argparse.Namespace(lr=lr, mask_lr=mask_lr, weight_decay=weight_decay)
    optimizer, mask_optimizer, n_backbone, n_mask = build_optimizers(module.clip, args)

    batch_all = global_batch(case)
    history = {}
    for step in range(1, steps + 1):
        if mode == 'rank':
            if case == 'G':
                # genuinely different per-rank batch sizes: rank0 keeps 3, rank1 keeps 2
                sizes = [LOCAL_BATCH, LOCAL_BATCH - 1]
                start = sum(sizes[:rank])
                batch = slice_batch(batch_all, start, start + sizes[rank])
            else:
                start = rank * LOCAL_BATCH
                batch = slice_batch(batch_all, start, start + LOCAL_BATCH)
        else:
            batch = slice_batch(batch_all, 0, GLOBAL_BATCH)

        out = cvssl_train_step(ddp_model, batch, optimizer, mask_optimizer,
                               torch.device('cpu'), torch.float32, amp_enabled=False,
                               mask_rng_seed=99, capture_grads=(step == 1),
                               mask_override=(torch.zeros(batch['image_a'].shape[0], DIM)
                                              if case == 'D' else None))
        if step in (1, 5, 20):
            payload = {
                'step': step,
                'loss_sidm': float(out['loss_sidm']),
                'loss_dism': float(out['loss_dism']),
                'loss_sparsity': float(out['loss_sparsity']),
                'loss_smart': float(out['loss_smart']),
                'loss_vssl_global_mean': float(out['loss_vssl_global_mean']),
                'loss_vssl_for_backward_local': float(out['loss_vssl_for_backward_local']),
                'loss_total_global_form': float(out['loss_smart'] + out['loss_vssl_global_mean']),
                'valid_ab': float(out['vssl_valid_ab']),
                'valid_ba': float(out['vssl_valid_ba']),
                'global_sample_count': float(out['global_sample_count']),
                'loss_scale': float(out['vssl_loss_scale']),
                'param_digest': _digest(module.state_dict()),
                'backbone_opt_digest': _opt_digest(optimizer),
                'mask_opt_digest': _opt_digest(mask_optimizer),
            }
            if step == 1:
                payload['grads'] = {name: (None if grad is None else grad.tolist())
                                    for name, grad in out['_grads'].items()}
                payload['params'] = {k: v.tolist() for k, v in module.state_dict().items()}
                payload['backbone_exp_avg'] = _opt_state(optimizer, 'exp_avg')
                payload['mask_exp_avg'] = _opt_state(mask_optimizer, 'exp_avg')
                payload['backbone_group_names'] = _group_names(optimizer, module)
                payload['mask_group_names'] = _group_names(mask_optimizer, module)
                payload['sample_id'] = batch['sample_id'].tolist()
                payload['image_id'] = batch['image_id'].tolist()
                payload['caption_said'] = list(batch['caption_said'])
                payload['n_backbone'] = n_backbone
                payload['n_mask'] = n_mask
            history[step] = payload

    with open(out_path.format(rank=rank), 'w') as handle:
        json.dump({'case': case, 'mode': mode, 'rank': rank, 'world': world,
                   'arm': arm, 'lambda_u': lambda_u, 'lr': lr, 'mask_lr': mask_lr,
                   'weight_decay': weight_decay, 'static_graph': static_graph,
                   'checkpoint_views': checkpoint_views,
                   'final_state': {k: v.detach().clone().tolist()
                                   for k, v in clip.state_dict().items()},
                   'history': history}, handle)
    if mode == 'rank':
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def _digest(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


def _opt_digest(optimizer):
    digest = hashlib.sha256()
    for group in optimizer.param_groups:
        for parameter in group['params']:
            state = optimizer.state.get(parameter, {})
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                value = state.get(key)
                if value is None:
                    continue
                digest.update(str(key).encode())
                digest.update(torch.as_tensor(value).detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


def _opt_state(optimizer, key):
    out = {}
    for index, group in enumerate(optimizer.param_groups):
        for position, parameter in enumerate(group['params']):
            state = optimizer.state.get(parameter, {})
            if key in state:
                out['g%d_p%d' % (index, position)] = torch.as_tensor(
                    state[key]).detach().float().cpu().tolist()
    return out


def _group_names(optimizer, module):
    wanted = {id(parameter) for group in optimizer.param_groups for parameter in group['params']}
    return [name for name, parameter in module.named_parameters() if id(parameter) in wanted]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True, choices=sorted(CASES))
    parser.add_argument('--mode', choices=['single', 'rank'], required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--mask_lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--static_graph', type=int, default=0)
    parser.add_argument('--checkpoint_views', type=int, default=0)
    parsed = parser.parse_args()
    run(parsed.case, parsed.mode, parsed.out, parsed.steps, parsed.lr, parsed.mask_lr,
        parsed.weight_decay, bool(parsed.static_graph), bool(parsed.checkpoint_views))
