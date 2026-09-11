"""Committed copy of the SAID-CLS-CVSSL v0.1 distributed-correctness diagnosis harness.

The measured results it produced are in docs/said_cls_cvssl/v01_ddp_debug_matrix.md;
the conclusions are in docs/said_cls_cvssl/v01_distributed_correctness_fix_report.md.
Collect everything with tools/diag/run_ddp_diagnosis.sh, which keeps the locked order
S0 -> per-term -> golden reference -> U-only -> R0.

Layered DDP diagnostic worker for SAID-CLS-CVSSL (read-only diagnosis; changes no math).
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

DIM = 16
TEXT_WIDTH = 24
TOKENS = 5
GLOBAL_BATCH = 6
CAPTIONS = [
    'a photo of a cat sitting on a wooden chair .',
    'a photo of a dog running on green grass .',
    'a photo of a red bus on a city street .',
    'a photo of a plate of pasta on a table .',
    'a photo of a mountain range under blue sky .',
    'a photo of a small boat on calm water .',
]


class StubMaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(TEXT_WIDTH, 12)
        self.out = nn.Linear(12, DIM)

    def forward(self, hidden):
        return self.out(torch.tanh(self.proj(hidden.mean(dim=1))))


class StubVisual(nn.Module):
    """Mirrors ``clip.visual``: accepts the 4D NCHW view tensor (as the real ViT does)."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(TEXT_WIDTH, DIM)

    def forward(self, images):
        return self.proj(images.flatten(1)[:, :TEXT_WIDTH])


class StubClip(nn.Module):
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
        hidden = tokens.to(self.text_proj.weight.dtype).mean(dim=1, keepdim=True).repeat(
            1, TEXT_WIDTH)
        hidden = hidden.unsqueeze(1).repeat(1, TOKENS, 1)
        pooled = self.text_proj(hidden[:, 0, :])
        return (pooled, hidden) if return_full else pooled


def build_init():
    torch.manual_seed(1234)
    clip = StubClip()
    with torch.no_grad():
        for parameter in clip.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
        clip.text_projection.add_(torch.randn_like(clip.text_projection) * 0.2)
    return clip


def global_batch():
    """The one canonical global batch; every rank slices it deterministically."""
    generator = torch.Generator().manual_seed(7)
    images = torch.randn(GLOBAL_BATCH, 3, 4, 4, generator=generator) * 0.3
    images_b = images + torch.randn(GLOBAL_BATCH, 3, 4, 4, generator=generator) * 0.05
    return {'image_a': images, 'image_b': images_b, 'image_id': torch.arange(GLOBAL_BATCH),
            'sample_id': torch.arange(GLOBAL_BATCH), 'caption': list(CAPTIONS)}


def slice_batch(batch, start, stop):
    return {'image_a': batch['image_a'][start:stop], 'image_b': batch['image_b'][start:stop],
            'image_id': batch['image_id'][start:stop], 'sample_id': batch['sample_id'][start:stop],
            'caption': batch['caption'][start:stop]}


def digest_tensor(value):
    return hashlib.sha256(torch.as_tensor(value).detach().float().cpu().contiguous()
                          .numpy().tobytes()).hexdigest()[:16]


def param_digest(module):
    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        digest.update(name.encode())
        digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


def opt_digest(optimizer):
    digest = hashlib.sha256()
    for group in optimizer.param_groups:
        for parameter in group['params']:
            state = optimizer.state.get(parameter, {})
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                value = state.get(key)
                if value is None:
                    continue
                digest.update(key.encode())
                digest.update(torch.as_tensor(value).detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


def named_grads(module):
    return {name: (None if parameter.grad is None
                   else parameter.grad.detach().clone().tolist())
            for name, parameter in module.named_parameters()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['single', 'rank'], required=True)
    parser.add_argument('--case', default='S0')
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--world', type=int, default=2)
    parser.add_argument('--ddp_only', type=int, default=1,
                        help='1: wrap in DistributedDataParallel (production path)')
    parser.add_argument('--ddp_gradient_averaging', type=int, default=1)
    parser.add_argument('--static_graph', type=int, default=0)
    parser.add_argument('--checkpoint_views', type=int, default=0)
    parser.add_argument('--term_grads', type=int, default=0)
    parser.add_argument('--terms', default='all',
                        help='which terms exist in the backward graph: all | u | smart '
                             '(diagnostic only; the objective itself is unchanged)')
    parser.add_argument('--backward_term', default='',
                        help='diagnostic: back-propagate exactly this single term through the real '
                             'DDP path (sidm|dism|sparse|smart|u|total); empty = production total')
    parser.add_argument('--opt', type=int, default=0)
    parser.add_argument('--lambda_u', type=float, default=None)
    parser.add_argument('--arm', default=None)
    parser.add_argument('--mask_rng_seed', type=int, default=99)
    parser.add_argument('--dtype', default='float32', choices=['float32', 'float64'],
                        help='diagnostic: run the whole step in this dtype to separate real scaling '
                             'errors from floating-point accumulation order')
    parser.add_argument('--out', required=True)
    parsed = parser.parse_args()

    import torch.distributed as dist
    if parsed.mode == 'rank':
        rank = int(os.environ['RANK'])
        world = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='gloo', rank=rank, world_size=world)
    else:
        rank, world = 0, 1
        # torch.distributed.nn.all_gather (the reference SmartCLIP gather) needs a process group
        # even for the single-process global-batch reference run.
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29799')
        dist.init_process_group(backend='gloo', rank=0, world_size=1)

    from model.said_cls_cvssl import SaidClsCvsslTrainModule
    from train_said_cls_cvssl import build_optimizers
    from model import longclip

    arm = parsed.arm or ('S0_smartclip' if parsed.case == 'S0' else 'C0_complement_vssl')
    lambda_u = parsed.lambda_u if parsed.lambda_u is not None else (
        0.0 if parsed.case == 'S0' else 1.0)

    local_batch = GLOBAL_BATCH // max(world, 1)
    payload = {'mode': parsed.mode, 'rank': rank, 'world': world, 'case': parsed.case,
               'arm': arm, 'lambda_u': lambda_u, 'static_graph': parsed.static_graph,
               'checkpoint_views': parsed.checkpoint_views}

    clip = build_init()
    if parsed.dtype == 'float64':
        clip = clip.double()
    payload['init_model_digest'] = param_digest(clip)
    payload['init_visual_digest'] = digest_tensor(clip.visual.proj.weight)
    payload['init_text_digest'] = digest_tensor(clip.text_proj.weight)
    payload['init_mask_digest'] = param_digest(clip.mask_net)
    payload['init_state'] = {name: value.detach().clone().tolist()
                             for name, value in clip.state_dict().items()}
    payload['n_backbone_param'] = len([p for p in clip.parameters()
                                       if not any(p is q for q in clip.mask_net.parameters())])
    payload['n_mask_param'] = len(list(clip.mask_net.parameters()))

    module = SaidClsCvsslTrainModule(
        clip, rank=rank, arm=arm, lambda_u=lambda_u, tau_u=0.1, rho=0.0,
        ddp_gradient_averaging=bool(parsed.ddp_gradient_averaging),
        grad_checkpoint_views=bool(parsed.checkpoint_views))
    if parsed.mode == 'rank' and parsed.ddp_only:
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            module, find_unused_parameters=True)
        if parsed.static_graph:
            ddp_model._set_static_graph()
    else:
        ddp_model = module

    args = argparse.Namespace(lr=1e-3, mask_lr=1e-2, weight_decay=1e-2)
    optimizer, mask_optimizer, n_backbone, n_mask = build_optimizers(module.clip, args)
    payload['n_backbone_optimized'] = n_backbone
    payload['n_mask_optimized'] = n_mask
    payload['backbone_group_names'] = [name for name, parameter in module.clip.named_parameters()
                                       if any(parameter is q
                                              for q in optimizer.param_groups[0]['params'])]
    payload['mask_group_names'] = [name for name, parameter in module.clip.named_parameters()
                                   if any(parameter is q
                                          for q in mask_optimizer.param_groups[0]['params'])]

    full = global_batch()
    start, stop = rank * local_batch, (rank + 1) * local_batch
    batch = slice_batch(full, start, stop) if parsed.mode == 'rank' else full
    text = longclip.tokenize(batch['caption'], truncate=True)
    payload['data'] = {
        'sample_id': batch['sample_id'].tolist(),
        'image_id': batch['image_id'].tolist(),
        'caption': batch['caption'],
        'text_token_digest': digest_tensor(text),
        'image_a_digest': digest_tensor(batch['image_a']),
        'image_b_digest': digest_tensor(batch['image_b']),
        'local_batch': int(batch['image_a'].shape[0]),
    }

    def forward():
        generator = torch.Generator().manual_seed(int(parsed.mask_rng_seed))
        return ddp_model(batch['image_a'].type(clip.dtype), batch['image_b'].type(clip.dtype),
                         text, batch['image_id'], generator)

    steps = []
    for step in range(1, parsed.steps + 1):
        out = forward()
        if parsed.terms == 'u':
            out['loss_total_for_backward'] = out['loss']            # gradient-isolation diagnostic
        elif parsed.terms == 'smart':
            out['loss_total_for_backward'] = out['loss_smart']      # gradient-isolation diagnostic
        record = {
            'step': step,
            'loss_sidm': float(out['loss_sidm']),
            'loss_dism': float(out['loss_dism']),
            'loss_sparsity': float(out['loss_sparsity']),
            'loss_smart': float(out['loss_smart']),
            'loss_smart_global_mean': float(out['loss_smart_global_mean']),
            'loss_vssl_local_backward': float(out['loss']),
            'loss_vssl_global_mean': float(out['loss_vssl_global_mean']),
            'loss_total_local': float(out['loss_total']),
            'loss_scale': float(out['vssl_loss_scale']),
            'global_sample_count': float(out['global_sample_count']),
            'valid_ab': float(out['vssl_valid_ab']),
            'valid_ba': float(out['vssl_valid_ba']),
            'actual_global_candidate_count': int(out['actual_global_candidate_count']),
        }
        if parsed.term_grads:
            terms = {
                'sidm': out['loss_sidm'],
                'dism': out['loss_dism'],
                'sparse': out['loss_sparsity'],
                'smart': out['loss_smart'],
                'u': lambda_u * out['loss'] if lambda_u > 0 else out['loss'] * 0.0,
                'total': out['loss_total_for_backward'],
            }
            parameters = [(name, parameter) for name, parameter in module.named_parameters()
                          if parameter.requires_grad]
            names = [name for name, _ in parameters]
            grads = [parameter for _, parameter in parameters]
            term_grads = {}
            for term_name, term in terms.items():
                if not term.requires_grad:
                    term_grads[term_name] = {name: None for name in names}
                    continue
                values = torch.autograd.grad(term, grads, retain_graph=True, allow_unused=True)
                term_grads[term_name] = {name: (None if value is None else value.tolist())
                                         for name, value in zip(names, values)}
            record['term_grads'] = term_grads
        else:
            if parsed.backward_term:
                gradient_term = {
                    'sidm': out['loss_sidm'],
                    'dism': out['loss_dism'],
                    'sparse': out['loss_sparsity'],
                    'smart': out['loss_smart'],
                    'u': lambda_u * out['loss'] if lambda_u > 0 else out['loss'] * 0.0,
                    'total': out['loss_total_for_backward'],
                }[parsed.backward_term]
                gradient_term.backward()
            else:
                out['loss_total_for_backward'].backward()
            record['grads'] = named_grads(module)
        if parsed.opt:
            optimizer.step()
            mask_optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            mask_optimizer.zero_grad(set_to_none=True)
            record['param_digest'] = param_digest(module)
            record['params'] = {name: value.detach().clone().tolist()
                                for name, value in module.named_parameters()}
            record['backbone_opt_digest'] = opt_digest(optimizer)
            record['mask_opt_digest'] = opt_digest(mask_optimizer)
            record['backbone_exp_avg'] = {
                name: optimizer.state[parameter]['exp_avg'].tolist()
                for name, parameter in module.named_parameters()
                if parameter in optimizer.state and 'exp_avg' in optimizer.state[parameter]}
            record['mask_exp_avg'] = {
                name: mask_optimizer.state[parameter]['exp_avg'].tolist()
                for name, parameter in module.named_parameters()
                if parameter in mask_optimizer.state and 'exp_avg' in mask_optimizer.state[parameter]}
        else:
            if parsed.term_grads:
                module.zero_grad(set_to_none=True)
        steps.append(record)
    payload['steps'] = steps

    with open(parsed.out.format(rank=rank), 'w') as handle:
        json.dump(payload, handle)
    if parsed.mode == 'rank':
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
