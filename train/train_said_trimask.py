"""SAID S0-TriMask trainer -- one trainer, two text-gate modes.

    soft    (v0.1, default)  mT = 0.1 + 0.9 * sigmoid(aT)
                             L_total = 10*L1 + 1*L2 + 1*L3 + 2*L_sparse_I
    hard_st (v0.2)           mT = hT + (pT - pT.detach()),  hT = (pT >= 0.5)
                             L_total = 10*L1 + 1*L2 + 1*L3 + 2*L_sparse_I + 0.2*L_sparse_T

The mode selects the objective/arm/phase names written into the config and the checkpoint, so a
checkpoint can never be silently reinterpreted in the other mode; the default stays ``soft``, which
keeps every v0.1 checkpoint and launcher exactly as it was.

Reused verbatim from the validated S0/C1 infrastructure (imported, never edited):
``seed_everything``, ``state_digest``, ``tensor_digest``, ``setup_distributed``, ``git_head``,
``load_init_state``, the reference dataset/caption stream, ``cosine_lr`` and the frozen canonical
evaluator. New here: the tri-path objective step, the parameter-group construction for the added
text branch, the pre-update gradient-finiteness gate and the checkpoint payload.

Distributed scaling, stated once because it is the part that is easy to get wrong:
the per-path cross-entropies are means over *this rank's* anchors, every cross-rank tensor is
gathered with the reference's autograd-aware ``all_gather``, and DDP averages the parameter
gradients over ranks by default. Under that combination

    mean_r dL_r/dp  ==  d(global anchor mean)/dp

so no extra ``world_size`` factor is applied anywhere -- including to the two sparsity terms, which
are plain means over this rank's own captions. This is the reference S0 convention (the baseline this
experiment is compared against) and it is verified against a single-process run in
``tests/test_said_trimask.py`` and ``tests/test_said_trimask_hs.py``. The frozen shared init is
loaded strictly, only the unused ``logit_scale`` (and the repository's frozen
``positional_embedding``) stay frozen and out of the optimizer, and the LR horizon stays at the full
three-epoch plan (``epochs * len(loader)``, ~3651) while ``--max_steps`` only truncates the run.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import longclip  # noqa: E402
from model.said_trimask import (GATE_MODE_TO_ARM, GATE_MODE_TO_OBJECTIVE,  # noqa: E402
                                GATE_MODE_TO_PHASE, HARD_GATE, LAMBDA_1, LAMBDA_2, LAMBDA_3,
                                LAMBDA_SPARSE_I, LAMBDA_SPARSE_T_HS, SOFT_GATE,
                                TEXT_GATE_MODES, TriMaskTrainModule)
from said_cvssl_data import Share4VCvsslDataset, cvssl_collate  # noqa: E402
from scheduler import cosine_lr  # noqa: E402
from train_said_cls_cvssl import (git_head, load_init_state, seed_everything,  # noqa: E402
                                  setup_distributed, state_digest, tensor_digest)

TOKENIZER_CONTEXT = 248
# This repository's CLIP builder freezes ``positional_embedding`` (the reference SmartCLIP/S0
# trainers inherit exactly this), so it is the only frozen tensor besides the unused logit_scale.
ALLOWED_FROZEN = ('positional_embedding', 'logit_scale')

GATE_DEFINITION = {
    SOFT_GATE: 'mT = 0.1 + 0.9 * sigmoid(aT); graded reweighting with a 0.1 floor',
    HARD_GATE: 'pT = sigmoid(aT), hT = (pT >= 0.5), mT = hT + (pT - pT.detach()); hard forward '
               '(exactly 0/1) with the sigmoid straight-through gradient, no floor',
}


def gradients_are_finite(module, device, world):
    """One all-rank-consistent finiteness verdict over every parameter gradient.

    Every rank evaluates the same predicate and participates in the same reduction, so all ranks
    agree; if any rank sees a non-finite gradient every rank refuses the update.
    """
    flags = [torch.isfinite(parameter.grad).all() for parameter in module.parameters()
             if parameter.grad is not None]
    if flags:
        finite = torch.stack(flags).all()
    else:
        finite = torch.ones((), dtype=torch.bool, device=device)
    verdict = torch.ones((), dtype=torch.float32, device=device)
    if not bool(finite):
        verdict.zero_()
    if world > 1 and dist.is_initialized():
        dist.all_reduce(verdict, op=dist.ReduceOp.MIN)
    return float(verdict) > 0.5


def inspected_module(model):
    """The underlying module: works for a DDP wrapper and for the bare module."""
    return getattr(model, 'module', model)


def check_checkpoint_compatibility(payload, arm, objective, text_gate_mode, lambda_sparse_t):
    """Refuse to resume a checkpoint in a different experiment.

    The gate mode is part of the experiment identity: a v0.1 payload carries no ``text_gate_mode``
    at all and may only be resumed by the soft mode, and a v0.2 payload may never be continued as
    the soft one. ``lambda_sparse_t`` is checked for the same reason -- the text sparsity term
    changes the objective, so continuing across a different coefficient is a different experiment.
    """
    if payload.get('objective') != objective or payload.get('arm') != arm:
        raise ValueError('resume objective/arm mismatch: checkpoint is %r/%r, this run is %r/%r'
                         % (payload.get('objective'), payload.get('arm'), objective, arm))
    checkpoint_mode = payload.get('text_gate_mode')
    if checkpoint_mode is None and text_gate_mode != SOFT_GATE:
        raise ValueError('this checkpoint predates text_gate_mode and is a soft-gate checkpoint; '
                         'it cannot be resumed as %r' % (text_gate_mode,))
    if checkpoint_mode is not None and checkpoint_mode != text_gate_mode:
        raise ValueError('resume text_gate_mode mismatch: %r vs %r'
                         % (checkpoint_mode, text_gate_mode))
    if float(payload.get('lambda_sparse_t', 0.0)) != float(lambda_sparse_t):
        raise ValueError('resume lambda_sparse_t mismatch: %r vs %r'
                         % (payload.get('lambda_sparse_t'), lambda_sparse_t))


def _json_scalar(value):
    """A JSON-safe scalar: non-finite floats become ``null`` instead of ``NaN``/``Infinity``.

    The dashboard parses these lines with a strict JSON reader, so emitting Python's non-standard
    ``NaN`` token would break it; an undefined diagnostic is reported as ``null``.
    """
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = float(value.detach())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def build_trimask_optimizers(clip_model, text_mask_net, args):
    """Two AdamW groups: the student backbone, and the original visual mask net plus the new
    text branch. Verified by parameter id: nothing registered twice, nothing left out, the frozen
    ``logit_scale`` excluded."""
    visual_mask_ids = {id(parameter) for parameter in clip_model.mask_net.parameters()}
    text_ids = {id(parameter) for parameter in text_mask_net.parameters()}
    if visual_mask_ids & text_ids:
        raise AssertionError('the visual and text mask branches share a parameter object')

    backbone, mask_group = [], []
    for parameter in clip_model.parameters():
        if not parameter.requires_grad:
            continue
        (mask_group if id(parameter) in visual_mask_ids else backbone).append(parameter)
    for parameter in text_mask_net.parameters():
        if parameter.requires_grad:
            mask_group.append(parameter)

    registered = [id(parameter) for parameter in backbone] + [id(parameter) for parameter in
                                                               mask_group]
    if len(registered) != len(set(registered)):
        raise AssertionError('a parameter is registered in more than one group')
    every_trainable = {id(parameter) for parameter in clip_model.parameters()
                       if parameter.requires_grad}
    every_trainable |= {id(parameter) for parameter in text_mask_net.parameters()
                        if parameter.requires_grad}
    if set(registered) != every_trainable:
        raise AssertionError('the optimizer groups do not cover exactly the trainable parameters')
    frozen = [name for name, parameter in clip_model.named_parameters()
              if not parameter.requires_grad]
    unexpected_frozen = [name for name in frozen if name not in ALLOWED_FROZEN]
    if unexpected_frozen:
        raise AssertionError('unexpected frozen parameters: %r' % (unexpected_frozen,))

    optimizer = torch.optim.AdamW(backbone, lr=args.lr, weight_decay=args.weight_decay)
    mask_optimizer = torch.optim.AdamW(mask_group, lr=args.mask_lr, weight_decay=0)
    return optimizer, mask_optimizer, len(backbone), len(mask_group)


def grad_health(module, grad_snapshot):
    """Per-group gradient norms from a post-backward, pre-update snapshot (rank-local)."""
    def norm_of(predicate):
        total = 0.0
        for name, grad in grad_snapshot.items():
            if grad is None or not predicate(name):
                continue
            total += float(grad.detach().float().pow(2).sum())
        return total ** 0.5

    missing = sum(1 for grad in grad_snapshot.values() if grad is None)
    zero = sum(1 for grad in grad_snapshot.values()
               if grad is not None and float(grad.detach().abs().max()) == 0.0)
    text_stem = norm_of(lambda name: name.startswith('text_mask_net.text_stem'))
    text_gate = norm_of(lambda name: name.startswith('text_mask_net.text_gate_projection'))
    visual_mask = norm_of(lambda name: name.startswith('clip.mask_net'))
    backbone = norm_of(lambda name: not name.startswith('clip.mask_net')
                       and not name.startswith('text_mask_net'))
    return {
        'grad_norm_backbone': backbone,
        'grad_norm_visual_mask_net': visual_mask,
        'grad_norm_text_stem': text_stem,
        'grad_norm_text_gate_projection': text_gate,
        'grad_norm_text_branch': (text_stem ** 2 + text_gate ** 2) ** 0.5,
        'grad_params_without_grad': float(missing),
        'grad_params_exactly_zero': float(zero),
    }


def trimask_train_step(ddp_model, batch, optimizer, mask_optimizer, device, amp_dtype,
                       amp_enabled=True, capture_grads=False, heavy_diagnostics=True,
                       world_size=1, reject_nonfinite=True):
    """One real training step: DDP forward -> backward -> finiteness gate -> both optimizer steps.

    No ``world_size`` scaling: see the module docstring. The ragged-batch guard runs before the
    first cross-rank gather so a length mismatch is reported instead of aborting inside gloo. The
    finiteness check happens *before* any optimizer step and is decided identically on every rank;
    there is no gradient clipping and no automatic loss adjustment.
    """
    from model import complement_visual_ssl as cvssl
    image_a = batch['image_a'].to(device, non_blocking=True)
    cvssl.assert_equal_local_batch(int(image_a.shape[0]))
    with torch.no_grad():
        text = longclip.tokenize(batch['caption_said'], truncate=True).to(device)

    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
        out = ddp_model(image_a, text, heavy_diagnostics=heavy_diagnostics)
        loss = out['loss_total_for_backward']
    loss.backward()
    module = inspected_module(ddp_model)
    finite = gradients_are_finite(module, device, world_size)
    out['grads_finite'] = torch.as_tensor(1.0 if finite else 0.0, device=device)
    if not finite:
        if reject_nonfinite:
            raise RuntimeError('non-finite parameter gradient: every rank refuses this update '
                               '(no clipping, no loss rescaling, no silent skip)')
        optimizer.zero_grad(set_to_none=True)
        mask_optimizer.zero_grad(set_to_none=True)
        return out
    if capture_grads:
        out['_grads'] = {name: (None if parameter.grad is None
                                else parameter.grad.detach().clone())
                         for name, parameter in module.named_parameters()}
        out['_grad_health'] = grad_health(module, out['_grads'])
    optimizer.step()
    mask_optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    mask_optimizer.zero_grad(set_to_none=True)
    return out


def write_mask_snapshot(path, out, completed, epoch, image_count, captions, git_head, gate_mode):
    """Atomically write the small per-coordinate mask snapshot the dashboard draws.

    Every number comes from tensors the step already computed (detached copies only): no extra
    forward, no evaluation, no GPU work for the page. The caption text recorded here is the sampled
    ``caption_said`` of the current batch -- the unsampled suffix never exists in the process.
    """
    mask_i = out['m_i'].detach().float()
    mask_t = out['m_t'].detach().float()
    hard_t = out.get('text_gate_hard')
    probability = out.get('text_gate_probability')
    first_caption = captions[0] if captions else None
    snapshot = {
        'completed_steps': int(completed),
        'epoch': int(epoch),
        'batch_size_local': int(image_count),
        'source': 'rank0/local_batch',
        'text_gate_mode': gate_mode,
        'git_head': git_head,
        'caption_note': 'current batch samples, not a fixed cohort; consecutive snapshots are '
                        'different captions and must not be chained into one sample trajectory',
        'mask_i_coordinate_mean': mask_i.mean(dim=0).cpu().tolist(),
        'mask_t_coordinate_mean': mask_t.mean(dim=0).cpu().tolist(),
        'mask_i_sample0': mask_i[0].cpu().tolist(),
        'mask_t_sample0': mask_t[0].cpu().tolist(),
        'intersection_coordinate_mean': (mask_i * mask_t).mean(dim=0).cpu().tolist(),
        'union_coordinate_mean': torch.maximum(mask_i, mask_t).mean(dim=0).cpu().tolist(),
        'text_gate_keep_ratio': float(mask_t.mean()),
        'visual_mask_keep_ratio': float(mask_i.mean()),
        'text_gate_zero_fraction': float((mask_t <= 0).float().mean()),
        'sample0': {'caption_said': first_caption,
                    'sha256': hashlib.sha256((first_caption or '').encode('utf-8')).hexdigest()[:16]},
    }
    if probability is not None:
        snapshot['text_gate_probability_sample0'] = probability.detach().float()[0].cpu().tolist()
    if hard_t is not None:
        snapshot['text_gate_hard_sample0'] = hard_t.detach().float()[0].cpu().tolist()
    temporary = path + '.tmp'
    with open(temporary, 'w') as handle:
        json.dump(snapshot, handle)
    os.replace(temporary, path)


def checkpoint_payload(clip_model, train_module, optimizer, mask_optimizer, args, config,
                       completed, epoch, step_in_epoch, digests, initial_digest,
                       lr_horizon_steps):
    return {
        'model': clip_model.state_dict(),
        'text_mask_net': train_module.text_mask_net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'mask_optimizer': mask_optimizer.state_dict(),
        'completed_steps': int(completed),
        'epoch': int(epoch),
        'step_in_epoch': int(step_in_epoch),
        'next_batch_index': int(step_in_epoch) + 1,
        'phase': config['phase'],
        'arm': config['arm'],
        'objective': config['objective'],
        'text_gate_mode': config['text_gate_mode'],
        'lambda_sparse_t': config['lambda_sparse_t'],
        'config': config,
        'precision': 'fp32 master + %s autocast' % args.amp_dtype,
        'lr_horizon_steps': int(lr_horizon_steps),
        'git_head': config['git_head'],
        'initial_state_sha256': initial_digest,
        'digests': digests,
    }


def main():
    parser = argparse.ArgumentParser(description='SAID S0-TriMask / S0-TriMask-HS')
    parser.add_argument('--text_gate_mode', default=SOFT_GATE, choices=list(TEXT_GATE_MODES),
                        help='soft = v0.1 graded gate (default, unchanged); '
                             'hard_st = v0.2 hard forward with the straight-through gradient')
    parser.add_argument('--arm', default=None, choices=list(GATE_MODE_TO_ARM.values()))
    parser.add_argument('--objective', default=None,
                        choices=list(GATE_MODE_TO_OBJECTIVE.values()))
    parser.add_argument('--lambda_1', type=float, default=LAMBDA_1)
    parser.add_argument('--lambda_2', type=float, default=LAMBDA_2)
    parser.add_argument('--lambda_3', type=float, default=LAMBDA_3)
    parser.add_argument('--lambda_sparse_i', type=float, default=LAMBDA_SPARSE_I)
    parser.add_argument('--lambda_sparse_t', type=float, default=None,
                        help='default: 0.0 for the soft mode (v0.1 behaviour) and %.2f for hard_st'
                             % LAMBDA_SPARSE_T_HS)
    parser.add_argument('--text_mask_width', type=int, default=512)
    parser.add_argument('--text_mask_layers', type=int, default=1)
    parser.add_argument('--text_mask_heads', type=int, default=8)
    parser.add_argument('--text_mask_seed', type=int, default=0)
    parser.add_argument('--base_model', default='B16')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--mask_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--warmup_length', type=int, default=200)
    parser.add_argument('--log_scale', type=float, default=4.6052)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--init_state', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--save_completed_steps', default='0')
    parser.add_argument('--log_every', type=int, default=10,
                        help='scalar log interval (the v0.2 default is 10 steps)')
    parser.add_argument('--heavy_log_every', type=int, default=25,
                        help='interval for the heavier detached statistics (intersection, variance '
                             'decomposition); lighter steps skip that block entirely')
    parser.add_argument('--grad_health_steps', default='1,20,100,250,500')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--amp_dtype', default='bf16', choices=['bf16', 'fp32'])
    parser.add_argument('--grad_checkpoint_views', type=int, default=1)
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()
    if args.base_model == 'B16':
        args.base_model = 'ViT-B/16'
    elif args.base_model == 'L14':
        args.base_model = 'ViT-L/14'
    # the mode owns the names: an explicit conflicting --arm/--objective is an error, never a
    # silently reinterpreted checkpoint
    arm = GATE_MODE_TO_ARM[args.text_gate_mode]
    objective = GATE_MODE_TO_OBJECTIVE[args.text_gate_mode]
    phase = GATE_MODE_TO_PHASE[args.text_gate_mode]
    if args.arm is not None and args.arm != arm:
        raise SystemExit('--arm %s contradicts --text_gate_mode %s (expected %s)'
                         % (args.arm, args.text_gate_mode, arm))
    if args.objective is not None and args.objective != objective:
        raise SystemExit('--objective %s contradicts --text_gate_mode %s (expected %s)'
                         % (args.objective, args.text_gate_mode, objective))
    args.arm, args.objective = arm, objective
    if args.lambda_sparse_t is None:
        args.lambda_sparse_t = (LAMBDA_SPARSE_T_HS if args.text_gate_mode == HARD_GATE else 0.0)
    for name in ('lambda_1', 'lambda_2', 'lambda_3', 'lambda_sparse_i', 'lambda_sparse_t'):
        if getattr(args, name) < 0.0:
            raise SystemExit('%s must be non-negative' % name)
    if args.text_gate_mode == SOFT_GATE and args.lambda_sparse_t != 0.0:
        raise SystemExit('the soft (v0.1) mode must keep lambda_sparse_t = 0')

    seed_everything(args.seed)
    rank, local_rank, world = setup_distributed()
    device = torch.device('cuda', local_rank)
    save_completed = sorted({int(s) for s in args.save_completed_steps.split(',') if s.strip()})
    grad_health_steps = {int(s) for s in args.grad_health_steps.split(',') if s.strip()}

    model, _ = longclip.load_from_clip(args.base_model, device='cpu', download_root=None,
                                       args=args)
    model.train()
    model.logit_scale = torch.nn.Parameter(torch.ones([]) * args.log_scale)
    model = model.to(device)
    load_init_state(model, args.init_state, rank)
    # the fixed 100x SmartCLIP scale is used by every path, so logit_scale is unused here: it keeps
    # its compatibility key but never receives a gradient and never enters the optimizer
    model.logit_scale.requires_grad_(False)
    initial_digest = state_digest(model.state_dict())

    train_module = TriMaskTrainModule(
        model, rank=rank, lambda_1=args.lambda_1, lambda_2=args.lambda_2,
        lambda_3=args.lambda_3, lambda_sparse_i=args.lambda_sparse_i,
        lambda_sparse_t=args.lambda_sparse_t,
        text_mask_width=args.text_mask_width, text_mask_layers=args.text_mask_layers,
        text_mask_heads=args.text_mask_heads, text_mask_seed=args.text_mask_seed,
        text_gate_mode=args.text_gate_mode,
        grad_checkpoint_views=bool(args.grad_checkpoint_views)).to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        train_module, device_ids=[local_rank], output_device=local_rank,
        find_unused_parameters=True)
    ddp_model._set_static_graph()
    clip_model = train_module.clip
    text_mask_net = train_module.text_mask_net

    optimizer, mask_optimizer, n_backbone, n_mask = build_trimask_optimizers(
        clip_model, text_mask_net, args)
    n_text_branch = sum(parameter.numel() for parameter in text_mask_net.parameters())
    use_amp = args.amp_dtype == 'bf16'
    amp_dtype = torch.bfloat16

    dtype_counts = {}
    for parameter in clip_model.parameters():
        key = str(parameter.dtype)
        dtype_counts[key] = dtype_counts.get(key, 0) + 1
    for parameter in text_mask_net.parameters():
        key = 'text_branch:' + str(parameter.dtype)
        dtype_counts[key] = dtype_counts.get(key, 0) + 1
    group_dtypes = {
        'backbone': sorted({str(p.dtype) for p in optimizer.param_groups[0]['params']}),
        'mask_group': sorted({str(p.dtype) for p in mask_optimizer.param_groups[0]['params']}),
    }
    print('DTYPE_AUDIT rank=%d parameter_dtypes=%s backbone_group=%s mask_group=%s '
          'text_branch_params=%d amp_enabled=%s amp_dtype=%s logit_scale_requires_grad=%s'
          % (rank, sorted(dtype_counts.items()), group_dtypes['backbone'],
             group_dtypes['mask_group'], n_text_branch, use_amp, args.amp_dtype,
             bool(clip_model.logit_scale.requires_grad)), flush=True)
    if dtype_counts != {'torch.float32': len(list(clip_model.parameters())),
                        'text_branch:torch.float32': len(list(text_mask_net.parameters()))}:
        raise RuntimeError('fp32 master weights expected, got %r' % (dtype_counts,))
    if group_dtypes != {'backbone': ['torch.float32'], 'mask_group': ['torch.float32']}:
        raise RuntimeError('optimizer groups must be fp32, got %r' % (group_dtypes,))

    dataset = Share4VCvsslDataset(seed=args.seed, augment_view_b=True,
                                  strict_manifest=os.environ.get('SHARE4V_FULL_AUDIT'))
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, seed=args.seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                                         num_workers=args.num_workers, pin_memory=True,
                                         collate_fn=cvssl_collate, drop_last=False)
    steps_per_epoch = len(loader)
    total_steps = args.epochs * steps_per_epoch
    scheduler = cosine_lr(optimizer, base_lr=args.lr, warmup_length=args.warmup_length,
                          steps=total_steps)
    mask_scheduler = cosine_lr(mask_optimizer, base_lr=args.mask_lr, warmup_length=0,
                               steps=total_steps)

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, 'salu_log.jsonl')
    config = {
        'objective': objective,
        'arm': arm,
        'phase': phase,
        'text_gate_mode': args.text_gate_mode,
        'paths': {
            'path1': 'MASK image - native text: 100*dot(Norm(v_i*mI_j), Norm(t_raw_j))',
            'path2': 'native image - MASK text: 100*dot(Norm(v_i), Norm(t_raw_j*mT_j))',
            'path3': 'MASK image - MASK text: 100*dot(Norm(v_i*mI_j), Norm(t_raw_j*mT_j))',
            'shared_masks': 'one mI and one mT per caption, reused by all three paths; no third '
                            'mask head, no resampling, no detach of mI/mT',
        },
        'loss': 'L_total = lambda1*L1 + lambda2*L2 + lambda3*L3 + lambda_sparse_i*L_sparse_I '
                '+ lambda_sparse_t*L_sparse_T',
        'Lk': 'Lk_I2T + Lk_T2I, both directions, no 0.5 factor',
        'lambda_1': args.lambda_1, 'lambda_2': args.lambda_2, 'lambda_3': args.lambda_3,
        'lambda_sparse_i': args.lambda_sparse_i, 'lambda_sparse_t': args.lambda_sparse_t,
        'lambda_adv': 0.0,
        'sparse_definition': 'mean_d abs(m_i,d) of the HARD straight-through mask, over this rank\'s '
                             'own captions, each side counted exactly once (never tripled for the '
                             'three paths); abs is kept so the subgradient at a closed coordinate is '
                             'zero, which is the definition this version adopts',
        'visual_mask': 'hard straight-through mask_net(C_S hidden) exactly as the S0 Said forward',
        'text_gate': GATE_DEFINITION[args.text_gate_mode],
        'text_mask_init': 'gate weight 0 and bias log(8) => pT = 8/9 => hT = 1 => '
                          'Norm(t*mT) == Norm(t) and Q3 == Q1 at step 0',
        'text_branch': {'width': args.text_mask_width, 'layers': args.text_mask_layers,
                        'heads': args.text_mask_heads, 'seed': args.text_mask_seed,
                        'params': n_text_branch,
                        'rng_isolation': 'torch.random.fork_rng() (CPU and every initialised CUDA '
                                         'generator), not CPU only'},
        'candidate_rule': 'reference compute_smartclip_terms: every global caption is a candidate; '
                          'no duplicate-caption filtering exists in S0 and none is added here',
        'float_scale': 'fixed 100.0 SmartCLIP scale on every path; logit_scale unused and frozen',
        'mask_generation_input': 'text only (mI and mT are both functions of the caption hidden '
                                 'state); the image never conditions a mask',
        'precision': 'fp32 master + %s autocast for the CLIP backbone; explicit fp32 for the text '
                     'branch, the gating, the normalisation, the masked norms, the similarities '
                     'and the two cross-entropies' % args.amp_dtype,
        'ddp_scaling': 'per-rank anchor means + autograd-aware all_gather + standard DDP averaging; '
                       'NO world_size factor (reference S0 convention)',
        'grad_finiteness_gate': 'all ranks evaluate the same predicate and all_reduce(MIN) it before '
                                'any optimizer step; a non-finite gradient aborts the run instead of '
                                'being clipped or skipped',
        'log_cadence': {'scalars_every': args.log_every, 'heavy_stats_every': args.heavy_log_every},
        'view_a': 'reference openai-clip _transform(224)',
        'view_b': 'produced by the loader to keep the reference caption stream identical but NOT '
                  'used by the model (one image view only, no visual SSL term)',
        'sampling': 'DistributedSampler(shuffle=True), no drop_last, steps_per_epoch=len(loader)',
        'batch_size_per_gpu': args.batch_size, 'global_pairs': args.batch_size * world,
        'world_size': world, 'epochs': args.epochs, 'loader_batches': steps_per_epoch,
        'lr_horizon_steps': total_steps, 'warmup_length': args.warmup_length,
        'mask_warmup_length': 0, 'lr': args.lr, 'mask_lr': args.mask_lr,
        'weight_decay': args.weight_decay, 'grad_checkpoint_views': bool(args.grad_checkpoint_views),
        'online_check': 'the first 20 steps are part of this single 500-step run, not a separate run',
        'tokenizer_context': TOKENIZER_CONTEXT,
        'init_state': args.init_state, 'initial_state_sha256': initial_digest,
        'git_head': git_head(),
    }
    if rank == 0:
        print('CONFIG ' + json.dumps(config, sort_keys=True), flush=True)
        print('DATASET_SIZE %d STEPS_PER_EPOCH %d LR_HORIZON %d BACKBONE_TENSORS %d '
              'MASK_TENSORS %d TEXT_BRANCH_PARAMS %d'
              % (len(dataset), steps_per_epoch, total_steps, n_backbone, n_mask, n_text_branch),
              flush=True)
        with open(os.path.join(args.output_dir, 'config.json'), 'w') as handle:
            json.dump(config, handle, indent=2, sort_keys=True)

    caption_digest = hashlib.sha256()
    sample_digest = hashlib.sha256()
    view_digest = hashlib.sha256()
    view_a_pixels_sha = None
    digests = {'caption_stream_sha256': None, 'sample_stream_sha256': None}
    completed = 0
    start_epoch, start_batch = 0, 0
    if args.resume:
        payload = torch.load(args.resume, map_location='cpu', weights_only=False)
        check_checkpoint_compatibility(payload, arm, objective, args.text_gate_mode,
                                       args.lambda_sparse_t)
        clip_model.load_state_dict(payload['model'])
        text_mask_net.load_state_dict(payload['text_mask_net'])
        optimizer.load_state_dict(payload['optimizer'])
        mask_optimizer.load_state_dict(payload['mask_optimizer'])
        completed = int(payload['completed_steps'])
        start_epoch = int(payload.get('epoch', 0))
        start_batch = int(payload.get('next_batch_index', int(payload.get('step_in_epoch', -1)) + 1))
        if completed != start_epoch * steps_per_epoch + start_batch:
            raise ValueError('resume cursor disagrees with completed steps')
        if int(payload['lr_horizon_steps']) != total_steps:
            raise ValueError('resume must preserve the LR horizon')
        if rank == 0:
            print('RESUMED from %s at %d' % (args.resume, completed), flush=True)
    elif 0 in save_completed:
        # the shared initialisation itself, saved as step 0 before any update
        if rank == 0:
            torch.save(checkpoint_payload(clip_model, train_module, optimizer, mask_optimizer,
                                          args, config, 0, 0, -1, dict(digests), initial_digest,
                                          total_steps),
                       os.path.join(args.output_dir, 'trimask_%s_step%06d.pt' % (arm, 0)))
            print('SAVED step 0 (shared initialisation, no update applied)', flush=True)

    t_start = time.time()
    compute_times = []
    stopped = False
    for epoch in range(args.epochs):
        if epoch < start_epoch:
            continue
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        for i, batch in enumerate(loader):
            if epoch == start_epoch and i < start_batch:
                continue
            if args.max_steps is not None and completed >= args.max_steps:
                stopped = True
                break
            is_last = args.max_steps is not None and (completed + 1) >= args.max_steps
            # the heavy detached statistics and the mask snapshot only on the requested cadence;
            # every rank takes the same branch, and the block is no_grad so the parameter set the
            # reducer sees is identical either way
            heavy = (completed == 0 or (completed + 1) % args.heavy_log_every == 0
                     or (completed + 1) in save_completed or is_last)
            t0 = time.time()
            scheduler(completed)
            mask_scheduler(completed)
            want_grads = (completed + 1) in grad_health_steps
            out = trimask_train_step(ddp_model, batch, optimizer, mask_optimizer, device,
                                     amp_dtype, amp_enabled=use_amp, capture_grads=want_grads,
                                     heavy_diagnostics=heavy, world_size=world)
            image_a = batch['image_a']
            completed += 1
            compute_times.append(time.time() - t0)

            caption_digest.update(('\n'.join(batch['caption_said'])).encode('utf-8'))
            sample_digest.update(batch['sample_id'].numpy().tobytes())
            view_digest.update(batch['view_b_resample_size'].numpy().tobytes())
            view_digest.update(batch['view_b_blur_sigma'].numpy().tobytes())
            digests['caption_stream_sha256'] = caption_digest.hexdigest()
            digests['sample_stream_sha256'] = sample_digest.hexdigest()
            if view_a_pixels_sha is None:
                view_a_pixels_sha = tensor_digest(batch['image_a'][0])

            if rank == 0 and (completed % args.log_every == 0 or completed in (1, 20)
                              or is_last or completed in save_completed):
                record = {
                    'completed_steps': completed,
                    'epoch': epoch,
                    'step_in_epoch': i,
                    'next_batch_index': i + 1,
                    'arm': arm,
                    'objective': objective,
                    'text_gate_mode': args.text_gate_mode,
                    'heavy_diagnostics': heavy,
                    'lr': optimizer.param_groups[0]['lr'],
                    'mask_lr': mask_optimizer.param_groups[0]['lr'],
                    'batch_size_local': int(image_a.shape[0]),
                    'global_pairs': int(image_a.shape[0]) * world,
                    'world_size': world,
                    'statistics_scope': 'rank0/local_batch',
                    'rank_param_digest': state_digest(clip_model.state_dict())[:16],
                    'rank_text_branch_digest': state_digest(text_mask_net.state_dict())[:16],
                    'sec_per_step': compute_times[-1],
                    'samples_per_sec': float(image_a.shape[0]) * world / max(compute_times[-1],
                                                                           1e-9),
                    'peak_memory_gb': torch.cuda.max_memory_allocated(device) / (1024 ** 3),
                    'batch_caption_sha256': hashlib.sha256(
                        ('\n'.join(batch['caption_said'])).encode('utf-8')).hexdigest()[:16],
                    'view_b_constructed_but_unused': True,
                }
                for key, value in out.items():
                    if torch.is_tensor(value) and value.numel() == 1:
                        record[key] = _json_scalar(value)
                    elif isinstance(value, (float, int)) or value is None:
                        record[key] = _json_scalar(value)
                if '_grad_health' in out:
                    record.update(out['_grad_health'])
                with open(log_path, 'a') as handle:
                    handle.write('LOG ' + json.dumps(record, sort_keys=True) + '\n')
                print('LOG ' + json.dumps(record, sort_keys=True), flush=True)
            if rank == 0 and heavy:
                # the dashboard's mask grid: a few kB, overwritten on the heavy cadence, built from
                # tensors this step already produced (independent of the scalar log cadence)
                write_mask_snapshot(os.path.join(args.output_dir, 'mask_snapshot.json'), out,
                                    completed, epoch, int(image_a.shape[0]),
                                    list(batch['caption_said']), config['git_head'],
                                    args.text_gate_mode)
            # the gradient snapshot is a test/health hook only: never keep it alive between steps
            out.pop('_grads', None)
            out.pop('_grad_health', None)
            # the trainer logs scalars only; the forward tensors must not be kept across steps
            for key in ('v_a', 't_raw', 'q1', 'q2', 'q3', 'm_i', 'm_t', 'soft_mask_i',
                        'mask_logits_i', 'text_gate_logits', 'text_gate_probability',
                        'text_gate_hard'):
                out.pop(key, None)

            if rank == 0 and completed in save_completed:
                path = os.path.join(args.output_dir,
                                    'trimask_%s_step%06d.pt' % (arm, completed))
                torch.save(checkpoint_payload(clip_model, train_module, optimizer, mask_optimizer,
                                              args, config, completed, epoch, i,
                                              {'initial_state_sha256': initial_digest,
                                               **digests,
                                               'view_a_pixels_sha256_step0': view_a_pixels_sha,
                                               'view_b_param_stream_sha256':
                                                   view_digest.hexdigest()},
                                              initial_digest, total_steps), path)
                print('SAVED ' + path, flush=True)
        if stopped:
            break

    if rank == 0:
        summary = {
            'arm': arm, 'objective': objective, 'phase': phase,
            'text_gate_mode': args.text_gate_mode, 'completed_steps': completed,
            'epochs': args.epochs, 'wall_sec': time.time() - t_start,
            'mean_sec_per_step': sum(compute_times) / max(len(compute_times), 1),
            'steps_per_epoch': steps_per_epoch, 'lr_horizon_steps': total_steps,
            'initial_state_sha256': initial_digest,
            'caption_stream_sha256': caption_digest.hexdigest(),
            'sample_stream_sha256': sample_digest.hexdigest(),
            'view_a_pixels_sha256_step0': view_a_pixels_sha,
            'peak_memory_gb': torch.cuda.max_memory_allocated(device) / (1024 ** 3),
            'lambda_1': args.lambda_1, 'lambda_2': args.lambda_2, 'lambda_3': args.lambda_3,
            'lambda_sparse_i': args.lambda_sparse_i, 'lambda_sparse_t': args.lambda_sparse_t,
        }
        with open(os.path.join(args.output_dir, 'run_summary.json'), 'w') as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print('RUN_SUMMARY ' + json.dumps(summary, sort_keys=True), flush=True)
        if args.max_steps is not None and completed != args.max_steps:
            print('INCOMPLETE_RUN completed=%d expected=%d' % (completed, args.max_steps), flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
