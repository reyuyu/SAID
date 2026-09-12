"""SAID-Token v1 trainer -- arms ``T1_said_token_reconstruction`` and ``T0_said_token_only``.

One objective, one graph:

    L = L_Said + lambda_rec * L_rec          (lambda_rec = 0.1 for T1, 0.0 for T0)

Everything that is *not* in that line is deliberately absent: no global image-text loss, no global
warm-up stage, no hidden alignment weight, no NCE/contrastive term, no mask sparsity penalty, no
EMA, no external teacher. The native CLS and the native EOS stay inside the fine-grained matching
set, and the native student CLS is what the retrieval evaluation reads.

Reused verbatim from the existing trainers: ``seed_everything``, ``state_digest``, ``tensor_digest``,
``setup_distributed``, ``git_head``, ``load_init_state``, ``cosine_lr`` and the data plumbing
(``Share4VCvsslDataset`` + ``cvssl_collate``). New here: the three optimizer groups, the DDP scaling
bookkeeping for the two loss terms, the step-0 save, and the fixed-cohort input-dependency
diagnostics.

Budget note, recorded rather than hidden: per sample T1 encodes the student image once (with graph),
the student text once, and the frozen reference image once (no grad). That is not the same FLOPs as
the C0 two-view step; the arms are comparable only because the step budget, the data and the init are
matched.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import longclip  # noqa: E402
from model.said_token_v1 import (ARM_LAMBDA_REC, ARM_T0, ARM_T1, ARMS, EPS,  # noqa: E402
                                 N_SLOTS, K_SAID, SaidTokenV1Module,
                                 reference_fingerprint, text_content_mask, complement_pool,
                                 intra_image_slot_cos, _pairwise_cos)
from said_cvssl_data import Share4VCvsslDataset, cvssl_collate  # noqa: E402
from scheduler import cosine_lr  # noqa: E402
from train_said_cls_cvssl import (git_head, load_init_state, seed_everything,  # noqa: E402
                                  setup_distributed, state_digest, tensor_digest)

TOKENIZER_CONTEXT = 248
EOT_ID = 49407


def inspected_module(model):
    """The underlying module: works for a DDP wrapper and for the bare module."""
    return getattr(model, 'module', model)


# --------------------------------------------------------------------------- #
# optimizers
# --------------------------------------------------------------------------- #
def build_optimizers(module, args):
    """Three disjoint groups, and a proof that they cover exactly the trainable parameters.

    * student backbone (contains the original image/text projections): lr 1e-6, wd 1e-2
    * the two aggregators and the router:                              lr 2e-4, wd 1e-2
    * the reconstruction decoder:                                      lr 1e-4, wd 0

    The frozen reference tower, the historic ``mask_net`` and the unused ``logit_scale`` are in no
    group: they stay in the checkpoint so the exported student still loads strictly, but nothing here
    can move them.
    """
    frozen = []
    for name in ('mask_net', 'logit_scale'):
        holder = getattr(module.clip, name, None)
        if holder is None:
            continue
        frozen.extend(list(holder.parameters()) if hasattr(holder, 'parameters') else [holder])
    frozen_ids = {id(parameter) for parameter in frozen}
    trainable = [p for p in module.parameters() if p.requires_grad]
    if frozen_ids & {id(p) for p in trainable}:
        raise RuntimeError('a tensor declared frozen still requires grad: mask_net or logit_scale')

    backbone = [p for p in module.clip.parameters() if p.requires_grad]
    extra = [p for p in list(module.image_aggregator.parameters())
             + list(module.text_aggregator.parameters())
             + list(module.router.parameters()) if p.requires_grad]
    decoder = [p for p in module.decoder.parameters() if p.requires_grad]
    grouped = backbone + extra + decoder
    if len({id(p) for p in grouped}) != len(grouped):
        raise RuntimeError('a parameter is in two optimizer groups')
    if {id(p) for p in grouped} != {id(p) for p in trainable}:
        missing = [n for n, p in module.named_parameters()
                   if p.requires_grad and id(p) not in {id(q) for q in grouped}]
        raise RuntimeError('trainable parameters outside every group: %r' % (missing[:8],))
    if not backbone or not extra or not decoder:
        raise RuntimeError('an optimizer group is empty: backbone %d module %d decoder %d'
                           % (len(backbone), len(extra), len(decoder)))

    optimizer = torch.optim.AdamW(backbone, lr=args.lr, weight_decay=args.weight_decay)
    module_optimizer = torch.optim.AdamW(extra, lr=args.module_lr, weight_decay=args.module_wd)
    decoder_optimizer = torch.optim.AdamW(decoder, lr=args.decoder_lr, weight_decay=args.decoder_wd)
    return (optimizer, module_optimizer, decoder_optimizer,
            len(backbone), len(extra), len(decoder))


def check_equal_local_batch(batch_size, device, world_size):
    """Every rank must feed the same local batch size, before any collective inside the model.

    The module builds a *global* pair grid by gathering the local batches, so unequal local sizes
    would misalign the grid and silently rank the wrong candidates. The check runs on all ranks and
    raises identically everywhere, which is what makes it safe to run before the first big gather.
    """
    if world_size <= 1:
        return batch_size
    local = torch.tensor([int(batch_size)], device=device, dtype=torch.long)
    gathered = [torch.zeros_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    sizes = [int(value.item()) for value in gathered]
    if len(set(sizes)) != 1:
        print('RAGGED_BATCH sizes=%r (drop_last must keep the ranks equal)' % (sizes,),
              file=sys.stderr, flush=True)
        raise RuntimeError('unequal per-rank batch sizes %r: the global pair grid would be '
                           'misaligned' % (sizes,))
    return sizes[0]


# --------------------------------------------------------------------------- #
# one training step
# --------------------------------------------------------------------------- #
def t1_train_step(ddp_model, batch, optimizers, device, amp_dtype, amp_enabled=True,
                  world_size=1, capture_grads=False, diagnostics=False, health_check=False):
    """DDP forward -> backward -> optimizer steps, plus the global means of both loss terms.

    The module already scales its backward value the way plain DDP averaging needs:
    ``W * local_sum / global_count`` for both terms. That is why the *global* mean is recomputed here
    from the all-reduced detached sums and counts rather than read off the amplified local scalar, and
    why the identity ``all_reduce(loss) / W == global_mean`` is asserted on every step.
    """
    optimizer, module_optimizer, decoder_optimizer = optimizers
    image_a = batch['image_a'].to(device, non_blocking=True)
    image_ids = batch['image_id'].to(device, non_blocking=True)
    with torch.no_grad():
        text = longclip.tokenize(batch['caption_said'], truncate=True).to(device)

    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
        out = ddp_model(image_a, text, image_ids, EOT_ID, diagnostics=diagnostics)

    loss = out['loss_total_for_backward']
    with torch.profiler.record_function('t1_backward_collectives'):
        loss.backward()

    if health_check:
        module = inspected_module(ddp_model)
        # DDP wraps returned tensors in its output sink; inspect the original graph tensor.
        positive = module._positive_live_for_check
        if positive is not None:
            if positive.grad is None:
                raise RuntimeError('positive score has no backward gradient')
            positive_grad = positive.grad.detach()
            out['positive_score_grad_mean'] = positive_grad.mean()
            out['positive_score_grad_min'] = positive_grad.min()
            out['positive_score_grad_max'] = positive_grad.max()
            if bool((positive_grad > 1e-8).any()):
                raise RuntimeError('positive score gradient has wrong sign')
        norms = []
        for prefix in ('clip.visual', 'clip.transformer', 'image_aggregator', 'text_aggregator', 'router', 'decoder'):
            grads = [p.grad.detach().float() for n, p in module.named_parameters()
                     if n.startswith(prefix) and p.grad is not None]
            norm = torch.stack([g.square().sum() for g in grads]).sum().sqrt() if grads else loss.new_zeros(())
            out['grad_norm_' + prefix.replace('.', '_')] = norm
            norms.append(norm)
        probe = torch.stack(norms)
        finite = torch.isfinite(probe).all().int()
        if world_size > 1:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            lo, hi = probe.clone(), probe.clone()
            dist.all_reduce(lo, op=dist.ReduceOp.MIN)
            dist.all_reduce(hi, op=dist.ReduceOp.MAX)
            if not torch.allclose(lo, hi, atol=1e-6, rtol=1e-5):
                raise RuntimeError('DDP gradient health check lost synchronization')
            out['grad_norm_rank_max_diff'] = (hi-lo).abs().max()
        if not bool(finite):
            raise RuntimeError('non-finite gradients; optimizer step refused')

    if capture_grads:
        module = inspected_module(ddp_model)
        out['_grads'] = {name: (None if p.grad is None else p.grad.detach().clone())
                         for name, p in module.named_parameters()}

    optimizer.step()
    module_optimizer.step()
    decoder_optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    module_optimizer.zero_grad(set_to_none=True)
    decoder_optimizer.zero_grad(set_to_none=True)

    # Both objectives use local work sums and globally reduced valid counts.
    totals = torch.stack([out['loss_said_i2t_hinge_sum'], out['loss_said_t2i_hinge_sum'],
        out['loss_said_pairs_local'], out['loss_rec_sum_local'], out['loss_rec_valid_local'],
        out['loss_said'].detach(), out['loss_rec'].detach(), out['positive_score_sum_local'],
        out['negative_score_sum_local'], out['margin_active_local']])
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(totals)
    first, second, pairs, rec_sum, rec_valid, said_bw, rec_bw, pos, neg, active = totals
    said_global = (first + second) / pairs.clamp_min(1)
    rec_global = rec_sum / rec_valid.clamp_min(1)
    out.update(loss_said_global_mean=said_global, loss_rec_global_mean=rec_global,
        loss_total_global_mean=said_global + float(out['lambda_rec'])*rec_global,
        said_scaling_abs_diff=(said_bw/world_size-said_global).abs(),
        rec_scaling_abs_diff=(rec_bw/world_size-rec_global).abs(),
        loss_said_backward_value=out['loss_said'].detach(),
        loss_rec_backward_value=out['loss_rec'].detach(),
        positive_said_score=pos/(image_a.shape[0]*world_size),
        negative_said_score=neg/pairs.clamp_min(1), margin_active_fraction=active/pairs.clamp_min(1))
    return out


# --------------------------------------------------------------------------- #
# fixed-cohort diagnostics
# --------------------------------------------------------------------------- #
def build_fixed_cohort(dataset, device, size=64):
    """The *same* 64 images/captions for every checkpoint of the run.

    Index-based and deterministic on purpose: view A is the reference 224 preprocessing and the
    sample-level seeding is stateless, so these tensors do not depend on the current epoch, on the
    shuffling, or on which batch happens to be in flight. (Using "the first 64 of the current batch"
    would make two checkpoints' numbers incomparable, which is exactly what this table is for.)
    """
    subset = torch.utils.data.Subset(dataset, list(range(min(size, len(dataset)))))
    loader = torch.utils.data.DataLoader(subset, batch_size=len(subset), shuffle=False,
                                         num_workers=0, collate_fn=cvssl_collate)
    batch = next(iter(loader))
    text = longclip.tokenize(batch['caption_said'], truncate=True).to(device)
    return {'images': batch['image_a'].to(device),
            'text': text,
            'image_id': batch['image_id'].to(device),
            'captions': list(batch['caption_said']),
            'sample_id': batch['sample_id'].tolist()}


def input_dependency_diagnostics(ddp_model, cohort, device):
    """No-grad intervention table for the decoder, evaluated on the fixed cohort.

    The mask is computed ONCE and held fixed across the interventions, so only the decoder inputs
    change. ``u_wrong`` is the complement of a *permuted* image's slots combined with this image's
    mask (a fixed derangement, never a self-match), which is the honest "wrong content, right mask"
    probe. ``R@1`` ranks each prediction against the frozen teacher bank of the same 64 images.
    """
    module = inspected_module(ddp_model)
    images = cohort['images'].float()
    text = cohort['text']
    rows = images.shape[0]
    with torch.no_grad():
        with torch.autocast(device_type=device.type, enabled=False):
            g_i_raw, patch_raw = module.encode_image_tokens(images)
            g_t_raw, text_local = module.encode_text_tokens(text)
            valid, empty = text_content_mask(text, EOT_ID)
            v_slots = module.image_aggregator(patch_raw)[0]
            t_slots = module.text_aggregator(text_local, valid=valid)[0]
            g_i = F.normalize(g_i_raw.float(), dim=-1, eps=EPS)
            g_t = F.normalize(g_t_raw.float(), dim=-1, eps=EPS)
            visual_local = torch.cat([g_i[:, None, :], v_slots], dim=1)
            text_local_all = torch.cat([g_t[:, None, :], t_slots], dim=1)
            text_valid = torch.cat([torch.ones_like(empty[:, None]), (~empty[:, None]).expand(-1, module.n_slots)], 1)
            scored = module.scorer.score_matched_pairs(visual_local, text_local_all, text_valid)
            index = torch.arange(rows, device=device)
            gate = scored['gate'].detach().float()          # [rows, V]
            m_u = 1.0 - gate
            u = (m_u[:, :, None] * v_slots).sum(dim=1) / m_u.sum(-1).clamp_min(1)[:, None]
            u_norm = F.normalize(u, dim=-1, eps=EPS)
            permutation = torch.arange(rows, device=device).roll(1)
            u_wrong = (m_u[:, :, None] * v_slots[permutation]).sum(dim=1) / m_u.sum(-1).clamp_min(1)[:, None]
            u_wrong_norm = F.normalize(u_wrong, dim=-1, eps=EPS)
            bank = module.reference_global(images)                        # frozen teacher bank
            bank_student = g_i

            def table(left, right, name, out):
                prediction = module.decoder(left, right)
                prediction_norm = F.normalize(prediction, dim=-1, eps=EPS)
                cosine = (prediction_norm * bank).sum(dim=-1)
                similarity = prediction_norm @ bank.t()
                rank1 = (similarity.argmax(dim=-1) == index).float().mean()
                out['diag_cos_error_' + name] = float((1.0 - cosine).mean())
                out['diag_pred_r1_teacher_' + name] = float(rank1)
                out['diag_pred_variance_' + name] = float(prediction_norm.var(dim=0).mean())
                out['diag_pred_norm_' + name] = float(prediction_norm.norm(dim=-1).mean())
                out['diag_pred_student_cls_cos_' + name] = float(
                    (prediction_norm * bank_student).sum(dim=-1).mean())

            results = {'diag_native_cls_pairwise_cos': float(_pairwise_cos(g_i)),
                       'diag_intra_image_slot_cos': float(intra_image_slot_cos(v_slots)),
                       'diag_all_image_slot_cos': float(_pairwise_cos(v_slots)),
                       'diag_soft_gate_mean': float(scored['logits'].sigmoid().mean())}
            table(u_norm, g_t, 'normal', results)
            table(u_norm, torch.zeros_like(g_t), 'zero_text', results)
            table(torch.zeros_like(u_norm), g_t, 'zero_u', results)
            table(u_norm, g_t[permutation], 'permuted_text', results)
            table(u_wrong_norm, g_t, 'wrong_u', results)
            base = results['diag_cos_error_normal']
            for name in ('zero_text', 'zero_u', 'permuted_text', 'wrong_u'):
                results['diag_delta_' + name] = results['diag_cos_error_' + name] - base
            mean_bank = bank.mean(dim=0)
            results['diag_constant_baseline_error'] = float(
                (1.0 - (mean_bank / mean_bank.norm().clamp_min(EPS) * bank).sum(dim=-1)).mean())
            results['diag_u_gate_kept'] = float(gate.sum(dim=-1).mean())
            results['diag_u_gate_unsaid'] = float(m_u.sum(dim=-1).mean())
            results['diag_gate_overlap_with_permuted'] = float(
                (gate[permutation] * gate).sum(dim=-1).mean() / float(module.k_said))
            results['diag_router_logit_std'] = float(scored['logits'].detach().std())
            results['diag_empty_text'] = int(empty.sum())
            results['diag_n_images'] = int(rows)
    return results


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description='SAID-Token v1 (T1 / T0)')
    parser.add_argument('--arm', default=ARM_T1, choices=list(ARMS))
    parser.add_argument('--objective', default='said_token', choices=['said_token'])
    parser.add_argument('--lambda_rec', type=float, default=None,
                        help='fixed constant, no warmup and no dynamic balancing; default 0.1 for T1 '
                             'and 0.0 for T0')
    parser.add_argument('--n_slots', type=int, default=N_SLOTS)
    parser.add_argument('--k_said', type=int, default=K_SAID)
    parser.add_argument('--router_tau', type=float, default=0.07)
    parser.add_argument('--router_dim', type=int, default=128)
    parser.add_argument('--surrogate_eta', type=float, default=0.1)
    parser.add_argument('--margin', type=float, default=0.2)
    parser.add_argument('--chunk_image', type=int, default=8)
    parser.add_argument('--chunk_text', type=int, default=32)
    parser.add_argument('--checkpoint_pairwise', type=int, default=1)
    parser.add_argument('--profile_steps', type=int, default=0)
    parser.add_argument('--module_seed', type=int, default=0)
    parser.add_argument('--base_model', default='B16')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--module_lr', type=float, default=2e-4)
    parser.add_argument('--module_wd', type=float, default=1e-2)
    parser.add_argument('--decoder_lr', type=float, default=1e-4)
    parser.add_argument('--decoder_wd', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--warmup_length', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--init_state', required=True,
                        help='the shared student init; the frozen reference is copied from the '
                             'student *after* this load, never from a trained checkpoint')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--save_completed_steps', default='0')
    parser.add_argument('--log_every', type=int, default=25)
    parser.add_argument('--online_check_steps', type=int, default=0,
                        help='>0: log every single step up to this step (the run\'s own online check)')
    parser.add_argument('--diag_every', type=int, default=0,
                        help='>0: run the fixed-cohort diagnostics every N steps (rank 0 only)')
    parser.add_argument('--cohort_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--amp_dtype', default='bf16', choices=['bf16', 'fp32'])
    parser.add_argument('--grad_checkpoint_views', type=int, default=0)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--reference_state_out', default=None)
    args = parser.parse_args()
    if args.base_model == 'B16':
        args.base_model = 'ViT-B/16'
    elif args.base_model == 'L14':
        args.base_model = 'ViT-L/14'
    if args.lambda_rec is None:
        args.lambda_rec = ARM_LAMBDA_REC[args.arm]
    if args.arm == ARM_T0 and args.lambda_rec != 0.0:
        raise SystemExit('T0_said_token_only must run with --lambda_rec 0')

    if args.resume:
        raise SystemExit('fix_v2 requires shared_init; resume is unsupported (RNG/data-position restoration not implemented)')
    seed_everything(args.seed)
    rank, local_rank, world = setup_distributed()
    device = torch.device('cuda', local_rank)
    save_completed = sorted({int(s) for s in args.save_completed_steps.split(',') if s.strip()})

    model, preprocess = longclip.load_from_clip(args.base_model, device='cpu',
                                                download_root=None, args=args)
    model.train()
    model.logit_scale = torch.nn.Parameter(torch.ones([]) * 4.6052)
    model = model.to(device)
    # 1. build and strictly load the shared student init, in fp32 master precision
    load_init_state(model, args.init_state, rank)
    initial_digest = state_digest(model.state_dict())

    train_module = SaidTokenV1Module(model, rank=rank, arm=args.arm, lambda_rec=args.lambda_rec,
                                     margin=args.margin, n_slots=args.n_slots,
                                     k_said=args.k_said, router_dim=args.router_dim,
                                     router_tau=args.router_tau, surrogate_eta=args.surrogate_eta,
                                     seed=args.module_seed,
                                     grad_checkpoint_views=bool(args.grad_checkpoint_views),
                                     chunk_image=args.chunk_image,
                                     chunk_text=args.chunk_text, checkpoint_pairwise=bool(args.checkpoint_pairwise)).to(device)
    # 2. the reference was deep-copied from the student *after* the init load
    reference_digest = reference_fingerprint(train_module.reference_visual)
    reference_zero_storage_overlap = not any(
        a.data_ptr() == b.data_ptr()
        for a in train_module.clip.visual.parameters()
        for b in train_module.reference_visual.parameters())
    if not reference_zero_storage_overlap:
        raise RuntimeError('the frozen reference shares storage with the student visual tower')
    if train_module.reference_visual.training:
        raise RuntimeError('the frozen reference is not in eval mode')

    ddp_model = torch.nn.parallel.DistributedDataParallel(
        train_module, device_ids=[local_rank], output_device=local_rank,
        find_unused_parameters=True)
    # deliberately NOT _set_static_graph(): the graph differs between the positive pass, the chunked
    # grid and the U branch, and a stale static graph would be a silent correctness hazard
    clip_model = train_module.clip

    (optimizer, module_optimizer, decoder_optimizer,
     n_backbone, n_module, n_decoder) = build_optimizers(train_module, args)
    use_amp = args.amp_dtype == 'bf16'
    amp_dtype = torch.bfloat16

    dtype_counts = {}
    for name, parameter in train_module.named_parameters():
        key = str(parameter.dtype)
        dtype_counts[key] = dtype_counts.get(key, 0) + 1
    reference_dtypes = sorted({str(p.dtype) for p in train_module.reference_visual.parameters()})
    reference_grads = sorted({bool(p.requires_grad)
                              for p in train_module.reference_visual.parameters()})
    print('DTYPE_AUDIT rank=%d parameter_dtypes=%s reference_dtypes=%s reference_requires_grad=%s '
          'backbone_tensors=%d module_tensors=%d decoder_tensors=%d amp_enabled=%s amp_dtype=%s '
          'reference_eval=%s mask_net_trainable=%s logit_scale_trainable=%s'
          % (rank, sorted(dtype_counts.items()), reference_dtypes, reference_grads, n_backbone,
             n_module, n_decoder, use_amp, args.amp_dtype,
             not train_module.reference_visual.training,
             any(p.requires_grad for p in clip_model.mask_net.parameters()),
             bool(clip_model.logit_scale.requires_grad)), flush=True)

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
    module_scheduler = cosine_lr(module_optimizer, base_lr=args.module_lr,
                                 warmup_length=args.warmup_length, steps=total_steps)
    decoder_scheduler = cosine_lr(decoder_optimizer, base_lr=args.decoder_lr,
                                  warmup_length=args.warmup_length, steps=total_steps)

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, 'salu_log.jsonl')
    reference_path = args.reference_state_out or os.path.join(args.output_dir,
                                                              'reference_visual_state.pt')
    config = {
        'objective': 'said_token',
        'implementation_version': 'fix_v2',
        'said_direction_reduction': 'sum',
        'pair_ownership': 'local image rows x global text candidates',
        'ddp_scaling': 'W*(local_i2t_sum+local_t2i_sum)/M; W*local_rec_sum/V',
        'arm': args.arm,
        'objective_line': 'L = L_Said + %g * L_rec' % args.lambda_rec,
        'lambda_rec': args.lambda_rec,
        'absent_by_design': [
            'separate global image-text loss', 'global warm-up stage', 'hidden alignment weight',
            'NCE / contrastive reconstruction', 'mask sparsity penalty', 'orthogonality or energy '
            'terms', 'EMA teacher', 'external strong teacher', 'USS or caption-suffix reading',
            'auto-tuned loss weights / keep ratio / lr',
        ],
        'interface': 'image: last-layer tokens of encode_image_with_patches (ln_post + visual.proj '
                     'already applied); text: encode_text(return_full=True) hidden state projected '
                     'once by text_projection; native CLS and native EOS are separate candidates',
        'aggregation': 'A(X) = softmax_L(f(X)^T) X, one aggregator per modality, f = LN -> Linear(102) '
                       '-> GELU -> Linear(%d), fixed logit scale 1, CLS/EOS never aggregated, text '
                       'padding/SOT/EOT never attended' % args.n_slots,
        'router': 'a[i,j,p] = <Norm(W_V V[i,p]), Norm(W_T g_T[j])> / tau_r, tau_r=%g, '
                  'W_V/W_T = Linear(512,%d,bias=False); hard = 1{p in TopK(a,16)} over the 32 '
                  'aggregated visual slots only, deterministic on ties, native CLS excluded; the '
                  'candidate text enters through its native global EOS feature' % (args.router_tau,
                                                                                   args.router_dim),
        'proxy': 'hard forward + score-layer backward: s = s_hard + (s_tilde - sg(s_tilde)) from a '
                 'detached cosine grid, eta=%g' % args.surrogate_eta,
        'said_set': '{native CLS} u {16 selected slots} (17) against {32 text slots} u {native EOS} '
                    '(33), per-token normalised, bidirectional max-similarity, hinge margin %g, mean '
                    'over all valid (anchor, negative) incidences' % args.margin,
        'u_branch': 'm_U = 1 - sg(h_ii); u = mean_p m_U,ip V_ip over the raw aggregated slots, '
                    'normalised afterwards; g_hat = D([Norm(u); sg(g_T)]); '
                    'L_rec = mean_i (1 - <Norm(g_hat_i), sg(g0_i)>)',
        'reconstruction_target': 'g0 = Norm(frozen_initial_visual(I_a)) in fp32, no_grad',
        'reference_source': 'deep copy of the student visual AFTER loading --init_state %s'
                            % args.init_state,
        'sampling': 'DistributedSampler(shuffle=True), no drop_last, steps_per_epoch=len(loader)',
        'precision': 'fp32 master + %s autocast for the student; fp32 frozen reference, router proxy, '
                     'cosine and reconstruction cores (autocast off)' % args.amp_dtype,
        'optimizer_groups': {
            'backbone': {'lr': args.lr, 'weight_decay': args.weight_decay, 'tensors': n_backbone},
            'aggregators_router': {'lr': args.module_lr, 'weight_decay': args.module_wd,
                                   'tensors': n_module},
            'decoder': {'lr': args.decoder_lr, 'weight_decay': args.decoder_wd,
                        'tensors': n_decoder},
        },
        'batch_size_per_gpu': args.batch_size,
        'global_pairs': args.batch_size * world,
        'world_size': world,
        'epochs': args.epochs,
        'loader_batches': steps_per_epoch,
        'lr_horizon_steps': total_steps,
        'warmup_length': args.warmup_length,
        'chunk_image': args.chunk_image,
        'chunk_text': args.chunk_text,
        'checkpoint_pairwise': bool(args.checkpoint_pairwise),
        'grad_checkpoint_views': bool(args.grad_checkpoint_views),
        'init_state': args.init_state,
        'initial_state_sha256': initial_digest,
        'reference_visual_state_sha256': reference_digest,
        'reference_state_path': reference_path,
        'git_head': git_head(),
        'tokenizer_context': TOKENIZER_CONTEXT,
        'eot_id': EOT_ID,
    }
    if rank == 0:
        print('CONFIG ' + json.dumps(config, sort_keys=True), flush=True)
        print('DATASET_SIZE %d STEPS_PER_EPOCH %d LR_HORIZON %d' % (len(dataset), steps_per_epoch,
                                                                    total_steps), flush=True)
        with open(os.path.join(args.output_dir, 'config.json'), 'w') as handle:
            json.dump(config, handle, indent=2, sort_keys=True)
        torch.save({'reference_visual': train_module.reference_visual.state_dict(),
                    'sha256': reference_digest, 'source': 'initial visual tower',
                    'init_state': args.init_state, 'initial_state_sha256': initial_digest},
                   reference_path)
        print('SAVED_REFERENCE ' + reference_path, flush=True)

    cohort = None
    if args.cohort_size > 0:
        cohort = build_fixed_cohort(dataset, device, size=args.cohort_size)
        if rank == 0:
            print('FIXED_COHORT n=%d sample_ids=%s' % (len(cohort['captions']),
                                                       cohort['sample_id'][:8]), flush=True)

    if args.resume:
        payload = torch.load(args.resume, map_location='cpu', weights_only=False)
        if payload.get('config', {}).get('arm') != args.arm:
            raise ValueError('resume arm mismatch')
        clip_model.load_state_dict(payload['model'])
        train_module.image_aggregator.load_state_dict(payload['image_aggregator'])
        train_module.text_aggregator.load_state_dict(payload['text_aggregator'])
        train_module.router.load_state_dict(payload['router'])
        train_module.decoder.load_state_dict(payload['decoder'])
        optimizer.load_state_dict(payload['optimizer'])
        module_optimizer.load_state_dict(payload['module_optimizer'])
        decoder_optimizer.load_state_dict(payload['decoder_optimizer'])
        start_step = int(payload['completed_steps'])
        if rank == 0:
            print('RESUMED from %s at %d' % (args.resume, start_step), flush=True)
    else:
        start_step = 0

    def checkpoint_payload(step, epoch, step_in_epoch):
        return {
            'model': clip_model.state_dict(),
            'image_aggregator': train_module.image_aggregator.state_dict(),
            'text_aggregator': train_module.text_aggregator.state_dict(),
            'router': train_module.router.state_dict(),
            'decoder': train_module.decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'module_optimizer': module_optimizer.state_dict(),
            'decoder_optimizer': decoder_optimizer.state_dict(),
            'completed_steps': step,
            'epoch': epoch,
            'step_in_epoch': step_in_epoch,
            'phase': 'said-token-v1',
            'objective': 'said_token',
            'arm': args.arm,
            'config': config,
            'precision': 'fp32 master + %s autocast' % args.amp_dtype,
            'lr_horizon_steps': total_steps,
            'git_head': config['git_head'],
            'reference_visual_state_sha256': reference_digest,
            'reference_state_path': reference_path,
            'digests': {
                'initial_state_sha256': initial_digest,
                'caption_stream_sha256': caption_digest.hexdigest(),
                'sample_stream_sha256': sample_digest.hexdigest(),
                'view_a_pixels_sha256_step0': view_a_pixels_sha,
            },
        }

    caption_digest = hashlib.sha256()
    sample_digest = hashlib.sha256()
    view_a_pixels_sha = None
    completed = start_step
    t_start = time.time()
    compute_times = []
    stopped = False
    severe_checks = 0

    if rank == 0 and 0 in save_completed:
        # step-0 checkpoint: the exact initial state, so a later analysis never has to guess which
        # init a run started from
        path = os.path.join(args.output_dir, 't1_%s_step%06d.pt' % (args.arm, 0))
        torch.save(checkpoint_payload(0, -1, -1), path)
        print('SAVED ' + path, flush=True)
        if cohort is not None:
            record = {'completed_steps': 0, 'arm': args.arm, 'objective': 'said_token',
                      'initial': True, 'reference_visual_state_sha256': reference_digest,
                      'rank_param_digest': state_digest(clip_model.state_dict())[:16]}
            record.update(input_dependency_diagnostics(ddp_model, cohort, device))
            with open(log_path, 'a') as handle:
                handle.write('LOG ' + json.dumps(record, sort_keys=True) + '\n')
            print('LOG ' + json.dumps(record, sort_keys=True), flush=True)
    if dist.is_initialized():
        dist.barrier()

    profiler = None
    if args.profile_steps:
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA], record_shapes=False, profile_memory=True)
        profiler.start()
    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        for i, batch in enumerate(loader):
            if args.max_steps is not None and completed >= args.max_steps:
                stopped = True
                break
            check_equal_local_batch(batch['image_a'].shape[0], device, world)
            torch.cuda.synchronize(device)
            t0 = time.time()
            scheduler(completed)
            module_scheduler(completed)
            decoder_scheduler(completed)
            out = t1_train_step(ddp_model, batch, (optimizer, module_optimizer, decoder_optimizer),
                                device, amp_dtype, amp_enabled=use_amp, world_size=world,
                                diagnostics=(completed < 20 or (completed+1) % args.log_every == 0
                                             or completed+1 in save_completed),
                                health_check=(completed+1 in (1, 5, 20)))
            torch.cuda.synchronize(device)
            image_a = batch['image_a']
            completed += 1
            compute_times.append(time.time() - t0)
            if profiler is not None:
                profiler.step()
                if completed >= args.profile_steps:
                    profiler.stop()
                    if rank == 0:
                        table = profiler.key_averages().table(sort_by='self_cuda_time_total', row_limit=30)
                        print('PROFILE ' + table, flush=True)
                        with open(os.path.join(args.output_dir, 'profile_table.txt'), 'w') as h:
                            h.write(table)
                        profiler.export_chrome_trace(os.path.join(args.output_dir, 'profile_trace.json'))
                    profiler = None

            caption_digest.update(('\n'.join(batch['caption_said'])).encode('utf-8'))
            sample_digest.update(batch['sample_id'].numpy().tobytes())
            if view_a_pixels_sha is None:
                view_a_pixels_sha = tensor_digest(batch['image_a'][0])

            is_last = args.max_steps is not None and completed >= args.max_steps
            want_log = (completed % args.log_every == 0 or completed == 1 or is_last
                        or completed in save_completed
                        or (args.online_check_steps > 0 and completed <= args.online_check_steps))
            failure = torch.zeros((), device=device, dtype=torch.int32)
            if rank == 0 and want_log:
                record = {
                    'completed_steps': completed,
                    'epoch': epoch,
                    'step_in_epoch': i,
                    'arm': args.arm,
                    'objective': 'said_token',
                    'lr': optimizer.param_groups[0]['lr'],
                    'module_lr': module_optimizer.param_groups[0]['lr'],
                    'decoder_lr': decoder_optimizer.param_groups[0]['lr'],
                    'lambda_rec': args.lambda_rec,
                    'batch_size_local': int(image_a.shape[0]),
                    'global_pairs': int(image_a.shape[0]) * world,
                    'world_size': world,
                    'reference_visual_state_sha256': reference_digest,
                    'core_dtypes': out['core_dtypes'],
                    'said_direction_reduction': 'sum',
                    'sec_per_step': compute_times[-1],
                    'samples_per_sec': float(image_a.shape[0]) * world / max(compute_times[-1], 1e-9),
                    'peak_memory_gb': torch.cuda.max_memory_allocated(device) / (1024 ** 3),
                    'batch_caption_sha256': hashlib.sha256(
                        ('\n'.join(batch['caption_said'])).encode('utf-8')).hexdigest()[:16],
                }
                for key, value in out.items():
                    if torch.is_tensor(value) and value.numel() == 1 and value.dtype.is_floating_point:
                        record[key] = float(value.detach())
                    elif isinstance(value, float):
                        record[key] = value
                record['loss_said'] = record['loss_said_global_mean']
                record['loss_rec'] = record['loss_rec_global_mean']
                record['weighted_rec'] = args.lambda_rec * record['loss_rec_global_mean']
                want_diag = (completed in save_completed
                             or (args.diag_every > 0 and completed % args.diag_every == 0))
                if want_diag and cohort is not None:
                    record.update(input_dependency_diagnostics(ddp_model, cohort, device))
                with open(log_path, 'a') as handle:
                    handle.write('LOG ' + json.dumps(record, sort_keys=True) + '\n')
                print('LOG ' + json.dumps(record, sort_keys=True), flush=True)
                # Conservative, predeclared stop on sustained total loss of discrimination.
                # Never change parameters/objectives to rescue a run. Evaluate only at log points.
                severe = (completed >= 100 and
                    abs(record['positive_said_score']-record['negative_said_score']) < .005 and
                    record.get('native_cls_pairwise_cos', 0) > .99 and
                    record.get('all_image_slot_cos', 0) > .99 and
                    record.get('intra_image_slot_cos', 0) > .99 and
                    record['margin_active_fraction'] > .99)
                severe_checks = severe_checks + 1 if severe else 0
                if severe_checks >= 3:
                    failure.fill_(1)
                    path = os.path.join(args.output_dir, 'stopped_degenerate_step%06d.pt' % completed)
                    torch.save(checkpoint_payload(completed, epoch, i), path)
                    with open(os.path.join(args.output_dir, 'STOPPED_DEGENERATE.json'), 'w') as handle:
                        json.dump({'completed_steps': completed, 'reason': 'three consecutive logged checks: score gap<.005, native/all/intra cosine>.99, active>.99', 'checkpoint': path}, handle)
            if world > 1:
                dist.broadcast(failure, src=0)
            if bool(failure):
                raise RuntimeError('sustained global representation degeneration; checkpoint saved; no automatic redesign')

            if rank == 0 and completed in save_completed:
                path = os.path.join(args.output_dir,
                                    't1_%s_step%06d.pt' % (args.arm, completed))
                torch.save(checkpoint_payload(completed, epoch, i), path)
                print('SAVED ' + path, flush=True)
            if world > 1 and dist.is_initialized() and completed in save_completed:
                dist.barrier()
        if stopped:
            break

    if args.max_steps is not None and completed != args.max_steps:
        raise RuntimeError('target step not reached: %d != %d' % (completed, args.max_steps))
    if rank == 0:
        final_reference_digest = reference_fingerprint(train_module.reference_visual)
        summary = {
            'arm': args.arm, 'objective': 'said_token', 'completed_steps': completed,
            'lambda_rec': args.lambda_rec,
            'epochs': args.epochs, 'wall_sec': time.time() - t_start,
            'mean_sec_per_step': sum(compute_times) / max(len(compute_times), 1),
            'timing_scope': 'CUDA synchronized train step including transfer/tokenization/forward/backward/optimizer/global logs; excludes loader wait, checkpoint save and cohort diagnostics',
            'steps_per_epoch': steps_per_epoch, 'lr_horizon_steps': total_steps,
            'initial_state_sha256': initial_digest,
            'reference_visual_state_sha256': reference_digest,
            'reference_visual_state_sha256_final': final_reference_digest,
            'reference_unchanged': reference_digest == final_reference_digest,
            'mask_net_requires_grad_any': any(p.requires_grad
                                              for p in clip_model.mask_net.parameters()),
            'logit_scale_requires_grad': bool(clip_model.logit_scale.requires_grad),
            'caption_stream_sha256': caption_digest.hexdigest(),
            'sample_stream_sha256': sample_digest.hexdigest(),
            'view_a_pixels_sha256_step0': view_a_pixels_sha,
            'peak_memory_gb': torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        }
        with open(os.path.join(args.output_dir, 'run_summary.json'), 'w') as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print('RUN_SUMMARY ' + json.dumps(summary, sort_keys=True), flush=True)
        if not summary['reference_unchanged']:
            print('REFERENCE_CHANGED (this is a hard error)', flush=True)
            sys.exit(3)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
