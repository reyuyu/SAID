"""Phase 2 Said-only training script: L = lambda_global * L_global + lambda_said * L_said.

This script is independent from ``train/train.py`` (the official SmartCLIP
baseline), which stays untouched and fully runnable.

It trains exactly two objectives:
  1. global CLIP symmetric InfoNCE on (z_global, t)
  2. Said symmetric InfoNCE on (z_s, t), where z_s is the caption-conditioned
     Said visual representation produced by the Said router.

Nothing else is trained: no prototype bank, no unsaid explorer, no sparsity /
entropy / overlap / reconstruction loss. SmartCLIP's ``mask_net`` is kept for
checkpoint compatibility but frozen and excluded from the optimizer.

Precision: fp32 master weights + bf16 autocast (default). fp16 autocast with
GradScaler's default scaling overflows in this graph, so fp16 is opt-in.

Throughput logging distinguishes:
  * compute_sec_per_step / compute_samples_per_sec: forward+backward+optimizer only
  * wall_sec_per_step / wall_samples_per_sec: full loop iteration (incl. data loading)

Validation cadence (Phase 2.7B protocol):
  * ``--val_every N`` drives **ShareGPT4V-1K only** (three frozen caption variants:
    first_sentence / fixed_sparse / full_dense), also run at epoch end and final.
  * COCO runs only at the cadence asked for: ``--eval_coco_initial`` (a dedicated
    step-0 call *before the first optimizer update*), ``--eval_coco_each_epoch``
    (epoch end), ``--eval_coco`` (final).
  * Every result is appended to ``<output_dir>/validation_history.jsonl`` (UTF-8,
    ``ensure_ascii=False``) with step / epoch / dataset / caption_variant / metrics /
    wall_sec / protocol / similarity_chunk; a repeated (step, dataset, variant) is
    never written twice.
  * All evaluation runs inside an RNG guard, so it cannot steer the training stream.
  * ``--legacy_eval_coco`` additionally runs the legacy SmartCLIP evaluator for
    protocol comparison.

Unsaid (Phase 2.8A, opt-in, no new parameters):
  * ``--lambda_unsaid`` (default ``0.0``) adds the minimal Unsaid complement loss
    ``L_U = mean_valid[1 - cos(z_U, target_U)]``, where ``z_U`` pools the same patch
    features with the same Said raw scores but anti-routed
    (``A^U = softmax(-s / tau_unsaid)``) and ``target_U`` is the stop-gradient
    complement of the Said direction inside the global representation.
  * ``--tau_unsaid`` (default ``0.07``) and ``--unsaid_residual_eps`` (default
    ``1e-4``) control that branch.
  * ``--lambda_unsaid 0`` skips the branch entirely, so the run stays the Said-only
    run: same losses, same RNG use, same speed. Unsaid diagnostics are then logged as
    ``null`` behind ``unsaid_enabled = false`` instead of fabricated numbers.
  * Unsaid monitors (loss, valid ratio, residual norm, target cosine, said/unsaid
    cosine, entropies, effective patch counts, attention overlap, attention JSD) are
    monitoring only and never enter a loss.

Example (4 GPUs, 676K no-SAM subset)::

    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
    SHARE4V_DATA_ROOT=/root/datasets/ShareGPT4V \
    SHARE4V_JSON=debug/share4v_smoke_nosam.json \
    COCO_DATA_ROOT=/root/datasets/coco \
    torchrun --nproc_per_node=4 --master_port=25961 train/train_salu.py \
        --base_model B16 --batch_size 256 --epochs 1 \
        --backbone_lr 1e-6 --head_lr 1e-4 \
        --lambda_global 1.0 --lambda_said 1.0 --tau_said 0.07 \
        --save_at 200,400
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TRAIN_DIR)
for _p in (REPO_ROOT, TRAIN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import longclip  # noqa: E402
from model.salu_model import SALUModel  # noqa: E402
from model.unsaid_core import DIAGNOSTIC_KEYS as UNSAID_DIAGNOSTIC_KEYS  # noqa: E402
from sharegpt4v import share4v_train_dataset, share4v_val_dataset  # noqa: E402
from train_utils import eval_coco  # noqa: E402  (legacy SmartCLIP-style COCO evaluation)
from eval.retrieval.coco_retrieval import (  # noqa: E402
    evaluate_coco as evaluate_coco_standard,
    DEFAULT_SIMILARITY_CHUNK as CANONICAL_SIMILARITY_CHUNK,
)
from eval.validation_protocol import (  # noqa: E402
    PROTOCOL_NAME as SHAREGPT4V1K_PROTOCOL,
    VARIANTS as SHAREGPT4V1K_VARIANTS,
    evaluate_all_variants,
    evaluate_variant,
    load_or_create_manifest,
    rng_guard,
)
from salu_reproducibility import seed_everything, seed_worker, state_digest, update_caption_digest


def _load_baseline_setup_distributed():
    """Reuse ``setup_distributed`` from ``train/train.py``.

    ``import train`` resolves to the ``train/`` package, so the baseline training
    module is loaded explicitly from its file path instead of shadowing it.
    """
    import importlib.util

    baseline_path = os.path.join(TRAIN_DIR, 'train.py')
    spec = importlib.util.spec_from_file_location('smartclip_baseline_train', baseline_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.setup_distributed


setup_distributed = _load_baseline_setup_distributed()


def lr_scale(step: int, warmup_length: int, total_steps: int) -> float:
    """Cosine decay with linear warmup, expressed as a multiplier."""
    if warmup_length > 0 and step < warmup_length:
        return float(step + 1) / float(warmup_length)
    e = max(0, step - warmup_length)
    es = max(1, total_steps - warmup_length)
    return 0.5 * (1.0 + math.cos(math.pi * e / es))


def set_lrs(optimizer, base_lrs, step, warmup_length, total_steps) -> float:
    scale = lr_scale(step, warmup_length, total_steps)
    for group, base in zip(optimizer.param_groups, base_lrs):
        group['lr'] = base * scale
    return scale


def summarize_throughput(global_batch, compute_times, wall_times):
    """Compute- and wall-time throughput metrics (logging only)."""
    def _avg(values):
        return float(sum(values)) / float(len(values)) if values else 0.0

    compute_sec = _avg(compute_times)
    wall_sec = _avg(wall_times)
    return {
        'compute_sec_per_step': compute_sec,
        'compute_samples_per_sec': (float(global_batch) / compute_sec) if compute_sec > 0 else 0.0,
        'wall_sec_per_step': wall_sec,
        'wall_samples_per_sec': (float(global_batch) / wall_sec) if wall_sec > 0 else 0.0,
    }


def append_validation_record(path, record):
    """Append one validation record as a UTF-8 JSON line (never ASCII-escaped)."""
    with open(path, 'a', encoding='utf-8') as fp:
        fp.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + '\n')


def validation_key(step, dataset, caption_variant):
    """Identity of a validation result: never write the same one twice."""
    return (int(step), str(dataset), str(caption_variant))


def tri_caption_collate(samples):
    """Collate the Phase 2.9A caption-view dicts into one batch."""
    return {
        'image': torch.stack([sample['image'] for sample in samples]),
        'caption_full': [sample['caption_full'] for sample in samples],
        'caption_said': [sample['caption_said'] for sample in samples],
        'caption_unsaid': [sample['caption_unsaid'] for sample in samples],
        'has_unsaid': torch.tensor([bool(sample['has_unsaid']) for sample in samples],
                                   dtype=torch.bool),
        'num_sentences': torch.tensor([sample['num_sentences'] for sample in samples],
                                      dtype=torch.long),
        'prefix_k': torch.tensor([sample['prefix_k'] for sample in samples], dtype=torch.long),
        'unsaid_sentence_index': torch.tensor([sample['unsaid_sentence_index'] for sample in samples],
                                              dtype=torch.long),
    }


def caption_view_stats(batch):
    """Data statistics of one tri-view batch (Phase 2.9A diagnostics)."""
    num_sentences = batch['num_sentences'].float()
    prefix_k = batch['prefix_k'].float()
    has_unsaid = batch['has_unsaid']
    unsaid_index = batch['unsaid_sentence_index'].float()
    suffix_length = num_sentences - prefix_k
    return {
        'data_num_sentences_mean': float(num_sentences.mean()),
        'data_num_sentences_median': float(num_sentences.median()),
        'data_prefix_k_mean': float(prefix_k.mean()),
        'data_suffix_length_mean': float(suffix_length.mean()),
        'data_unsaid_position_mean': (float(unsaid_index[has_unsaid].mean())
                                      if bool(has_unsaid.any()) else 0.0),
        'data_has_unsaid_ratio': float(has_unsaid.float().mean()),
    }


def debug_caption_examples(batch, count=3):
    """Human-readable Full / Said / Unsaid examples that also verify the suffix property."""
    lines = []
    for position in range(min(int(count), len(batch['caption_said']))):
        full = batch['caption_full'][position]
        said = batch['caption_said'][position]
        unsaid = batch['caption_unsaid'][position]
        num_sentences = int(batch['num_sentences'][position])
        prefix_k = int(batch['prefix_k'][position])
        index = int(batch['unsaid_sentence_index'][position])
        in_suffix = bool(batch['has_unsaid'][position]) and index > prefix_k and unsaid in full
        lines.append('DEBUG_EXAMPLE %d N=%d K=%d J=%d has_unsaid=%s unsaid_in_suffix=%s\n'
                     '  Full: %s\n  Said: %s\n  Unsaid: %s'
                     % (position, num_sentences, prefix_k, index,
                        bool(batch['has_unsaid'][position]), in_suffix, full, said, unsaid))
    return '\n'.join(lines)


def plan_validation(args, step, is_epoch_end, is_final_step, is_initial=False):
    """Which validation jobs run at this step (cadence only, no dedupe).

    ``--val_every`` drives ShareGPT4V-1K only, at interval steps, epoch end and final.
    COCO runs only at the cadence explicitly asked for: ``--eval_coco_initial`` (the
    dedicated step-0 call, ``is_initial=True``) / ``--eval_coco_each_epoch`` (epoch
    end) / ``--eval_coco`` or its ``--val_coco`` alias (final); when an epoch end is
    also the final step the caller runs COCO once.
    """
    jobs = []
    if args.val_sharegpt4v and not is_initial:
        interval = args.val_every > 0 and step % args.val_every == 0
        if interval or is_epoch_end or is_final_step:
            reason = 'interval' if interval else ('epoch_end' if is_epoch_end else 'final')
            for variant in SHAREGPT4V1K_VARIANTS:
                jobs.append({'dataset': 'sharegpt4v1k', 'caption_variant': variant, 'reason': reason})
    coco_reason = None
    if is_initial:
        # step 0 is owned by the dedicated ``run_initial_validation`` call only, so it
        # is never reached implicitly through an interval step
        coco_reason = 'initial' if args.eval_coco_initial else None
    elif args.eval_coco_each_epoch and is_epoch_end:
        coco_reason = 'epoch_end'
    elif (args.eval_coco or args.val_coco) and is_final_step:
        coco_reason = 'final'
    if coco_reason:
        jobs.append({'dataset': 'coco_val2017', 'caption_variant': 'coco_5captions', 'reason': coco_reason})
    return jobs


def run_validation_job(args, job, model, preprocess, cohort, image_root, device,
                       step, epoch, history_path, seen):
    """Run one validation job under an RNG guard and append its record.

    Returns the record, or ``None`` when (step, dataset, caption_variant) was
    already written — a repeated ``final`` never produces a duplicate record.
    """
    key = validation_key(step, job['dataset'], job['caption_variant'])
    if key in seen:
        return None
    seen.add(key)
    started = time.time()
    model.eval()
    try:
        with rng_guard():
            if job['dataset'] == 'sharegpt4v1k':
                result = evaluate_variant(
                    model, cohort, image_root, job['caption_variant'], preprocess,
                    batch_size=args.val_batch_size, similarity_chunk=CANONICAL_SIMILARITY_CHUNK,
                    device=device,
                )
                metrics = {'retrieval': result['retrieval'], 'diagnostics': result['diagnostics']}
                protocol = SHAREGPT4V1K_PROTOCOL
            elif job['dataset'] == 'coco_val2017':
                metrics = evaluate_coco_standard(
                    model, preprocess, batch_size=args.val_batch_size,
                    similarity_chunk=CANONICAL_SIMILARITY_CHUNK, device=device,
                )
                protocol = 'coco-val2017-5captions-v1'
            else:
                raise ValueError('unknown validation dataset %r' % (job['dataset'],))
    finally:
        model.train()
    record = {
        'step': int(step),
        'epoch': int(epoch),
        'dataset': job['dataset'],
        'caption_variant': job['caption_variant'],
        'metrics': metrics,
        'wall_sec': time.time() - started,
        'protocol': protocol,
        'similarity_chunk': CANONICAL_SIMILARITY_CHUNK,
        'reason': job.get('reason'),
    }
    append_validation_record(history_path, record)
    return record


def run_planned_validation(args, model, preprocess, cohort, image_root, device,
                           step, epoch, is_epoch_end, is_final_step, history_path, seen,
                           is_initial=False):
    """Run every planned job for this step (rank 0 only) and return the records."""
    records = []
    for job in plan_validation(args, step, is_epoch_end, is_final_step, is_initial=is_initial):
        record = run_validation_job(args, job, model, preprocess, cohort, image_root, device,
                                    step, epoch, history_path, seen)
        if record is not None:
            records.append(record)
    return records


def run_initial_validation(args, model, preprocess, cohort, image_root, device,
                           history_path, seen):
    """Canonical COCO validation at step 0, before the first optimizer update.

    ``main()`` calls this once, immediately before entering the training loop, so the
    very first optimizer update of the run already sees the initial record. It is only
    reached when the run really starts at step 0 (a resumed run has already passed it).

    Only ``--eval_coco_initial`` schedules a job here; the record carries
    ``step=0, epoch=0, reason='initial'`` and the canonical ``similarity_chunk``.
    The job runs under the same RNG guard as every other validation point and
    ``run_validation_job`` restores ``train()`` mode afterwards, so training is
    bit-identical whether or not the initial evaluation ran.
    """
    return run_planned_validation(args, model, preprocess, cohort, image_root, device,
                                  0, 0, False, False, history_path, seen, is_initial=True)


def resolve_total_steps(args, steps_per_epoch):
    """``(stop_steps, lr_total_steps)``: run length vs LR-schedule horizon.

    Default (``--lr_total_steps`` unset) keeps the old behaviour: ``--max_steps``
    truncates the run *and* shrinks the cosine schedule, so both values are the same.
    Setting ``--lr_total_steps`` decouples them: ``--max_steps`` only stops the run,
    while the LR schedule keeps the horizon of the full run it stands in for (e.g. a
    500-step probe of a 3-epoch run uses ``lr_total_steps = 3 * steps_per_epoch``).
    """
    stop_steps = args.max_steps if args.max_steps is not None else args.epochs * steps_per_epoch
    if args.lr_total_steps is None:
        return int(stop_steps), int(stop_steps)
    lr_total = int(args.lr_total_steps)
    if lr_total <= 0:
        raise ValueError('--lr_total_steps must be positive, got %r' % (args.lr_total_steps,))
    return int(stop_steps), lr_total


def build_optimizer(model: SALUModel, backbone_lr, head_lr, weight_decay):
    backbone = model.backbone_parameters()
    head = model.said_head_parameters()
    if not head:
        raise RuntimeError('Said router has no trainable parameters')
    optimizer = torch.optim.AdamW(
        [
            {'params': backbone, 'lr': backbone_lr, 'weight_decay': weight_decay, 'name': 'backbone'},
            {'params': head, 'lr': head_lr, 'weight_decay': weight_decay, 'name': 'said_router'},
        ]
    )
    return optimizer, len(backbone), len(head)


def save_checkpoint(path, ddp_model, optimizer, scaler, step, epoch, args, base_lrs, total_steps):
    torch.save(
        {
            'model': ddp_model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'scheduler': {
                'type': 'cosine_with_warmup',
                'base_lrs': list(base_lrs),
                'warmup_length': args.warmup_length,
                'total_steps': total_steps,
                'step': step,
            },
            'step': step,
            'epoch': epoch,
            'args': vars(args),
            'phase': 'phase2-said-only',
        },
        path,
    )
    # a plain CLIP-only state dict so standard tooling (eval/retrieval/coco.py)
    # can load the backbone without knowing about SALU
    clip_path = os.path.join(os.path.dirname(path), 'clip_only_' + os.path.basename(path))
    torch.save(ddp_model.module.clip.state_dict(), clip_path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Phase 2 Said-only training')
    parser.add_argument('--base_model', default='B16', help='B16 or L14')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--said_feature_source', choices=['residual', 'attention_delta'], default='residual')
    parser.add_argument('--save_initial', action='store_true', help='save model and args before training')
    parser.add_argument('--batch_size', type=int, default=256, help='batch per GPU')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max_steps', type=int, default=None,
                        help='debug early stop; None = full run')
    parser.add_argument('--lr_total_steps', type=int, default=None,
                        help='LR-schedule horizon for the cosine decay; None = follow --max_steps / epochs '
                             '(old behaviour). Setting it makes --max_steps only truncate the run.')
    parser.add_argument('--backbone_lr', type=float, default=1e-6)
    parser.add_argument('--head_lr', type=float, default=1e-4)
    parser.add_argument('--lambda_global', type=float, default=1.0)
    parser.add_argument('--lambda_said', type=float, default=1.0)
    parser.add_argument('--lambda_unsaid', type=float, default=0.0,
                        help='weight of the minimal Unsaid complement loss (0 = Said-only path)')
    parser.add_argument('--tau_unsaid', type=float, default=0.07,
                        help='temperature of the Unsaid anti-routing softmax')
    parser.add_argument('--unsaid_residual_eps', type=float, default=1e-4,
                        help='residual-norm threshold below which an Unsaid target is invalid')
    parser.add_argument('--unsaid_mode', default='residual',
                        choices=['residual', 'debiased_suffix'],
                        help='residual = Phase 2.8A ablation; debiased_suffix = Phase 2.9A '
                             'withheld-suffix alignment')
    parser.add_argument('--global_caption_view', default='prefix', choices=['prefix', 'full'],
                        help="text view used by the global alignment (default 'prefix' = legacy)")
    parser.add_argument('--unsaid_gate_floor', type=float, default=0.1,
                        help='lower bound of the soft Said-suppression gate (debiased_suffix)')
    parser.add_argument('--unsaid_gate_temperature', type=float, default=1.0)
    parser.add_argument('--unsaid_suppression_beta', type=float, default=1.0)
    parser.add_argument('--unsaid_candidate_chunk_size', type=int, default=0,
                        help='chunk the candidate-text dimension of the Unsaid attention '
                             '(0 = Phase 2.9A unchunked behaviour, e.g. 32 for 2.9B)')
    parser.add_argument('--tau_said', type=float, default=0.07)
    parser.add_argument('--said_loss_mode', default='identifiable', choices=['positive', 'identifiable'],
                        help='positive = Phase 2 ablation; identifiable = Phase 2.2 routing objective')
    parser.add_argument('--pair_chunk_size', type=int, default=64,
                        help='caption candidates per chunk for pairwise routing (0 = all at once)')
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--warmup_length', type=int, default=200)
    parser.add_argument('--output_dir', default='runs_salu')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='save a checkpoint every N steps (0 = only at the end)')
    parser.add_argument('--save_at', default='',
                        help='comma-separated steps to checkpoint, e.g. "200,400"')
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--download_root', default=None)
    parser.add_argument('--amp_dtype', default='bf16', choices=['bf16', 'fp16', 'fp32'],
                        help='autocast dtype; bf16 needs no loss scaling (default)')
    parser.add_argument('--scaler_init_scale', type=float, default=1024.0)
    parser.add_argument('--resume', default=None, help='SALU checkpoint to resume from')
    parser.add_argument('--eval_coco', action='store_true',
                        help='run the canonical COCO val2017 retrieval evaluation at the end (rank 0)')
    parser.add_argument('--eval_coco_each_epoch', action='store_true',
                        help='also run COCO retrieval at every epoch end')
    parser.add_argument('--eval_coco_initial', action='store_true',
                        help='run canonical COCO retrieval at step 0, before the first optimizer update')
    parser.add_argument('--val_every', type=int, default=0,
                        help='ShareGPT4V-1K validation every N steps (0 = only epoch end / final)')
    parser.add_argument('--val_sharegpt4v', action='store_true',
                        help='validate the fixed ShareGPT4V-1K cohort with all three caption variants')
    parser.add_argument('--val_coco', action='store_true',
                        help='deprecated alias of --eval_coco (COCO at final)')
    parser.add_argument('--val_batch_size', type=int, default=64,
                        help='images per forward pass during validation')
    parser.add_argument('--validation_manifest', default=os.path.join('outputs', 'validation',
                                                                     'sharegpt4v1k_manifest.json'),
                        help='frozen ShareGPT4V-1K caption manifest (created once, never resampled)')
    parser.add_argument('--legacy_eval_coco', action='store_true',
                        help='additionally run the legacy train_utils.eval_coco for comparison')
    parser.add_argument('--strict_manifest', action='store_true')
    return parser.parse_args(argv)


def main():
    args = parse_args()

    if args.base_model == 'B16':
        args.base_model = 'ViT-B/16'
    elif args.base_model == 'L14':
        args.base_model = 'ViT-L/14'
    save_at = sorted({int(s) for s in args.save_at.split(',') if s.strip()})

    rank, local_rank = setup_distributed()
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    world_size = dist.get_world_size()
    seed_everything(args.seed)

    clip_model, preprocess = longclip.load_from_clip(
        args.base_model, device='cpu', download_root=args.download_root, args=args
    )
    salu = SALUModel(clip_model, tau_said=args.tau_said, said_loss_mode=args.said_loss_mode,
                     pair_chunk_size=(args.pair_chunk_size or None), said_feature_source=args.said_feature_source)
    initial_digest = state_digest(salu.state_dict())
    salu = salu.to(device)
    ddp_model = DDP(salu, device_ids=[local_rank], find_unused_parameters=True)

    optimizer, n_backbone, n_head = build_optimizer(
        salu, args.backbone_lr, args.head_lr, args.weight_decay
    )
    base_lrs = [args.backbone_lr, args.head_lr]
    use_amp = args.amp_dtype != 'fp32'
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(args.amp_dtype == 'fp16'),
                                  init_scale=args.scaler_init_scale)

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print('args', vars(args), flush=True)
        print('world_size', world_size, 'backbone_tensors', n_backbone, 'said_head_tensors', n_head, flush=True)
        print('mask_net_trainable',
              any(p.requires_grad for p in salu.clip.mask_net.parameters()) if hasattr(salu.clip, 'mask_net') else 'n/a',
              flush=True)

    if args.strict_manifest or os.environ.get('SHARE4V_FULL_AUDIT'):
        from tools.data.full_data_gate import require_full_data, resolve_json_path
        root = os.environ.get('SHARE4V_DATA_ROOT', '../datasets/ShareGPT4V')
        # SHARE4V_JSON follows the dataset convention: relative to SHARE4V_DATA_ROOT
        jp = resolve_json_path(root, os.environ.get('SHARE4V_JSON', 'share-captioner_coco_lcs_sam_1246k_1107.json'))
        ap = os.environ.get('SHARE4V_FULL_AUDIT', os.path.join(REPO_ROOT, 'outputs/data_audit/sharegpt4v_full_audit.json'))
        require_full_data(ap, jp, root)
        if rank == 0:
            print('FULL_DATA_GATE_PASS json=%s audit=%s' % (jp, ap), flush=True)
    tri_mode = args.unsaid_mode == 'debiased_suffix'
    train_set = share4v_train_dataset(caption_views=tri_mode, suffix_seed=args.seed)
    sampler = DistributedSampler(train_set, shuffle=True, seed=args.seed)
    loader_generator = torch.Generator().manual_seed(args.seed + rank)
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=loader_generator,
        collate_fn=tri_caption_collate if tri_mode else None,
    )
    steps_per_epoch = len(loader)
    stop_steps, total_steps = resolve_total_steps(args, steps_per_epoch)
    if rank == 0:
        print('steps_per_epoch %d stop_steps %d lr_total_steps %d lr_warmup %d'
              % (steps_per_epoch, stop_steps, total_steps, args.warmup_length), flush=True)
    global_batch = args.batch_size * world_size

    cohort = None
    cohort_root = None
    if args.val_sharegpt4v and rank == 0:
        data_root = os.environ.get('SHARE4V_DATA_ROOT', '../datasets/ShareGPT4V')
        json_name = os.environ.get('SHARE4V_JSON', 'share-captioner_coco_lcs_sam_1246k_1107.json')
        json_path = json_name if os.path.isabs(json_name) else os.path.join(data_root, json_name)
        manifest = load_or_create_manifest(args.validation_manifest, json_path)
        cohort = manifest['samples']
        cohort_root = data_root
        print('sharegpt4v1k_cohort %d variants %s manifest=%s json_sha256=%s' % (
            len(cohort), ','.join(manifest['variants']), args.validation_manifest,
            manifest['dataset_json_sha256'][:16]), flush=True)

    start_step, start_epoch = 0, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        if ckpt.get('args', {}).get('said_feature_source', 'residual') != args.said_feature_source:
            raise ValueError('resume checkpoint feature source differs from CLI')
        ddp_model.module.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scaler' in ckpt and ckpt['scaler']:
            scaler.load_state_dict(ckpt['scaler'])
        start_step = int(ckpt.get('step', 0))
        start_epoch = int(ckpt.get('epoch', 0))
        if rank == 0:
            print('resumed from %s at step %d epoch %d' % (args.resume, start_step, start_epoch), flush=True)

    log_path = os.path.join(args.output_dir, 'salu_log.jsonl')
    validation_history_path = os.path.join(args.output_dir, 'validation_history.jsonl')
    seen_validation = set()
    caption_digest = hashlib.sha256()
    full_digest = hashlib.sha256()
    unsaid_digest = hashlib.sha256()
    has_unsaid_digest = hashlib.sha256()
    sampler_digest = hashlib.sha256()
    if rank == 0 and args.save_initial and not args.resume:
        torch.save({'model': salu.state_dict(), 'args': vars(args), 'step': 0,
                    'phase': 'phase2.5-local-evidence-router'},
                   os.path.join(args.output_dir, 'salu_initial.pt'))
    ddp_model.train()
    step = start_step
    stopped = False
    t_start = time.time()
    compute_times = []
    wall_times = []

    # --eval_coco_initial: canonical COCO validation at step 0 (reason='initial'),
    # strictly before the first optimizer update. Every rank waits at the barriers
    # around the rank-0-only evaluation, and only rank 0 writes the record.
    if args.eval_coco_initial and start_step == 0:
        if dist.is_initialized():
            dist.barrier()
        if rank == 0:
            for record in run_initial_validation(args, ddp_model, preprocess, cohort, cohort_root,
                                                 device, validation_history_path, seen_validation):
                print('VAL_INITIAL ' + json.dumps(record, sort_keys=True, ensure_ascii=False), flush=True)
        if dist.is_initialized():
            dist.barrier()

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        sampler_digest.update(torch.tensor(list(sampler), dtype=torch.int64).numpy().tobytes())
        for batch in loader:
            t_wall0 = time.time()
            if args.max_steps is not None and step >= args.max_steps:
                stopped = True
                break

            caption_batch = None
            texts_full = texts_unsaid = has_unsaid = None
            if tri_mode:
                caption_batch = batch
                images = batch['image']
                texts = batch['caption_said']
                texts_full = batch['caption_full']
                texts_unsaid = batch['caption_unsaid']
                has_unsaid = batch['has_unsaid'].to(device)
                if rank == 0 and step == 0:
                    print(debug_caption_examples(batch), flush=True)
            else:
                images, texts = batch

            batch_digests = None
            if rank == 0 and (step % args.log_every == 0 or step == 0):
                # per-step stream identity, used by the RNG-safety matched smoke
                batch_digests = {
                    'batch_image_sha256': hashlib.sha256(images.numpy().tobytes()).hexdigest()[:16],
                    'batch_caption_sha256': hashlib.sha256('\n'.join(texts).encode('utf-8')).hexdigest()[:16],
                }
                if tri_mode:
                    batch_digests.update({
                        'batch_prefix_sha256': batch_digests['batch_caption_sha256'],
                        'batch_full_sha256': hashlib.sha256(
                            '\n'.join(batch['caption_full']).encode('utf-8')).hexdigest()[:16],
                        'batch_unsaid_sha256': hashlib.sha256(
                            '\n'.join(batch['caption_unsaid']).encode('utf-8')).hexdigest()[:16],
                        'batch_has_unsaid_sha256': hashlib.sha256(
                            ','.join(str(bool(value)) for value in batch['has_unsaid']
                                     ).encode('utf-8')).hexdigest()[:16],
                    })
            images = images.to(device, non_blocking=True)
            update_caption_digest(caption_digest, texts)
            if tri_mode:
                update_caption_digest(full_digest, batch['caption_full'])
                update_caption_digest(unsaid_digest, batch['caption_unsaid'])
                update_caption_digest(has_unsaid_digest,
                                      ['%s' % bool(value) for value in batch['has_unsaid']])
            text_tokens = longclip.tokenize(texts, truncate=True).to(device)
            full_tokens = (longclip.tokenize(texts_full, truncate=True).to(device)
                           if texts_full is not None else None)
            unsaid_tokens = (longclip.tokenize(texts_unsaid, truncate=True).to(device)
                             if texts_unsaid is not None else None)

            scale_factor = set_lrs(optimizer, base_lrs, step, args.warmup_length, total_steps)
            t_compute0 = time.time()
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                out = ddp_model(images, text_tokens, args.lambda_global, args.lambda_said,
                                lambda_unsaid=args.lambda_unsaid,
                                tau_unsaid=args.tau_unsaid,
                                unsaid_residual_eps=args.unsaid_residual_eps,
                                texts_full=full_tokens,
                                texts_unsaid=unsaid_tokens,
                                has_unsaid=has_unsaid,
                                global_caption_view=args.global_caption_view,
                                unsaid_mode=args.unsaid_mode,
                                unsaid_gate_floor=args.unsaid_gate_floor,
                                unsaid_gate_temperature=args.unsaid_gate_temperature,
                                unsaid_suppression_beta=args.unsaid_suppression_beta,
                                unsaid_candidate_chunk_size=args.unsaid_candidate_chunk_size)
                loss = out['loss_total']
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            compute_times.append(time.time() - t_compute0)
            wall_times.append(time.time() - t_wall0)

            is_last_step = (
                (args.max_steps is not None and step == args.max_steps - 1)
                or (args.max_steps is None and step == stop_steps - 1)
            )
            if rank == 0 and (step % args.log_every == 0 or step == 0 or is_last_step or step + 1 in save_at):
                throughput = summarize_throughput(global_batch, compute_times, wall_times)
                record = {
                    'step': step,
                    'completed_steps': step + 1,
                    'epoch': epoch,
                    'lr_scale': scale_factor,
                    'backbone_lr': optimizer.param_groups[0]['lr'],
                    'head_lr': optimizer.param_groups[1]['lr'],
                    'loss_global': float(out['loss_global'].detach()),
                    'loss_said': float(out['loss_said'].detach()),
                    'loss_route': float(out['loss_route'].detach()),
                    'loss_evidence': float(out['loss_evidence'].detach()),
                    'loss_unsaid': float(out['loss_unsaid'].detach()),
                    'unsaid_enabled': bool(out['unsaid_enabled']),
                    'route_top1_acc': float(out['route_top1_acc'].detach()),
                    'evidence_top1_acc': float(out['evidence_top1_acc'].detach()),
                    'route_margin': float(out['route_margin'].detach()),
                    'evidence_margin': float(out['evidence_margin'].detach()),
                    'loss_total': float(out['loss_total'].detach()),
                    'said_attention_entropy': float(out['said_attention_entropy']),
                    'said_effective_patch_count': float(out['said_effective_patch_count']),
                    'said_attention_max': float(out['said_attention_max']),
                    'said_attention_min': float(out['said_attention_min']),
                    'said_feature_norm': float(out['said_feature_norm']),
                    'router_input_feature_norm': float(out['router_input_feature_norm']),
                    'global_feature_norm': float(out['global_feature_norm']),
                }
                record.update(throughput)
                if batch_digests:
                    record.update(batch_digests)
                for key in ('pair_gap_full', 'pair_gap_said', 'balancing_gain',
                            'relative_balancing_gain'):
                    record[key] = float(out[key])
                record['representation_gap_scope'] = 'rank0_local_training_batch'
                if caption_batch is not None:
                    record.update(caption_view_stats(caption_batch))
                # Unsaid monitors: None (JSON null) when the branch is disabled, so a
                # skipped branch can never be read as a measurement.
                for key in UNSAID_DIAGNOSTIC_KEYS:
                    value = out.get(key)
                    record[key] = None if value is None else float(value)
                if torch.cuda.is_available():
                    record['peak_gpu_mem_gb'] = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                record['sec_per_step_avg'] = throughput['compute_sec_per_step']  # legacy alias (compute-only)
                with open(log_path, 'a') as fp:
                    fp.write(json.dumps(record, sort_keys=True) + '\n')
                print('LOG ' + json.dumps(record, sort_keys=True), flush=True)

            step += 1
            if rank == 0 and ((args.save_every and step % args.save_every == 0) or step in save_at):
                ckpt_path = os.path.join(args.output_dir, 'salu_said_only_step%06d.pt' % step)
                save_checkpoint(ckpt_path, ddp_model, optimizer, scaler, step, epoch, args, base_lrs, total_steps)
                print('SAVED ' + ckpt_path, flush=True)

            if (args.val_every > 0 and step % args.val_every == 0
                    and (args.val_sharegpt4v or args.eval_coco or args.val_coco)):
                # keep every rank in lockstep around the rank-0-only evaluation
                if dist.is_initialized():
                    dist.barrier()
                if rank == 0:
                    records = run_planned_validation(
                        args, ddp_model, preprocess, cohort, cohort_root, device,
                        step, epoch, False, False, validation_history_path, seen_validation)
                    for record in records:
                        print('VAL ' + json.dumps(record, sort_keys=True, ensure_ascii=False), flush=True)
                if dist.is_initialized():
                    dist.barrier()

        if (args.val_sharegpt4v or args.eval_coco_each_epoch) and not stopped:
            if dist.is_initialized():
                dist.barrier()
            if rank == 0:
                records = run_planned_validation(
                    args, ddp_model, preprocess, cohort, cohort_root, device,
                    step, epoch, True, False, validation_history_path, seen_validation)
                for record in records:
                    print('VAL_EPOCH_END ' + json.dumps(record, sort_keys=True, ensure_ascii=False), flush=True)
            if dist.is_initialized():
                dist.barrier()

        if stopped:
            break

    if rank == 0:
        final_path = os.path.join(args.output_dir, 'salu_said_only_last.pt')
        save_checkpoint(final_path, ddp_model, optimizer, scaler, step, epoch, args, base_lrs, total_steps)
        elapsed = time.time() - t_start
        throughput = summarize_throughput(global_batch, compute_times, wall_times)
        summary = {
            'steps': step - start_step,
            'wall_sec_total': elapsed,
            'final_checkpoint': final_path,
        }
        summary.update(throughput)
        print('SUMMARY ' + json.dumps(summary, sort_keys=True), flush=True)
        with open(os.path.join(args.output_dir, 'salu_summary.json'), 'w') as fp:
            json.dump(summary, fp, indent=2, sort_keys=True)

        if args.val_sharegpt4v or args.eval_coco or args.val_coco:
            records = run_planned_validation(
                args, ddp_model, preprocess, cohort, cohort_root, device,
                step, epoch, False, True, validation_history_path, seen_validation)
            for record in records:
                print('VAL_FINAL ' + json.dumps(record, sort_keys=True, ensure_ascii=False), flush=True)

        if args.legacy_eval_coco:
            ddp_model.eval()
            result = eval_coco(ddp_model, preprocess)
            print('COCO_RETRIEVAL_LEGACY ' + json.dumps(result, sort_keys=True), flush=True)
            with open(os.path.join(args.output_dir, 'coco_retrieval_legacy.json'), 'w') as fp:
                json.dump(result, fp, indent=2, sort_keys=True)

    if dist.is_initialized():
        audit = {'rank': rank, 'seed': args.seed, 'initial_state_sha256': initial_digest,
                 'sampler_order_sha256': sampler_digest.hexdigest(),
                 'caption_stream_sha256': caption_digest.hexdigest(),
                 'prefix_caption_stream_sha256': caption_digest.hexdigest(),
                 'full_caption_stream_sha256': full_digest.hexdigest() if tri_mode else None,
                 'unsaid_caption_stream_sha256': unsaid_digest.hexdigest() if tri_mode else None,
                 'has_unsaid_stream_sha256': has_unsaid_digest.hexdigest() if tri_mode else None,
                 'steps': step, 'dataset_size': len(train_set)}
        with open(os.path.join(args.output_dir, 'reproducibility_rank%d.json' % rank), 'w') as fp:
            json.dump(audit, fp, indent=2, sort_keys=True)
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
