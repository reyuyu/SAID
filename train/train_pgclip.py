"""PG-CLIP v0.1 trainer: native alignment + pre-projection text-conditioned visual selection.

One configuration only -- 4 x 256 pairs, exactly 500 optimizer updates, two AdamW optimizers
(CLIP at 1e-6 / weight decay 1e-2 / warmup 200, the new 768-d gate at 1e-3 / weight decay 0 /
warmup 0), betas (0.9, 0.999), eps 1e-8, cosine horizon ``3 * len(loader)`` (the real loader, never a
hard-coded 3651) and ``--max_steps`` only truncating the run.

Per optimizer step
------------------
1. every rank encodes its local images **once** (one visual pass exposing ``ln_post`` output before
   ``visual.proj``) and its local captions **once** (one text pass exposing the full ``ln_final``
   sequence);
2. every rank builds its local 768-d masks from its own captions;
3. text features and masks are gathered with an autograd-aware row gather;
4. local image rows x **global** texts give ``QG_local`` / ``QP_local``;
5. the score rows are gathered so each rank holds the global scalar matrices;
6. I2T uses the local image rows, T2I uses the columns of the local texts;
7. the sparsity term uses the local captions' masks, computed once.

The route is the mature one: per-rank anchor means, autograd-aware gather, standard DDP parameter
gradient averaging -- no extra world-size factor is applied anywhere. Training always goes through
``ddp_model(...) -> backward() -> optimizer.step()``; ``.module`` is only used for parameter
grouping, state reads, export and logging.
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (os.path.dirname(os.path.abspath(__file__)), REPO):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from model import longclip                                                    # noqa: E402
from model.pgclip import (ARM, IMAGE_CHUNK_DEFAULT, LAMBDA_GLOBAL, LAMBDA_PREPROJ,  # noqa: E402
                          LAMBDA_SPARSE, LOSS_WEIGHTS, NORM_EPS, OBJECTIVE,
                          TEXT_CHUNK_DEFAULT, PreProjectionGate, assert_equal_local_batch,
                          build_optimizers, check_resume_compatible, checkpoint_metadata,
                          conditional_scores, config_dict, energy_metrics, file_sha256,
                          gate_init_report, gather_rows, git_head, global_targets,
                          mask_statistics, model_shapes, native_scores, sparse_term,
                          state_digest, total_loss)
from model.longclip import _tokenizer as PGCLIP_TOKENIZER                       # noqa: E402
from scheduler import cosine_lr                                                # noqa: E402
from said_cvssl_data import Share4VCvsslDataset, cvssl_collate                 # noqa: E402

TOKENIZER_CONTEXT = 248
PRECISION_NOTE = ('fp32 master weights; bf16 autocast for the visual/text transformers; the '
                  'pre-projection core (native projection, conditional projection, final text '
                  'projection, gate, normalise, scores, losses) is explicitly fp32')


# --------------------------------------------------------------------------- small helpers
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed(backend='nccl'):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
    else:
        rank, world, local_rank = 0, 1, 0
    if world > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, rank=rank, world_size=world)
    return rank, local_rank, world


def tensor_digest(tensor) -> str:
    array = tensor.detach().to(torch.float32).cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def load_init_state(model, path: str, rank: int):
    """Load the complete frozen initial state or fail loudly (adapted from the S0 lineage).

    A bare CLIP layout and a ``clip.``-prefixed layout are both accepted. Only ``mask_net`` keys may
    be absent (the compatibility-only frozen head is not part of this experiment's mathematics).
    """
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(payload, dict) and 'model' in payload:
        payload = payload['model']
    target = set(model.state_dict())
    mapped, ignored = {}, []
    for key, value in payload.items():
        if key in target:
            mapped[key] = value
        elif key.startswith('clip.') and key[len('clip.'):] in target:
            mapped[key[len('clip.'):]] = value
        elif ('clip.' + key) in target:
            mapped['clip.' + key] = value
        else:
            ignored.append(key)
    if not mapped:
        raise ValueError('--init_state %s matches no model tensor' % path)
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    bad_missing = [key for key in missing if not key.startswith('mask_net')]
    if bad_missing or list(unexpected):
        raise RuntimeError('--init_state did not load cleanly: missing %r unexpected %r'
                           % (bad_missing, list(unexpected)))
    if rank == 0:
        print('INIT_STATE_LOADED %s tensors=%d mask_net_absent=%d ignored=%r'
              % (path, len(mapped), len([k for k in missing if k.startswith('mask_net')]),
                 ignored[:5]), flush=True)


def raw_token_lengths(captions):
    """Reference tokeniser length check (no forward): how many captions exceed the 248 capacity."""
    over = 0
    longest = 0
    for caption in captions:
        length = len(PGCLIP_TOKENIZER.encode(caption)) + 2          # SOT + EOT
        longest = max(longest, length)
        if length > TOKENIZER_CONTEXT:
            over += 1
    return over, longest


def grads_finite(parameters) -> bool:
    for parameter in parameters:
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            return False
    return True


def grad_norm(parameters, device) -> float:
    total = torch.zeros((), device=device, dtype=torch.float32)
    for parameter in parameters:
        if parameter.grad is not None:
            total = total + parameter.grad.detach().float().pow(2).sum()
    return float(total.sqrt())


# --------------------------------------------------------------------------- training module
class PgClipTrainModule(torch.nn.Module):
    """The PG-CLIP objective: one CLIP model (shared trunk and projections) plus one new 768-d gate.

    ``clip`` is registered exactly once and the gate is registered outside it, so ``visual.proj`` is
    a single Parameter of a single module and cannot be duplicated by the optimizer.
    """

    def __init__(self, clip: torch.nn.Module, gate: PreProjectionGate, rank: int = 0,
                 image_chunk: int = IMAGE_CHUNK_DEFAULT, text_chunk: int = TEXT_CHUNK_DEFAULT,
                 qp_checkpoint: bool = True):
        super().__init__()
        self.clip = clip
        self.gate = gate
        self.rank = int(rank)
        self.image_chunk = int(image_chunk)
        self.text_chunk = int(text_chunk)
        self.qp_checkpoint = bool(qp_checkpoint)
        self.objective = OBJECTIVE

    # -- encoding -------------------------------------------------------------------------
    def encode(self, images, text, amp_dtype, amp_enabled: bool):
        """One visual pass and one text pass under autocast, then the fp32 pre-projection core."""
        device_type = 'cuda' if images.is_cuda else 'cpu'
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
            h_pre, v_raw = self.clip.encode_image_with_preprojection(images)
            text_hidden = self.clip.encode_text_final_hidden(text)
        with torch.autocast(device_type=device_type, enabled=False):
            h = h_pre.float()
            hidden = text_hidden.float()
            eot, effective_length = self.clip.eot_indices(text)
            t_raw = hidden[torch.arange(hidden.shape[0], device=hidden.device), eot] \
                @ self.clip.text_projection.float()
            t_unit = torch.nn.functional.normalize(t_raw, dim=-1, eps=NORM_EPS)
            masks, probabilities = self.gate(hidden)
        return {'h': h, 'v_raw': v_raw, 't_raw': t_raw, 't_unit': t_unit, 'masks': masks,
                'probabilities': probabilities, 'effective_length': effective_length}

    # -- objective ------------------------------------------------------------------------
    def forward(self, images, text, amp_dtype=torch.bfloat16, amp_enabled: bool = True):
        encoded = self.encode(images, text, amp_dtype, amp_enabled)
        h, masks = encoded['h'], encoded['masks']
        W = self.clip.visual.proj
        local_size = int(h.shape[0])
        assert_equal_local_batch(local_size)

        t_unit_all = gather_rows(encoded['t_unit'])                 # autograd-aware, never detached
        masks_all = gather_rows(masks)
        qg_local = native_scores(h, W, t_unit_all)
        qp_local = conditional_scores(h, W, t_unit_all, masks_all, image_chunk=self.image_chunk,
                                      text_chunk=self.text_chunk, use_checkpoint=self.qp_checkpoint)
        qg = gather_rows(qg_local)
        qp = gather_rows(qp_local)

        world = dist.get_world_size() if dist.is_initialized() else 1
        targets = global_targets(local_size, self.rank, qg.device)
        columns = slice(self.rank * local_size, (self.rank + 1) * local_size)

        # I2T: the local images are the anchors, every candidate text is scored with its own mask.
        # T2I: this rank's texts are the anchors, the candidates are all images and each fixed text
        # keeps its single mask (the columns of the global matrix).
        lg_i2t = torch.nn.functional.cross_entropy(qg_local, targets)
        lg_t2i = torch.nn.functional.cross_entropy(qg[:, columns].t(), targets)
        lp_i2t = torch.nn.functional.cross_entropy(qp_local, targets)
        lp_t2i = torch.nn.functional.cross_entropy(qp[:, columns].t(), targets)

        loss_global = lg_i2t + lg_t2i
        loss_preproj = lp_i2t + lp_t2i
        loss_sparse = sparse_term(masks)
        loss_total = total_loss(loss_global, loss_preproj, loss_sparse)

        with torch.no_grad():
            def path_stats(rows):
                """Positive / strongest negative / max margin / LSE margin / top1 for one direction."""
                positive = rows.gather(1, targets.view(-1, 1)).squeeze(1)
                masked = rows.clone()
                masked.scatter_(1, targets.view(-1, 1), float('-inf'))
                strongest = masked.max(dim=1).values
                margin = positive - strongest
                lse = torch.logsumexp(rows, dim=1) - positive
                top1 = (rows.argmax(dim=1) == targets).float().mean()
                return {'positive_mean': float(positive.mean()),
                        'strongest_negative_mean': float(strongest.mean()),
                        'max_margin_mean': float(margin.mean()),
                        'max_margin_min': float(margin.min()),
                        'lse_margin_mean': float(lse.mean()),
                        'top1': float(top1),
                        'positive_win_fraction': float((margin > 0).float().mean())}

            stats = {
                'path_global_i2t': path_stats(qg_local),
                'path_global_t2i': path_stats(qg[:, columns].t()),
                'path_preproj_i2t': path_stats(qp_local),
                'path_preproj_t2i': path_stats(qp[:, columns].t()),
                'global_batch': int(qg.shape[0]), 'local_batch': local_size,
                'mask_all_one': bool((masks.detach() >= 0.5).all())}
        return {'loss_global_i2t': lg_i2t, 'loss_global_t2i': lg_t2i, 'loss_global': loss_global,
                'loss_preproj_i2t': lp_i2t, 'loss_preproj_t2i': lp_t2i,
                'loss_preproj': loss_preproj, 'loss_sparse': loss_sparse, 'loss_total': loss_total,
                'weighted_global': LAMBDA_GLOBAL * loss_global,
                'weighted_preproj': LAMBDA_PREPROJ * loss_preproj,
                'weighted_sparse': LAMBDA_SPARSE * loss_sparse,
                'stats': stats, 'encoded': encoded}


# --------------------------------------------------------------------------- one real step
def pgclip_train_step(ddp_model, batch, optimizers, device, amp_dtype, amp_enabled=True,
                      capture_grads=False, grad_health=0.0, completed_steps=0, text_ids=None):
    """DDP forward -> backward -> gradient-health collective -> both optimizer steps.

    ``text_ids`` is an optional test hook: when given, the batch is not re-tokenised, so a two-rank
    run and a single-process reference can be fed byte-identical tokens.
    """
    images = batch['image_a'].to(device, non_blocking=True)
    text = (text_ids if text_ids is not None
            else longclip.tokenize(batch['caption_said'], truncate=True)).to(device)
    out = ddp_model(images, text, amp_dtype=amp_dtype, amp_enabled=amp_enabled)
    loss = out['loss_total']
    loss.backward()

    module = getattr(ddp_model, 'module', ddp_model)
    clip_parameters = [p for p in module.clip.parameters() if p.requires_grad]
    gate_parameters = [p for p in module.gate.parameters() if p.requires_grad]
    if capture_grads:
        out['_grads'] = {name: (None if parameter.grad is None else parameter.grad.detach().clone())
                         for name, parameter in module.named_parameters()}
    local_ok = grads_finite(clip_parameters) and grads_finite(gate_parameters)
    health = {'clip_grad_norm': grad_norm(clip_parameters, device),
              'gate_grad_norm': grad_norm(gate_parameters, device),
              'gate_output_grad_norm': float(module.gate.projection.weight.grad.detach().norm())
              if module.gate.projection.weight.grad is not None else 0.0,
              'gate_stem_grad_norm': math.sqrt(sum(
                  float(p.grad.detach().float().pow(2).sum()) for p in module.gate.stem.parameters()
                  if p.grad is not None)),
              'visual_proj_grad_norm': float(module.clip.visual.proj.grad.detach().norm())
              if module.clip.visual.proj.grad is not None else 0.0,
              'text_projection_grad_norm': float(module.clip.text_projection.grad.detach().norm())
              if module.clip.text_projection.grad is not None else 0.0}
    if grad_health > 0:
        health['grad_health_threshold'] = grad_health
    # a single non-finite gradient on any rank must stop every rank, so the flag is collective and
    # runs on ALL ranks (no rank-0-only collective on a path the others skip)
    flag = torch.tensor([0.0 if local_ok else 1.0], device=device)
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if float(flag.item()) > 0:
        raise SystemExit('REFUSING the optimizer step at step %d: a non-finite gradient was found '
                         'on rank %d of %d (no clipping, no nan_to_num, no skipping)'
                         % (completed_steps, module.rank,
                            dist.get_world_size() if dist.is_initialized() else 1))

    optimizers['clip'].step()
    optimizers['gate'].step()
    optimizers['clip'].zero_grad(set_to_none=True)
    optimizers['gate'].zero_grad(set_to_none=True)
    out['_grad_health'] = health
    out['_grads_finite'] = bool(local_ok)
    return out


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description='PG-CLIP v0.1 (%s)' % OBJECTIVE)
    parser.add_argument('--base_model', default='B16')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-6, help='CLIP backbone + native projections')
    parser.add_argument('--gate_lr', type=float, default=1e-3, help='the new 768-d gate')
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--warmup_length', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--init_state', default=None)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--save_completed_steps', default='0,20,100,250,500')
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--heavy_log_every', type=int, default=25)
    parser.add_argument('--grad_health_steps', type=int, default=0,
                        help='record per-group gradient norms at this completed step (0 = never)')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--amp_dtype', default='bf16', choices=['bf16', 'fp32'])
    parser.add_argument('--image_chunk', type=int, default=IMAGE_CHUNK_DEFAULT)
    parser.add_argument('--text_chunk', type=int, default=TEXT_CHUNK_DEFAULT)
    parser.add_argument('--qp_checkpoint', type=int, default=1,
                        help='non-reentrant activation checkpointing inside the conditional score '
                             'blocks (memory only; the arithmetic is unchanged)')
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()
    if args.base_model == 'B16':
        args.base_model = 'ViT-B/16'

    seed_everything(args.seed)
    rank, local_rank, world = setup_distributed()
    device = torch.device('cuda', local_rank)
    save_completed = sorted({int(s) for s in args.save_completed_steps.split(',') if s.strip()})

    model, _preprocess = longclip.load_from_clip(args.base_model, device='cpu', download_root=None,
                                                 args=args)
    model.train()
    # compatibility key only: PG-CLIP scores with the fixed 100x scale, logit_scale is never used
    model.logit_scale = torch.nn.Parameter(torch.ones([]) * math.log(100.0))
    model = model.to(device)
    init_file_sha = None
    if args.init_state:
        if not os.path.isfile(args.init_state):
            raise SystemExit('shared init missing: %s' % args.init_state)
        init_file_sha = file_sha256(args.init_state)
        load_init_state(model, args.init_state, rank)
    initial_state_digest = state_digest(model.state_dict())

    gate = PreProjectionGate(device=device).to(device)
    with torch.no_grad():
        probe_hidden = torch.zeros(2, TOKENIZER_CONTEXT, model.text_projection.shape[0],
                                   device=device)
        init_report = gate_init_report(gate, probe_hidden)
    if rank == 0:
        print('GATE_INIT ' + json.dumps(init_report, sort_keys=True), flush=True)

    train_module = PgClipTrainModule(model, gate, rank=rank, image_chunk=args.image_chunk,
                                     text_chunk=args.text_chunk,
                                     qp_checkpoint=bool(args.qp_checkpoint)).to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        train_module, device_ids=[local_rank], output_device=local_rank,
        find_unused_parameters=True)
    ddp_model._set_static_graph()
    clip_handle = train_module.clip                       # read-only handle for checkpoints/logs

    optimizers = build_optimizers(clip_handle, train_module.gate, clip_lr=args.lr,
                                  gate_lr=args.gate_lr, weight_decay=args.weight_decay)
    partition = optimizers['partition']
    use_amp = args.amp_dtype == 'bf16'
    amp_dtype = torch.bfloat16

    dtype_counts = {}
    for parameter in list(clip_handle.parameters()) + list(train_module.gate.parameters()):
        key = str(parameter.dtype)
        dtype_counts[key] = dtype_counts.get(key, 0) + 1
    group_dtypes = {name: sorted({str(p.dtype) for p in optimizer.param_groups[0]['params']})
                    for name, optimizer in optimizers.items() if name in ('clip', 'gate')}
    if rank == 0:
        print('DTYPE_AUDIT parameter_dtypes=%s clip_group=%s gate_group=%s frozen=%s'
              % (sorted(dtype_counts.items()), group_dtypes['clip'], group_dtypes['gate'],
                 partition['frozen']), flush=True)
    if set(dtype_counts) != {'torch.float32'}:
        raise RuntimeError('fp32 master weights expected, got %r' % (dtype_counts,))
    if group_dtypes != {'clip': ['torch.float32'], 'gate': ['torch.float32']}:
        raise RuntimeError('optimizer groups must be fp32, got %r' % (group_dtypes,))

    dataset = Share4VCvsslDataset(seed=args.seed, augment_view_b=False,
                                  strict_manifest=os.environ.get('SHARE4V_FULL_AUDIT'))
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, seed=args.seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                                        num_workers=args.num_workers, pin_memory=True,
                                        collate_fn=cvssl_collate, drop_last=False)
    steps_per_epoch = len(loader)
    total_steps = args.epochs * steps_per_epoch
    scheduler = cosine_lr(optimizers['clip'], base_lr=args.lr, warmup_length=args.warmup_length,
                          steps=total_steps)
    gate_scheduler = cosine_lr(optimizers['gate'], base_lr=args.gate_lr, warmup_length=0,
                               steps=total_steps)

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, 'salu_log.jsonl')
    identity = config_dict()
    config = dict(identity)
    config.update({
        'loss_weights': dict(LOSS_WEIGHTS), 'loss_combination': '5*(LG_i2t+LG_t2i) + 5*(LP_i2t+LP_t2i) + mean(|mask|)',
        'two_paths': {'native': 'Norm(h @ clip.visual.proj)', 'preproj': 'Norm((h * mask_j) @ clip.visual.proj)'},
        'h_source': 'ln_post(CLS of visual block 12) before visual.proj, one visual forward',
        'text_source': 'ln_final sequence, EOT = argmax(token ids), native text_projection in fp32',
        'gate': train_module.gate.config(),
        'candidate_rule': 'fixed image: each candidate text uses its own mask; fixed text: all '
                          'candidate images share that mask; positives and negatives identical',
        'grader': 'none: LG never touches the gate, LP touches all shared parameters and the gate, '
                  'LS (mean|mask|) touches only the gate',
        'precision': PRECISION_NOTE,
        'chunking': {'image_chunk': args.image_chunk, 'text_chunk': args.text_chunk,
                     'qp_checkpoint': bool(args.qp_checkpoint),
                     'note': 'exact blocked (h*m)@W with fp32 normalisation; no [B,B,768] activation '
                             'is ever resident'},
        'ddp_route': 'per-rank anchor means + autograd-aware gather + standard DDP averaging; no '
                     'world-size factor',
        'no_world_size_factor': True,
        'sampling': 'DistributedSampler(shuffle=True), no drop_last, steps_per_epoch=len(loader)=%d'
                    % steps_per_epoch,
        'view': 'image_a only (reference openai-clip _transform(224)); no second view, no new '
                'augmentation task',
        'caption_stream': 'reference prefix draw caption.split(". ")[:k], k ~ randint(1, n)',
        'lr': args.lr, 'gate_lr': args.gate_lr, 'weight_decay': args.weight_decay,
        'warmup_length': args.warmup_length, 'epochs': args.epochs,
        'batch_size_per_gpu': args.batch_size, 'world_size': world,
        'global_batch': args.batch_size * world,
        'loader_batches': steps_per_epoch, 'lr_horizon_steps': total_steps,
        'max_steps': args.max_steps, 'seed': args.seed, 'init_state': args.init_state,
        'init_file_sha256': init_file_sha, 'initial_state_digest': initial_state_digest,
        'git_head': git_head(REPO), 'tokenizer_context': TOKENIZER_CONTEXT,
        'model_shapes': model_shapes(clip_handle),
        'statistics_scope': 'rank0 local batch unless a field says otherwise',
    })
    if rank == 0:
        print('CONFIG ' + json.dumps(config, sort_keys=True), flush=True)
        print('DATASET_SIZE %d STEPS_PER_EPOCH %d LR_HORIZON %d CLIP_TENSORS %d GATE_TENSORS %d'
              % (len(dataset), steps_per_epoch, total_steps, partition['clip_count'],
                 partition['gate_count']), flush=True)
        with open(os.path.join(args.output_dir, 'config.json'), 'w') as handle:
            json.dump(config, handle, indent=2, sort_keys=True)

    if args.resume:
        payload = torch.load(args.resume, map_location='cpu', weights_only=False)
        check_resume_compatible(payload, identity)
        clip_handle.load_state_dict(payload['clip'])
        train_module.gate.load_state_dict(payload['gate'])
        optimizers['clip'].load_state_dict(payload['optimizer_clip'])
        optimizers['gate'].load_state_dict(payload['optimizer_gate'])
        start_step = int(payload['completed_steps'])
        if rank == 0:
            print('RESUMED from %s at completed step %d' % (args.resume, start_step), flush=True)
    else:
        start_step = 0

    caption_digest = hashlib.sha256()
    sample_digest = hashlib.sha256()
    image_id_digest = hashlib.sha256()
    prefix_digest = hashlib.sha256()

    def save_checkpoint(steps, epoch, step_in_epoch, health=None):
        """Write one self-describing checkpoint (rank 0 only; the state is identical on all ranks)."""
        digests = {'init_file_sha256': init_file_sha,
                   'initial_state_digest': initial_state_digest,
                   'caption_stream_sha256': caption_digest.hexdigest(),
                   'sample_stream_sha256': sample_digest.hexdigest(),
                   'image_id_stream_sha256': image_id_digest.hexdigest(),
                   'prefix_k_stream_sha256': prefix_digest.hexdigest()}
        payload = {
            'clip': clip_handle.state_dict(),
            'gate': train_module.gate.state_dict(),
            'optimizer_clip': optimizers['clip'].state_dict(),
            'optimizer_gate': optimizers['gate'].state_dict(),
            'epoch': epoch, 'step_in_epoch': step_in_epoch,
            'rng': {'torch': torch.get_rng_state(),
                    'cuda': torch.cuda.get_rng_state(device) if torch.cuda.is_available() else None},
        }
        payload.update(checkpoint_metadata(
            steps=steps, rank=rank, world=world, config=config, digests=digests,
            batch_size=args.batch_size,
            chunking={'image_chunk': args.image_chunk, 'text_chunk': args.text_chunk,
                      'qp_checkpoint': bool(args.qp_checkpoint)},
            precision=PRECISION_NOTE))
        payload['grad_health'] = health
        path = os.path.join(args.output_dir, 'pgclip_%s_step%06d.pt' % (ARM, steps))
        torch.save(payload, path)
        print('SAVED ' + path, flush=True)

    if rank == 0 and start_step == 0 and 0 in save_completed:
        # step 0 is the frozen shared init, written before the first update (never the 501st step)
        save_checkpoint(0, 0, -1)
    in_loop_saves = sorted(step for step in save_completed if step > start_step and step != 0)

    completed = start_step
    t_start = time.time()
    compute_times = []
    over_capacity = 0
    caption_count = 0
    max_raw_tokens = 0
    max_effective = 0
    effective_lengths = []
    stopped = False

    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        for i, batch in enumerate(loader):
            if args.max_steps is not None and completed >= args.max_steps:
                stopped = True
                break
            t0 = time.time()
            scheduler(completed)
            gate_scheduler(completed)
            out = pgclip_train_step(ddp_model, batch, optimizers, device, amp_dtype,
                                    amp_enabled=use_amp, completed_steps=completed,
                                    grad_health=1.0 if completed in (0, 1) else 0.0)
            local_over, longest = raw_token_lengths(batch['caption_said'])
            over_capacity += local_over
            caption_count += len(batch['caption_said'])
            max_raw_tokens = max(max_raw_tokens, longest)
            effective = out['encoded']['effective_length']
            max_effective = max(max_effective, int(effective.max()))
            effective_lengths.append(float(effective.float().mean()))
            completed += 1
            compute_times.append(time.time() - t0)

            caption_digest.update(('\n'.join(batch['caption_said'])).encode('utf-8'))
            sample_digest.update(batch['sample_id'].numpy().tobytes())
            image_id_digest.update(batch['image_id'].numpy().tobytes())
            prefix_digest.update(batch['prefix_k'].numpy().tobytes())

            log_now = (completed % args.log_every == 0 or completed == 1
                       or completed in save_completed)
            heavy = (args.heavy_log_every > 0 and completed % args.heavy_log_every == 0)
            if rank == 0 and (log_now or heavy):
                stats = out['stats']
                record = {
                    'completed_steps': completed, 'epoch': epoch, 'step_in_epoch': i,
                    'arm': ARM, 'objective': OBJECTIVE, 'world_size': world,
                    'batch_size_local': stats['local_batch'], 'global_pairs': stats['global_batch'],
                    'lr': optimizers['clip'].param_groups[0]['lr'],
                    'gate_lr': optimizers['gate'].param_groups[0]['lr'],
                    'loss_global_i2t': float(out['loss_global_i2t']),
                    'loss_global_t2i': float(out['loss_global_t2i']),
                    'loss_global': float(out['loss_global']),
                    'loss_preproj_i2t': float(out['loss_preproj_i2t']),
                    'loss_preproj_t2i': float(out['loss_preproj_t2i']),
                    'loss_preproj': float(out['loss_preproj']),
                    'weights_5_5_1': [LAMBDA_GLOBAL, LAMBDA_PREPROJ, LAMBDA_SPARSE],
                    'weighted_loss_global': float(out['weighted_global']),
                    'weighted_loss_preproj': float(out['weighted_preproj']),
                    'weighted_loss_sparse': float(out['weighted_sparse']),
                    'loss_sparse': float(out['loss_sparse']),
                    'loss_total': float(out['loss_total']),
                    'optimizer_steps_before_update': int(completed) - 1,
                    'sec_per_step': compute_times[-1],
                    'samples_per_sec': float(stats['local_batch']) * world / max(compute_times[-1], 1e-9),
                    'peak_memory_gb': torch.cuda.max_memory_allocated(device) / (1024 ** 3),
                    'rank_param_digest': state_digest(clip_handle.state_dict())[:16],
                    'gate_param_digest': state_digest(train_module.gate.state_dict())[:16],
                    'batch_caption_sha256': hashlib.sha256(
                        ('\n'.join(batch['caption_said'])).encode('utf-8')).hexdigest()[:16],
                    'batch_image_id_sha256': tensor_digest(batch['image_id']),
                    'statistics_scope': 'rank0_local_batch',
                    'over_capacity_captions_cumulative': over_capacity,
                    'captions_seen': caption_count,
                    'max_raw_token_length': max_raw_tokens,
                    'max_effective_length': max_effective,
                    'conditional_chunking': {'image_chunk': args.image_chunk,
                                             'text_chunk': args.text_chunk,
                                             'qp_checkpoint': bool(args.qp_checkpoint)},
                }
                for path in ('path_global_i2t', 'path_global_t2i', 'path_preproj_i2t',
                             'path_preproj_t2i'):
                    for key, value in stats[path].items():
                        record['%s_%s' % (path, key)] = value
                health = out['_grad_health']
                record.update(health)
                if heavy:
                    masks = out['encoded']['masks']
                    probabilities = out['encoded']['probabilities']
                    h = out['encoded']['h']
                    record.update(mask_statistics(masks, probabilities))
                    record.update(energy_metrics(h, clip_handle.visual.proj, masks))
                    record['h_norm_mean'] = float(h.norm(dim=-1).mean())
                    record['mask_all_one'] = bool(stats['mask_all_one'])
                    record['effective_length_mean'] = float(np.mean(effective_lengths[-20:]))
                with open(log_path, 'a') as handle:
                    handle.write(json.dumps(record, sort_keys=True) + '\n')
                print('LOG ' + json.dumps(record, sort_keys=True), flush=True)

            if rank == 0 and completed in in_loop_saves:
                save_checkpoint(completed, epoch, i, health=out['_grad_health'])
        if stopped:
            break

    if rank == 0:
        summary = {
            'arm': ARM, 'objective': OBJECTIVE, 'completed_steps': completed,
            'max_steps': args.max_steps, 'epochs': args.epochs,
            'wall_sec': time.time() - t_start,
            'mean_sec_per_step': sum(compute_times) / max(len(compute_times), 1),
            'steps_per_epoch': steps_per_epoch, 'lr_horizon_steps': total_steps,
            'world_size': world, 'batch_size_per_gpu': args.batch_size,
            'global_batch': args.batch_size * world,
            'peak_memory_gb': torch.cuda.max_memory_allocated(device) / (1024 ** 3),
            'init_file_sha256': init_file_sha, 'initial_state_digest': initial_state_digest,
            'final_state_digest': state_digest(clip_handle.state_dict()),
            'final_gate_digest': state_digest(train_module.gate.state_dict()),
            'caption_stream_sha256': caption_digest.hexdigest(),
            'sample_stream_sha256': sample_digest.hexdigest(),
            'image_id_stream_sha256': image_id_digest.hexdigest(),
            'prefix_k_stream_sha256': prefix_digest.hexdigest(),
            'captions_seen': caption_count,
            'over_capacity_captions': over_capacity,
            'max_effective_length': max_effective,
            'precision': PRECISION_NOTE,
            'statistics_scope': 'rank0 local batch',
        }
        with open(os.path.join(args.output_dir, 'run_summary.json'), 'w') as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print('RUN_SUMMARY ' + json.dumps(summary, sort_keys=True), flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
