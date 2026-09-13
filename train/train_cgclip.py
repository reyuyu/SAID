"""CG-CLIP v0.1 trainer: native alignment + text-gated final-block CLS attention.

One configuration only -- 4 x 256 pairs, exactly 500 optimizer updates, two AdamW optimizers
(CLIP at 1e-6 / weight decay 1e-2 / warmup 200, the new 64-d caption gate at 1e-3 / weight decay 0 /
warmup 0), betas (0.9, 0.999), eps 1e-8, cosine horizon ``3 * len(loader)`` (the real loader, never a
hard-coded 3651) and ``--max_steps`` only truncating the run.

Per optimizer step
-----------------
1. every rank runs the visual trunk ONCE, up to (and excluding) the last visual block, exposing
   ``X11`` [B, 197, 768];
2. every rank runs the text trunk ONCE up to ``ln_final`` and applies the native ``text_projection``
   to the EOT row in fp32;
3. the native last-block CLS row is computed once (fp32) and gives ``vG``;
4. the caption gate is applied to the gathered GLOBAL captions x this rank's local images, producing
   one 196-d straight-through patch gate per (text, image) pair;
5. the local image rows x global texts give ``QG_local`` / ``QA_local``; the score rows are gathered
   so each rank holds the global scalar matrices;
6. I2T uses the local image rows, T2I uses the columns of the local texts;
7. the sparse term uses only the TRUE positive pairs' patch gates, taken from the tile that already
   contains them (the gate network is never evaluated a second time).

The route is the mature one: per-rank anchor means, autograd-aware gather, standard DDP parameter
gradient averaging -- no extra world-size factor is applied anywhere. Training always goes through
``ddp_model(...) -> backward() -> optimizer.step()``; ``.module`` is only used for parameter grouping,
state reads, export and logging.
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
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (os.path.dirname(os.path.abspath(__file__)), REPO):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from model import longclip                                                    # noqa: E402
from model.cgclip import (ARM, FIXED_SCALE, IMAGE_CHUNK_DEFAULT, LAMBDA_ATTENTION,  # noqa: E402
                          LAMBDA_GLOBAL, LAMBDA_SPARSE, LOSS_WEIGHTS, NORM_EPS, OBJECTIVE,
                          TEXT_CHUNK_DEFAULT, CaptionGate, assert_equal_local_batch,
                          attention_map_grid, attention_read_diagnostics, build_optimizers,
                          check_resume_compatible, checkpoint_metadata, conditional_cls_scores,
                          config_dict, file_sha256, gate_config, gate_init_report,
                          gate_statistics, gather_rows, git_head, global_targets, native_cls_row,
                          path_statistics, positive_global_columns, sparse_positive_term,
                          state_digest, total_loss, visual_spec)
from model.longclip import _tokenizer as CGCLIP_TOKENIZER                       # noqa: E402
from scheduler import cosine_lr                                                # noqa: E402
from said_cvssl_data import Share4VCvsslDataset, cvssl_collate                 # noqa: E402

TOKENIZER_CONTEXT = 248
GRID_SAMPLES = 2
PRECISION_NOTE = ('fp32 master weights; bf16 autocast for the first 11 visual blocks and the text '
                  'transformer; the final visual CLS core (ln_1, Q/K/V, the native softmax, the '
                  'caption gate, the renormalisation, out_proj, both residuals, ln_2/MLP, ln_post, '
                  'visual.proj), the text projection, the normalisation, the scores and every loss '
                  'are explicitly fp32')
STATS_SCOPE = 'rank0 local batch unless the field name says otherwise'


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
    """Load the complete frozen initial state or fail loudly (same rule as the S0/PG lineage)."""
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
        length = len(CGCLIP_TOKENIZER.encode(caption)) + 2                  # SOT + EOT
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


def parameter_grad_norm(parameter) -> float:
    if parameter is None or parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().float().norm())


def tf32_report() -> dict:
    backends = torch.backends
    report = {'matmul_allow_tf32': bool(backends.cuda.matmul.allow_tf32),
              'cudnn_allow_tf32': bool(backends.cudnn.allow_tf32)}
    if hasattr(backends.cuda, 'matmul'):
        fp32_precision = getattr(backends.cuda.matmul, 'fp32_precision', None)
        if fp32_precision is not None:
            report['matmul_fp32_precision'] = str(fp32_precision)
    return report


# --------------------------------------------------------------------------- training module
class CgClipTrainModule(torch.nn.Module):
    """The CG-CLIP objective: one CLIP model (shared trunk and projections) plus one caption gate.

    ``clip`` is registered exactly once and the gate is registered outside it, so ``visual.proj`` and
    the last visual block stay a single set of Parameters owned by a single module.
    """

    def __init__(self, clip: torch.nn.Module, gate: CaptionGate, rank: int = 0,
                 image_chunk: int = IMAGE_CHUNK_DEFAULT, text_chunk: int = TEXT_CHUNK_DEFAULT,
                 cond_checkpoint: bool = True):
        super().__init__()
        self.clip = clip
        self.gate = gate
        self.rank = int(rank)
        self.image_chunk = int(image_chunk)
        self.text_chunk = int(text_chunk)
        self.cond_checkpoint = bool(cond_checkpoint)
        self.objective = OBJECTIVE

    # -- encoding -------------------------------------------------------------------------
    def encode(self, images, text, amp_dtype, amp_enabled: bool):
        """One visual pass (blocks 0..10) and one text pass under autocast, then the fp32 core."""
        device_type = 'cuda' if images.is_cuda else 'cpu'
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
            tokens = self.clip.encode_visual_prefinal(images)
            text_hidden = self.clip.encode_text_final_hidden(text)
        with torch.autocast(device_type=device_type, enabled=False):
            x11 = tokens['x11_raw'].float()
            hidden = text_hidden.float()
            eot, effective_length = self.clip.eot_indices(text)
            t_raw = hidden[torch.arange(hidden.shape[0], device=hidden.device), eot] \
                @ self.clip.text_projection.float()
            t_unit = torch.nn.functional.normalize(t_raw, dim=-1, eps=NORM_EPS)
        return {'x11': x11, 'cls11': tokens['cls11_raw'].float(), 't_raw': t_raw,
                't_unit': t_unit, 'effective_length': effective_length}

    # -- objective ------------------------------------------------------------------------
    def forward(self, images, text, amp_dtype=torch.bfloat16, amp_enabled: bool = True,
                want_capture: bool = False):
        encoded = self.encode(images, text, amp_dtype, amp_enabled)
        x11 = encoded['x11']
        visual = self.clip.visual
        last = visual.transformer.resblocks[-1]
        ln_post = visual.ln_post
        W = visual.proj
        local_size = int(x11.shape[0])
        assert_equal_local_batch(local_size)

        native = native_cls_row(last, x11, ln_post, W, want_attention=True)
        vG = torch.nn.functional.normalize(native['projected'], dim=-1, eps=NORM_EPS)
        t_unit_all = gather_rows(encoded['t_unit'])                       # autograd-aware, kept live
        qT_all = self.gate.project_query(t_unit_all)                      # A(stop_grad(t))
        kG = self.gate.project_key(native['u'][:, 1:, :])                 # B(stop_grad(U patches))
        qg_local = FIXED_SCALE * (vG @ t_unit_all.t())

        positive_columns = positive_global_columns(local_size, self.rank)
        capture = {} if want_capture else None
        qa_local, positive_mask = conditional_cls_scores(
            x11, native, last, self.gate, t_unit_all, qT_all, kG, W, ln_post,
            image_chunk=self.image_chunk, text_chunk=self.text_chunk,
            use_checkpoint=self.cond_checkpoint, positive_columns=positive_columns,
            capture=capture)
        if capture is not None:
            i0, i1, j0, j1 = capture['block']
            capture['native_attention'] = native['attention'][i0:i1].detach()
            capture['native_cls'] = native['cls'][i0:i1].detach()
            capture['native_projected'] = native['projected'][i0:i1].detach()
            capture['native_out'] = native['out'][i0:i1].detach()
            capture['native_c'] = native['c'][i0:i1].detach()

        qg = gather_rows(qg_local)
        qa = gather_rows(qa_local)
        targets = global_targets(local_size, self.rank, qg.device)
        columns = slice(self.rank * local_size, (self.rank + 1) * local_size)

        # I2T: the local images are the anchors, every candidate text is scored with its own gate.
        # T2I: this rank's texts are the anchors and every candidate image is scored against them.
        # Both are single cross entropies with LOCAL anchors and a GLOBAL candidate set; the sum of
        # the two directions is the per-rank objective and DDP averages the parameter gradients.
        lg_i2t = F.cross_entropy(qg_local, targets)
        lg_t2i = F.cross_entropy(qg[:, columns].t(), targets)
        la_i2t = F.cross_entropy(qa_local, targets)
        la_t2i = F.cross_entropy(qa[:, columns].t(), targets)
        loss_global = lg_i2t + lg_t2i
        loss_attention = la_i2t + la_t2i
        loss_sparse = sparse_positive_term(positive_mask)
        loss_total = total_loss(loss_global, loss_attention, loss_sparse)

        with torch.no_grad():
            stats = {
                'path_global_i2t': path_statistics(qg_local, targets),
                'path_global_t2i': path_statistics(qg[:, columns].t(), targets),
                'path_attention_i2t': path_statistics(qa_local, targets),
                'path_attention_t2i': path_statistics(qa[:, columns].t(), targets),
                'global_batch': int(qg.shape[0]), 'local_batch': local_size,
                'all_gates_open': bool((positive_mask.detach() >= 0.5).all()),
                'any_gate_closed': bool((positive_mask.detach() < 0.5).any()),
                'positive_gate_mean': float(positive_mask.detach().mean()),
            }
        return {'loss_global_i2t': lg_i2t, 'loss_global_t2i': lg_t2i, 'loss_global': loss_global,
                'loss_attention_i2t': la_i2t, 'loss_attention_t2i': la_t2i,
                'loss_attention': loss_attention, 'loss_sparse': loss_sparse,
                'loss_total': loss_total,
                'weighted_global': LAMBDA_GLOBAL * loss_global,
                'weighted_attention': LAMBDA_ATTENTION * loss_attention,
                'weighted_sparse': LAMBDA_SPARSE * loss_sparse,
                'qg_local': qg_local, 'qa_local': qa_local, 'qg': qg, 'qa': qa,
                'targets': targets, 'positive_columns': positive_columns,
                'positive_mask': positive_mask, 'stats': stats, 'encoded': encoded,
                'capture': capture, 'native': native}


# --------------------------------------------------------------------------- one real step
def cgclip_train_step(ddp_model, batch, optimizers, device, amp_dtype, amp_enabled=True,
                      capture_grads=False, grad_health=0.0, completed_steps=0, text_ids=None,
                      want_capture=False):
    """DDP forward -> backward -> gradient-health collective -> both optimizer steps.

    ``text_ids`` is an optional test hook: when given, the batch is not re-tokenised, so a two-rank
    run and a single-process reference can be fed byte-identical tokens.
    """
    images = batch['image_a'].to(device, non_blocking=True)
    text = (text_ids if text_ids is not None
            else longclip.tokenize(batch['caption_said'], truncate=True)).to(device)
    out = ddp_model(images, text, amp_dtype=amp_dtype, amp_enabled=amp_enabled,
                    want_capture=want_capture)
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
              # A, B and b are reported separately: with A = 0 the key projection B receives no
              # gradient on the first step, which is expected and must be visible
              'gate_query_grad_norm_A': parameter_grad_norm(module.gate.query.weight),
              'gate_key_grad_norm_B': parameter_grad_norm(module.gate.key.weight),
              'gate_bias_grad_b': parameter_grad_norm(module.gate.bias),
              'visual_proj_grad_norm': parameter_grad_norm(module.clip.visual.proj),
              'text_projection_grad_norm': parameter_grad_norm(module.clip.text_projection),
              'last_block_grad_norm': math.sqrt(sum(
                  float(p.grad.detach().float().pow(2).sum())
                  for p in module.clip.visual.transformer.resblocks[-1].parameters()
                  if p.grad is not None))}
    if grad_health > 0:
        health['grad_health_threshold'] = grad_health
        health['named_grad_norms'] = {
            name: parameter_grad_norm(parameter)
            for name, parameter in module.named_parameters()
            if parameter.requires_grad and (name.startswith('gate.')
                                             or name.startswith('clip.visual.proj')
                                             or name.startswith('clip.text_projection')
                                             or name.startswith(
                                                 'clip.visual.transformer.resblocks.11.'))}
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


# --------------------------------------------------------------------------- heavy log block
def heavy_record(out, gate, clip_handle, device) -> dict:
    """Patch-gate, attention-read and CLS-residual diagnostics from the forward that just ran."""
    record = {}
    mask = out['positive_mask'].detach()
    record.update(dict(('positive_pairs_' + key, value)
                       for key, value in gate_statistics(mask, mask).items()))
    record['positive_pairs_gate_scope'] = ('the TRUE positive (image, caption) pairs of rank0 local '
                                           'batch only; CLS slot excluded')
    capture = out.get('capture') or {}
    if 'mask' in capture:
        tile_mask = capture['mask']
        tile_probability = capture['probability']
        tile = gate_statistics(tile_mask, tile_probability)
        record.update(dict(('tile_' + key, value) for key, value in tile.items()))
        record['tile_scope'] = ('one (text_chunk x image_chunk) tile of rank0 local batch: %d texts x '
                                '%d images, all %d patch gates each' % (tile_mask.shape[0],
                                                                        tile_mask.shape[1],
                                                                        tile_mask.shape[2]))
        record['cls_slot_gate_value'] = 1.0
        record['cls_slot_gate_note'] = ('the CLS slot keeps gate 1 by construction; it is excluded '
                                        'from the sparse term and is never drawn in the 14x14 grid')
        record['gate_head_sharing'] = 'one 196-d patch gate shared by all 12 vision heads'
        record['gate_grid_samples'] = [
            dict(attention_map_grid(tile_mask[0, index]), sample=index)
            for index in range(min(GRID_SAMPLES, tile_mask.shape[1]))]
        record.update(attention_read_diagnostics(
            {'attention': capture['native_attention'], 'cls': capture['native_cls'],
             'projected': capture['native_projected'], 'out': capture['native_out'],
             'c': capture['native_c']},
            {'attention_conditional': capture['attention_conditional'], 'cls': capture['cls'],
             'projected': capture['projected']}))
        native_attention = capture['native_attention']                    # [bi,H,1,197]
        conditional_attention = capture['attention_conditional']          # [bj,bi,H,197]
        record['native_attention_cls_self_mass'] = [
            float(v) for v in native_attention[..., 0].mean(dim=0).reshape(-1)]
        record['conditional_attention_cls_self_mass'] = [
            float(v) for v in conditional_attention[..., 0].mean(dim=(0, 1)).reshape(-1)]
        record['attention_head_axis_note'] = ('per-head CLS-self mass, listed over the 12 vision '
                                              'heads in head order (head-averaged values are the '
                                              '*_mean fields above)')
        record['gate_pair_variation_sample'] = int(tile.get('gate_pair_sample', 0) or 0)
    record['native_attention_shape'] = list(out['native']['attention'].shape)
    record['hidden_width'] = int(out['native']['cls'].shape[-1])
    record['cls_self_mass_unit'] = 'softmax probability mass on the CLS token itself'
    record['device'] = str(device)
    record['gate_bias_value'] = float(gate.bias.detach())
    record['gate_query_weight_norm'] = float(gate.query.weight.detach().norm())
    record['gate_key_weight_norm'] = float(gate.key.weight.detach().norm())
    record['clip_state_digest_prefix'] = state_digest(clip_handle.state_dict())[:16]
    return record


# --------------------------------------------------------------------------- checkpoint writer
def write_checkpoint(path, clip, gate, optimizers, steps, epoch, step_in_epoch, config, digests,
                     batch_size, chunking, lr_horizon_steps, gate_report, spec, rng=None,
                     health=None, rank=0, world=1, precision=PRECISION_NOTE, clip_digest=None):
    """The production checkpoint writer, usable from tests and from the training loop.

    The gate *tensors* live under ``gate_state`` and the gate *description* under ``gate_config``:
    the two keys can never collide, and a checkpoint whose gate weights went missing is detectable
    both by the missing key and by the digest mismatch. The file is written atomically.
    """
    gate_digest = state_digest(gate.state_dict())
    if clip_digest is None:
        clip_digest = state_digest(clip.state_dict())
    payload = {
        'clip': clip.state_dict(),
        'gate_state': gate.state_dict(),
        'optimizer_clip': optimizers['clip'].state_dict(),
        'optimizer_gate': optimizers['gate'].state_dict(),
        'epoch': int(epoch), 'step_in_epoch': int(step_in_epoch),
        'rng': dict(rng or {}),
    }
    payload.update(checkpoint_metadata(
        steps=steps, rank=rank, world=world, config=config, digests=digests, batch_size=batch_size,
        chunking=chunking, precision=precision, gate_digest=gate_digest, clip_digest=clip_digest,
        lr_horizon_steps=lr_horizon_steps, gate_report=gate_report, spec=spec))
    if not isinstance(payload.get('gate_state'), dict) or not payload['gate_state']:
        raise RuntimeError('checkpoint payload is missing the gate tensors (gate_state)')
    if not isinstance(payload.get('gate_config'), dict) or not payload['gate_config']:
        raise RuntimeError('checkpoint payload is missing the gate description (gate_config)')
    if payload.get('gate_state_digest') != gate_digest:
        raise RuntimeError('gate_state_digest header does not describe the written gate state')
    if payload.get('clip_state_digest') != clip_digest:
        raise RuntimeError('clip_state_digest header does not describe the written clip state')
    payload['grad_health'] = health
    temporary = path + '.tmp'
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return payload


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description='CG-CLIP v0.1 (%s)' % OBJECTIVE)
    parser.add_argument('--base_model', default='B16')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-6, help='CLIP backbone + native projections')
    parser.add_argument('--gate_lr', type=float, default=1e-3, help='the new 64-d caption gate')
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
    parser.add_argument('--cond_checkpoint', type=int, default=1,
                        help='non-reentrant activation checkpointing inside the conditional CLS '
                             'tiles (memory only; the arithmetic is unchanged)')
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()
    if args.base_model == 'B16':
        args.base_model = 'ViT-B/16'

    seed_everything(args.seed)
    rank, local_rank, world = setup_distributed()
    device = torch.device('cuda', local_rank)
    save_completed = sorted({int(s) for s in args.save_completed_steps.split(',') if s.strip()})

    # fp32 core mandate: no TF32 reassociation inside the final CLS core or in the scores
    torch.backends.cuda.matmul.allow_tf32 = False
    tf32 = tf32_report()

    model, _preprocess = longclip.load_from_clip(args.base_model, device='cpu', download_root=None,
                                                 args=args)
    model.train()
    # compatibility key only: CG-CLIP scores with the fixed 100x scale, logit_scale is never used
    model.logit_scale = torch.nn.Parameter(torch.ones([]) * math.log(100.0))
    model = model.to(device)
    init_file_sha = None
    if args.init_state:
        if not os.path.isfile(args.init_state):
            raise SystemExit('shared init missing: %s' % args.init_state)
        init_file_sha = file_sha256(args.init_state)
        load_init_state(model, args.init_state, rank)
    initial_state_digest = state_digest(model.state_dict())
    spec = visual_spec(model)

    gate = CaptionGate(device=device).to(device)
    last = model.visual.transformer.resblocks[-1]
    with torch.no_grad():
        probe_x11 = torch.zeros(2, 197, model.visual.conv1.out_channels, device=device)
        probe_u = last.ln_1(probe_x11).float()[:, 1:, :]
        probe_t = torch.nn.functional.normalize(
            torch.zeros(2, model.text_projection.shape[0], device=device), dim=-1, eps=NORM_EPS)
        init_report = gate_init_report(gate, probe_t, probe_u)
    if rank == 0:
        print('GATE_INIT ' + json.dumps(init_report, sort_keys=True), flush=True)
        print('TF32_AUDIT ' + json.dumps(tf32, sort_keys=True), flush=True)

    train_module = CgClipTrainModule(model, gate, rank=rank, image_chunk=args.image_chunk,
                                     text_chunk=args.text_chunk,
                                     cond_checkpoint=bool(args.cond_checkpoint)).to(device)
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
        'loss_weights': dict(LOSS_WEIGHTS),
        'loss_combination': ('5*(LG_i2t+LG_t2i) + 5*(LA_i2t+LA_t2i) + LS, LS = mean of the 196 '
                             'patch gates on the true positive pairs'),
        'two_paths': {
            'native': 'Norm(ln_post(CLS after the native last block) @ visual.proj)',
            'attention': 'Norm(ln_post(CLS after the same last block with the native CLS attention '
                         'weights renormalised by the caption gate) @ visual.proj)'},
        'attention_route': {
            'cls_query': 'native last-block CLS query, never replaced by text',
            'keys_values': 'native last-block keys/values, shared by both paths',
            'gate_sharing': 'one 196-d patch gate shared by the 12 vision heads',
            'renormalisation': 'a_cond = (a * m_full) / sum_p (a * m_full); no detached '
                               'denominator, no clamp',
            'cls_slot': 'fixed gate 1, never trainable, excluded from LS',
            'conditional_path_use': 'training only; not used at inference'},
        'x11_source': 'tokens ENTERING the last visual block, from ONE visual forward over blocks 0..10',
        'text_source': 'ln_final sequence, EOT = argmax(token ids), native text_projection in fp32',
        'gate': gate_config(train_module.gate),
        'gate_init': init_report,
        'candidate_rule': 'fixed image: every candidate text uses its own patch gate; fixed text: '
                          'every candidate image is scored with the gate of that (text, image) pair; '
                          'positives and negatives follow the identical rule',
        'grader': 'none: LG never touches the gate, LA touches the shared CLIP parameters and the '
                  'gate, LS (mean of the positive pairs patch gates) touches only the gate',
        'precision': PRECISION_NOTE, 'tf32': tf32,
        'chunking': {'image_chunk': args.image_chunk, 'text_chunk': args.text_chunk,
                     'cond_checkpoint': bool(args.cond_checkpoint),
                     'note': 'the conditional CLS runs tile by tile in fp32; no [B,B,12,197] or '
                             '[B,B,768] activation is ever resident'},
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
        'visual_spec': spec,
        'statistics_scope': STATS_SCOPE,
    })
    if rank == 0:
        print('CONFIG ' + json.dumps(config, sort_keys=True), flush=True)
        print('DATASET_SIZE %d STEPS_PER_EPOCH %d LR_HORIZON %d CLIP_TENSORS %d GATE_TENSORS %d '
              'GLOBAL_BATCH %d'
              % (len(dataset), steps_per_epoch, total_steps, partition['clip_count'],
                 partition['gate_count'], args.batch_size * world), flush=True)
        with open(os.path.join(args.output_dir, 'config.json'), 'w') as handle:
            json.dump(config, handle, indent=2, sort_keys=True)

    if args.resume:
        payload = torch.load(args.resume, map_location='cpu', weights_only=False)
        check_resume_compatible(payload, identity)
        clip_handle.load_state_dict(payload['clip'])
        train_module.gate.load_state_dict(payload['gate_state'])
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

    def local_stream_digests():
        return {'caption_stream_sha256': caption_digest.hexdigest(),
                'sample_stream_sha256': sample_digest.hexdigest(),
                'image_id_stream_sha256': image_id_digest.hexdigest(),
                'prefix_k_stream_sha256': prefix_digest.hexdigest()}

    def save_checkpoint(steps, epoch, step_in_epoch, health=None):
        """Write one self-describing checkpoint (rank 0 only; the state is identical on all ranks)."""
        digests = dict(local_stream_digests())
        digests['init_file_sha256'] = init_file_sha
        digests['initial_state_digest'] = initial_state_digest
        path = os.path.join(args.output_dir, '%s_step%06d.pt' % (ARM, steps))
        write_checkpoint(
            path=path, clip=clip_handle, gate=train_module.gate, optimizers=optimizers,
            steps=steps, epoch=epoch, step_in_epoch=step_in_epoch, config=config, digests=digests,
            batch_size=args.batch_size,
            chunking={'image_chunk': args.image_chunk, 'text_chunk': args.text_chunk,
                      'cond_checkpoint': bool(args.cond_checkpoint)},
            lr_horizon_steps=total_steps, gate_report=init_report, spec=spec,
            rng={'torch': torch.get_rng_state(),
                 'cuda': torch.cuda.get_rng_state(device) if torch.cuda.is_available() else None},
            health=health, rank=rank, world=world)
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
    presentations = 0
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
            heavy = (args.heavy_log_every > 0 and (completed + 1) % args.heavy_log_every == 0)
            out = cgclip_train_step(ddp_model, batch, optimizers, device, amp_dtype,
                                    amp_enabled=use_amp, completed_steps=completed,
                                    grad_health=1.0 if (completed in (0, 1)
                                                        or completed == args.grad_health_steps)
                                    else 0.0,
                                    want_capture=heavy)
            local_over, longest = raw_token_lengths(batch['caption_said'])
            over_capacity += local_over
            caption_count += len(batch['caption_said'])
            max_raw_tokens = max(max_raw_tokens, longest)
            effective = out['encoded']['effective_length']
            max_effective = max(max_effective, int(effective.max()))
            effective_lengths.append(float(effective.float().mean()))
            completed += 1
            compute_times.append(time.time() - t0)
            presentations += args.batch_size * world

            caption_digest.update(('\n'.join(batch['caption_said'])).encode('utf-8'))
            sample_digest.update(batch['sample_id'].numpy().tobytes())
            image_id_digest.update(batch['image_id'].numpy().tobytes())
            prefix_digest.update(batch['prefix_k'].numpy().tobytes())

            log_now = (completed % args.log_every == 0 or completed == 1
                       or completed in save_completed)
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
                    'loss_global_sum': float(out['loss_global']),
                    'loss_attention_i2t': float(out['loss_attention_i2t']),
                    'loss_attention_t2i': float(out['loss_attention_t2i']),
                    'loss_attention_sum': float(out['loss_attention']),
                    'loss_sparse': float(out['loss_sparse']),
                    'weights_5_5_1': [LAMBDA_GLOBAL, LAMBDA_ATTENTION, LAMBDA_SPARSE],
                    'weighted_loss_global': float(out['weighted_global']),
                    'weighted_loss_attention': float(out['weighted_attention']),
                    'weighted_loss_sparse': float(out['weighted_sparse']),
                    'loss_total': float(out['loss_total']),
                    'optimizer_steps_before_update': int(completed) - 1,
                    'sec_per_step': compute_times[-1],
                    'samples_per_sec': float(stats['local_batch']) * world
                    / max(compute_times[-1], 1e-9),
                    'pair_presentations_per_sec': float(stats['global_batch']) * world
                    / max(compute_times[-1], 1e-9),
                    'peak_memory_gb': torch.cuda.max_memory_allocated(device) / (1024 ** 3),
                    'peak_memory_gi_b': torch.cuda.max_memory_allocated(device) / (1024 ** 3),
                    'peak_memory_note': 'GiB (1024^3 bytes)',
                    'rank_param_digest': state_digest(clip_handle.state_dict())[:16],
                    'gate_param_digest': state_digest(train_module.gate.state_dict())[:16],
                    'batch_caption_sha256': hashlib.sha256(
                        ('\n'.join(batch['caption_said'])).encode('utf-8')).hexdigest()[:16],
                    'batch_image_id_sha256': tensor_digest(batch['image_id']),
                    'rank_local_stream_digests': local_stream_digests(),
                    'synchronized_pair_presentations': presentations,
                    'statistics_scope': 'rank0_local_batch',
                    'all_gates_open': stats['all_gates_open'],
                    'any_gate_closed': stats['any_gate_closed'],
                    'positive_gate_mean': stats['positive_gate_mean'],
                    'over_capacity_captions_cumulative': over_capacity,
                    'captions_seen': caption_count,
                    'max_raw_token_length': max_raw_tokens,
                    'max_effective_length': max_effective,
                    'conditional_chunking': {'image_chunk': args.image_chunk,
                                             'text_chunk': args.text_chunk,
                                             'cond_checkpoint': bool(args.cond_checkpoint)},
                    'tf32': tf32,
                }
                for path in ('path_global_i2t', 'path_global_t2i', 'path_attention_i2t',
                             'path_attention_t2i'):
                    for key, value in stats[path].items():
                        record['%s_%s' % (path, key)] = value
                health = out['_grad_health']
                record.update(health)
                if heavy:
                    record.update(heavy_record(out, train_module.gate, clip_handle, device))
                    record['effective_length_mean'] = float(np.mean(effective_lengths[-20:]))
                with open(log_path, 'a') as handle:
                    handle.write(json.dumps(record, sort_keys=True) + '\n')
                print('LOG ' + json.dumps({k: v for k, v in record.items()
                                           if k != 'gate_grid_samples'}, sort_keys=True),
                      flush=True)

            if rank == 0 and completed in in_loop_saves:
                save_checkpoint(completed, epoch, i, health=out['_grad_health'])
        if stopped:
            break

    # per-rank stream digests, all ranks, for the provenance record
    all_rank_digests = None
    if dist.is_initialized():
        mine = local_stream_digests()
        packed = [None] * world
        dist.all_gather_object(packed, mine)
        all_rank_digests = packed
    if rank == 0:
        summary = {
            'arm': ARM, 'objective': OBJECTIVE, 'completed_steps': completed,
            'max_steps': args.max_steps, 'epochs': args.epochs,
            'wall_sec': time.time() - t_start,
            'mean_sec_per_step': sum(compute_times) / max(len(compute_times), 1),
            'median_sec_per_step': float(np.median(compute_times)) if compute_times else 0.0,
            'steps_per_epoch': steps_per_epoch, 'lr_horizon_steps': total_steps,
            'world_size': world, 'batch_size_per_gpu': args.batch_size,
            'global_batch': args.batch_size * world,
            'global_pairs_per_step': args.batch_size * world,
            'synchronized_pair_presentations': presentations,
            'expected_pair_presentations_at_500': 500 * args.batch_size * world,
            'single_rank_pair_presentations': presentations // max(world, 1),
            'peak_memory_gb': torch.cuda.max_memory_allocated(device) / (1024 ** 3),
            'peak_memory_note': 'GiB (1024^3 bytes)',
            'init_file_sha256': init_file_sha, 'initial_state_digest': initial_state_digest,
            'final_state_digest': state_digest(clip_handle.state_dict()),
            'final_gate_digest': state_digest(train_module.gate.state_dict()),
            'stream_digests': local_stream_digests(),
            'stream_digests_all_ranks': all_rank_digests,
            'captions_seen': caption_count,
            'over_capacity_captions': over_capacity,
            'max_effective_length': max_effective,
            'precision': PRECISION_NOTE, 'tf32': tf32,
            'statistics_scope': STATS_SCOPE,
        }
        with open(os.path.join(args.output_dir, 'run_summary.json'), 'w') as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print('RUN_SUMMARY ' + json.dumps(summary, sort_keys=True), flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
