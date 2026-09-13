"""SAID-S0-Suffix v0.1 trainer.

    L = L_S0 + lambda_suffix * L_suffix

``L_S0 = 10 * (L_SIDM + L_DISM) + 2 * L_sparse_S`` is the frozen S0 objective, called through the
existing S0 helper on the FULL global candidate pool; a sample whose suffix is empty is never removed
from it. The suffix task adds the prefix-conditioned readout of the new branch and is scored on the
global valid subset ``J`` (``V = |J|``) with the ``world_size / V`` DDP scaling of the spec.

One configuration only -- 4 x 256 pairs, exactly 500 optimizer updates, three AdamW optimizers (CLIP
backbone + projections at 1e-6 / weight decay 1e-2 / warmup 200, the EXISTING S0 mask network at
1e-3 / weight decay 0 / warmup 0, the NEW F at 1e-4 / weight decay 0 / warmup 0 in the mask arm
only), betas (0.9, 0.999), eps 1e-8, cosine horizon ``3 * len(loader)`` (the real loader, never a
hard-coded 3651) and ``--max_steps`` only truncating the run.

Per optimizer step
-----------------
1. every rank tokenises its own prefix captions and its own suffix strings (TWO independent
   tokenisations) and encodes the image ONCE;
2. the S0 path runs verbatim on the full global batch (``L_SIDM + L_DISM + L_sparse_S``);
3. ``suffix_valid`` is all-gathered, J and V are formed identically on every rank, and both the
   suffix images and the suffix texts are restricted to J;
4. the suffix scores are produced tile by tile (``[Bi, Bj]`` score tiles only, never
   ``[B_global, B_global, 1024]``, and never another rank's image rows);
5. both directions are SUMMED over the local valid anchors and scaled by ``world_size / V``;
6. a collective gradient-health check runs on every rank; then exactly one update of each group.

The first 20 steps belong to this same 500-step run: there is no separate smoke run, and the 501st
optimizer update is never executed.
"""
import argparse
import hashlib
import importlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (os.path.dirname(os.path.abspath(__file__)), REPO, os.path.join(REPO, 'model')):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from model import longclip                                                      # noqa: E402
from model.said_prefix_suffix import (ARM_MASK, ARM_NATIVE, ARMS, BATCH_SIZE_PER_GPU,  # noqa: E402
                                      CHECKPOINT_FILENAME, CHECKPOINT_KEYS, CLIP_LR,
                                      CLIP_WEIGHT_DECAY, DEFAULT_SAVE_COMPLETED_STEPS,
                                      EXPECTED_PRESENTATIONS_AT_STEP_500,
                                      IMAGE_CHUNK_DEFAULT, LAMBDA_ALIGN, LAMBDA_SPARSE,
                                      LAMBDA_SUFFIX, MAX_STEPS, OBJECTIVE, PHASE, S0_MASK_LR,
                                      S0_MASK_WARMUP, STREAM_SCOPE, SUFFIX_LR, SUFFIX_MASK_SEED,
                                      SUFFIX_WARMUP, TEXT_CHUNK_DEFAULT, TOKENIZER_CONTEXT,
                                      SaidPrefixSuffixTrainModule, SuffixMask,
                                      assert_equal_local_batch, build_optimizers,
                                      checkpoint_metadata, split_batch_prefix_suffix,
                                      state_digest, suffix_mask_config, suffix_mask_init_report)
from said_cvssl_data import Share4VCvsslDataset, cvssl_collate                 # noqa: E402
from scheduler import cosine_lr                                                # noqa: E402

PHASE_NAME = PHASE
#: the two frozen arms of this objective, re-exported so a runner or a test never restates them
ARMS = tuple(ARMS)
PRECISION_NOTE = ('fp32 master weights; the existing bf16 autocast policy for the CLIP backbone; the '
                  'new F, the g normalisation, the suffix scoring and both CE cores explicitly fp32, '
                  'identically in both arms')
STATS_SCOPE = ('rank-local values unless the field name says otherwise; fields named global_* are '
               'all-gathered over ranks for that step')


# --------------------------------------------------------------------------- small helpers
def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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


def jsonable(value):
    """JSON-safe scalars for the log line (numpy/torch scalars included)."""
    if isinstance(value, (str, bool)) or value is None:
        return value
    if torch.is_tensor(value):
        return jsonable(value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return dict((str(key), jsonable(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (int, float)):
        return value
    return str(value)


def tensor_digest(tensor) -> str:
    array = tensor.detach().to(torch.float32).cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(repo=None) -> str:
    import subprocess
    try:
        return subprocess.run(['git', '-C', repo or REPO, 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, timeout=20).stdout.strip() or 'unknown'
    except Exception:                                            # pragma: no cover
        return 'unknown'


def load_init_state(model, path: str, rank: int):
    """Load the complete frozen initial state, or fail loudly (the S0/PG lineage rule).

    Both arms start from the SAME shared initialisation. A ``clip.``-prefixed SALU layout and a bare
    CLIP layout are accepted; a bare CLIP checkpoint legitimately has no ``mask_net``, in which case
    the deterministically seeded freshly built ``mask_net`` is kept.
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
              % (path, len(mapped), len([key for key in missing if key.startswith('mask_net')]),
                 ignored[:5]), flush=True)


def grads_finite(parameters) -> bool:
    for parameter in parameters:
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            return False
    return True


def group_grad_norm(parameters, device) -> float:
    total = torch.zeros((), device=device, dtype=torch.float32)
    for parameter in parameters:
        if parameter.grad is not None:
            total = total + parameter.grad.detach().float().pow(2).sum()
    return float(total.sqrt())


def parameter_grad_norm(parameter) -> float:
    if parameter is None or parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().float().norm())


def run_schedulers(schedulers: dict, step: int, optimizers: dict = None) -> dict:
    """Advance every group's scheduler and return the learning rate each group will use.

    ``cosine_lr`` returns a CALLABLE that assigns the step's rate to the optimizer it owns, so the
    values of ``schedulers`` are plain callables; groups with ``warmup_length = 0`` keep their base
    rate for the whole run by construction. The rates are read back from the optimizers afterwards, so
    the logged value is the one the optimizer really uses.
    """
    for name, scheduler in schedulers.items():
        if scheduler is None:
            continue
        if callable(scheduler):
            scheduler(step)
        else:                                                     # pragma: no cover - defensive
            raise TypeError('scheduler for group %r is not callable: %r' % (name, scheduler))
    if optimizers is None:
        return {}
    return {name: lr_of(optimizers.get(name)) for name in schedulers}


def lr_of(optimizer) -> float:
    return float(optimizer.param_groups[0]['lr']) if optimizer is not None else 0.0


def peak_memory_gib(device) -> float:
    """Peak allocated GPU memory in GiB (1024^3 bytes); 0.0 off CUDA, so a CPU test can still log."""
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024 ** 3))


def tokenized(batch, device):
    """The TWO independent tokenisations of the step: the prefix and the suffix.

    The split runs on the FULL caption (``caption_full``), never on ``caption_said``, because
    ``caption_said`` is ALREADY the prefix the pipeline chopped at ``prefix_k`` -- splitting it again
    yields an empty suffix for every row (measured: ``global_valid_V = 0`` on the first smoke test).
    ``prefix_k`` is the K the S0 stream actually drew and is never re-drawn here.

    ``caption_said`` is then used as a hard, character-for-character INVARIANT: the prefix this file
    produces from the full caption must equal it exactly, so the S0 prefix stream provably cannot
    drift. An invalid suffix keeps the EMPTY STRING as its placeholder and is marked ``valid=False``
    downstream, so a placeholder can never enter the suffix loss nor the suffix candidate pool.
    """
    prefix_texts = [str(text) for text in batch['caption_said']]
    full_texts = [str(text) for text in batch.get('caption_full', batch['caption_said'])]
    prefix_k = [int(value) for value in batch['prefix_k'].tolist()]
    split = split_batch_prefix_suffix(full_texts, prefix_k)
    suffix_texts = list(split['suffix'])
    prefix_from_full = [str(text) for text in split['prefix']]
    mismatched = [index for index, (left, right) in enumerate(zip(prefix_from_full, prefix_texts))
                  if left != right]
    if mismatched:
        raise RuntimeError('the prefix rebuilt from the full caption differs from the pipeline prefix '
                           'in %d of %d rows (first at %d): %r vs %r -- the S0 prefix stream would '
                           'have drifted, refusing to train'
                           % (len(mismatched), len(prefix_texts), mismatched[0],
                              prefix_from_full[mismatched[0]], prefix_texts[mismatched[0]]))
    prefix_ids = longclip.tokenize(prefix_texts, truncate=True).to(device)
    suffix_ids = longclip.tokenize(suffix_texts, truncate=True).to(device)
    return {'prefix': prefix_ids, 'suffix': suffix_ids}, prefix_texts, suffix_texts, split


# --------------------------------------------------------------------------- one real step
def suffix_train_step(ddp_model, batch, optimizers, device, amp_dtype, amp_enabled: bool = True,
                      capture_grads: bool = False, completed_steps: int = 0,
                      want_statistics: bool = False, image_chunk: int = None,
                      text_chunk: int = None, text_ids=None, suffix_texts=None):
    """DDP forward -> backward -> collective gradient health -> one update of all three groups.

    This is the *production* step function: the acceptance tests call it directly, so what is
    verified is what is run. ``lambda_suffix`` is never dropped here -- it is applied inside the
    objective, so a test that compares a one-step update cannot silently lose it.
    """
    image_a = batch['image_a'].to(device, non_blocking=True)
    suffix_texts = None
    if text_ids is None:
        text_ids, _prefix_texts, suffix_texts, _split = tokenized(batch, device)
    if suffix_texts is None:
        suffix_texts = [''] * int(image_a.shape[0])
    # a ragged batch must be refused BEFORE the first cross-rank gather
    module = getattr(ddp_model, 'module', ddp_model)
    assert_equal_local_batch(int(image_a.shape[0]), world_size=module.objective._world(),
                             device=device)

    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
        out = ddp_model(image_a, text_ids,
                        prefix_captions=list(batch.get('caption_full', batch['caption_said'])),
                        batch_prefix_texts=list(batch['caption_said']),
                        suffix_texts=suffix_texts, prefix_k=batch['prefix_k'],
                        image_chunk=image_chunk, text_chunk=text_chunk,
                        want_statistics=want_statistics)
    loss = out['loss_total_for_backward']
    loss.backward()

    inspected = getattr(ddp_model, 'module', ddp_model)
    partition = optimizers['partition']
    local_ok = (bool(torch.isfinite(loss.detach()).all())
                and grads_finite(inspected.clip.parameters())
                and grads_finite(inspected.clip.mask_net.parameters())
                and (inspected.suffix_mask is None
                     or grads_finite(inspected.suffix_mask.parameters())))
    # one non-finite loss or gradient on any rank must stop every rank: the flag is collective and
    # is executed on ALL ranks before any optimizer step
    flag = torch.tensor([0.0 if local_ok else 1.0], device=device)
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if float(flag.item()) > 0:
        raise SystemExit('REFUSING the optimizer step at completed step %d: a non-finite loss or '
                         'gradient was found on at least one of %d ranks (no nan_to_num, no LR or '
                         'threshold change, no skipped step)'
                         % (completed_steps, dist.get_world_size() if dist.is_initialized() else 1))

    health = {
        'grad_norm_clip_group': group_grad_norm(partition['backbone'], device),
        'grad_norm_mask_group': group_grad_norm(partition['mask_net'], device),
        'grad_norm_suffix_group': group_grad_norm(partition['suffix'], device),
        'grad_norm_mask_attn_pool_weight': parameter_grad_norm(
            inspected.clip.mask_net.attn_pool.attention.weight),
        'grad_norm_suffix_layer1_weight': parameter_grad_norm(
            None if inspected.suffix_mask is None else inspected.suffix_mask.layer1.weight),
        'grad_norm_suffix_layer2_weight': parameter_grad_norm(
            None if inspected.suffix_mask is None else inspected.suffix_mask.layer2.weight),
        'grad_norm_suffix_layer2_bias': parameter_grad_norm(
            None if inspected.suffix_mask is None else inspected.suffix_mask.layer2.bias),
        'loss_finite': bool(torch.isfinite(loss.detach()).all()),
        'grads_finite': bool(local_ok),
        'mask_grad_nonzero': bool(group_grad_norm(partition['mask_net'], device) > 0.0),
    }
    if capture_grads:
        out['_grads'] = {name: (None if parameter.grad is None else parameter.grad.detach().clone())
                         for name, parameter in inspected.named_parameters()}
    optimizers['clip'].step()
    optimizers['mask'].step()
    if optimizers['suffix'] is not None:
        optimizers['suffix'].step()
    optimizers['clip'].zero_grad(set_to_none=True)
    optimizers['mask'].zero_grad(set_to_none=True)
    if optimizers['suffix'] is not None:
        optimizers['suffix'].zero_grad(set_to_none=True)
    out['_grad_health'] = health
    out['_loss_value'] = float(loss.detach())
    return out


# --------------------------------------------------------------------------- checkpoint writer
def write_checkpoint(path, clip=None, clip_model=None, suffix_mask=None, optimizers=None, steps=0,
                     arm=None, config=None, digests=None, suffix_mask_state=None, objective=None,
                     scheduler_config=None, scheduler_state=None, data_cursor=None, rng_states=None,
                     loss_config=None, sampling_config=None, provenance=None, chunking=None,
                     rank=0, world=1, precision=PRECISION_NOTE, horizon_steps=None, batch_size=None,
                     suffix_mask_report=None, completed_steps=None, **ignored) -> dict:
    """The PRODUCTION checkpoint writer (module level, so a test calls exactly what the run calls).

    Exactly the frozen key list of spec section 10 is written, and the writer accepts a sparse call:
    every argument except ``path`` has a default, and the tensors are taken from the modules when the
    caller does not pass the state dicts explicitly. The suffix-mask TENSORS live under
    ``suffix_mask_state`` and the description under ``suffix_mask_config``: the two keys can never
    collide, and a checkpoint whose gate weights went missing is detectable both by the missing key
    and by the digest mismatch. ``suffix_mask_state`` and ``optimizer_suffix`` are ``None`` for the
    native arm. The file is written atomically through a temporary file plus ``os.replace``.
    """
    del ignored
    clip = clip if clip is not None else clip_model
    if clip is None:
        raise ValueError('write_checkpoint needs the CLIP model (clip= or clip_model=)')
    steps = int(steps if completed_steps is None else completed_steps)
    if arm is None:
        arm = ARM_MASK if suffix_mask is not None else ARM_NATIVE
    if arm not in (ARM_NATIVE, ARM_MASK):
        raise ValueError('arm must be one of %r, got %r' % (ARMS, arm))
    optimizers = optimizers or {}
    clip_state = clip.state_dict()
    clip_digest = state_digest(clip_state)
    if not suffix_mask_state and suffix_mask is not None and arm == ARM_MASK:
        # the tensors always come from the module when the caller did not pass them explicitly
        suffix_mask_state = suffix_mask.state_dict()
    if arm == ARM_NATIVE:
        # the native arm carries no suffix tensors at all: suffix_mask_state is explicitly None
        suffix_state = None
    else:
        suffix_state = suffix_mask_state
    suffix_digest = None if suffix_state is None else state_digest(suffix_state)
    optimizer_suffix = optimizers.get('suffix')
    payload = {
        'clip_state': clip_state,
        'suffix_mask_state': suffix_state,
        'suffix_mask_config': suffix_mask_config(suffix_mask),
        'optimizer_clip': (None if optimizers.get('clip') is None
                           else optimizers['clip'].state_dict()),
        'optimizer_mask': (None if optimizers.get('mask') is None
                           else optimizers['mask'].state_dict()),
        'optimizer_suffix': (None if optimizer_suffix is None else optimizer_suffix.state_dict()),
        'scheduler_config': scheduler_config,
        'scheduler_state': scheduler_state,
        'completed_steps': steps,
        'data_cursor': data_cursor,
        'rng_states': rng_states,
        'loss_config': loss_config,
        'sampling_config': sampling_config,
        'provenance': provenance,
    }
    payload.update(checkpoint_metadata(
        suffix_mask=suffix_mask, steps=steps, rank=rank, world=world, config=config,
        digests=digests, batch_size=int(batch_size if batch_size is not None else BATCH_SIZE_PER_GPU),
        chunking=chunking or {}, precision=precision, clip_digest=clip_digest,
        suffix_mask_digest=suffix_digest, horizon_steps=horizon_steps,
        scheduler_config=scheduler_config, scheduler_state=scheduler_state,
        data_cursor=data_cursor, rng_states=rng_states, loss_config=loss_config,
        sampling_config=sampling_config, provenance=provenance, arm=arm))
    if objective is not None:
        payload['objective'] = objective
    if suffix_mask_report is not None:
        payload['suffix_mask_init_report'] = suffix_mask_report
    if payload['objective'] != OBJECTIVE:
        raise RuntimeError('the checkpoint objective must be %r, got %r'
                           % (OBJECTIVE, payload['objective']))
    missing = [key for key in CHECKPOINT_KEYS if key not in payload]
    if missing:
        raise RuntimeError('the checkpoint payload is missing the frozen key(s) %r' % missing)
    if arm == ARM_NATIVE and payload['suffix_mask_state'] is not None:
        raise RuntimeError('the native arm must write suffix_mask_state = None')
    if arm == ARM_MASK:
        state = payload['suffix_mask_state']
        if not isinstance(state, dict) or not state:
            raise RuntimeError('the mask arm must write a non-empty suffix_mask_state')
        expected = {'layer1.weight': (512, 1024), 'layer1.bias': (512,),
                    'layer2.weight': (512, 512), 'layer2.bias': (512,)}
        present = {}
        for tensor_name, tensor in state.items():
            # strip a leading module prefix only: ``rsplit`` keeps 'layer1.weight' intact while
            # dropping a 'suffix_mask.' prefix, which is what the exporter also accepts
            name = str(tensor_name)
            if name.startswith('suffix_mask.'):
                name = name[len('suffix_mask.'):]
            present[name] = tuple(int(dim) for dim in tuple(tensor.shape))
        absent = sorted(set(expected) - set(present))
        wrong = sorted('%s=%s' % (name, list(present[name])) for name in present
                       if name in expected and expected[name] != present[name])
        if absent or wrong:
            raise RuntimeError('suffix_mask_state does not match the suffix mask module: absent %r, '
                               'wrong shapes %r (state holds %r)'
                               % (absent, wrong, sorted(present)))
    if not isinstance(payload['suffix_mask_config'], dict) or not payload['suffix_mask_config']:
        raise RuntimeError('the checkpoint payload is missing the suffix mask description')
    if payload.get('clip_state_digest') != clip_digest:
        raise RuntimeError('the clip_state_digest header does not describe the written clip state')
    if payload.get('suffix_mask_state_digest') != suffix_digest:
        raise RuntimeError('the suffix_mask_state_digest header does not describe the written state')
    temporary = path + '.tmp'
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return payload


#: the name the runner records for the production writer of this repository
write_production_checkpoint = write_checkpoint


def build_objective(clip=None, clip_model=None, suffix_model=None, model=None, module=None,
                    suffix_mask=None, arm=ARM_NATIVE, rank: int = 0, lambda_suffix: float = 0.0,
                    device=None, soft_mask: bool = False, **ignored):
    """Build the objective of ``model/said_prefix_suffix.py`` from either spelling of the CLIP model.

    ``lambda_suffix = 0.0`` (the default here) is the S0-equivalence switch of spec section 9 E: the
    returned objective then reproduces the ORIGINAL S0 loss and its gradients exactly.
    """
    del soft_mask, ignored
    clip = clip if clip is not None else (clip_model if clip_model is not None else model)
    if clip is None:
        raise ValueError('build_objective needs the CLIP model (clip= / clip_model= / model=)')
    suffix_model = suffix_model if suffix_model is not None else module
    if suffix_model is None:
        module = importlib.import_module('model.said_prefix_suffix')
        suffix_model = module
    builder = getattr(suffix_model, 'SaidPrefixSuffixObjective')
    objective = builder(clip, rank=int(rank), arm=arm, suffix_mask=suffix_mask,
                        lambda_suffix=float(lambda_suffix))
    if device is not None:
        objective = objective.to(device) if hasattr(objective, 'to') else objective
    return objective


# --------------------------------------------------------------------------- logging
def build_log_record(out, batch, optimizers, module, args, epoch, step_in_epoch, completed,
                     compute_time, presentations, synchronized_presentations, rank, world, device,
                     digests) -> dict:
    """One JSON object per logged step, with the field names the dashboard reads."""
    statistics = out.get('_statistics') or {}
    health = out.get('_grad_health') or {}
    validity = out['suffix_validity']
    local_size = int(batch['image_a'].shape[0])
    global_size = int(out['global_batch_size'])
    record = {
        'completed_steps': int(completed),
        'optimizer_steps_before_update': int(completed) - 1,
        'epoch': int(epoch),
        'step_in_epoch': int(step_in_epoch),
        'arm': args.arm,
        'objective': OBJECTIVE,
        'phase': PHASE_NAME,
        'rank': int(rank),
        'world_size': int(world),
        'batch_size_local': local_size,
        'global_batch_size': global_size,
        'global_pairs': global_size,
        # ---- S0, verbatim -------------------------------------------------------------------
        'loss_sidm': float(out['loss_sidm']),
        'loss_dism': float(out['loss_dism']),
        'loss_sparse_S': float(out['loss_sparsity']),
        'loss_s0_unweighted': float(out['loss_s0_unweighted']),
        'weighted_loss_sidm': LAMBDA_ALIGN * float(out['loss_sidm'].detach()),
        'weighted_loss_dism': LAMBDA_ALIGN * float(out['loss_dism'].detach()),
        'weighted_loss_sparse_S': LAMBDA_SPARSE * float(out['loss_sparsity'].detach()),
        'lambda_align': LAMBDA_ALIGN,
        'lambda_sparse': LAMBDA_SPARSE,
        'loss_s0': float(out['loss_s0']),
        'sidm_top1': float(out['sidm_top1']),
        'dism_top1': float(out['dism_top1']),
        's0_candidate_count': int(out['actual_global_candidate_count']),
        's0_candidate_pool': 'FULL global batch; empty-suffix samples are NOT removed',
        # ---- the suffix task -----------------------------------------------------------------
        'loss_suffix': float(out['loss_suffix'].detach()),
        'loss_suffix_I2T': float(out['loss_suffix_i2t_sum'].detach()),
        'loss_suffix_T2I': float(out['loss_suffix_t2i_sum'].detach()),
        'loss_suffix_local_sum': float(out['loss_suffix_local_sum'].detach()),
        'weighted_loss_suffix': float(out['loss_suffix_weighted'].detach()),
        'loss_suffix_times_lambda_suffix': float(out['loss_suffix_weighted'].detach()),
        'lambda_suffix': float(out['lambda_suffix']),
        'suffix_scale_world_over_V': float(out['suffix_scale_world_over_V']),
        'suffix_scaling_rule': 'local (I2T CE sum + T2I CE sum) * world_size / V, then standard DDP '
                               'averaging; per-rank valid means are never averaged',
        'loss_total': float(out['loss_total'].detach()),
        # ---- valid subset --------------------------------------------------------------------
        'valid_suffix_count_rank0': int(validity['valid_counts_per_rank'][0]),
        'valid_suffix_count_rank1': int(validity['valid_counts_per_rank'][1])
        if len(validity['valid_counts_per_rank']) > 1 else None,
        'valid_suffix_count_rank2': int(validity['valid_counts_per_rank'][2])
        if len(validity['valid_counts_per_rank']) > 2 else None,
        'valid_suffix_count_rank3': int(validity['valid_counts_per_rank'][3])
        if len(validity['valid_counts_per_rank']) > 3 else None,
        'valid_suffix_counts_per_rank': [int(value) for value in validity['valid_counts_per_rank']],
        'valid_suffix_count_local_n_r': int(out['local_valid_count']),
        'global_valid_V': int(out['global_valid_V']),
        'empty_suffix_fraction_local': 1.0 - float(out['suffix_valid_ratio_local']),
        'empty_suffix_fraction_global': float(out['empty_suffix_fraction_global']),
        'valid_subset_rule': 'J = all_gather(suffix_valid); both the suffix images and the suffix '
                             'texts are restricted to J; V < 2 gives an exactly zero suffix loss',
        # ---- token lengths and truncation ----------------------------------------------------
        'prefix_raw_length_max': int(statistics.get('prefix_raw_length_max', 0)),
        'suffix_raw_length_max': int(statistics.get('suffix_raw_length_max', 0)),
        'prefix_effective_length_mean': float(statistics.get('prefix_effective_length_mean', 0.0)),
        'suffix_effective_length_mean': float(statistics.get('suffix_effective_length_mean', 0.0)),
        'prefix_effective_length_max': int(statistics.get('prefix_effective_length_max', 0)),
        'suffix_effective_length_max': int(statistics.get('suffix_effective_length_max', 0)),
        'prefix_truncated_count': int(statistics.get('prefix_truncated_count', 0)),
        'suffix_truncated_count': int(statistics.get('suffix_truncated_count', 0)),
        'tokenizer_context': TOKENIZER_CONTEXT,
        'token_length_scope': statistics.get('token_length_scope', STATS_SCOPE),
        # ---- mask keep rates, scopes spelled out ---------------------------------------------
        'mS_keep_ratio_rank_local_all_candidates':
            float(statistics.get('mS_keep_ratio_rank_local_all_candidates', 0.0)),
        'mS_keep_ratio_positive_pair_global':
            float(statistics.get('mS_keep_ratio_positive_pair_global', 0.0)),
        'mS_mean_positive_pair_global': float(statistics.get('mS_mean_positive_pair_global', 0.0)),
        'mU_keep_ratio_global_valid_tiles':
            float(statistics.get('mU_keep_ratio_global_valid_tiles', 0.0)),
        'mU_keep_ratio_local_rows_global_valid_columns':
            float(statistics.get('mU_keep_ratio_local_rows_global_valid_columns', 0.0)),
        'mU_probability_mean': float(statistics.get('mU_probability_mean', 0.0)),
        'mU_probability_min': float(statistics.get('mU_probability_min', 0.0)),
        'mU_probability_max': float(statistics.get('mU_probability_max', 0.0)),
        'mask_keep_scope_note': ('mS_*_rank_local_* is this rank\'s batch; *_positive_pair_global is '
                                 'the gathered batch restricted to each image\'s own prefix; mU_* '
                                 'is the global valid subset J (rows = I2T anchors, columns = the '
                                 'candidate prefixes of J)'),
        # ---- suffix margins ------------------------------------------------------------------
        'suffix_positive_mean': float(statistics.get('positive_mean', 0.0)),
        'suffix_strongest_negative_mean': float(statistics.get('strongest_negative_mean', 0.0)),
        'suffix_max_margin_mean': float(statistics.get('max_margin_mean', 0.0)),
        'suffix_lse_margin_mean': float(statistics.get('lse_margin_mean', 0.0)),
        'suffix_lse_margin_min': float(statistics.get('lse_margin_min', 0.0)),
        'suffix_positive_win_fraction': float(statistics.get('positive_win_fraction', 0.0)),
        'suffix_top1': float(statistics.get('top1', 0.0)),
        'suffix_lse_margin_definition': 'positive - logsumexp(valid negatives only); logsumexp(all) - '
                                        'positive is CE and is never reported as the LSE margin',
        'suffix_anchor_count_local': int(statistics.get('anchor_count', 0)),
        'suffix_candidate_count_V': int(statistics.get('candidate_count', 0)),
        'near_zero_readout_norm_count': int(out['near_zero_readout_norm_count']),
        'readout_pair_count': int(out['readout_pair_count']),
        # ---- learning rates and step health --------------------------------------------------
        'lr': lr_of(optimizers['clip']),
        'mask_lr': lr_of(optimizers['mask']),
        'suffix_lr': lr_of(optimizers['suffix']),
        'lr_clip_group': lr_of(optimizers['clip']),
        'lr_mask_group': lr_of(optimizers['mask']),
        'lr_suffix_group': lr_of(optimizers['suffix']),
        'sec_per_step': float(compute_time),
        'sec_per_step_synchronized': float(compute_time),
        'synchronized_step_time_note': 'wall time of the whole DDP step (forward, backward, both '
                                       'collectives) measured to the update on this rank',
        'samples_per_sec': local_size * world / max(compute_time, 1e-9),
        'peak_memory_gib': peak_memory_gib(device),
        'peak_memory_gb': peak_memory_gib(device),
        'peak_memory_unit': 'GiB (1024^3 bytes)',
        # ---- presentation counters -----------------------------------------------------------
        'global_presentations': int(synchronized_presentations),
        'synchronized_pair_presentations': int(synchronized_presentations),
        'rank_local_presentations': int(presentations),
        'pair_presentations_expected_at_500': EXPECTED_PRESENTATIONS_AT_STEP_500,
        'presentation_definition': ('global_presentations and synchronized_pair_presentations both '
                                    'count the synchronized global pairs (local batch * world_size) '
                                    'over the run and must reach 512000 at step 500 = 500 * 1024; '
                                    'rank_local_presentations counts this rank\'s own pairs only'),
        'gpu_memory_note': 'peak_memory_gib is GPU memory in GiB (1024^3 bytes)',
        # ---- optimizers and precision --------------------------------------------------------
        'optimizer_groups': {'clip': len(optimizers['partition']['backbone']),
                             'mask': len(optimizers['partition']['mask_net']),
                             'suffix': len(optimizers['partition']['suffix']),
                             'trainable_total': len(optimizers['partition']['backbone'])
                             + len(optimizers['partition']['mask_net'])
                             + len(optimizers['partition']['suffix'])},
        'group_learning_rates': {'clip': {'lr': CLIP_LR, 'weight_decay': CLIP_WEIGHT_DECAY,
                                          'warmup': 200},
                                 'mask': {'lr': S0_MASK_LR, 'weight_decay': 0.0,
                                          'warmup': S0_MASK_WARMUP},
                                 'suffix': {'lr': SUFFIX_LR, 'weight_decay': 0.0,
                                            'warmup': SUFFIX_WARMUP}},
        'precision': PRECISION_NOTE,
        'image_chunk': int(out['image_chunk']), 'text_chunk': int(out['text_chunk']),
        'amp_dtype': args.amp_dtype,
        'suffix_mask_seed': SUFFIX_MASK_SEED,
        'statistics_scope': STATS_SCOPE,
        'stream_scope': STREAM_SCOPE,
        'rank_param_digest': state_digest(module.clip.state_dict())[:16],
        'rank_suffix_mask_digest': (None if module.suffix_mask is None
                                    else state_digest(module.suffix_mask.state_dict())[:16]),
        'batch_caption_sha256': hashlib.sha256(
            ('\n'.join(batch['caption_said'])).encode('utf-8')).hexdigest()[:16],
        'batch_image_id_sha256': tensor_digest(batch['image_id']),
        'rank_local_stream_digests': digests,
        'momentary_loss_total_for_backward': float(out.get('_loss_value', 0.0)),
        # the reused S0 code returns ``loss_smart_unweighted`` from its high-level forward only; the
        # low-level helper returns the two direction losses plus the sparsity term. Recompute the
        # unweighted sum from whichever shape is present instead of assuming one of them.
        'loss_smart_unweighted': float(out.get(
            'loss_smart_unweighted',
            float(out.get('loss_sidm', 0.0)) + float(out.get('loss_dism', 0.0)))),
        's0_term_keys_present': sorted(key for key in out if key.startswith('loss_')),
        'weights_note': 'adding the two CE directions is not halved; no new sparsity term is added '
                        'for mU',
    }
    record.update(dict((key, jsonable(value)) for key, value in health.items()))
    return record


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description='SAID-S0-Suffix v0.1 (%s)' % OBJECTIVE)
    parser.add_argument('--arm', default=ARM_NATIVE, choices=list(ARMS))
    parser.add_argument('--base_model', default='B16')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=BATCH_SIZE_PER_GPU)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=CLIP_LR,
                        help='CLIP backbone + both projections')
    parser.add_argument('--mask_lr', type=float, default=S0_MASK_LR,
                        help='the EXISTING S0 mask network (it keeps training in both arms)')
    parser.add_argument('--suffix_lr', type=float, default=SUFFIX_LR,
                        help='the NEW F (mask arm only)')
    parser.add_argument('--weight_decay', type=float, default=CLIP_WEIGHT_DECAY)
    parser.add_argument('--warmup_length', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lambda_suffix', type=float, default=LAMBDA_SUFFIX)
    parser.add_argument('--init_state', default=None)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS,
                        help='%d by default: the 501st optimizer update is never executed' % MAX_STEPS)
    parser.add_argument('--save_completed_steps',
                        default=','.join(str(step) for step in DEFAULT_SAVE_COMPLETED_STEPS))
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--heavy_log_every', type=int, default=25)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--amp_dtype', default='bf16', choices=['bf16', 'fp32'])
    parser.add_argument('--image_chunk', type=int, default=IMAGE_CHUNK_DEFAULT)
    parser.add_argument('--text_chunk', type=int, default=TEXT_CHUNK_DEFAULT)
    parser.add_argument('--suffix_checkpoint', type=int, default=1,
                        help='1 (default): write the frozen checkpoint key list at the save steps')
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()
    if args.base_model == 'B16':
        args.base_model = 'ViT-B/16'
    elif args.base_model == 'L14':
        args.base_model = 'ViT-L/14'

    seed_everything(args.seed)
    rank, local_rank, world = setup_distributed()
    device = torch.device('cuda', local_rank)
    save_completed = sorted({int(step) for step in args.save_completed_steps.split(',')
                             if step.strip()})

    model, _preprocess = longclip.load_from_clip(args.base_model, device='cpu', download_root=None,
                                                 args=args)
    model.train()
    model.logit_scale = torch.nn.Parameter(torch.ones([]) * 4.6052)
    model = model.to(device)
    init_file_sha = None
    if args.init_state:
        if not os.path.isfile(args.init_state):
            raise SystemExit('shared init missing: %s' % args.init_state)
        init_file_sha = file_sha256(args.init_state)
        load_init_state(model, args.init_state, rank)
    initial_state_digest = state_digest(model.state_dict())

    # the new gate is built INSIDE the module constructor's isolated RNG context, so constructing it
    # here cannot perturb the CLIP initialisation or the data stream
    suffix_mask = SuffixMask() if args.arm == ARM_MASK else None
    if suffix_mask is not None:
        suffix_mask = suffix_mask.to(device)
        init_report = suffix_mask_init_report(suffix_mask)
    else:
        init_report = None
    train_module = SaidPrefixSuffixTrainModule(
        model, rank=rank, arm=args.arm, lambda_suffix=args.lambda_suffix,
        suffix_mask=suffix_mask, image_chunk=args.image_chunk, text_chunk=args.text_chunk,
        world_size=world).to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        train_module, device_ids=[local_rank], output_device=local_rank,
        find_unused_parameters=True)
    ddp_model._set_static_graph()
    clip_handle = train_module.clip                        # read-only handle for checkpoints/logs

    optimizers = build_optimizers(clip_handle, train_module.suffix_mask,
                                 include_suffix=bool(args.arm == ARM_MASK), clip_lr=args.lr,
                                 mask_lr=args.mask_lr, suffix_lr=args.suffix_lr,
                                 weight_decay=args.weight_decay)
    partition = optimizers['partition']
    if rank == 0:
        print('OPT_GROUPS ' + json.dumps({
            'backbone': partition['backbone_count'], 'mask_net': partition['mask_net_count'],
            'suffix': partition['suffix_count'], 'trainable_total': partition['trainable_count'],
            'frozen': partition['frozen'],
            'note': 'partitioned by parameter id: no parameter missing, none duplicated; the S0 mask '
                    'network keeps training'}), flush=True)

    use_amp = args.amp_dtype == 'bf16'
    amp_dtype = torch.bfloat16
    dtype_counts = {}
    for parameter in list(clip_handle.parameters()) + ([] if train_module.suffix_mask is None
                                                       else list(train_module.suffix_mask.parameters())):
        key = str(parameter.dtype)
        dtype_counts[key] = dtype_counts.get(key, 0) + 1
    group_dtypes = {'clip': sorted({str(p.dtype) for p in optimizers['clip'].param_groups[0]['params']}),
                    'mask': sorted({str(p.dtype) for p in optimizers['mask'].param_groups[0]['params']})}
    if optimizers['suffix'] is not None:
        group_dtypes['suffix'] = sorted({str(p.dtype)
                                         for p in optimizers['suffix'].param_groups[0]['params']})
    if rank == 0:
        print('DTYPE_AUDIT parameter_dtypes=%s groups=%s amp_enabled=%s amp_dtype=%s'
              % (sorted(dtype_counts.items()), group_dtypes, use_amp, args.amp_dtype), flush=True)
        if init_report is not None:
            print('SUFFIX_MASK_INIT ' + json.dumps(jsonable(init_report), sort_keys=True), flush=True)
    if set(dtype_counts) != {'torch.float32'}:
        raise RuntimeError('fp32 master weights expected, got %r' % (dtype_counts,))
    for group_name, dtypes in group_dtypes.items():
        if dtypes != ['torch.float32']:
            raise RuntimeError('optimizer group %s must be fp32, got %r' % (group_name, dtypes))

    dataset = Share4VCvsslDataset(seed=args.seed, augment_view_b=False,
                                  strict_manifest=os.environ.get('SHARE4V_FULL_AUDIT'))
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, seed=args.seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                                        num_workers=args.num_workers, pin_memory=True,
                                        collate_fn=cvssl_collate, drop_last=False)
    steps_per_epoch = len(loader)
    total_steps = args.epochs * steps_per_epoch
    scheduler_clip = cosine_lr(optimizers['clip'], base_lr=args.lr,
                               warmup_length=args.warmup_length, steps=total_steps)
    scheduler_mask = cosine_lr(optimizers['mask'], base_lr=args.mask_lr, warmup_length=0,
                               steps=total_steps)
    scheduler_suffix = (None if optimizers['suffix'] is None else
                        cosine_lr(optimizers['suffix'], base_lr=args.suffix_lr, warmup_length=0,
                                  steps=total_steps))
    schedulers = {'clip': scheduler_clip, 'mask': scheduler_mask, 'suffix': scheduler_suffix}
    scheduler_config = {
        'clip': {'base_lr': args.lr, 'warmup_length': args.warmup_length,
                 'weight_decay': args.weight_decay, 'steps': total_steps},
        'mask': {'base_lr': args.mask_lr, 'warmup_length': 0, 'weight_decay': 0.0,
                 'steps': total_steps},
        'suffix': None if optimizers['suffix'] is None else
        {'base_lr': args.suffix_lr, 'warmup_length': 0, 'weight_decay': 0.0, 'steps': total_steps},
        'kind': 'cosine_lr (scheduler.cosine_lr), horizon = epochs * len(loader)',
        'note': '--max_steps only truncates the run, it never compresses the schedule',
    }

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, 'salu_log.jsonl')
    loss_config = {
        'loss_total': 'L_S0 + lambda_suffix * L_suffix',
        'L_S0': 'lambda_align * (L_SIDM + L_DISM) + lambda_sparse * mean|mS|',
        'lambda_align': LAMBDA_ALIGN, 'lambda_sparse': LAMBDA_SPARSE,
        'lambda_suffix': args.lambda_suffix,
        'L_suffix_native': 'CE(QN, y) + CE(QN.T, y), QN[i,j] = 100 * dot(Norm(g_i), tR_j)',
        'L_suffix_mask': 'CE(QU, y) + CE(QU.T, y), QU[i,j] = 100 * dot(Norm(g_i * mU_ij), tR_j)',
        'two_directions': 'summed, never multiplied by 0.5',
        'fixed_scale': 100.0,
        'new_sparsity_term_for_mU': None,
        'ddp_scaling_suffix': 'sum over local valid anchors * world_size / V, then standard DDP '
                              'averaging',
        'ddp_scaling_s0': 'unchanged S0 averaging, no world_size factor',
        'stopgrad_rule': 'S0: candidates detached, anchor live, mS detached, both directions summed; '
                         'suffix: xU fully detached (g and rS), g live only through g * mU, tR live',
        'gradient_responsibility': {
            's0': 'visual trunk + text trunk + S0 mask_net',
            'suffix': 'visual trunk through g * mU, text trunk through tR, and F; never the S0 mask '
                      'parameters directly, never the visual trunk through xU',
        },
    }
    sampling_config = {
        'dataset': 'Share4VCvsslDataset (reference json slicing, total_len=1000)',
        'sampler': 'DistributedSampler(shuffle=True), no drop_last',
        'batch_size_per_gpu': args.batch_size, 'world_size': world, 'global_batch': args.batch_size * world,
        'loader_batches': steps_per_epoch, 'lr_horizon_steps': total_steps,
        'num_workers': args.num_workers, 'pin_memory': True,
        'caption_stream': 'caption.replace("\\n"," "), sentences = caption.split(". "), '
                          'K = rng.randint(1, len(sentences)), P = ". ".join(sentences[:K]); the '
                          'objective reads K from the batch prefix_k and never re-draws it',
        'suffix_stream': 'R = ". ".join(s for s in sentences[K:last] if s non-empty), last = index of '
                         'the last non-empty fragment of the ORIGINAL list; empty suffixes keep an '
                         'empty-string placeholder and valid=False',
        'view': 'image_a only (reference openai-clip _transform(224)); no second view',
        'gradient_accumulation': 1, 'optimizer_updates_per_arm': args.max_steps,
        'stream_scope': STREAM_SCOPE,
    }
    config = {
        'objective': OBJECTIVE, 'phase': PHASE_NAME, 'arm': args.arm,
        'arm_mask': 'native' if args.arm == ARM_NATIVE else 'suffix_mask_F',
        'suffix_branch': 'prefix_conditioned_suffix_readout',
        'base_sha': '5676666', 'branch': 'codex/s0-suffix-v01',
        'precision': PRECISION_NOTE, 'amp_dtype': args.amp_dtype,
        'lr': args.lr, 'mask_lr': args.mask_lr, 'suffix_lr': args.suffix_lr,
        'weight_decay': args.weight_decay, 'warmup_length': args.warmup_length,
        'epochs': args.epochs, 'batch_size_per_gpu': args.batch_size, 'world_size': world,
        'loader_batches': steps_per_epoch, 'lr_horizon_steps': total_steps,
        'max_steps': args.max_steps, 'seed': args.seed, 'init_state': args.init_state,
        'init_file_sha256': init_file_sha, 'initial_state_digest': initial_state_digest,
        'git_head': git_head(), 'tokenizer_context': TOKENIZER_CONTEXT,
        'chunking': {'image_chunk': args.image_chunk, 'text_chunk': args.text_chunk},
        'suffix_mask': suffix_mask_config(train_module.suffix_mask),
        'suffix_checkpoint': bool(args.suffix_checkpoint),
        'statistics_scope': STATS_SCOPE,
    }
    if rank == 0:
        print('CONFIG ' + json.dumps(jsonable(config), sort_keys=True), flush=True)
        print('DATASET_SIZE %d STEPS_PER_EPOCH %d LR_HORIZON %d BACKBONE_TENSORS %d MASK_TENSORS %d '
              'SUFFIX_TENSORS %d GLOBAL_BATCH %d'
              % (len(dataset), steps_per_epoch, total_steps, partition['backbone_count'],
                 partition['mask_net_count'], partition['suffix_count'], args.batch_size * world),
              flush=True)
        with open(os.path.join(args.output_dir, 'config.json'), 'w') as handle:
            json.dump(jsonable(config), handle, indent=2, sort_keys=True)

    if args.resume:
        payload = torch.load(args.resume, map_location='cpu', weights_only=False)
        if payload.get('arm') != args.arm:
            raise ValueError('resume arm mismatch: %r vs %r' % (payload.get('arm'), args.arm))
        clip_handle.load_state_dict(payload['clip_state'])
        if train_module.suffix_mask is not None and payload.get('suffix_mask_state') is not None:
            train_module.suffix_mask.load_state_dict(payload['suffix_mask_state'])
        optimizers['clip'].load_state_dict(payload['optimizer_clip'])
        optimizers['mask'].load_state_dict(payload['optimizer_mask'])
        if optimizers['suffix'] is not None and payload.get('optimizer_suffix') is not None:
            optimizers['suffix'].load_state_dict(payload['optimizer_suffix'])
        start_step = int(payload['completed_steps'])
        if rank == 0:
            print('RESUMED from %s at completed step %d' % (args.resume, start_step), flush=True)
    else:
        start_step = 0

    caption_digest = hashlib.sha256()
    sample_digest = hashlib.sha256()
    image_id_digest = hashlib.sha256()
    prefix_k_digest = hashlib.sha256()
    suffix_valid_digest = hashlib.sha256()

    def local_stream_digests():
        return {'caption_stream_sha256': caption_digest.hexdigest(),
                'sample_stream_sha256': sample_digest.hexdigest(),
                'image_id_stream_sha256': image_id_digest.hexdigest(),
                'prefix_k_stream_sha256': prefix_k_digest.hexdigest(),
                'suffix_valid_stream_sha256': suffix_valid_digest.hexdigest()}

    def provenance(digests, horizon):
        return {
            'objective': OBJECTIVE, 'phase': PHASE_NAME, 'arm': args.arm,
            'base_sha': '5676666', 'branch': 'codex/s0-suffix-v01',
            'init_state': args.init_state, 'init_file_sha256': init_file_sha,
            'initial_state_digest': initial_state_digest,
            'git_head': config['git_head'],
            'stream_digests': digests, 'stream_scope': STREAM_SCOPE,
            'lr_horizon_steps': horizon, 'world_size': world,
            'rank_local_stream_digests_recorded_per_rank': True,
            'suffix_mask': suffix_mask_config(train_module.suffix_mask),
            'suffix_mask_init_report': init_report,
            'suffix_valid_rule': 'R non-empty AND at least one content token after tokenisation',
            'candidate_protocol': 'column j is (P_j, R_j); image i is scored against R_j with the mask '
                                  'generated from P_j',
            'valid_subset': 'J = all_gather(suffix_valid), V = |J|, suffix images and texts both '
                            'restricted to J, backward factor world_size / V',
            'precision': PRECISION_NOTE,
        }

    def save_checkpoint(steps, epoch, step_in_epoch, health=None):
        """Rank 0 writes one checkpoint with the frozen key list."""
        digests = dict(local_stream_digests())
        digests['init_file_sha256'] = init_file_sha
        digests['initial_state_digest'] = initial_state_digest
        digests['provenance'] = provenance(digests, total_steps)
        scheduler_state = {
            'clip': {'lr': lr_of(optimizers['clip'])},
            'mask': {'lr': lr_of(optimizers['mask'])},
            'suffix': {'lr': lr_of(optimizers['suffix'])},
            'completed_steps': int(steps),
        }
        path = os.path.join(args.output_dir, CHECKPOINT_FILENAME % (args.arm, steps))
        write_checkpoint(
            path=path, clip=clip_handle, suffix_mask=train_module.suffix_mask, optimizers=optimizers,
            steps=steps, arm=args.arm, config=config, digests=digests,
            objective=OBJECTIVE,
            scheduler_config=scheduler_config, scheduler_state=scheduler_state,
            data_cursor={'epoch': int(epoch), 'step_in_epoch': int(step_in_epoch),
                         'completed_steps': int(steps), 'loader_batches': steps_per_epoch,
                         'global_batch': args.batch_size * world,
                         'rank_local_stream_digests': digests},
            rng_states={'torch': torch.get_rng_state(),
                        'cuda': (torch.cuda.get_rng_state(device)
                                 if torch.cuda.is_available() else None)},
            loss_config=loss_config, sampling_config=sampling_config,
            provenance=provenance(digests, total_steps),
            chunking={'image_chunk': args.image_chunk, 'text_chunk': args.text_chunk},
            rank=rank, world=world, horizon_steps=total_steps, batch_size=args.batch_size,
            suffix_mask_report=init_report)
        if health is not None:
            print('SAVED %s %s' % (path, json.dumps(jsonable(health), sort_keys=True)), flush=True)
        else:
            print('SAVED ' + path, flush=True)

    if rank == 0 and args.suffix_checkpoint and start_step == 0 and 0 in save_completed:
        # step 0 is the frozen shared init, written before the first update (never the 501st step)
        save_checkpoint(0, 0, -1)
    in_loop_saves = sorted(step for step in save_completed if step > start_step and step != 0)

    completed = start_step
    t_start = time.time()
    compute_times = []
    presentations = 0
    synchronized_presentations = 0
    stopped = False

    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        for i, batch in enumerate(loader):
            if args.max_steps is not None and completed >= args.max_steps:
                stopped = True
                break
            t0 = time.time()
            group_lrs = run_schedulers(schedulers, completed, optimizers)
            heavy = bool(args.heavy_log_every > 0 and (completed + 1) % args.heavy_log_every == 0)
            out = suffix_train_step(ddp_model, batch, optimizers, device, amp_dtype,
                                    amp_enabled=use_amp, completed_steps=completed,
                                    want_statistics=heavy, image_chunk=args.image_chunk,
                                    text_chunk=args.text_chunk)
            completed += 1
            step_time = time.time() - t0
            compute_times.append(step_time)
            local_size = int(batch['image_a'].shape[0])
            presentations += local_size
            synchronized_presentations += local_size * world

            caption_digest.update(('\n'.join(batch['caption_said'])).encode('utf-8'))
            sample_digest.update(batch['sample_id'].numpy().tobytes())
            image_id_digest.update(batch['image_id'].numpy().tobytes())
            prefix_k_digest.update(batch['prefix_k'].numpy().tobytes())
            suffix_valid_digest.update(
                out['suffix_valid_flags'].detach().cpu().to(torch.uint8).numpy().tobytes())

            log_now = (completed % args.log_every == 0 or completed == 1
                       or completed in save_completed)
            if rank == 0 and (log_now or heavy):
                record = build_log_record(out, batch, optimizers, train_module, args, epoch, i,
                                         completed, step_time, presentations,
                                         synchronized_presentations, rank, world, device,
                                         local_stream_digests())
                for key, value in out.items():
                    if key.startswith('_'):
                        continue
                    if torch.is_tensor(value) and value.numel() == 1:
                        record.setdefault(key, float(value.detach()))
                with open(log_path, 'a') as handle:
                    handle.write(json.dumps(jsonable(record), sort_keys=True) + '\n')
                print('LOG ' + json.dumps(jsonable(record), sort_keys=True), flush=True)

            if rank == 0 and args.suffix_checkpoint and completed in in_loop_saves:
                save_checkpoint(completed, epoch, i, health=out.get('_grad_health'))
        if stopped:
            break

    all_rank_digests = None
    if dist.is_initialized():
        mine = dict(local_stream_digests())
        mine['rank'] = rank
        packed = [None] * world
        dist.all_gather_object(packed, mine)
        all_rank_digests = packed
    if rank == 0:
        summary = {
            'arm': args.arm, 'objective': OBJECTIVE, 'completed_steps': completed,
            'optimizer_updates': completed, 'max_steps': args.max_steps, 'epochs': args.epochs,
            'wall_sec': time.time() - t_start,
            'mean_sec_per_step': sum(compute_times) / max(len(compute_times), 1),
            'median_sec_per_step': float(np.median(compute_times)) if compute_times else 0.0,
            'steps_per_epoch': steps_per_epoch, 'lr_horizon_steps': total_steps,
            'world_size': world, 'batch_size_per_gpu': args.batch_size,
            'global_batch': args.batch_size * world,
            # ONE convention across the log and the summary: the headline presentation count is the
            # GLOBAL figure (500 * 1024 = 512000 at step 500), and the rank-local figure is labelled
            # as such instead of hiding behind the same key name
            'global_presentations': synchronized_presentations,
            'synchronized_pair_presentations': synchronized_presentations,
            'rank_local_presentations': presentations,
            'expected_pair_presentations_at_500': EXPECTED_PRESENTATIONS_AT_STEP_500,
            'presentation_definition': 'global_presentations and synchronized_pair_presentations both '
                                       'count (local pairs * world_size), i.e. 500 * 1024 = 512000 at '
                                       'step 500; rank_local_presentations counts this rank\'s own '
                                       'pairs only (128000 at step 500)',
            'peak_memory_gib': peak_memory_gib(device),
            'peak_memory_unit': 'GiB (1024^3 bytes)',
            'init_file_sha256': init_file_sha, 'initial_state_digest': initial_state_digest,
            'final_clip_state_digest': state_digest(clip_handle.state_dict()),
            'final_suffix_mask_digest': (None if train_module.suffix_mask is None
                                         else state_digest(train_module.suffix_mask.state_dict())),
            'stream_digests': local_stream_digests(),
            'stream_digests_all_ranks': all_rank_digests,
            'precision': PRECISION_NOTE, 'statistics_scope': STATS_SCOPE,
            'saved_steps': sorted(step for step in save_completed if step <= completed),
            'chunking': {'image_chunk': args.image_chunk, 'text_chunk': args.text_chunk},
        }
        with open(os.path.join(args.output_dir, 'run_summary.json'), 'w') as handle:
            json.dump(jsonable(summary), handle, indent=2, sort_keys=True)
        print('RUN_SUMMARY ' + json.dumps(jsonable(summary), sort_keys=True), flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
