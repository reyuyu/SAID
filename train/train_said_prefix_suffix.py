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
5. both directions are SUMMED over the local valid anchors (``A_r``) and scaled ONCE by
   ``world_size / V``: ``L_suffix_backward_local = (W / V) * A_r``. That per-rank scalar is a
   different quantity from the reported global suffix loss and is logged under a different name;
6. a collective reduction of the DETACHED local CE sums and of the gradient health runs on every
   rank; then exactly one update of each group. The reported global suffix loss is
   ``sum_r(local CE sum) / V``, never the local backward scalar.

Two index spaces are named and never mixed: this rank's rows are the LOCAL space, the concatenation
of every rank's rows in rank order is the GLOBAL space (``world_size * B`` rows) and the filtered
valid rows are the VALID space (``V`` rows). The models' ``to_valid`` map is the only conversion
between them; a local tensor is never selected with a global index and an already-filtered tensor is
never selected again.

``--resume`` is REFUSED for training (see :data:`RESUME_UNSUPPORTED`): the data cursor and the
per-rank RNG/data state are not restored, so continuing from a checkpoint would silently restart the
sample stream. Read-only acceptance loading lives in :func:`load_checkpoint_for_acceptance`.

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
                                      state_digest, suffix_mask_config, suffix_mask_init_report,
                                      suffix_scaling)
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

#: the two quantities the suffix path produces at every step, named apart on purpose: the reported
#: global value is reduced from the DETACHED local CE sums of every rank, while the scalar that is
#: back-propagated is this rank's own ``(W / V) * A_r``. They are never the same number.
LOG_FIELD_SUFFIX_GLOBAL = 'loss_suffix_global_reduced'
LOG_FIELD_SUFFIX_BACKWARD_LOCAL = 'loss_suffix_backward_local'

#: index spaces. ``LOCAL_FULL`` is this rank's own ``B`` rows, ``GLOBAL_FULL`` is the concatenation of
#: every rank's rows in rank order (``world_size * B``) and ``GLOBAL_VALID`` is the ``V`` rows left
#: after filtering. The names are recorded in the log so a reader can tell which space a number is in.
INDEX_SPACE_LOCAL_FULL = 'LOCAL_FULL'
INDEX_SPACE_GLOBAL_FULL = 'GLOBAL_FULL'
INDEX_SPACE_GLOBAL_VALID = 'GLOBAL_VALID'

#: the explicit refusal text for the unsupported resume path (spec section 10 wants an honest status)
RESUME_UNSUPPORTED = ('RESUME_UNSUPPORTED: training resume is disabled for objective %s. The '
                      'checkpoint is complete and loadable (clip_state, suffix_mask_state, '
                      'optimizer_clip, optimizer_mask, optimizer_suffix, the real config and '
                      'completed_steps), but this trainer does NOT restore the data cursor or the '
                      'per-rank RNG/data state, so continuing from a checkpoint would silently '
                      'replay epoch 0 batch 0 while reporting the checkpoint step. Run a fresh arm '
                      'from --init_state instead, or load the checkpoint read-only with '
                      'load_checkpoint_for_acceptance() for verification.')

#: the suffix-mask tensors and their exact shapes; the mask arm refuses to load without them
SUFFIX_MASK_EXPECTED_SHAPES = {'layer1.weight': (512, 1024), 'layer1.bias': (512,),
                               'layer2.weight': (512, 512), 'layer2.bias': (512,)}

#: the config keys whose drift between the checkpoint and this invocation must be refused
CHECKPOINT_CONFIG_IDENTITY = ('objective', 'phase', 'arm', 'base_sha', 'branch', 'init_state',
                              'init_file_sha256', 'batch_size_per_gpu', 'world_size',
                              'loader_batches', 'epochs')


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

    The entry-point contract is explicit, never inferred from text length: the production batch MUST
    carry ``caption_full``. A caption whose every fragment is shorter than ``K``, a one-sentence
    caption, ``K == N`` and a whole batch with no suffix are all LEGAL batches, so no heuristic may
    decide whether the caller passed a prefix instead of the full caption; the only evidence accepted
    here is that the field is present (guaranteed by ``cvssl_collate`` of ``said_cvssl_data``) and
    that the rebuilt prefix equals ``caption_said`` character for character.

    ``caption_said`` is used as that hard, character-for-character INVARIANT: the prefix produced from
    the full caption must equal it exactly, so the S0 prefix stream provably cannot drift.

    An empty (or content-token-free) suffix keeps the REAL tokenisation of the empty string as its
    placeholder -- ``longclip.tokenize('', truncate=True)``, i.e. SOT + EOT + padding, exactly what
    ``tokenize('')`` produces -- and is marked ``valid=False`` downstream. A placeholder is therefore
    ordinary text, never an arbitrary all-zero token tensor standing in for equivalent text, and it can
    never enter the suffix loss nor the suffix candidate pool.
    """
    if 'caption_full' not in batch:
        raise RuntimeError('the batch carries no caption_full field: the suffix branch needs the FULL '
                           'caption together with the prefix_k the S0 stream drew. Deriving P/R from '
                           'caption_said (already a prefix) makes every suffix empty, and guessing '
                           'from the text length is forbidden (a legal K == N, a one-sentence caption '
                           'and a suffix-free batch all look identical to a wrongly passed prefix). '
                           'Batch keys: %r' % sorted(batch))
    prefix_texts = [str(text) for text in batch['caption_said']]
    full_texts = [str(text) for text in batch['caption_full']]
    prefix_k = [int(value) for value in batch['prefix_k'].tolist()]
    if len(full_texts) != len(prefix_texts):
        raise RuntimeError('caption_full has %d rows but caption_said has %d'
                           % (len(full_texts), len(prefix_texts)))
    # ONE split function, shared by the trainer, the read-only diagnostic and the docs
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
    # the placeholder is the REAL tokenisation of the empty string for every row whose suffix is empty
    suffix_ids = longclip.tokenize([text if valid else ''
                                    for text, valid in zip(suffix_texts, split['valid'])],
                                   truncate=True).to(device)
    return {'prefix': prefix_ids, 'suffix': suffix_ids}, prefix_texts, suffix_texts, split


# --------------------------------------------------------------------------- reduced / log helpers
def _first_scalar(out, keys, default=0.0) -> float:
    """The first present scalar of ``keys`` as a float, else ``default``.

    The trainer owns the LOG NAMES of this objective and the model owns its output names; looking a
    value up through a short alias list keeps the two in step while the model file is being finished,
    and never turns a missing diagnostic into a crash inside the step.
    """
    for key in keys:
        if key in out and torch.is_tensor(out[key]):
            return float(out[key].detach())
        if key in out and isinstance(out[key], (int, float)):
            return float(out[key])
    return float(default)


def reduce_suffix_metrics(out, device, world_size: int = None, distributed: bool = None) -> dict:
    """Reduce the DETACHED local CE sums and counts of every rank: the reported suffix quantity.

    This is a NECESSARY collective: it runs on EVERY rank, from the main step path and never from a
    rank-0-only logging branch. With ``V`` global valid anchors the global reported value is

        loss_suffix_global_reduced = (sum_r local CE sum_r) / V

    which is the ``L_U_global`` of the spec. The scalar that is back-propagated is a DIFFERENT number,
    ``(W / V) * A_r`` on each rank, and is reported under its own name
    (``loss_suffix_backward_local``) so the two can never be confused in a log or in a report.
    """
    local_i2t = _first_scalar(out, ('loss_suffix_i2t_sum',))
    local_t2i = _first_scalar(out, ('loss_suffix_t2i_sum',))
    local_sum = _first_scalar(out, ('loss_suffix_local_ce_sum', 'loss_suffix_local_sum'),
                              default=local_i2t + local_t2i)
    V = int(out.get('global_valid_V', 0))
    total = int(out.get('global_batch_size', 0))
    lam = float(out.get('lambda_suffix', LAMBDA_SUFFIX))
    if distributed is None:
        distributed = dist.is_initialized()
    if distributed:
        packed = torch.tensor([local_i2t, local_t2i, local_sum], dtype=torch.float64, device=device)
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        global_i2t, global_t2i, global_sum = (float(value) for value in packed.tolist())
        world = int(world_size if world_size is not None else dist.get_world_size())
    else:
        global_i2t, global_t2i, global_sum = local_i2t, local_t2i, local_sum
        world = int(world_size if world_size is not None else 1)
    divisor = float(V) if V >= 2 else 0.0
    global_reduced = (global_sum / divisor) if divisor else 0.0
    return {
        'loss_suffix_global_reduced': global_reduced,
        'loss_suffix_global_reduced_i2t': (global_i2t / divisor) if divisor else 0.0,
        'loss_suffix_global_reduced_t2i': (global_t2i / divisor) if divisor else 0.0,
        'loss_suffix_global_ce_sum': global_sum,
        'loss_suffix_global_ce_sum_i2t': global_i2t,
        'loss_suffix_global_ce_sum_t2i': global_t2i,
        'loss_suffix_backward_local': _first_scalar(
            out, (LOG_FIELD_SUFFIX_BACKWARD_LOCAL, 'loss_suffix')),
        'weighted_loss_suffix_global_reduced': lam * global_reduced,
        'loss_suffix_local_ce_sum': local_sum,
        'loss_suffix_local_ce_sum_i2t': local_i2t,
        'loss_suffix_local_ce_sum_t2i': local_t2i,
        'suffix_reduction_world_size': world,
        'suffix_reduction_global_valid_V': V,
        'suffix_reduction_divisor': divisor,
        'suffix_reduction_rule': ('loss_suffix_global_reduced = (sum over ranks of the DETACHED local '
                                  'I2T/T2I CE sums) / V; the per-rank backward scalar '
                                  '(W / V) * A_r is reported separately as '
                                  'loss_suffix_backward_local and is never the reported global value'),
    }


def assert_forward_shape_and_index_legality(out: dict, batch, device, rank: int, world: int) -> dict:
    """The all-rank shape/index legality check, run BEFORE any dangerous GPU indexing.

    Every rank reaches the same verdict, because the values checked (``V``, the per-rank counts, the
    local batch size and the world size) were all produced by the collectives of this step. The check
    reports its full context -- every shape, the index space of each tensor and the local/global valid
    counts -- instead of letting a stray index abort the backend on a device-side assertion with no
    information about which space the index came from.
    """
    context = {
        'rank': int(rank), 'world_size': int(world), 'device': str(device),
        'local_batch': int(batch['image_a'].shape[0]),
        'global_batch': int(out.get('global_batch_size', 0)),
        'global_valid_V': int(out.get('global_valid_V', -1)),
        'local_valid_count': int(out.get('local_valid_count', -1)),
        'valid_counts_per_rank': [int(value) for value in out.get('valid_counts_per_rank', [])],
        'index_spaces': {
            'LOCAL_FULL': 'this rank\'s rows [0, local_batch)',
            'GLOBAL_FULL': 'every rank\'s rows in rank order [0, world_size * local_batch)',
            'GLOBAL_VALID': 'the filtered valid rows [0, V)',
        },
        'global_batch_size_key_present': 'global_batch_size' in out,
        'suffix_valid_flags_shape': list(out['suffix_valid_flags'].shape)
        if torch.is_tensor(out.get('suffix_valid_flags')) else None,
    }
    problems = []
    if context['global_batch'] != context['world_size'] * context['local_batch']:
        problems.append('global_batch_size %d != world_size * local_batch = %d'
                        % (context['global_batch'], context['world_size'] * context['local_batch']))
    if context['suffix_valid_flags_shape'] not in (None, [context['local_batch']]):
        problems.append('suffix_valid_flags has shape %s, expected [local_batch=%d]'
                        % (context['suffix_valid_flags_shape'], context['local_batch']))
    if context['valid_counts_per_rank']:
        counts = context['valid_counts_per_rank']
        if len(counts) != context['world_size']:
            problems.append('valid_counts_per_rank has %d entries but world_size is %d'
                            % (len(counts), context['world_size']))
        elif sum(counts) != context['global_valid_V']:
            problems.append('valid_counts_per_rank sums to %d but global_valid_V is %d'
                            % (sum(counts), context['global_valid_V']))
        elif counts[int(rank)] != context['local_valid_count']:
            problems.append('this rank\'s count %d disagrees with the gathered vector (%d)'
                            % (counts[int(rank)], context['local_valid_count']))
    if context['global_valid_V'] > context['global_batch']:
        problems.append('global_valid_V %d exceeds the global batch %d: the valid index set can only '
                        'name GLOBAL_FULL rows' % (context['global_valid_V'], context['global_batch']))
    V = context['global_valid_V']
    if V >= 2 and context['local_valid_count'] > V:
        problems.append('this rank owns %d valid anchors but the global valid pool has only %d'
                        % (context['local_valid_count'], V))
    if problems:
        raise RuntimeError('the forward is not shape/index legal on every rank, refusing to index '
                           'with it: %r | context=%r' % (problems, context))
    return context


# --------------------------------------------------------------------------- one real step
def suffix_train_step(ddp_model, batch, optimizers, device, amp_dtype, amp_enabled: bool = True,
                      capture_grads: bool = False, completed_steps: int = 0,
                      want_statistics: bool = False, image_chunk: int = None,
                      text_chunk: int = None, text_ids=None, suffix_texts=None):
    """DDP forward -> backward -> collective reductions -> one update of all three groups.

    This is the *production* step function: the acceptance tests call it directly, so what is
    verified is what is run. ``lambda_suffix`` is never dropped here -- it is applied inside the
    objective, so a test that compares a one-step update cannot silently lose it.

    The suffix path is entirely inside the DDP ``forward``, so every rank executes the same necessary
    collectives and their backward: a rank with ``n_r = 0`` while the global ``V >= 2`` still builds
    its own scores and still takes part in the differentiable gather, and a step with ``V < 2`` makes
    every rank skip ONLY the candidate scoring communication, keeping ``L_suffix = 0`` and continuing
    with S0. Nothing is fabricated to make an unused parameter look trained: a genuinely unused F
    parameter legitimately has ``None`` gradient, which is exactly what plain
    ``find_unused_parameters=True`` DDP expects (no ``_set_static_graph`` anywhere in this file).

    The returned dict is safe to hand back across the DDP boundary: the live tensors and the modules
    of the forward are replaced by detached, scalar-only diagnostics, because DDP walks the FORWARD
    OUTPUT in ``prepare_for_backward`` and a live non-loss tensor or a ``Module`` in that structure
    breaks its graph analysis.
    """
    image_a = batch['image_a'].to(device, non_blocking=True)
    suffix_texts = None
    if text_ids is None:
        text_ids, _prefix_texts, suffix_texts, _split = tokenized(batch, device)
    if suffix_texts is None:
        suffix_texts = [''] * int(image_a.shape[0])
    # a ragged batch must be refused BEFORE the first cross-rank gather
    module = getattr(ddp_model, 'module', ddp_model)
    world = int(module.objective._world())
    assert_equal_local_batch(int(image_a.shape[0]), world_size=world, device=device)

    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
        out = ddp_model(image_a, text_ids,
                        prefix_captions=list(batch['caption_full']),
                        batch_prefix_texts=list(batch['caption_said']),
                        suffix_texts=suffix_texts, prefix_k=batch['prefix_k'],
                        image_chunk=image_chunk, text_chunk=text_chunk,
                        want_statistics=want_statistics)

    # every rank checks the shapes and the index spaces of this step BEFORE anything indexes with
    # them; the values were produced collectively, so all ranks reach the same verdict
    index_context = assert_forward_shape_and_index_legality(out, batch, device, rank=module.rank,
                                                            world=world)

    loss = out['loss_total_for_backward']
    # the scalar that is back-propagated on THIS rank, named apart from the reported global value
    out[LOG_FIELD_SUFFIX_BACKWARD_LOCAL] = _first_scalar(out, ('loss_suffix',))
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

    # the reported GLOBAL suffix loss, reduced from the DETACHED local CE sums of every rank. This is
    # a necessary collective on the main step path (never a rank-0-only branch), and it is NOT the
    # per-rank backward scalar.
    reduced = reduce_suffix_metrics(out, device, world_size=world)

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
        'suffix_mask_parameter_grad_is_none': bool(
            inspected.suffix_mask is not None
            and all(parameter.grad is None for parameter in inspected.suffix_mask.parameters())),
        'suffix_mask_parameter_grad_unused_note': ('a None gradient on a genuinely unused F parameter '
                                                   'is the correct DDP state for find_unused_'
                                                   'parameters=True; no zero is fabricated'),
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
    out['_suffix_metrics'] = reduced
    out['_index_context'] = index_context
    # ---- the DDP boundary: only detached scalars and the short summaries may cross it --------------
    # ``g_live`` / ``mask_s`` / ``t_r_all`` / ``suffix_mask`` are live tensors and a Module of the
    # forward that do NOT contribute to the loss. They are replaced by detached scalar diagnostics so
    # that DDP's graph analysis sees a clean structure, and so that a caller cannot accidentally
    # backward through a stale graph.
    detached_scalars = {}
    for key, source in (('g_live_norm_mean', 'g_live'), ('mask_s_keep_ratio_local', 'mask_s'),
                        ('t_r_norm_mean', 't_r_all')):
        value = out.get(source)
        if torch.is_tensor(value):
            detached_scalars[key] = float(value.detach().float().mean())
    for key in ('g_live', 'mask_s', 'mask_s_positive_pair_local', 't_r_all', 'suffix_mask'):
        out.pop(key, None)
    out['_detached_diagnostics'] = detached_scalars
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


#: the suffix-mask TENSOR key family and the DESCRIPTION key family. They are disjoint by construction
#: (spec section 10) and the loader refuses any checkpoint that merged the two.
SUFFIX_TENSOR_KEYS = ('suffix_mask_state',)
SUFFIX_DESCRIPTION_KEYS = ('suffix_mask_config', 'suffix_mask_descriptor')


def load_checkpoint_for_acceptance(path, config=None, expect_arm=None, clip=None,
                                   clip_model=None, suffix_mask=None, strict_identity: bool = True,
                                   device=None) -> dict:
    """Load a full checkpoint for a READ-ONLY acceptance check, and prove it is complete.

    This is the loading half of spec section 10 and is deliberately NOT wired into training: the
    training resume path is refused (see :data:`RESUME_UNSUPPORTED`) because the data cursor and the
    per-rank RNG/data state are not restored there. Verification, export and the acceptance tests use
    this function instead, and it never touches the sample stream.

    It refuses, loudly and before any state reaches a model:

    * a checkpoint whose frozen key list is incomplete (``clip_state``, ``suffix_mask_state``,
      ``suffix_mask_config``, ``optimizer_clip``, ``optimizer_mask``, ``optimizer_suffix``, the real
      config and ``completed_steps``);
    * a mask-arm checkpoint with NO F tensors, instead of silently continuing with a random F;
    * a state whose shape or digest disagrees with its own header, or a description dict that smuggled
      tensors under the tensor key (or vice versa);
    * an identity drift: objective, arm, base SHA, init state and the real config of the invocation
      that wrote it.

    Returns a report of everything that was verified; when ``clip`` (or the new gate) is given, the
    verified tensors are also loaded into those modules, so the caller proves the round trip.
    """
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError('%s does not hold a checkpoint payload (got %s)'
                           % (path, type(payload).__name__))
    missing = [key for key in CHECKPOINT_KEYS if key not in payload]
    if missing:
        raise RuntimeError('the checkpoint %s is incomplete: the frozen key(s) %r are absent'
                           % (path, missing))
    arm = payload.get('arm')
    if arm not in ARMS:
        raise RuntimeError('the checkpoint %s names arm %r, not one of %r' % (path, arm, ARMS))
    if expect_arm is not None and arm != expect_arm:
        raise RuntimeError('the checkpoint %s was written by arm %r, not %r' % (path, arm, expect_arm))
    if payload.get('objective') != OBJECTIVE:
        raise RuntimeError('the checkpoint %s carries objective %r instead of %r'
                           % (path, payload.get('objective'), OBJECTIVE))
    clip_state = payload['clip_state']
    if not isinstance(clip_state, dict) or not clip_state:
        raise RuntimeError('the checkpoint %s carries no clip_state tensor dict' % path)
    recorded_clip_digest = payload.get('clip_state_digest')
    if recorded_clip_digest is not None and state_digest(clip_state) != recorded_clip_digest:
        raise RuntimeError('clip_state does not match its own header digest in %s (recorded %s, '
                           'recomputed %s)' % (path, recorded_clip_digest, state_digest(clip_state)))

    # ---- the suffix mask: TENSORS and DESCRIPTION live under different keys -----------------------
    suffix_state = payload.get('suffix_mask_state')
    suffix_config = payload.get('suffix_mask_config')
    if not isinstance(suffix_config, dict) or not suffix_config:
        raise RuntimeError('the checkpoint %s carries no suffix mask DESCRIPTION under %r'
                           % (path, SUFFIX_DESCRIPTION_KEYS))
    if any(torch.is_tensor(value) for value in suffix_config.values()):
        raise RuntimeError('the suffix mask description of %s carries tensors: the description and the '
                           'tensor state must never be merged' % path)
    if not set(SUFFIX_TENSOR_KEYS) & set(payload) or set(SUFFIX_TENSOR_KEYS) & set(suffix_config):
        raise RuntimeError('the suffix tensor key family %r and the description family %r must stay '
                           'disjoint (ckpt %s)' % (SUFFIX_TENSOR_KEYS, SUFFIX_DESCRIPTION_KEYS, path))
    if arm == ARM_NATIVE:
        if suffix_state is not None:
            raise RuntimeError('the native arm must carry suffix_mask_state = None, %s carries %s'
                               % (path, type(suffix_state).__name__))
        if int(payload.get('suffix_mask_count') or 0) != 0:
            raise RuntimeError('the native arm checkpoint %s claims %r suffix mask parameters: the '
                               'native arm must not own F' % (path, payload.get('suffix_mask_count')))
    else:
        if not isinstance(suffix_state, dict) or not suffix_state:
            raise RuntimeError('REFUSING to load %s: the mask arm checkpoint carries no F tensors '
                               '(suffix_mask_state = %r). Continuing would train or score with a '
                               'randomly initialised gate while claiming the conditional model was '
                               'restored' % (path, suffix_state))
        stripped = {str(name).split('.', 1)[-1] if str(name).startswith('suffix_mask.')
                    else str(name): value for name, value in suffix_state.items()}
        if any(not torch.is_tensor(value) for value in stripped.values()):
            raise RuntimeError('the suffix mask state of %s is not a pure tensor dict' % path)
        absent = sorted(set(SUFFIX_MASK_EXPECTED_SHAPES) - set(stripped))
        wrong = sorted('%s=%s' % (name, list(stripped[name].shape)) for name in stripped
                       if name in SUFFIX_MASK_EXPECTED_SHAPES
                       and tuple(int(dim) for dim in stripped[name].shape)
                       != SUFFIX_MASK_EXPECTED_SHAPES[name])
        if absent or wrong:
            raise RuntimeError('the F tensors of %s do not match the module: absent %r, wrong shapes '
                               '%r (state holds %r)' % (path, absent, wrong, sorted(stripped)))
        recorded_suffix_digest = payload.get('suffix_mask_state_digest')
        if recorded_suffix_digest is None:
            raise RuntimeError('the mask arm checkpoint %s records no suffix_mask_state_digest: a '
                               'missing F state must not be silently accepted' % path)
        if state_digest(suffix_state) != recorded_suffix_digest:
            raise RuntimeError('the F tensors of %s do not match their own header digest (recorded %s, '
                               'recomputed %s)' % (path, recorded_suffix_digest,
                                                   state_digest(suffix_state)))
    if not isinstance(payload.get('completed_steps'), int):
        raise RuntimeError('the checkpoint %s carries no integer completed_steps' % path)

    # ---- identity: the invocation that wrote it must be the one loading it -----------------------
    drift = {}
    recorded_config = payload.get('config')
    if recorded_config is not None and not isinstance(recorded_config, dict):
        raise RuntimeError('the checkpoint config of %s is a %s, not a dict'
                           % (path, type(recorded_config).__name__))
    if strict_identity and isinstance(config, dict) and isinstance(recorded_config, dict):
        for key in CHECKPOINT_CONFIG_IDENTITY:
            if key in recorded_config and key in config and recorded_config[key] != config[key]:
                drift[key] = {'checkpoint': recorded_config[key], 'invocation': config[key]}
    if drift:
        raise RuntimeError('the checkpoint %s was written by a different configuration: %r'
                           % (path, drift))
    loaded = {'clip': False, 'suffix_mask': False}
    target_clip = clip if clip is not None else clip_model
    if target_clip is not None:
        target_clip.load_state_dict(clip_state, strict=True)
        loaded['clip'] = True
    if suffix_mask is not None and arm == ARM_MASK:
        if device is not None:
            suffix_mask = suffix_mask.to(device)
        suffix_mask.load_state_dict(suffix_state, strict=True)
        loaded['suffix_mask'] = True
    return {
        'path': os.path.abspath(path),
        'arm': arm, 'objective': payload.get('objective'),
        'completed_steps': int(payload['completed_steps']),
        'clip_state_digest': recorded_clip_digest,
        'suffix_mask_state_digest': payload.get('suffix_mask_state_digest'),
        'suffix_tensor_keys': sorted(SUFFIX_TENSOR_KEYS),
        'suffix_description_key': sorted(SUFFIX_DESCRIPTION_KEYS),
        'optimizers_present': {name: payload.get('optimizer_' + name) is not None
                               for name in ('clip', 'mask', 'suffix')},
        'config_present': isinstance(recorded_config, dict),
        'identity_keys_compared': list(CHECKPOINT_CONFIG_IDENTITY),
        'data_cursor': payload.get('data_cursor'),
        'rng_states_present': bool(payload.get('rng_states')),
        'loaded_into': loaded,
        'training_resume_used': False,
        'training_resume_note': RESUME_UNSUPPORTED % OBJECTIVE,
    }


# --------------------------------------------------------------------------- logging
def build_log_record(out, batch, optimizers, module, args, epoch, step_in_epoch, completed,
                     compute_time, presentations, synchronized_presentations, rank, world, device,
                     digests) -> dict:
    """One JSON object per logged step, with the field names the dashboard reads."""
    statistics = out.get('_statistics') or {}
    health = out.get('_grad_health') or {}
    metrics = out.get('_suffix_metrics') or {}
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
        # TWO different quantities, two different names. The headline suffix number of this objective
        # is the GLOBAL one, reduced from the DETACHED local CE sums of every rank and divided by V;
        # the per-rank scalar that is actually back-propagated is (W / V) * A_r and is reported only
        # under the ``*_backward_local`` name, never as the global value.
        LOG_FIELD_SUFFIX_GLOBAL: float(metrics.get(LOG_FIELD_SUFFIX_GLOBAL, 0.0)),
        LOG_FIELD_SUFFIX_BACKWARD_LOCAL: float(out.get(LOG_FIELD_SUFFIX_BACKWARD_LOCAL, 0.0)),
        'weighted_loss_suffix_global_reduced': float(
            metrics.get('weighted_loss_suffix_global_reduced', 0.0)),
        'loss_suffix_global_reduced_I2T': float(metrics.get('loss_suffix_global_reduced_i2t', 0.0)),
        'loss_suffix_global_reduced_T2I': float(metrics.get('loss_suffix_global_reduced_t2i', 0.0)),
        'loss_suffix_global_ce_sum': float(metrics.get('loss_suffix_global_ce_sum', 0.0)),
        'loss_suffix_local_ce_sum': float(metrics.get('loss_suffix_local_ce_sum', 0.0)),
        'loss_suffix_local_ce_sum_I2T': float(metrics.get('loss_suffix_local_ce_sum_i2t', 0.0)),
        'loss_suffix_local_ce_sum_T2I': float(metrics.get('loss_suffix_local_ce_sum_t2i', 0.0)),
        'suffix_reduction_divisor_V': float(metrics.get('suffix_reduction_divisor', 0.0)),
        'suffix_reduction_rule': metrics.get(
            'suffix_reduction_rule',
            'the reported global suffix loss is (sum over ranks of the detached local CE sums) / V'),
        'suffix_loss_naming_note': ('loss_suffix_global_reduced is the all-gathered DETACHED result '
                                    'and is the only field to quote as the global suffix loss; '
                                    'loss_suffix_backward_local is this rank\'s (W / V) * A_r and is '
                                    'a different quantity'),
        'suffix_local_equal_global_invariant': ('the per-rank local_ce_sum values SUM to the global '
                                                'CE sum, and DDP\'s final parameter gradients equal '
                                                'an independent global-mean objective; '
                                                'local_ce_sum == (W/V)*global_ce_sum is NOT an '
                                                'invariant and is never asserted'),
        'lambda_suffix': float(out['lambda_suffix']),
        'suffix_scale_world_over_V': float(out['suffix_scale_world_over_V']),
        'suffix_scaling_rule': 'A_r = local (I2T CE sum + T2I CE sum); L_suffix_backward_local = '
                               '(world_size / V) * A_r applied EXACTLY once, then standard DDP '
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
        'valid_suffix_counts_source_scope': INDEX_SPACE_GLOBAL_FULL + ' + ' + INDEX_SPACE_GLOBAL_VALID,
        'empty_suffix_fraction_local': 1.0 - _first_scalar(out, ('suffix_valid_ratio_local',), 0.0),
        'empty_suffix_fraction_global': _first_scalar(out, ('empty_suffix_fraction_global',), 1.0),
        'valid_subset_rule': 'J = all_gather(suffix_valid) over the %s index space; both the suffix '
                             'images and the suffix texts are restricted to J once; V < 2 gives an '
                             'exactly zero suffix loss and every rank skips only the candidate '
                             'scoring communication' % INDEX_SPACE_GLOBAL_FULL,
        'index_space_names': {INDEX_SPACE_LOCAL_FULL: 'this rank\'s rows',
                              INDEX_SPACE_GLOBAL_FULL: 'world_size * local_batch rows in rank order',
                              INDEX_SPACE_GLOBAL_VALID: 'the V filtered valid rows'},
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
    # the detached scalars that REPLACED the live forward tensors (``g_live`` / ``mask_s`` /
    # ``t_r_all``) at the DDP boundary: what used to be a whole tensor is now a named scalar
    diagnostics = out.get('_detached_diagnostics') or {}
    record['detached_diagnostics'] = {
        'scope': 'rank-local rows, detached at the DDP boundary (the live tensors of the forward are '
                 'not part of the returned dict)',
        'g_live_norm_mean': diagnostics.get('g_live_norm_mean'),
        'mask_s_keep_ratio_local': diagnostics.get('mask_s_keep_ratio_local'),
        't_r_norm_mean': diagnostics.get('t_r_norm_mean'),
    }
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
    parser.add_argument('--resume', default=None,
                        help='REFUSED for training: the data cursor and the per-rank RNG/data state '
                             'are not restored, so the trainer stops with RESUME_UNSUPPORTED instead '
                             'of silently restarting the sample stream. Read-only acceptance loading '
                             'uses load_checkpoint_for_acceptance().')
    args = parser.parse_args()
    if args.base_model == 'B16':
        args.base_model = 'ViT-B/16'
    elif args.base_model == 'L14':
        args.base_model = 'ViT-L/14'
    if args.resume:
        # Honest refusal, BEFORE any process-group, CUDA or data work: the previous behaviour restored
        # ``completed_steps`` and then continued from epoch 0 batch 0, which silently replayed the
        # sample stream while claiming a resume. Either the data cursor and the per-rank RNG/data state
        # are restored and PROVEN with an interrupted-vs-uninterrupted comparison, or resume is refused.
        raise SystemExit(RESUME_UNSUPPORTED % OBJECTIVE)

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
    # NO ``_set_static_graph()``: this objective's valid subset J changes from step to step (V = 0 ->
    # valid -> V = 1 -> valid), so DDP must re-derive the unused-parameter set on every iteration.
    # A static graph would freeze the first iteration's set and silently drop the synchronisation of a
    # parameter that becomes used later. Genuinely unused F parameters keep a None gradient, which is
    # the correct state for find_unused_parameters=True; no zero is ever fabricated for them.
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

    # ---- training resume: REFUSED, and refused unreachably --------------------------------------
    # ``--resume`` already stopped the process before the process group was created, so this branch is
    # unreachable in production. It is kept as the single documented place where the loading half of
    # the checkpoint exists for READ-ONLY acceptance, and it is asserted here as well so that a future
    # edit cannot quietly reinstate the "restore completed_steps and continue from epoch 0 batch 0"
    # behaviour, nor the silent "keep the random F when the checkpoint has none" fallback.
    if args.resume:                                            # pragma: no cover - guarded in main()
        raise SystemExit(RESUME_UNSUPPORTED % OBJECTIVE)
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
                            'restricted to J exactly once, backward factor world_size / V applied '
                            'exactly once',
            'precision': PRECISION_NOTE,
            'resume_support': RESUME_UNSUPPORTED % OBJECTIVE,
            'resume_status': 'RESUME_UNSUPPORTED',
            'read_only_loader': 'load_checkpoint_for_acceptance',
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
                         'rank_local_stream_digests': digests,
                         # the cursor is RECORDED for provenance but is NOT restored by this trainer:
                         # that is exactly why training resume is refused rather than faked
                         'restored_by_trainer': False,
                         'resume_status': 'RESUME_UNSUPPORTED'},
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
            valid_flags = out.get('suffix_valid_flags')
            if torch.is_tensor(valid_flags):        # the field exists in every arm and at every V
                suffix_valid_digest.update(
                    valid_flags.detach().cpu().to(torch.uint8).numpy().tobytes())

            log_now = (completed % args.log_every == 0 or completed == 1
                       or completed in save_completed)
            # NOTE: this branch is rank-0-only and therefore contains NO collective. Every collective
            # of the step (the gradient-health flag and the suffix reduction) already ran on all ranks
            # inside suffix_train_step.
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
                metrics = out.get('_suffix_metrics') or {}
                if metrics:
                    # the reduced, global fields are recorded under their OWN names at the top level
                    # of the record, so a reader never has to know a helper's return dict
                    record['loss_suffix_global_reduced'] = float(
                        metrics.get('loss_suffix_global_reduced', 0.0))
                    record['loss_suffix_backward_local'] = float(
                        out.get(LOG_FIELD_SUFFIX_BACKWARD_LOCAL, 0.0))
                    record['suffix_reduction_fields_source'] = 'reduce_suffix_metrics (all-rank reduce)'
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
            # ---- the honest resume status of this arm -------------------------------------------
            'resume_support': RESUME_UNSUPPORTED % OBJECTIVE,
            'resume_status': 'RESUME_UNSUPPORTED',
            'resume_restores': ['nothing: this trainer never continues a run'],
            'resume_refused_for_training': True,
            'read_only_checkpoint_loader': 'load_checkpoint_for_acceptance',
            'checkpoint_contains': ['clip_state', 'suffix_mask_state', 'suffix_mask_config',
                                    'optimizer_clip', 'optimizer_mask', 'optimizer_suffix',
                                    'config', 'completed_steps', 'data_cursor (recorded only)',
                                    'rng_states (recorded only)'],
            'checkpoint_verified_by': ('load_checkpoint_for_acceptance (frozen key list, shape and '
                                       'digest checks, refusal to load a mask arm without F tensors)'),
            'ddp_notes': {
                'static_graph': False,
                'find_unused_parameters': True,
                'unused_parameter_policy': ('a genuinely unused F parameter keeps grad None; no zero is '
                                            'fabricated to make it look trained'),
                'forward_output_policy': ('live tensors and modules of the forward are replaced by '
                                          'detached scalar diagnostics before the dict leaves the '
                                          'step, so DDP analyses a clean graph'),
                'collectives_on_every_rank': True,
            },
            'suffix_log_fields': {
                'global_reported': LOG_FIELD_SUFFIX_GLOBAL,
                'local_backward': LOG_FIELD_SUFFIX_BACKWARD_LOCAL,
                'note': ('the two are different quantities; the global one is reduced from detached '
                         'local CE sums and divided by V'),
            },
        }
        with open(os.path.join(args.output_dir, 'run_summary.json'), 'w') as handle:
            json.dump(jsonable(summary), handle, indent=2, sort_keys=True)
        print('RUN_SUMMARY ' + json.dumps(jsonable(summary), sort_keys=True), flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
