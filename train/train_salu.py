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

Objective modes (Phase 3.0A, ``--objective_mode``):
  * ``legacy`` (default): the Phase 2 losses described above, unchanged.
  * ``gap_completion``: the **base** objective, *Said-Conditioned Visual Complement
    Discovery*, on ``(Image I, C_S)`` only::

        L_total = lambda_said * L_said
                + lambda_gap_discover * L_gap_discover
                + lambda_global_absorb * L_global_absorb

    It has no Global-text InfoNCE and no Phase 2.8A / 2.9A Unsaid term, so
    ``--lambda_global 0 --lambda_unsaid 0`` are required and checked before training starts
    (``validate_objective_args``). The DataLoader asks ``share4v_train_dataset`` for
    ``caption_views=False``, so a batch is exactly ``(I, C_S)``: no ``caption_full``, no
    ``caption_unsaid``, no ``has_unsaid``, and only ``C_S`` is tokenized. Logged
    ``loss_global`` / ``loss_unsaid`` are JSON ``null`` because those terms do not exist in
    this objective, and the semantic flags (``visual_complement_enabled``,
    ``legacy_unsaid_enabled``, ``global_text_alignment_enabled``) say which objective ran.

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
import contextlib
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
from model import exgap  # noqa: E402
from model.complement_diagnostics import COMPLEMENT_DIAGNOSTIC_KEYS  # noqa: E402
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


def tri_gap_record(out):
    """Explicit tri-alignment gaps for the training log (Phase 2.9C logging fix).

    ``forward_train`` already returns ``gap_global_to_full_text`` /
    ``gap_global_to_said_text`` / ``gap_said_to_said_text``; Phase 2.9B computed them but
    never serialized them. ``None`` (e.g. no full-caption view) stays ``None`` -> JSON null.
    Logging only: no loss or forward math is touched.
    """
    return {key: (None if out.get(key) is None else float(out[key]))
            for key in ('gap_global_to_full_text', 'gap_global_to_said_text',
                        'gap_said_to_said_text')}


def validation_key(step, dataset, caption_variant):
    """Identity of a validation result: never write the same one twice."""
    return (int(step), str(dataset), str(caption_variant))


# --------------------------------------------------------------------------- #
# Phase 3.0A base objective: (I, C_S)-only batch + its logger payload
# --------------------------------------------------------------------------- #
GAP_BATCH_KEYS = ('images', 'texts_said')


def resolve_gap_mode(args):
    """Data-shape policy of the chosen objective, as one testable decision.

    Returns ``(gap_mode, tri_mode, caption_views, collate_fn)``:

    * ``gap_mode`` -- the Phase 3.0A base objective is running.
    * ``tri_mode`` -- the Phase 2.9A debiased-suffix Unsaid branch is running.
    * ``caption_views`` -- what ``share4v_train_dataset`` is asked for. The gap objective
      must get ``False``: that is exactly the ``(I, C_S)`` tuple, and it is the only way
      ``caption_full`` / ``caption_unsaid`` / ``has_unsaid`` can appear.
    * ``collate_fn`` -- ``tri_caption_collate`` only for the tri-view dataset; the gap run
      keeps the default collate, so it cannot even represent extra caption views.
    """
    tri_mode = args.unsaid_mode == 'debiased_suffix'
    gap_mode = args.objective_mode == 'gap_completion'
    caption_views = bool(tri_mode) and not gap_mode
    return gap_mode, tri_mode, caption_views, (tri_caption_collate if caption_views else None)


def resolve_objective_mode(args):
    """``(gap_mode, exgap_mode, tri_mode, caption_views, collate_fn)`` for the chosen objective.

    ``gap_completion``, ``said_exgap`` and ``said_exgap_finalcls`` all consume only ``(I, C_S)``, so
    they ask the dataset for ``caption_views=False`` and keep the default collate; only the Phase
    2.9A suffix branch builds the tri-caption view dicts.
    """
    objective = getattr(args, 'objective_mode', 'legacy')
    exgap_mode = objective in ('said_exgap', 'said_exgap_finalcls')
    gap_mode = objective == 'gap_completion'
    tri_mode = getattr(args, 'unsaid_mode', 'residual') == 'debiased_suffix'
    caption_views = bool(tri_mode) and not (gap_mode or exgap_mode)
    return gap_mode, exgap_mode, tri_mode, caption_views, \
        (tri_caption_collate if caption_views else None)


def resolve_finalcls_mode(args):
    """``True`` when the SAID-ExGAP v1.5 (pre-final routing + native final CLS) objective runs."""
    return getattr(args, 'objective_mode', 'legacy') == 'said_exgap_finalcls'


def build_exgap_log_fields(out):
    """Logger payload of one ``said_exgap`` step.

    ``loss_global`` / ``loss_unsaid`` do not exist in this objective (v1 has no global-text CLIP
    loss and no Phase 2.x Unsaid branch) and stay ``None`` so the JSON record carries ``null``.
    """
    def _value(key):
        value = out.get(key)
        if value is None:
            return None
        return float(value.detach()) if torch.is_tensor(value) else value

    fields = {key: _value(key) for key in (
        'loss_total', 'loss_said', 'loss_route', 'loss_evidence', 'loss_exgap',
        'loss_global', 'loss_unsaid',
        'S_gc_mean', 'S_sc_mean', 'S_uc_mean',
        'S_sc_minus_S_gc_mean', 'S_gc_minus_S_uc_mean', 'S_uc_minus_S_gc_mean', 'D_U_mean',
        'S_gc_p50', 'S_sc_p50', 'S_uc_p50',
        'explanatory_gap_raw_mean', 'explanatory_gap_norm_mean', 'gap_weight_mean',
        'explanatory_gap_positive_fraction',
        'gap_p25', 'gap_p50', 'gap_p75', 'gap_p90',
        'mask_keep_ratio', 'mask_drop_ratio',
        'said_above_threshold_fraction', 'masked_patch_count_mean',
        'masked_patch_count_min', 'masked_patch_count_max', 'fallback_fraction',
        'said_relevance_mean', 'said_relevance_std',
        'said_attention_entropy', 'said_effective_patch_count',
        'global_attention_entropy', 'global_effective_patch_count',
        'global_attention_l1_deviation', 'global_attention_max_deviation',
        'route_top1_acc', 'evidence_top1_acc', 'route_margin', 'evidence_margin',
        'global_feature_norm', 'said_feature_norm', 'unsaid_feature_norm',
        'router_input_feature_norm', 'global_raw_pool_norm', 'unsaid_raw_pool_norm',
        'patch_norm_deviation', 'exgap_valid_fraction',
        # collapse monitors (the Full-Base arm collapsed: pairwise cosine -> 0.977)
        'global_pairwise_cos', 'said_pairwise_cos', 'unsaid_pairwise_cos',
        'global_pairwise_cos_max', 'said_pairwise_cos_max', 'unsaid_pairwise_cos_max',
        # embedding spread on the fixed cohort (std of the cohort features, per dimension)
        'global_embedding_std', 'said_embedding_std', 'unsaid_embedding_std',
    )}
    fields['objective_mode'] = out.get('objective_mode')
    fields['said_loss_mode'] = out.get('said_loss_mode')
    fields['exgap_global_pool'] = out.get('exgap_global_pool')
    fields['exgap_mask_threshold'] = out.get('exgap_mask_threshold')
    fields['exgap_temperature'] = out.get('exgap_temperature')
    fields['lambda_said'] = out.get('lambda_said')
    fields['lambda_exgap'] = out.get('lambda_exgap')
    fields['global_text_alignment_enabled'] = bool(
        out.get('global_text_alignment_enabled', False))
    return fields


def build_finalcls_log_fields(out):
    """Logger payload of one ``said_exgap_finalcls`` (v1.5) step.

    No ``loss_global`` / ``loss_unsaid`` exist here (v1.5 has no global-text CLIP loss and no
    Phase 2.x Unsaid branch); they stay ``None`` so the record carries ``null`` instead of a
    fabricated number. ``said_fraction_gt_0_6`` is a DIAGNOSTIC threshold fraction, never the
    training mask: v1.5 trains with the live soft gate ``log r``.
    """
    def _value(key):
        value = out.get(key)
        if value is None:
            return None
        return float(value.detach()) if torch.is_tensor(value) else value

    fields = {key: _value(key) for key in (
        'loss_total', 'loss_said', 'loss_route', 'loss_evidence', 'loss_exgap',
        'loss_global', 'loss_unsaid',
        'S_gc_mean', 'S_sc_mean', 'S_uc_mean',
        'S_sc_minus_S_gc_mean', 'S_uc_minus_S_gc_mean', 'S_gc_minus_S_uc_mean', 'D_U_mean',
        'S_gc_p50', 'S_sc_p50', 'S_uc_p50',
        'explanatory_gap_raw_mean', 'explanatory_gap_norm_mean', 'gap_weight_mean',
        'explanatory_gap_positive_fraction',
        'gap_p10', 'gap_p25', 'gap_p50', 'gap_p75', 'gap_p90',
        'said_relevance_mean', 'said_relevance_std',
        'said_relevance_p10', 'said_relevance_p25', 'said_relevance_p50',
        'said_relevance_p75', 'said_relevance_p90',
        'said_fraction_gt_0_6', 'said_fraction_gt_0_7', 'diagnostic_threshold',
        'said_effective_patch_count', 'unsaid_effective_patch_count',
        'said_to_global_cos', 'unsaid_to_global_cos', 'said_to_unsaid_cos',
        'said_to_global_cos_std', 'said_minus_global_l2', 'unsaid_minus_global_l2',
        'route_top1_acc', 'evidence_top1_acc', 'route_margin', 'evidence_margin',
        'global_feature_norm', 'said_feature_norm', 'unsaid_feature_norm',
        'router_input_feature_norm', 'patch11_norm', 'exgap_valid_fraction',
        # native-CLS geometry on the fixed cohort (v1.5 primary representation)
        'global_cls_pairwise_cos', 'global_cls_pairwise_cos_max', 'global_cls_std',
        '64way_i2t_at1', '64way_t2i_at1',
        'global_identity_max_abs_diff', 'global_identity_relative',
    )}
    fields['objective_mode'] = out.get('objective_mode')
    fields['said_loss_mode'] = out.get('said_loss_mode')
    fields['finalcls_pair_chunk_size'] = out.get('finalcls_pair_chunk_size')
    fields['exgap_temperature'] = out.get('exgap_temperature')
    fields['lambda_said'] = out.get('lambda_said')
    fields['lambda_exgap'] = out.get('lambda_exgap')
    fields['global_text_alignment_enabled'] = bool(
        out.get('global_text_alignment_enabled', False))
    return fields


def build_gap_train_batch(batch, tokenize):
    """Build the Phase 3.0A base training batch from an ``(I, C_S)`` tuple.

    ``batch`` must be exactly the ``share4v_train_dataset(caption_views=False)`` item:
    a 2-tuple ``(image_tensor, caption_said)``. Any other shape -- in particular a
    tri-caption dict carrying ``caption_full`` / ``caption_unsaid`` / ``has_unsaid`` --
    is rejected instead of being silently fed to the gap objective, because the gap
    objective is defined on ``Image I`` + ``C_S`` only.

    ``tokenize`` is called exactly once, on ``C_S``: the full caption and the withheld
    suffix are never text-encoded anywhere in this path.
    """
    if isinstance(batch, dict):
        raise ValueError('gap_completion takes an (I, C_S) tuple batch, got a dict with keys %r'
                         % (sorted(batch),))
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise ValueError('gap_completion takes an (I, C_S) tuple batch, got %r'
                         % (type(batch).__name__,))
    images, texts_said = batch
    if isinstance(texts_said, torch.Tensor):
        raise ValueError('gap_completion expects caption strings as C_S, got a tensor')
    return {'images': images, 'texts_said': tokenize(list(texts_said))}


def build_gap_log_fields(out):
    """Logger payload of one gap_completion step.

    ``loss_global`` / ``loss_unsaid`` do not exist in this objective and stay ``None`` so
    that the JSON record carries ``null`` instead of a fabricated number -- and so that no
    caller ever evaluates ``out['loss_global'].detach()`` on a missing term.

    ``global_feature_norm`` / ``said_feature_norm`` / ``unsaid_feature_norm`` are L2 norms
    of *already normalised* representations, so they are ~1 by construction and are kept
    only for compatibility: they are not magnitude diagnostics.
    """
    def _value(key):
        value = out.get(key)
        if value is None:
            return None
        return float(value.detach()) if torch.is_tensor(value) else value

    fields = {key: _value(key) for key in (
        'loss_global', 'loss_unsaid',
        'loss_said', 'loss_gap_discover', 'loss_global_absorb', 'loss_total',
        'loss_route', 'loss_evidence',
        'route_top1_acc', 'evidence_top1_acc', 'route_margin', 'evidence_margin',
        'gap_before_mean', 'gap_after_mean', 'gap_reduction_mean',
        'gap_closure_ratio_mean', 'gap_closure_positive_fraction',
        'said_unsaid_feature_cosine', 'unsaid_novel_component_norm',
        'said_attention_entropy', 'unsaid_attention_entropy',
        'said_unsaid_attention_overlap', 'said_unsaid_attention_jsd',
        'said_attention_max', 'said_attention_min',
        'global_feature_norm', 'said_feature_norm', 'unsaid_feature_norm',
        'router_input_feature_norm',
        # Phase 3.0A.1c complement-emergence diagnostics (monitoring only)
        'patch_pair_cosine_mean', 'patch_pair_cosine_std',
        'patch_to_global_cosine_mean', 'patch_to_global_cosine_std',
        'patch_centered_energy',
        'said_raw_pool_norm', 'unsaid_raw_pool_norm',
        'raw_pool_cosine', 'raw_pool_norm_ratio',
        'cos_global_said', 'cos_global_unsaid', 'cos_said_unsaid',
        # every effective weight is recorded, including a deliberate zero
        'lambda_said', 'lambda_gap_discover', 'lambda_global_absorb',
    )}
    fields['objective_mode'] = out.get('objective_mode')
    fields['said_loss_mode'] = out.get('said_loss_mode')
    fields['gap_anti_temperature'] = out.get('gap_anti_temperature')
    # explicit, non-overloaded semantics for this objective
    fields['visual_complement_enabled'] = bool(out.get('visual_complement_enabled', False))
    fields['legacy_unsaid_enabled'] = bool(out.get('legacy_unsaid_enabled', False))
    fields['global_text_alignment_enabled'] = bool(out.get('global_text_alignment_enabled', False))
    return fields


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


def objective_checkpoint_metadata(args):
    """``phase`` / ``objective_mode`` written into a checkpoint.

    Compatibility rule: ``legacy`` runs keep their historical phase tag and do **not** grow
    an ``objective_mode`` field, so every pre-Phase-3.0A checkpoint stays readable byte for
    byte. A ``gap_completion`` run is tagged as its own phase and carries the objective, so
    a later resume can refuse to silently mix objectives.
    """
    if getattr(args, 'objective_mode', 'legacy') == 'gap_completion':
        return 'phase3.0a-gap-completion', 'gap_completion'
    if getattr(args, 'objective_mode', 'legacy') == 'said_exgap':
        return 'said-exgap-v1', 'said_exgap'
    if getattr(args, 'objective_mode', 'legacy') == 'said_exgap_finalcls':
        return 'said-exgap-v1.5-finalcls', 'said_exgap_finalcls'
    return 'phase2-said-only', None


def validate_resume_objective(checkpoint, args, path='<checkpoint>'):
    """Refuse to resume across objectives.

    A checkpoint **with** ``objective_mode`` must match the CLI exactly. A checkpoint
    **without** it predates Phase 3.0A and is therefore a legacy run: ``legacy`` may still
    resume it, ``gap_completion`` may not (it would silently continue the wrong objective).
    """
    current = getattr(args, 'objective_mode', 'legacy')
    stored = checkpoint.get('objective_mode')
    if stored is None:
        if current != 'legacy':
            raise ValueError(
                "cannot resume a legacy checkpoint (no objective_mode metadata) with "
                "--objective_mode %s: %s" % (current, path))
        return 'legacy'
    if str(stored) != str(current):
        raise ValueError('resume objective mismatch: checkpoint declares objective_mode=%r '
                         'but the CLI asks for %r (%s)' % (stored, current, path))
    # an ExGAP run is only comparable under the same configuration, so refuse a silent change
    if current == 'said_exgap':
        stored_config = checkpoint.get('exgap_config')
        if stored_config is None:
            raise ValueError('resume of a said_exgap checkpoint without exgap_config: %s' % path)
        current_config = exgap.checkpoint_config(
            global_pool=args.exgap_global_pool, mask_threshold=args.exgap_mask_threshold,
            temperature=args.exgap_temperature, lambda_said=args.lambda_said,
            lambda_exgap=args.lambda_exgap, normalize_gap=args.exgap_normalize_gap)
        differences = {key: (stored_config.get(key), current_config[key])
                       for key in current_config if stored_config.get(key) != current_config[key]}
        if differences:
            raise ValueError('resume exgap_config mismatch (stored, current): %r (%s)'
                             % (differences, path))
    if current == 'said_exgap_finalcls':
        stored_config = checkpoint.get('finalcls_config')
        if stored_config is None:
            raise ValueError('resume of a said_exgap_finalcls checkpoint without finalcls_config: '
                             '%s' % path)
        current_config = finalcls_checkpoint_config(args)
        differences = {key: (stored_config.get(key), current_config[key])
                       for key in current_config if stored_config.get(key) != current_config[key]}
        if differences:
            raise ValueError('resume finalcls_config mismatch (stored, current): %r (%s)'
                             % (differences, path))
    return str(stored)


def finalcls_checkpoint_config(args):
    """Configuration block that makes two v1.5 runs comparable (travels with the checkpoint)."""
    return {
        'objective_mode': 'said_exgap_finalcls',
        'pair_chunk_size': int(getattr(args, 'finalcls_pair_chunk_size', 32)),
        'temperature': float(args.exgap_temperature),
        'lambda_said': float(args.lambda_said),
        'lambda_exgap': float(args.lambda_exgap),
        'normalize_gap': bool(args.exgap_normalize_gap),
        'route_source': 'H11_raw -> ln_post -> proj',
        'readout': 'original final block, CLS row only',
    }


def validate_finalcls_args(args):
    """Reject an illegal SAID-ExGAP v1.5 configuration before any model or data is built."""
    problems = []
    if int(getattr(args, 'finalcls_pair_chunk_size', 32)) <= 0:
        problems.append('--finalcls-pair-chunk-size must be positive, got %r'
                        % (args.finalcls_pair_chunk_size,))
    if float(args.exgap_temperature) <= 0.0:
        problems.append('--exgap-temperature must be positive, got %r'
                        % (args.exgap_temperature,))
    if float(args.lambda_said) <= 0.0:
        problems.append('--lambda-said must be positive, got %r' % (args.lambda_said,))
    if not 0.0 <= float(args.lambda_exgap) < float('inf'):
        problems.append('--lambda-exgap must be finite and >= 0, got %r' % (args.lambda_exgap,))
    if float(args.lambda_global) != 0.0:
        problems.append('--lambda-global must be 0 for said_exgap_finalcls (v1.5 has no '
                        'global-text CLIP loss), got %r' % (args.lambda_global,))
    if float(args.lambda_unsaid) != 0.0:
        problems.append('--lambda-unsaid must be 0 for said_exgap_finalcls (no Phase 2.x Unsaid '
                        'branch), got %r' % (args.lambda_unsaid,))
    if problems:
        raise ValueError('illegal said_exgap_finalcls configuration: ' + '; '.join(problems))


def save_checkpoint(path, ddp_model, optimizer, scaler, step, epoch, args, base_lrs, total_steps):
    phase, objective_mode = objective_checkpoint_metadata(args)
    payload = {
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
        'phase': phase,
    }
    if objective_mode is not None:
        payload['objective_mode'] = objective_mode
    if objective_mode == 'said_exgap':
        # SAID-ExGAP is only comparable to other runs under the same configuration, so the
        # whole exgap_config block travels with the checkpoint and is compared on resume.
        payload['exgap_config'] = exgap.checkpoint_config(
            global_pool=args.exgap_global_pool,
            mask_threshold=args.exgap_mask_threshold,
            temperature=args.exgap_temperature,
            lambda_said=args.lambda_said,
            lambda_exgap=args.lambda_exgap,
            normalize_gap=args.exgap_normalize_gap,
        )
    if objective_mode == 'said_exgap_finalcls':
        payload['finalcls_config'] = finalcls_checkpoint_config(args)
    torch.save(payload, path)
    # a plain CLIP-only state dict so standard tooling (eval/retrieval/coco.py)
    # can load the backbone without knowing about SALU
    clip_path = os.path.join(os.path.dirname(path), 'clip_only_' + os.path.basename(path))
    torch.save(ddp_model.module.clip.state_dict(), clip_path)


def batch_identity_sha256(images, captions):
    """Content hash of one training batch ``(I, C_S)`` (Phase 3.0A.1c sweeps).

    Used to prove that every temperature / feature-source probe saw the *same* images and
    the *same* prefix captions, so a difference in the metrics can only come from the
    setting under test.
    """
    digest = hashlib.sha256()
    if torch.is_tensor(images):
        digest.update(images.detach().float().cpu().contiguous().numpy().tobytes())
    else:
        digest.update(repr(images).encode('utf-8'))
    digest.update(b'\x00')
    for caption in captions:
        digest.update(str(caption).encode('utf-8'))
        digest.update(b'\x01')
    return digest.hexdigest()


def parse_step_list(value):
    """``'0,20,50,100'`` -> ``[0, 20, 50, 100]`` (empty string -> no diagnostic steps)."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(part) for part in str(value).split(',') if str(part).strip()]


def unwrap_model(model):
    """Return the underlying ``SALUModel`` whether or not ``model`` is DDP-wrapped."""
    return getattr(model, 'module', model)


def ddp_no_sync(model):
    """``model.no_sync()`` for a DDP wrapper, a no-op context otherwise."""
    if dist.is_available() and dist.is_initialized() and hasattr(model, 'no_sync'):
        return model.no_sync()
    return contextlib.nullcontext()


def grad_norm_summary(model):
    """L2 norm of the accumulated gradients, split backbone vs Said router.

    Reads the gradients of the *single* training backward pass: no extra backward is run.
    Returns ``grad_norm_backbone``, ``grad_norm_said_router``, ``grad_norm_total`` and the
    number of tensors that actually received a gradient. ``model`` may be the raw
    ``SALUModel`` or its DDP wrapper.
    """
    model = unwrap_model(model)
    head_ids = {id(parameter) for parameter in model.said_router.parameters()}
    sums = {'backbone': 0.0, 'router': 0.0}
    counts = {'backbone': 0, 'router': 0}
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        group = 'router' if id(parameter) in head_ids else 'backbone'
        value = float(parameter.grad.detach().float().pow(2).sum())
        sums[group] += value
        counts[group] += 1
    total = math.sqrt(sums['backbone'] + sums['router'])
    return {
        'grad_norm_backbone': math.sqrt(sums['backbone']),
        'grad_norm_said_router': math.sqrt(sums['router']),
        'grad_norm_total': total,
        'grad_tensor_count_backbone': counts['backbone'],
        'grad_tensor_count_said_router': counts['router'],
    }


def backward_and_step(loss, optimizer, scaler, retain_graph=False):
    """One optimizer step with the same scaler semantics as before (extracted verbatim).

    ``retain_graph`` is only ever true on the Phase 3.0A diagnostic steps, where the
    per-term gradient norms need to differentiate the same forward graph a second time.
    """
    if scaler.is_enabled():
        scaler.scale(loss).backward(retain_graph=retain_graph)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward(retain_graph=retain_graph)
        optimizer.step()


def grad_contribution_norms(model, images, text_tokens, args, diagnostic_steps, step):
    """Per-term gradient norms on *diagnostic steps only* (Phase 3.0A.1c, Section 11).

    ``grad_norm_gap_contribution`` / ``grad_norm_absorb_contribution`` are measured on a
    **fresh forward graph** built here, not on the training graph. Reusing the training
    graph is not viable: DDP's reducer releases it during the training backward (verified
    with ``find_unused_parameters=True`` on gloo), so a second backward through it raises
    "Trying to backward through the graph a second time".

    The cost is one extra forward + backward per term, and only on the requested steps
    (step 0 / 20 / 50 / 100 in the 100-step diagnostic); ``{}`` is returned on every other
    step. The passes run inside ``no_sync`` (when DDP is active) and consume their own
    graph, so the real training step and its gradient synchronisation are untouched. When
    a weight is zero the corresponding term is absent from ``loss_total``, which is exactly
    the matched control being measured.
    """
    if step not in set(diagnostic_steps):
        return {}
    contributions = {}
    for name, weight in (('gap', args.lambda_gap_discover),
                         ('absorb', args.lambda_global_absorb)):
        model.zero_grad(set_to_none=True)
        # a fresh context per term: contexts are single-use and the first backward frees
        # the graph it built
        with ddp_no_sync(model):
            # a fresh graph: enable_grad because the caller may sit inside no_grad
            with torch.enable_grad():
                out = model(images, text_tokens, 0.0, args.lambda_said, lambda_unsaid=0.0,
                            objective_mode='gap_completion',
                            lambda_gap_discover=args.lambda_gap_discover,
                            lambda_global_absorb=args.lambda_global_absorb,
                            gap_anti_temperature=args.gap_anti_temperature)
                term = out['loss_gap_discover'] if name == 'gap' else out['loss_global_absorb']
                weighted = term if float(weight) != 0.0 else term * 1.0
                weighted.backward(retain_graph=False)
        summary = grad_norm_summary(model)
        contributions['grad_norm_%s_contribution' % name] = summary['grad_norm_total']
        contributions['grad_norm_%s_contribution_backbone' % name] = summary['grad_norm_backbone']
        contributions['grad_norm_%s_contribution_router' % name] = summary['grad_norm_said_router']
        contributions['grad_norm_%s_contribution_weight' % name] = float(weight)
        del out, term, weighted
    model.zero_grad(set_to_none=True)
    return contributions


def exgap_grad_attribution(module, images, text_tokens, args, steps, completed):
    """Per-term gradient norms for SAID-ExGAP on diagnostic steps only (Phase ExGAP-1B).

    Measures, on a **fresh** graph built from the unwrapped module (so DDP's reducer is never
    involved), the L2 norm of ``d L_S`` and of ``d L_ExGAP`` for every parameter group, and the
    ratio ``R_grad = G_ExGAP / (G_S + eps)``. ``torch.autograd.grad`` is used per term instead of
    two ``backward`` calls, so neither term's gradient can contaminate the other and the training
    step's own gradients/synchronisation are untouched.

    The ExGAP term is measured *even when* ``lambda_exgap == 0`` (the matched-control arm M0):
    the forward always builds the full geometry, so ``G_ExGAP`` there answers "what would ExGAP
    have done", while the training step itself ignores it. ``R_grad`` in M0 is therefore a
    diagnostic, not an applied gradient.

    Returns ``{}`` on every step that is not in ``steps``.
    """
    if completed not in set(steps):
        return {}
    objective = getattr(args, 'objective_mode', 'said_exgap')
    groups = exgap_grad_groups(module)
    if objective == 'said_exgap_finalcls':
        forward_kwargs = dict(objective_mode='said_exgap_finalcls',
                              lambda_exgap=args.lambda_exgap,
                              exgap_temperature=args.exgap_temperature,
                              exgap_normalize_gap=args.exgap_normalize_gap,
                              finalcls_pair_chunk_size=args.finalcls_pair_chunk_size)
    else:
        forward_kwargs = dict(objective_mode='said_exgap',
                              lambda_exgap=args.lambda_exgap,
                              exgap_global_pool=args.exgap_global_pool,
                              exgap_mask_threshold=args.exgap_mask_threshold,
                              exgap_temperature=args.exgap_temperature,
                              exgap_normalize_gap=args.exgap_normalize_gap)
    with torch.enable_grad():
        out = module(images, text_tokens, 0.0, args.lambda_said, lambda_unsaid=0.0,
                     **forward_kwargs)
        record = {'grad_attr_objective_mode': objective}
        norms = {}
        for term_name, term in (('said', out['loss_said']), ('exgap', out['loss_exgap'])):
            term_norms = {}
            for group, parameters in groups.items():
                grads = torch.autograd.grad(term, parameters, retain_graph=True,
                                            allow_unused=True)
                total = 0.0
                for grad in grads:
                    if grad is not None:
                        total += float(grad.detach().float().pow(2).sum())
                term_norms[group] = math.sqrt(total)
            norms[term_name] = term_norms
            if term_name == 'said':
                record['grad_attr_loss_said'] = float(term.detach())
            else:
                record['grad_attr_loss_exgap'] = float(term.detach())
        for group in groups:
            g_s = norms['said'][group]
            g_e = norms['exgap'][group]
            record['grad_attr_gS_%s' % group] = g_s
            record['grad_attr_gE_%s' % group] = g_e
            record['grad_attr_R_%s' % group] = g_e / (g_s + 1e-12)
        record['grad_attr_lambda_exgap'] = float(args.lambda_exgap)
        record['grad_attr_groups'] = ';'.join(
            '%s=%d' % (group, len(parameters)) for group, parameters in groups.items())
        del out
    return record


def exgap_grad_groups(module):
    """Parameter groups used by the ExGAP gradient attribution.

    The real architecture has no separate patch projection: the patch tokens come out of the ViT
    itself, so the "patch pathway" IS ``clip.visual`` there. The test stub does expose
    ``clip.patch_proj`` / ``clip.global_proj``, so both spellings are classified, and
    ``patch_pathway`` is listed explicitly because the contract under test is "L_ExGAP reaches
    the shared visual backbone and nothing else".
    """
    visual_prefixes = ('clip.visual.', 'clip.patch_proj', 'clip.global_proj')
    patch_prefixes = ('clip.visual.', 'clip.patch_proj')
    final_block_prefix = 'clip.visual.transformer.resblocks.11.'
    text_prefixes = ('clip.transformer.', 'clip.token_embedding.', 'clip.positional_embedding',
                     'clip.ln_final', 'clip.text_projection', 'clip.text_proj')
    groups = {'visual_backbone': [], 'patch_pathway': [], 'final_block': [], 'text_encoder': [],
              'said_router': [], 'exgap_pooling': [], 'other': []}
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith('said_router.'):
            groups['said_router'].append(parameter)
        elif name.startswith('exgap.'):
            groups['exgap_pooling'].append(parameter)
        elif name.startswith(text_prefixes):
            groups['text_encoder'].append(parameter)
        else:
            if name.startswith(visual_prefixes):
                groups['visual_backbone'].append(parameter)
            if name.startswith(patch_prefixes):
                groups['patch_pathway'].append(parameter)
            if name.startswith(final_block_prefix):
                groups['final_block'].append(parameter)
            if not name.startswith(visual_prefixes):
                groups['other'].append(parameter)
    return {group: parameters for group, parameters in groups.items() if parameters}


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
    parser.add_argument('--lambda_said', '--lambda-said', dest='lambda_said', type=float,
                        default=1.0,
                        help='weight of L_S (SAID-ExGAP accepts the hyphenated spelling too)')
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
    parser.add_argument('--save_at', default='',                        help='comma-separated steps to checkpoint, e.g. "200,400"')
    parser.add_argument('--save_completed_steps', '--save-completed-steps',
                        dest='save_completed_steps', default='',
                        help='comma-separated numbers of COMPLETED optimizer updates to '
                             'checkpoint as salu_exgap_step%06d.pt; 0 saves the initial state '
                             'before the first update (matched-control provenance)')
    parser.add_argument('--grad_attribution_steps', '--grad-attribution-steps',
                        dest='grad_attribution_steps', default='20,100,500',
                        help='completed-step numbers at which the per-term gradient attribution '
                             '(G_S / G_ExGAP / R_grad) is measured on a fresh diagnostic graph')
    parser.add_argument('--finalcls_pair_chunk_size', '--finalcls-pair-chunk-size',
                        dest='finalcls_pair_chunk_size', type=int, default=32,
                        help='caption candidates per chunk for the v1.5 pairwise final-CLS readout '
                             '(never materialises [B, B, 197, 768])')
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
    # ------------------------------------------------------------------ #
    # Phase 3.0A base objective: Said-Conditioned Visual Complement Discovery
    # ------------------------------------------------------------------ #
    parser.add_argument('--objective_mode', default='legacy',
                        choices=['legacy', 'gap_completion', 'said_exgap', 'said_exgap_finalcls'],
                        help="legacy = Phase 2 losses (L_global + L_said [+ L_unsaid]); "
                             "gap_completion = Phase 3.0A base objective on (I, C_S) only; "
                             "said_exgap = v1 (patch-space ExGAP); said_exgap_finalcls = v1.5 "
                             "(pre-final H11 routing + native final-block CLS readout)")
    parser.add_argument('--lambda_gap_discover', type=float, default=1.0,
                        help='weight of L_gap_discover = mean(gap_after) (gap_completion only)')
    parser.add_argument('--lambda_global_absorb', type=float, default=1.0,
                        help='weight of L_global_absorb (gap_completion only)')
    parser.add_argument('--gap_anti_temperature', type=float, default=1.0,
                        help='temperature of the soft anti-Said attention A_U (gap_completion only)')
    parser.add_argument('--grad_contribution_steps', type=parse_step_list, default=[0],
                        help='comma-separated steps at which the extra per-term gradient '
                             'norms are measured (gap_completion only; each costs one '
                             'backward pass). Empty string disables them.')
    # ------------------------------------------------------------------ #
    # SAID-ExGAP v1: ExGAP learning. v1 has NO global-text CLIP loss.
    # ------------------------------------------------------------------ #
    parser.add_argument('--exgap-global-pool', default='mean',
                        choices=list(exgap.GLOBAL_POOL_MODES),
                        help="global pooling: 'mean' (parameter-free ablation) or 'attention' "
                             '(image-only attention pooling, exactly uniform at init)')
    parser.add_argument('--exgap-mask-threshold', type=float,
                        default=exgap.DEFAULT_MASK_THRESHOLD,
                        help='tau_M: patches with said relevance <= tau_M stay available to the '
                             'masked complementary representation (default 0.6)')
    parser.add_argument('--exgap-temperature', type=float,
                        default=exgap.DEFAULT_GAP_TEMPERATURE,
                        help='tau of the softplus ranking term in L_ExGAP (default 0.05)')
    parser.add_argument('--exgap-normalize-gap', dest='exgap_normalize_gap',
                        action='store_true', default=True,
                        help='divide the explanatory gap by (1 - S_gc) (default: on)')
    parser.add_argument('--no-exgap-normalize-gap', dest='exgap_normalize_gap',
                        action='store_false',
                        help='use the raw positive part of the explanatory gap instead')
    parser.add_argument('--lambda-exgap', type=float, default=1.0,
                        help='weight of L_ExGAP (said_exgap only)')
    parser.add_argument('--exgap-collapse-every', type=int, default=50,
                        help='compute the representation-collapse monitors every N steps '
                             '(0 = never); they use a fixed image cohort')
    return parser.parse_args(argv)


def validate_objective_args(args):
    """Reject a Phase 3.0A configuration that would silently mix objectives.

    The base gap objective is built from ``Image I`` + ``C_S`` alone, so a Global-text
    InfoNCE term, a Phase 2.8A residual Unsaid term, a Phase 2.9A candidate conditioned
    Unsaid term or a full-caption alignment would all change what the run *is*. They are
    therefore configuration errors, raised before any model or data is built.
    """
    if args.objective_mode == 'said_exgap':
        # SAID-ExGAP v1 has no global-text CLIP loss and no Phase 2.x Unsaid branch, and it
        # reads Global / Said / Masked-Unsaid from one patch set, so those weights are errors.
        exgap.validate_exgap_config(args)
        return args
    if args.objective_mode == 'said_exgap_finalcls':
        # v1.5 keeps the v1 loss contract and adds the pre-final routing / native final-CLS
        # configuration, so both validators run.
        validate_finalcls_args(args)
        return args
    if args.objective_mode != 'gap_completion':
        return args
    problems = []
    if float(args.lambda_global) != 0.0:
        problems.append('--lambda_global must be 0 for --objective_mode gap_completion, got %r'
                        % (args.lambda_global,))
    if float(args.lambda_unsaid) != 0.0:
        problems.append('--lambda_unsaid must be 0 for --objective_mode gap_completion, got %r'
                        % (args.lambda_unsaid,))
    if args.global_caption_view != 'prefix':
        problems.append("--global_caption_view must be 'prefix' for --objective_mode "
                        'gap_completion, got %r' % (args.global_caption_view,))
    if float(args.lambda_said) <= 0.0:
        problems.append('--lambda_said must be positive; the Said router is trained by L_said')
    # A zero gap / absorb weight is legal and meaningful: it is the matched control arm.
    #   L_S            -> (0, 0)   Said-only control
    #   L_S + L_gap    -> (1, 0)   discovery only
    #   L_S + L_gap + L_absorb -> (1, 1)  full base objective
    # Only finiteness and non-negativity are required.
    for flag, value in (('--lambda_gap_discover', args.lambda_gap_discover),
                        ('--lambda_global_absorb', args.lambda_global_absorb)):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            problems.append('%s must be finite and >= 0 (0 = matched control), got %r'
                            % (flag, value))
    if float(args.gap_anti_temperature) <= 0.0:
        problems.append('--gap_anti_temperature must be positive, got %r'
                        % (args.gap_anti_temperature,))
    if problems:
        raise ValueError('illegal gap_completion configuration: ' + '; '.join(problems))
    return args


def main():
    args = parse_args()
    # fail on an illegal Phase 3.0A configuration before any DDP / model / data setup
    validate_objective_args(args)

    if args.base_model == 'B16':
        args.base_model = 'ViT-B/16'
    elif args.base_model == 'L14':
        args.base_model = 'ViT-L/14'
    save_at = sorted({int(s) for s in args.save_at.split(',') if s.strip()})
    save_completed = sorted({int(s) for s in args.save_completed_steps.split(',') if s.strip()})
    grad_attribution_steps = {int(s) for s in args.grad_attribution_steps.split(',') if s.strip()}

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
    if resolve_objective_mode(args)[1]:
        # SAID-ExGAP: register the image-only attention pooling agent *before* the initial-state
        # digest and before DDP, for both pooling modes. Doing it here (rather than lazily in the
        # first forward) keeps the agent inside the process group -- otherwise it is created on
        # the CPU after DDP wrapping, which both raises a device-mismatch error and silently
        # leaves its gradients unsynchronised -- and makes the recorded initial state identical
        # across the mean / attention arms that have to be compared.
        salu.build_exgap_module()
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
    # Phase 3.0A base objective: the DataLoader must hand over (I, C_S) and nothing else.
    # ``share4v_train_dataset`` defaults to exactly that tuple; ``caption_views=True`` is
    # the only producer of C_F / C_U / has_unsaid, so the gap run asks for False and keeps
    # the default collate (never ``tri_caption_collate``).
    gap_mode, exgap_mode, tri_mode, caption_views, train_collate = resolve_objective_mode(args)
    finalcls_mode = resolve_finalcls_mode(args)
    train_set = share4v_train_dataset(caption_views=caption_views, suffix_seed=args.seed)
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
        collate_fn=train_collate,
    )
    steps_per_epoch = len(loader)
    stop_steps, total_steps = resolve_total_steps(args, steps_per_epoch)
    if rank == 0:
        print('steps_per_epoch %d stop_steps %d lr_total_steps %d lr_warmup %d'
              % (steps_per_epoch, stop_steps, total_steps, args.warmup_length), flush=True)
    global_batch = args.batch_size * world_size

    # ---------------------------------------------------------------- #
    # SAID-ExGAP: fixed 64-image cohort for the representation-collapse monitors. The Full-Base
    # arm collapsed (inter-image cosine -> 0.977) while its internal losses looked perfect, so
    # the geometry is monitored by default. The cohort is built from the *dataset* (only the
    # preprocess transform, no image augmentation) so it is deterministic and identical on every
    # rank; it never participates in training.
    # ---------------------------------------------------------------- #
    collapse_cohort = None
    collapse_cohort_texts = None
    if exgap_mode and args.exgap_collapse_every:
        try:
            indices = list(range(64))
            tensors = []
            captions = []
            for index in indices:
                sample = train_set[index]
                tensors.append(sample[0] if isinstance(sample, (tuple, list)) else sample['image'])
                captions.append(sample[1] if isinstance(sample, (tuple, list))
                                else sample['caption_said'])
            collapse_cohort = torch.stack(tensors).to(device)
            # v1.5 also reports the 64-way image-to-text R@1 on this cohort, which needs the
            # cohort's own captions. Tokenised once, on rank 0 only (identical on every rank).
            if finalcls_mode and rank == 0:
                collapse_cohort_texts = longclip.tokenize(
                    captions, truncate=True).to(device)
            if rank == 0:
                print('COLLAPSE_COHORT %r from dataset indices 0-%d texts=%s'
                      % (tuple(collapse_cohort.shape), indices[-1],
                         'yes' if collapse_cohort_texts is not None else 'no'), flush=True)
        except Exception as error:      # never let monitoring break training
            collapse_cohort = None
            collapse_cohort_texts = None
            if rank == 0:
                print('COLLAPSE_COHORT unavailable: %r' % (error,), flush=True)

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
        validate_resume_objective(ckpt, args, args.resume)
        if ckpt.get('args', {}).get('said_feature_source', 'residual') != args.said_feature_source:
            raise ValueError('resume checkpoint feature source differs from CLI')
        if ckpt.get('args', {}).get('gap_anti_temperature') != args.gap_anti_temperature:
            raise ValueError('resume checkpoint gap_anti_temperature differs from CLI')
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
        initial_phase, initial_objective = objective_checkpoint_metadata(args)
        initial_payload = {'model': salu.state_dict(), 'args': vars(args), 'step': 0,
                           'phase': initial_phase}
        if initial_objective is not None:
            initial_payload['objective_mode'] = initial_objective
        torch.save(initial_payload, os.path.join(args.output_dir, 'salu_initial.pt'))
    if rank == 0 and 0 in save_completed and not args.resume:
        # completed-step 0 = the exact initial state, before any optimizer update. Saved with the
        # same file-name convention as every other completed step so a matched-control pair can be
        # compared as (step 0, step 100, step 500) without special cases.
        initial_phase, initial_objective = objective_checkpoint_metadata(args)
        initial_payload = {'model': salu.state_dict(), 'args': vars(args), 'step': 0,
                           'phase': initial_phase}
        if initial_objective is not None:
            initial_payload['objective_mode'] = initial_objective
        initial_config = exgap.checkpoint_config(
            args.exgap_global_pool, args.exgap_mask_threshold, args.exgap_temperature,
            args.lambda_said, args.lambda_exgap, args.exgap_normalize_gap)
        initial_payload['exgap_config'] = initial_config
        initial_path = os.path.join(args.output_dir, 'salu_exgap_step%06d.pt' % 0)
        torch.save(initial_payload, initial_path)
        print('SAVED_EXGAP_STEP ' + initial_path, flush=True)
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
            gap_batch = None
            if exgap_mode:
                # SAID-ExGAP v1 / v1.5 consume exactly (I, C): no full caption, no unsaid text
                gap_batch = build_gap_train_batch(
                    batch, lambda captions: longclip.tokenize(captions, truncate=True).to(device))
                images = gap_batch['images']
                texts = batch[1]
            elif gap_mode:
                # Phase 3.0A: exactly (I, C_S); C_F and C_U are never constructed here.
                gap_batch = build_gap_train_batch(
                    batch, lambda captions: longclip.tokenize(captions, truncate=True).to(device))
                images = gap_batch['images']
                texts = batch[1]
            elif tri_mode:
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
                if tri_mode and not gap_mode and not exgap_mode:
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
            if tri_mode and not gap_mode and not exgap_mode:
                update_caption_digest(full_digest, batch['caption_full'])
                update_caption_digest(unsaid_digest, batch['caption_unsaid'])
                update_caption_digest(has_unsaid_digest,
                                      ['%s' % bool(value) for value in batch['has_unsaid']])
            if gap_mode or exgap_mode:
                # C_S was tokenized exactly once, inside build_gap_train_batch
                text_tokens = gap_batch['texts_said']
                full_tokens = unsaid_tokens = None
            else:
                text_tokens = longclip.tokenize(texts, truncate=True).to(device)
                full_tokens = (longclip.tokenize(texts_full, truncate=True).to(device)
                               if texts_full is not None else None)
                unsaid_tokens = (longclip.tokenize(texts_unsaid, truncate=True).to(device)
                                 if texts_unsaid is not None else None)

            scale_factor = set_lrs(optimizer, base_lrs, step, args.warmup_length, total_steps)
            t_compute0 = time.time()
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                if exgap_mode:
                    # ExGAP gets (I, C) plus its own hyper-parameters; no global / Unsaid term.
                    # v1.5 keeps the same outer contract and swaps the representation family.
                    out = ddp_model(images, text_tokens, 0.0, args.lambda_said,
                                    lambda_unsaid=0.0,
                                    objective_mode=('said_exgap_finalcls' if finalcls_mode
                                                    else 'said_exgap'),
                                    lambda_exgap=args.lambda_exgap,
                                    exgap_global_pool=args.exgap_global_pool,
                                    exgap_mask_threshold=args.exgap_mask_threshold,
                                    exgap_temperature=args.exgap_temperature,
                                    exgap_normalize_gap=args.exgap_normalize_gap,
                                    finalcls_pair_chunk_size=args.finalcls_pair_chunk_size,
                                    collapse_cohort=collapse_cohort
                                    if (args.exgap_collapse_every
                                        and step % args.exgap_collapse_every == 0) else None,
                                    collapse_cohort_texts=collapse_cohort_texts
                                    if (finalcls_mode and args.exgap_collapse_every
                                        and step % args.exgap_collapse_every == 0) else None,
                                    global_identity_check=(finalcls_mode and step == 0))
                elif gap_mode:
                    # the gap objective gets (I, C_S) and its own weights; no texts_full,
                    # no texts_unsaid, no has_unsaid, no global / Unsaid term
                    out = ddp_model(images, text_tokens, 0.0, args.lambda_said,
                                    lambda_unsaid=0.0,
                                    objective_mode='gap_completion',
                                    lambda_gap_discover=args.lambda_gap_discover,
                                    lambda_global_absorb=args.lambda_global_absorb,
                                    gap_anti_temperature=args.gap_anti_temperature)
                else:
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
                # fp16 path
                backward_and_step(loss, optimizer, scaler)
            else:
                # bf16 / fp32 path
                loss.backward()
                optimizer.step()
            # cheap: reads the gradients of the training backward pass, no extra backward
            grad_summary = grad_norm_summary(salu) if gap_mode else None
            optimizer.zero_grad(set_to_none=True)
            # Phase ExGAP-1B: per-term gradient attribution on a fresh diagnostic graph, on the
            # requested completed steps only. Runs after the training step (its own graph), on the
            # unwrapped module, so it can never steer the matched-control pair.
            grad_attribution = (exgap_grad_attribution(salu, images, text_tokens, args,
                                                       grad_attribution_steps, step + 1)
                                if exgap_mode else None)
            # per-term diagnostics build their own graph, so only on the requested steps
            grad_contributions = (grad_contribution_norms(ddp_model, images, text_tokens,
                                                          args, args.grad_contribution_steps,
                                                          step)
                                  if gap_mode else None)
            compute_times.append(time.time() - t_compute0)
            wall_times.append(time.time() - t_wall0)

            is_last_step = (
                (args.max_steps is not None and step == args.max_steps - 1)
                or (args.max_steps is None and step == stop_steps - 1)
            )
            if rank == 0 and (step % args.log_every == 0 or step == 0 or is_last_step
                              or step + 1 in save_at or step + 1 in grad_attribution_steps):
                throughput = summarize_throughput(global_batch, compute_times, wall_times)
                if finalcls_mode:
                    # SAID-ExGAP v1.5 payload: the native final-CLS readout is the primary
                    # representation, so the collapse monitor is the NATIVE CLS geometry.
                    record = {
                        'step': step,
                        'completed_steps': step + 1,
                        'epoch': epoch,
                        'lr_scale': scale_factor,
                        'backbone_lr': optimizer.param_groups[0]['lr'],
                        'head_lr': optimizer.param_groups[1]['lr'],
                        'unsaid_enabled': False,     # legacy alias only
                        'image_representation': 'native_final_cls',
                    }
                    record.update(build_finalcls_log_fields(out))
                    if grad_attribution:
                        record.update(grad_attribution)
                        record['grad_attribution_point'] = 'fresh_graph_after_optimizer_step'
                    cls_cos = record.get('global_cls_pairwise_cos')
                    if cls_cos is not None and float(cls_cos) >= 0.9 and rank == 0:
                        print('WARNING REPRESENTATION_COLLAPSE global_cls_pairwise_cos=%.4f '
                              'at step %d' % (float(cls_cos), step), flush=True)
                elif exgap_mode:
                    # SAID-ExGAP v1 payload: loss_global / loss_unsaid do not exist here and are
                    # written as JSON null; the legacy 2.8A/2.9A monitors are not part of it.
                    record = {
                        'step': step,
                        'completed_steps': step + 1,
                        'epoch': epoch,
                        'lr_scale': scale_factor,
                        'backbone_lr': optimizer.param_groups[0]['lr'],
                        'head_lr': optimizer.param_groups[1]['lr'],
                        'unsaid_enabled': False,     # legacy alias only
                    }
                    record.update(build_exgap_log_fields(out))
                    if grad_attribution:
                        record.update(grad_attribution)
                        record['grad_attribution_point'] = ('fresh_graph_after_optimizer_step')
                    gap_weight = record.get('global_pairwise_cos')
                    if gap_weight is not None and float(gap_weight) >= 0.9 and rank == 0:
                        print('WARNING REPRESENTATION_COLLAPSE global_pairwise_cos=%.4f at step %d'
                              % (float(gap_weight), step), flush=True)
                elif finalcls_mode:
                    raise AssertionError('unreachable: the v1.5 payload is handled first')
                elif gap_mode:
                    # Phase 3.0A payload: loss_global / loss_unsaid do not exist here and
                    # are written as JSON null (never ``out[...].detach()`` on a missing
                    # term). The legacy 2.8A/2.9A monitors are not part of this objective.
                    record = {
                        'step': step,
                        'completed_steps': step + 1,
                        'epoch': epoch,
                        'lr_scale': scale_factor,
                        'backbone_lr': optimizer.param_groups[0]['lr'],
                        'head_lr': optimizer.param_groups[1]['lr'],
                        'unsaid_enabled': False,   # legacy alias only, see the flags below
                        'said_effective_patch_count': float(out['said_effective_patch_count']),
                    }
                    record.update(build_gap_log_fields(out))
                    if grad_summary is not None:
                        record.update(grad_summary)
                        record['grad_norm_point'] = 'post_optimizer_step'
                    if grad_contributions:
                        record.update(grad_contributions)
                else:
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
                if not gap_mode and not exgap_mode:
                    for key in ('pair_gap_full', 'pair_gap_said', 'balancing_gain',
                                'relative_balancing_gain'):
                        record[key] = float(out[key])
                    record['representation_gap_scope'] = 'rank0_local_training_batch'
                    record.update(tri_gap_record(out))
                if caption_batch is not None:
                    record.update(caption_view_stats(caption_batch))
                if not gap_mode and not exgap_mode:
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
            if rank == 0 and save_completed and step in save_completed:
                # ``step`` is now the number of completed optimizer updates, so
                # ``salu_exgap_step000100.pt`` unambiguously means "the model after 100 updates"
                ckpt_path = os.path.join(args.output_dir, 'salu_exgap_step%06d.pt' % step)
                save_checkpoint(ckpt_path, ddp_model, optimizer, scaler, step, epoch, args,
                                base_lrs, total_steps)
                print('SAVED_EXGAP_STEP ' + ckpt_path, flush=True)

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
        # the full / unsaid caption streams only exist in the tri-view run; in gap mode the
        # dataset never produced them, so their digests must stay null
        tri_streams = bool(tri_mode) and not gap_mode and not exgap_mode
        audit = {'rank': rank, 'seed': args.seed, 'initial_state_sha256': initial_digest,
                 'sampler_order_sha256': sampler_digest.hexdigest(),
                 'caption_stream_sha256': caption_digest.hexdigest(),
                 'prefix_caption_stream_sha256': caption_digest.hexdigest(),
                 'full_caption_stream_sha256': full_digest.hexdigest() if tri_streams else None,
                 'unsaid_caption_stream_sha256': unsaid_digest.hexdigest() if tri_streams else None,
                 'has_unsaid_stream_sha256': has_unsaid_digest.hexdigest() if tri_streams else None,
                 'objective_mode': args.objective_mode,
                 'steps': step, 'dataset_size': len(train_set)}
        with open(os.path.join(args.output_dir, 'reproducibility_rank%d.json' % rank), 'w') as fp:
            json.dump(audit, fp, indent=2, sort_keys=True)
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
