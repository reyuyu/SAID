"""Phase 3.0A.1d runner: fixed-cohort checkpoint evaluation on the frozen USR cohort.

Evaluates the *same* frozen cohort for every checkpoint -- same images, same ``C_S``, same
``C_U`` targets, same candidate pool -- so that differences between checkpoints are
longitudinal rather than a change of batch. No optimizer step is ever taken.

Per checkpoint this reports

* internal gap metrics on the cohort (Q1),
* Phase 3 target-independent USR: ``global`` / ``unsaid`` / ``complete`` (Q2, Q3),
* patch common-mode geometry (Q5),

and optionally the canonical ShareGPT4V-1K variants and COCO val2017 retrieval (Q6).

Usage::

    python tools/phase30a_fixed_cohort_eval.py \
        --cohort usr \
        --checkpoints 'initial:...,step100:...' \
        --label A_said_only \
        --output outputs/phase30a_fixed_semantic_eval/A_said_only.json
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, 'train')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval.phase30a_fixed_cohort import (  # noqa: E402
    DECOMPOSITION_FEATURE_KEYS,
    DECOMPOSITION_SCORERS,
    PHASE3_SCORERS,
    cohort_mean,
    common_mode_diagnosis,
    common_mode_metrics,
    decomposition_features,
    decomposition_scorer_matrices,
    phase3_scorer_matrices,
    precompute_query_features,
)
from eval.unsaid_retrieval import (  # noqa: E402
    PROTOCOL_NAME as USR_PROTOCOL,
    load_or_create_usr_manifest,
    rank_of,
    retrieval_report,
)
from model import longclip  # noqa: E402
from model.complement_diagnostics import patch_homogeneity_metrics  # noqa: E402
from model.gap_completion import (  # noqa: E402
    gap_completion_terms,
    soft_anti_said_attention,
    unsaid_feature_from_attention,
)
from model.salu_model import SALUModel  # noqa: E402
from model.unsaid_core import attention_jsd, attention_overlap  # noqa: E402
from model.salu_modules import SaidRouter  # noqa: E402
from eval.paired_statistics import compare_methods, margin_per_query  # noqa: E402
from eval.unsaid_retrieval import rank_of, retrieval_report  # noqa: E402

# per-sample keys produced by evaluate_cohort_gap; the cohort report exposes them with the
# same names the training log uses (``*_mean``) so the two are directly comparable
COHORT_SAMPLE_KEYS = (
    'said_attention_entropy', 'unsaid_attention_entropy',
    'said_unsaid_attention_overlap', 'said_unsaid_attention_jsd',
    'cos_said_unsaid', 'unsaid_novel_component_norm',
    'gap_before', 'gap_after', 'gap_reduction', 'gap_closure_ratio',
    'cos_global_said', 'cos_global_unsaid',
    'patch_pair_cosine_mean', 'patch_centered_energy',
    'said_raw_pool_norm', 'unsaid_raw_pool_norm', 'raw_pool_cosine',
    'patch_centroid_norm', 'common_mode_ratio',
    'centered_said_norm', 'centered_unsaid_norm',
    'centered_pool_cosine', 'common_mode_fraction_said', 'common_mode_fraction_unsaid',
)
# sample key -> reported cohort key
COHORT_REPORT_ALIASES = {
    'gap_before': 'gap_before_mean',
    'gap_after': 'gap_after_mean',
    'gap_reduction': 'gap_reduction_mean',
    'gap_closure_ratio': 'gap_closure_ratio_mean',
}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def normalise_state_dict(state, model_state):
    """Map a checkpoint's key names onto the evaluator model's key names.

    ``train_salu.py`` saves ``SALUModel.state_dict()`` (``clip.*`` + ``said_router.*``), while
    the original SmartCLIP ``train.py`` saves the state dict of a *bare* ``CLIP`` module
    (``visual.*``, ``transformer.*``, ``ln_final``, ...). ``load_state_dict(strict=False)``
    would silently load nothing in the second case, so the prefix is added explicitly and any
    key the evaluator cannot accept is reported instead of being ignored.
    """
    target_keys = set(model_state)
    if all(key in target_keys for key in state):
        return {key: value for key, value in state.items() if key in target_keys}, []
    prefixed = {}
    for key, value in state.items():
        candidate = 'clip.' + key
        if candidate in target_keys:
            prefixed[candidate] = value
    if not prefixed:
        raise ValueError('checkpoint keys match neither the SALU nor the bare-CLIP layout; '
                         'refusing to load nothing silently')
    skipped = sorted(key for key in state
                     if ('clip.' + key) not in target_keys and key not in target_keys)
    return prefixed, skipped


def load_checkpoint_state(path):
    """Load a checkpoint and return ``(weight_state_dict, metadata_dict)``.

    Two on-disk conventions exist in this repository: ``train_salu.py`` writes a wrapper
    dict containing ``model`` / ``optimizer`` / ``step`` / ``phase``, while the original
    SmartCLIP ``train.py`` writes the bare ``state_dict`` under a ``smartclip_epochN.pt``
    name. Both must be loadable so the two methods can be compared with one evaluator.
    """
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(payload, dict) and 'model' in payload:
        return payload['model'], payload
    if isinstance(payload, dict):
        # a bare state dict: every value is a tensor, so there is no extra metadata
        return payload, {'step': None, 'phase': None, 'objective_mode': None}
    raise ValueError('unsupported checkpoint format at %s: %r' % (path, type(payload)))


def parse_tag_list(value):
    """``'initial,Aend'`` -> ``['initial', 'Aend']`` (empty -> ``[]``, meaning the default)."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value).split(',') if part.strip()]


def parse_checkpoints(spec):
    entries = []
    for entry in str(spec).split(','):
        entry = entry.strip()
        if not entry:
            continue
        if ':' not in entry:
            raise ValueError('checkpoints must be "tag:path", got %r' % (entry,))
        tag, path = entry.split(':', 1)
        entries.append((tag.strip(), path.strip()))
    if not entries:
        raise ValueError('at least one checkpoint is required')
    return entries


def cohort_samples(queries):
    """One fixed sample list: images + C_S (+ C_U targets) in manifest order, never resampled."""
    return [{'image_path': query['image_path'],
             'caption_said': query['caption_said'],
             'target_text': query['target_text'],
             'json_index': int(query['json_index']),
             'K': int(query['K']), 'N': int(query['N']), 'J': int(query['J'])}
            for query in queries]


@torch.no_grad()
def cohort_pass(model, samples, image_root, preprocess, device, batch_size,
                gap_anti_temperature, collect_decomposition=False):
    """One forward pass per sample over the frozen cohort; weights are never updated.

    Returns ``(rows, decomposition)`` where ``rows`` holds the per-sample gap / geometry
    metrics and ``decomposition`` (only when requested) holds the stacked query-side feature
    tensors of the common-mode decomposition. Every query feature is computed here, from
    ``(I, C_S)`` alone -- the candidate pool is not touched until scoring.
    """
    core = model.module if hasattr(model, 'module') else model
    was_training = core.training
    core.eval()
    rows = []
    collected = {key: [] for key in DECOMPOSITION_FEATURE_KEYS}
    try:
        for start in range(0, len(samples), batch_size):
            chunk = samples[start:start + batch_size]
            tensors = []
            for sample in chunk:
                with Image.open(os.path.join(image_root, sample['image_path'])) as image:
                    tensors.append(preprocess(image.convert('RGB')))
            images = torch.stack(tensors).to(device)
            tokens = longclip.tokenize([sample['caption_said'] for sample in chunk],
                                       truncate=True).to(device)
            z_global, patches = core.encode_router_input(images)
            text = F.normalize(core.encode_text(tokens), dim=-1)
            details = core.said_router.forward_with_details(text, patches)
            attention_said = details['attention']
            z_said = details['said']
            attention_unsaid = soft_anti_said_attention(
                details['scores'], temperature=gap_anti_temperature)
            z_unsaid = unsaid_feature_from_attention(attention_unsaid, patches)
            terms = gap_completion_terms(z_global, z_said, z_unsaid)

            a_said = attention_said.detach().float()
            a_unsaid = attention_unsaid.detach().float()
            entropy_said = SaidRouter.attention_entropy(a_said)
            entropy_unsaid = SaidRouter.attention_entropy(a_unsaid)
            unit_g = F.normalize(z_global.detach().float(), dim=-1)
            unit_s = F.normalize(z_said.detach().float(), dim=-1)
            unit_u = F.normalize(z_unsaid.detach().float(), dim=-1)
            pooled_said = torch.einsum('bp,bpd->bd', a_said, patches.detach().float())
            pooled_unsaid = torch.einsum('bp,bpd->bd', a_unsaid, patches.detach().float())
            # every per-sample metric must be a [B] tensor: the shared helpers return
            # attention-shaped tensors for JSD / overlap, which are aggregated over patches
            batch_size_actual = images.shape[0]
            per_sample = {
                'said_attention_entropy': entropy_said.reshape(-1),
                'unsaid_attention_entropy': entropy_unsaid.reshape(-1),
                'said_unsaid_attention_overlap': attention_overlap(a_said, a_unsaid).reshape(-1),
                'said_unsaid_attention_jsd': attention_jsd(a_said, a_unsaid).reshape(-1),
                'cos_said_unsaid': (unit_s * unit_u).sum(dim=-1),
                'unsaid_novel_component_norm': terms['novel_norm'].detach(),
                'gap_before': terms['gap_before'].detach(),
                'gap_after': terms['gap_after'].detach(),
                'gap_closure_ratio': terms['closure'].detach(),
                'cos_global_said': (unit_g * unit_s).sum(dim=-1),
                'cos_global_unsaid': (unit_g * unit_u).sum(dim=-1),
                'said_raw_pool_norm': pooled_said.norm(dim=-1),
                'unsaid_raw_pool_norm': pooled_unsaid.norm(dim=-1),
                'raw_pool_cosine': (F.normalize(pooled_said, dim=-1)
                                    * F.normalize(pooled_unsaid, dim=-1)).sum(dim=-1),
            }
            per_sample = {key: value.reshape(-1).detach().float()
                          for key, value in per_sample.items()}
            for key, value in per_sample.items():
                if value.shape[0] != batch_size_actual:
                    raise RuntimeError('metric %s has shape %r, expected [%d]'
                                       % (key, tuple(value.shape), batch_size_actual))
            per_sample['gap_reduction'] = per_sample['gap_before'] - per_sample['gap_after']
            per_sample['gap_closure_positive'] = (per_sample['gap_closure_ratio'] > 0).float()
            for index in range(batch_size_actual):
                row = {key: float(value[index]) for key, value in per_sample.items()}
                row.update(common_mode_metrics(patches[index:index + 1],
                                               a_said[index:index + 1],
                                               a_unsaid[index:index + 1]))
                # patch-granularity metrics come from the shared 3.0A.1c implementation
                homogeneity = patch_homogeneity_metrics(patches[index:index + 1],
                                                        z_global[index:index + 1])
                row['patch_pair_cosine_mean'] = float(homogeneity['patch_pair_cosine_mean'])
                row['patch_centered_energy'] = float(homogeneity['patch_centered_energy'])
                rows.append(row)
            if collect_decomposition:
                pieces = decomposition_features(patches, a_said, a_unsaid, z_global, z_said,
                                                F.normalize(z_said + terms['u_new'],
                                                            dim=-1))
                for key in DECOMPOSITION_FEATURE_KEYS:
                    collected[key].append(pieces[key].detach().float().cpu())
    finally:
        if was_training:
            core.train()
    decomposition = {key: torch.cat(value) for key, value in collected.items()} \
        if collect_decomposition else None
    return rows, decomposition


def evaluate_cohort_gap(model, samples, image_root, preprocess, device, batch_size,
                        gap_anti_temperature):
    """Per-sample internal gap + geometry metrics on one frozen cohort (weights fixed)."""
    rows, _ = cohort_pass(model, samples, image_root, preprocess, device, batch_size,
                          gap_anti_temperature)
    return rows


def aggregate_cohort(rows, extra_keys=()):
    """Cohort-level means, using the same names the training log reports."""
    keys = tuple(COHORT_SAMPLE_KEYS) + tuple(extra_keys)
    aggregate = cohort_mean(rows, keys)
    for sample_key, report_key in COHORT_REPORT_ALIASES.items():
        aggregate[report_key] = aggregate.pop(sample_key)
    aggregate['gap_closure_positive_fraction'] = float(
        sum(row['gap_closure_positive'] for row in rows) / len(rows))
    aggregate['sample_count'] = len(rows)
    return aggregate


@torch.no_grad()
def evaluate_cohort_usr(model, samples, image_root, preprocess, device, batch_size,
                        gap_anti_temperature, candidate_chunk_size=256):
    """Phase 3 target-independent USR on the frozen cohort.

    ``z_U`` (and the complete feature) are precomputed for every query *before* the candidate
    pool is touched; the score matrices are then plain inner products over the full pool.
    """
    core = model.module if hasattr(model, 'module') else model
    was_training = core.training
    core.eval()
    try:
        features = {'global_feature': [], 'unsaid_feature': [], 'complete_feature': [],
                    'said_feature': [], 'u_new': []}
        for start in range(0, len(samples), batch_size):
            chunk = samples[start:start + batch_size]
            tensors = []
            for sample in chunk:
                with Image.open(os.path.join(image_root, sample['image_path'])) as image:
                    tensors.append(preprocess(image.convert('RGB')))
            images = torch.stack(tensors).to(device)
            tokens = longclip.tokenize([sample['caption_said'] for sample in chunk],
                                       truncate=True).to(device)
            encoded = precompute_query_features(core, images, tokens, gap_anti_temperature)
            for key in features:
                features[key].append(encoded[key].cpu())
        query_features = {key: torch.cat(value) for key, value in features.items()}
        del query_features['said_feature'], query_features['u_new']

        candidate_texts = [sample['target_text'] for sample in samples]
        candidate_blocks = []
        for start in range(0, len(candidate_texts), candidate_chunk_size):
            block = candidate_texts[start:start + candidate_chunk_size]
            tokens = longclip.tokenize(block, truncate=True).to(device)
            candidate_blocks.append(core.encode_text(tokens).detach().float().cpu())
        candidate_features = torch.cat(candidate_blocks)

        scale = float(core.clip.logit_scale.exp().clamp(max=100))
        matrices = phase3_scorer_matrices(query_features, candidate_features, scale)
        reports = {name: retrieval_report(matrices[name]) for name in PHASE3_SCORERS}
        labels = torch.arange(len(samples))
        cross = {}
        for left in PHASE3_SCORERS:
            for right in PHASE3_SCORERS:
                if left >= right:
                    continue
                left_ranks = rank_of(matrices[left], labels).float()
                right_ranks = rank_of(matrices[right], labels).float()
                delta = left_ranks - right_ranks
                cross['%s_vs_%s' % (right, left)] = {
                    'mean_delta_rank': float(delta.mean()),
                    'median_delta_rank': float(delta.median()),
                    'improved_fraction': float((delta > 0).float().mean()),
                    'worsened_fraction': float((delta < 0).float().mean()),
                }
        return {'reports': reports, 'cross_scorer_rank_delta': cross,
                'candidate_pool': int(candidate_features.shape[0]),
                'query_count': int(query_features['global_feature'].shape[0])}
    finally:
        if was_training:
            core.train()


def evaluate_decomposition(features, candidate_features, logit_scale, samples):
    """Score every decomposition scorer over the full pool and add per-query margins.

    All eight scorers come from the same precomputed query features, so none of them can be
    candidate-conditioned; ``C_U`` enters only as the candidate text.
    """
    matrices = decomposition_scorer_matrices(features, candidate_features, logit_scale)
    reports, margins_mean, margins_best = {}, {}, {}
    for name in DECOMPOSITION_SCORERS:
        reports[name] = retrieval_report(matrices[name])
        margins_mean[name] = margin_per_query(matrices[name], 'mean')
        margins_best[name] = margin_per_query(matrices[name], 'best')
    return {'reports': reports,
            'margin_mean': margins_mean,
            'margin_best': margins_best,
            'matrices': matrices,
            'candidate_pool': int(candidate_features.shape[0]),
            'query_count': int(len(samples))}


# checkpoints that take part in the decomposition comparison (initial / A / B / C / D)
DECOMPOSITION_TAGS = ('initial', 'A100', 'A_step100', 'B100', 'B_step100', 'C100',
                      'C_step100', 'D100', 'D_step100')


def rank_key(run_tag):
    """Ordinal position of a checkpoint label (initial < A < B < C < D)."""
    if run_tag.startswith('initial'):
        return 0
    if run_tag.startswith('A'):
        return 1
    if run_tag.startswith('B'):
        return 2
    if run_tag.startswith('C'):
        return 3
    if run_tag.startswith('D'):
        return 4
    return 5


# (label, left tag, left scorer, right tag, right scorer); tags are rank tokens
COMPARISONS = (
    ('B_minus_A_zU_unsaid_raw', 'B', 'unsaid_raw', 'A', 'unsaid_raw'),
    ('C_minus_B_global', 'C', 'global', 'B', 'global'),
    ('C_minus_A_global', 'C', 'global', 'A', 'global'),
    ('C_minus_A_zU_unsaid_raw', 'C', 'unsaid_raw', 'A', 'unsaid_raw'),
    ('unsaid_centered_minus_centroid', 'C', 'unsaid_centered', 'C', 'centroid'),
    ('anti_minus_said_vs_unsaid_raw', 'C', 'anti_minus_said', 'C', 'unsaid_raw'),
    ('unsaid_centered_vs_said_centered', 'C', 'unsaid_centered', 'C', 'said_centered'),
    ('initial_minus_C_zU_unsaid_raw', 'initial', 'unsaid_raw', 'C', 'unsaid_raw'),
    ('C_minus_B_zU_unsaid_raw', 'C', 'unsaid_raw', 'B', 'unsaid_raw'),
    ('D_minus_C_zU_unsaid_raw', 'D', 'unsaid_raw', 'C', 'unsaid_raw'),
)


def resolve_tag(available, tag):
    """Map a short comparison tag (``A``) onto the key the run actually stored (``A100``).

    Checkpoints are labelled with a step suffix so a multi-checkpoint run can hold both the
    initial state and step 100 of the same arm, which makes the exact key depend on the
    caller's ``--checkpoints`` spelling. Accepting the plausible spellings keeps the
    comparison list readable without silently skipping a pair.
    """
    for candidate in (tag, '%s100' % tag, '%s_step100' % tag, '%s_step100' % tag.lower()):
        if candidate in available:
            return candidate
    return None


# scorers compared between the two full-training arms at every matched step
CROSS_ARM_SCORERS = ('unsaid_raw', 'unsaid_centered', 'global', 'centroid', 'said_raw',
                     'anti_minus_said')
# matched-step suffixes to pair up automatically (A<step> vs C<step>)
CROSS_ARM_STEPS = ('initial', '100', '200', '500', '1000', '1216', '1800', '2432', '3000',
                   'end')


def cross_arm_comparisons(decomposition_by_tag, replicates, seed):
    """Every ``C<step> - A<step>`` comparison on the fixed cohort, with paired CIs.

    The two full-training arms are matched by construction, so each matched step is an exact
    paired comparison on the same queries and the same 868-candidate pool.
    """
    available = set(decomposition_by_tag)
    output = {}
    for step in CROSS_ARM_STEPS:
        left_tag = 'C%s' % step if step != 'initial' else 'initial'
        right_tag = 'A%s' % step if step != 'initial' else 'initial'
        left_key = resolve_tag(available, left_tag)
        right_key = resolve_tag(available, right_tag)
        if left_key is None or right_key is None or left_key == right_key:
            continue
        for scorer in CROSS_ARM_SCORERS:
            label = 'C%s_minus_A%s_%s' % (step, step, scorer)
            output[label] = {
                'step': step,
                'scorer': scorer,
                'left': {'checkpoint': left_key, 'scorer': scorer},
                'right': {'checkpoint': right_key, 'scorer': scorer},
                **compare_methods(decomposition_by_tag[left_key]['matrices'][scorer],
                                  decomposition_by_tag[right_key]['matrices'][scorer],
                                  replicates=replicates, seed=seed),
            }
    return output


def paired_comparisons(decomposition_by_tag, replicates, seed):
    """Paired bootstrap + McNemar for every comparison that has both checkpoints present."""
    available = set(decomposition_by_tag)
    output = {}
    for label, left_tag, left_scorer, right_tag, right_scorer in COMPARISONS:
        left_key = resolve_tag(available, left_tag)
        right_key = resolve_tag(available, right_tag)
        if left_key is None or right_key is None:
            continue
        left = decomposition_by_tag[left_key]['matrices'][left_scorer]
        right = decomposition_by_tag[right_key]['matrices'][right_scorer]
        output[label] = {
            'left': {'checkpoint': left_key, 'scorer': left_scorer},
            'right': {'checkpoint': right_key, 'scorer': right_scorer},
            **compare_methods(left, right, replicates=replicates, seed=seed),
        }
    return output


def proxy_from_outcomes(reference_rows, outcomes, scorer='unsaid_raw'):
    """Proxy correlations from the stored per-query outcomes (no score matrices needed).

    ``reference_rows`` holds the internal per-sample numbers; ``outcomes`` holds the semantic
    per-query ranks and margins for every scorer. A useful proxy would correlate with
    ``-rank`` positively and with ``gap_after`` negatively.
    """
    from eval.paired_statistics import spearman
    ranks = outcomes['ranks'][scorer]
    margin_mean = outcomes['margin_mean'][scorer]
    margin_best = outcomes['margin_best'][scorer]
    if not (len(reference_rows) == len(ranks) == len(margin_mean) == len(margin_best)):
        raise ValueError('internal rows and per-query outcomes must cover the same queries')
    closure = [row['gap_closure_ratio'] for row in reference_rows]
    reduction = [row['gap_reduction'] for row in reference_rows]
    gap_after = [row['gap_after'] for row in reference_rows]
    negative_rank = [-float(value) for value in ranks]
    return {
        'scorer': scorer,
        'closure_vs_negative_rank': spearman(closure, negative_rank),
        'closure_vs_margin_mean': spearman(closure, margin_mean),
        'closure_vs_margin_best': spearman(closure, margin_best),
        'gap_reduction_vs_negative_rank': spearman(reduction, negative_rank),
        'gap_reduction_vs_margin_mean': spearman(reduction, margin_mean),
        'gap_after_vs_rank': spearman(gap_after, ranks),
        'gap_after_vs_margin_mean': spearman(gap_after, margin_mean),
        'rank_mean': float(sum(ranks) / len(ranks)),
        'rank_median': float(sorted(ranks)[len(ranks) // 2]),
        'query_count': len(reference_rows),
    }


def proxy_correlations(reference_rows, decomposition, scorer='unsaid_raw'):
    """Does the internal gap track withheld-semantic quality, per query?

    ``closure_i``, ``gap_reduction_i`` and ``gap_after_i`` are the internal numbers;
    ``-rank_i``, ``margin_mean_i`` and ``margin_best_i`` are the semantic ones. A useful proxy
    would correlate with ``-rank`` positively and with ``gap_after`` negatively.
    """
    from eval.paired_statistics import spearman
    if len(reference_rows) != decomposition['query_count']:
        raise ValueError('internal rows and decomposition features must cover the same queries')
    closure = [row['gap_closure_ratio'] for row in reference_rows]
    reduction = [row['gap_reduction'] for row in reference_rows]
    gap_after = [row['gap_after'] for row in reference_rows]
    ranks = rank_of(decomposition['matrices'][scorer]).float()
    negative_rank = (-ranks).tolist()
    margin_mean = decomposition['margin_mean'][scorer].tolist()
    margin_best = decomposition['margin_best'][scorer].tolist()
    return {
        'scorer': scorer,
        'closure_vs_negative_rank': spearman(closure, negative_rank),
        'closure_vs_margin_mean': spearman(closure, margin_mean),
        'closure_vs_margin_best': spearman(closure, margin_best),
        'gap_reduction_vs_negative_rank': spearman(reduction, negative_rank),
        'gap_reduction_vs_margin_mean': spearman(reduction, margin_mean),
        'gap_after_vs_rank': spearman(gap_after, ranks.tolist()),
        'gap_after_vs_margin_mean': spearman(gap_after, margin_mean),
        'rank_distribution': {'mean': float(ranks.mean()), 'median': float(ranks.median())},
    }


def evaluate_canonical(model, args, preprocess):
    """Standard CLIP CLS retrieval: ShareGPT4V-1K (3 variants) and COCO val2017."""
    from eval.validation_protocol import (evaluate_all_variants, load_or_create_manifest)
    from eval.retrieval.coco_retrieval import evaluate_coco

    output = {}
    if args.sharegpt4v_manifest:
        data_root = args.data_root
        json_name = os.environ.get('SHARE4V_JSON',
                                   'share-captioner_coco_lcs_sam_1246k_1107.json')
        json_path = json_name if os.path.isabs(json_name) else os.path.join(data_root, json_name)
        manifest = load_or_create_manifest(args.sharegpt4v_manifest, json_path)
        variants = evaluate_all_variants(model, manifest['samples'], args.image_root,
                                         preprocess, batch_size=args.image_batch_size,
                                         device=args.device)
        output['sharegpt4v1k'] = {
            variant: {'retrieval': result['retrieval']} for variant, result in variants.items()}
    if args.coco:
        output['coco_val2017'] = evaluate_coco(model, preprocess, root=args.coco_root,
                                               batch_size=args.image_batch_size,
                                               device=args.device)
    return output


def main():
    parser = argparse.ArgumentParser(description='Phase 3.0A.1d fixed-cohort evaluation')
    parser.add_argument('--checkpoints', required=True,
                        help='comma-separated "tag:path"; tag a step100 checkpoint as '
                             'A100/B100/C100/D100 so it enters the paired comparison')
    parser.add_argument('--label', required=True, help='arm label, e.g. A_said_only')
    parser.add_argument('--gap_anti_temperature', type=float, required=True,
                        help='the trained arm temperature; must match the arm')
    parser.add_argument('--usr_manifest',
                        default='/root/SAID/outputs/validation/sharegpt4v1k_usr_manifest.json')
    parser.add_argument('--source_manifest',
                        default='/root/SAID/outputs/validation/sharegpt4v1k_manifest.json')
    parser.add_argument('--data_root', default=os.environ.get('SHARE4V_DATA_ROOT',
                                                              '/root/datasets/ShareGPT4V'))
    parser.add_argument('--image_root', default=None,
                        help='image root for the cohorts (defaults to --data_root)')
    parser.add_argument('--output', required=True)
    parser.add_argument('--image_batch_size', type=int, default=32)
    parser.add_argument('--canonical', action='store_true')
    parser.add_argument('--canonical_only', action='store_true',
                        help='skip the cohort (USR / gap / decomposition) passes entirely and '
                             'run only the canonical retrieval evaluation')
    parser.add_argument('--canonical_tags', type=parse_tag_list, default=[],
                        help='comma-separated checkpoint tags to evaluate canonically; empty '
                             'means initial/step100/final')
    parser.add_argument('--canonical_names', type=parse_tag_list, default=[],
                        help='optional display names matching --canonical_tags, used only to '
                             'label the canonical results (the run/label is unchanged)')
    parser.add_argument('--decomposition', action='store_true',
                        help='also score the 8 common-mode decomposition scorers and run the '
                             'paired bootstrap / proxy correlations')
    parser.add_argument('--bootstrap_replicates', type=int, default=10000)
    parser.add_argument('--bootstrap_seed', type=int, default=20260911)
    parser.add_argument('--coco', action='store_true')
    parser.add_argument('--coco_root', default=None)
    parser.add_argument('--sharegpt4v_manifest',
                        default='/root/SAID/outputs/validation/sharegpt4v1k_manifest.json')
    parser.add_argument('--base_model', default='ViT-B/16')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    image_root = args.image_root or args.data_root
    manifest = load_or_create_usr_manifest(args.usr_manifest, args.source_manifest)
    samples = cohort_samples(manifest['queries'])
    cohort_sha = hashlib.sha256(json.dumps(samples, sort_keys=True).encode('utf-8')).hexdigest()
    print('cohort: %d queries, candidate_pool=%d, sha256=%s'
          % (len(samples), manifest['candidate_count'], cohort_sha[:16]), flush=True)

    clip_model, preprocess = longclip.load_from_clip(args.base_model, device='cpu', args=args)
    core = SALUModel(clip_model, tau_said=0.07, said_loss_mode='identifiable',
                     pair_chunk_size=64, said_feature_source='residual').float().to(args.device)

    payload = {
        'protocol': 'phase30a-1d-fixed-cohort',
        'arm_label': args.label,
        'gap_anti_temperature': args.gap_anti_temperature,
        'usr_protocol': USR_PROTOCOL,
        'usr_manifest': args.usr_manifest,
        'usr_manifest_sha256': file_sha256(args.usr_manifest),
        'cohort_sha256': cohort_sha,
        'cohort_query_count': len(samples),
        'cohort_candidate_pool': int(manifest['candidate_count']),
        'c_u_is_evaluation_target_only': True,
        'notes': ('z_U and the complete feature are built from (I, C_S) before any candidate '
                  'exists; score matrices are plain inner products over the full pool. '
                  'C_U never enters A_U, z_U, g or any feature construction. No optimizer '
                  'step is taken during evaluation.'),
        'checkpoints': {},
    }
    decomposition_by_tag = {}
    reference_by_tag = {}
    per_query_outcomes = {}
    for tag, path in parse_checkpoints(args.checkpoints):
        if args.canonical_only:
            # canonical retrieval does not need the cohort passes at all: skip the USR and
            # gap work entirely so this mode is cheap
            continue
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        missing, unexpected = core.load_state_dict(checkpoint['model'], strict=False)
        if list(missing) or list(unexpected):
            raise RuntimeError('checkpoint %s does not match the model' % path)
        started = time.time()
        rows, _ = cohort_pass(core, samples, image_root, preprocess, args.device,
                              args.image_batch_size, args.gap_anti_temperature)
        internal = aggregate_cohort(rows)
        internal['common_mode_diagnosis'] = common_mode_diagnosis(internal)
        usr = evaluate_cohort_usr(core, samples, image_root, preprocess, args.device,
                                  args.image_batch_size, args.gap_anti_temperature)
        entry = {
            'checkpoint': path,
            'checkpoint_sha256': file_sha256(path),
            'checkpoint_step': int(checkpoint.get('step', 0)),
            'checkpoint_phase': checkpoint.get('phase'),
            'checkpoint_objective_mode': checkpoint.get('objective_mode'),
            'internal': internal,
            'usr': usr,
            'wall_sec': time.time() - started,
        }
        if args.decomposition and tag in DECOMPOSITION_TAGS:
            # a second pass: the decomposition needs the query features themselves, which
            # are far too large to keep for the whole cohort in one go
            _, features = cohort_pass(core, samples, image_root, preprocess, args.device,
                                      args.image_batch_size, args.gap_anti_temperature,
                                      collect_decomposition=True)
            candidate_blocks = []
            candidate_texts = [sample['target_text'] for sample in samples]
            for start in range(0, len(candidate_texts), 256):
                block = candidate_texts[start:start + 256]
                tokens = longclip.tokenize(block, truncate=True).to(args.device)
                candidate_blocks.append(core.encode_text(tokens).detach().float().cpu())
            candidate_features = torch.cat(candidate_blocks)
            logit_scale = float(core.clip.logit_scale.exp().clamp(max=100))
            decomposition = evaluate_decomposition(features, candidate_features, logit_scale,
                                                   samples)
            rank_token = (tag.split('_')[0] if '_' in tag and not tag.startswith('initial')
                          else tag)
            if rank_token in decomposition_by_tag and rank_token != tag:
                rank_token = tag
            decomposition_by_tag[rank_token] = {
                'matrices': decomposition['matrices'],
                'reports': decomposition['reports'],
                'margin_mean': decomposition['margin_mean'],
                'margin_best': decomposition['margin_best'],
                'query_count': decomposition['query_count'],
                'candidate_pool': decomposition['candidate_pool'],
            }
            reference_by_tag[rank_token] = {
                'rows': rows,
                'query_count': decomposition['query_count'],
                'candidate_pool': decomposition['candidate_pool'],
            }
            entry['decomposition'] = {
                'reports': decomposition['reports'],
                'query_count': decomposition['query_count'],
                'candidate_pool': decomposition['candidate_pool'],
                'scorers': list(DECOMPOSITION_SCORERS),
                'rank_token': rank_token,
            }
            # per-query outcomes: the proxy correlations need the semantic side per query and
            # cannot be recovered from the aggregate report, so store the ranks and margins
            per_query_outcomes[rank_token] = {
                'ranks': {name: rank_of(decomposition['matrices'][name]).tolist()
                          for name in DECOMPOSITION_SCORERS},
                'margin_mean': {name: decomposition['margin_mean'][name].tolist()
                                for name in DECOMPOSITION_SCORERS},
                'margin_best': {name: decomposition['margin_best'][name].tolist()
                                for name in DECOMPOSITION_SCORERS},
            }
            del features
        payload['checkpoints'][tag] = entry
        summary = usr['reports']
        print('EVAL %-10s %-10s closure=%+.6f gap_after=%.6f cos(zS,zU)=%.6f | '
              'USR global R@1=%.4f MRR=%.4f  unsaid R@1=%.4f MRR=%.4f  complete R@1=%.4f MRR=%.4f'
              % (args.label, tag, internal['gap_closure_ratio_mean'],
                 internal['gap_after_mean'],
                 internal['cos_said_unsaid'], summary['global']['R@1'], summary['global']['MRR'],
                 summary['unsaid']['R@1'], summary['unsaid']['MRR'],
                 summary['complete']['R@1'], summary['complete']['MRR']), flush=True)

    if args.decomposition and decomposition_by_tag:
        payload['paired_statistics'] = {
            'replicates': int(args.bootstrap_replicates),
            'seed': int(args.bootstrap_seed),
            'note': ('All methods are scored on the same 868 queries with the same frozen '
                     '868-candidate pool, so every comparison is paired; the bootstrap '
                     'resamples query indices once and applies the same resample to both '
                     'methods. Independent binomial error bars are not used.'),
            'comparisons': paired_comparisons(decomposition_by_tag, args.bootstrap_replicates,
                                              args.bootstrap_seed),
        }
        payload['cross_arm_paired_statistics'] = {
            'replicates': int(args.bootstrap_replicates),
            'seed': int(args.bootstrap_seed),
            'note': ('C - A at every matched step, on the same 868 queries and the same '
                     '868-candidate pool. Paired bootstrap over query indices with one shared '
                     'resample, plus exact two-sided McNemar on R@1.'),
            'comparisons': cross_arm_comparisons(decomposition_by_tag,
                                                 args.bootstrap_replicates,
                                                 args.bootstrap_seed),
        }
        payload['proxy_correlation'] = {}
        for rank_token in sorted(decomposition_by_tag, key=rank_key):
            reference = reference_by_tag[rank_token]
            outcomes = per_query_outcomes[rank_token]
            payload['proxy_correlation'][rank_token] = proxy_from_outcomes(
                reference['rows'], outcomes, 'unsaid_raw')
        payload['per_query_rank_statistics'] = {
            rank_token: {'rank_mean': float(np.mean(outcomes['ranks']['unsaid_raw'])),
                         'rank_median': float(np.median(outcomes['ranks']['unsaid_raw'])),
                         'query_count': len(outcomes['ranks']['unsaid_raw'])}
            for rank_token, outcomes in per_query_outcomes.items()}
        print('PAIRED_STATISTICS %d comparisons' % len(payload['paired_statistics']['comparisons']),
              flush=True)

    if args.canonical:
        payload['canonical'] = {}
        wanted = set(args.canonical_tags) if args.canonical_tags else {'initial', 'step100',
                                                                      'final'}
        labels = dict(zip(args.canonical_tags, args.canonical_names)) \
            if args.canonical_names else {}
        for tag, path in parse_checkpoints(args.checkpoints):
            if tag not in wanted:
                continue
            state, _ = load_checkpoint_state(path)
            state, skipped = normalise_state_dict(state, core.state_dict())
            incompatible = core.load_state_dict(state, strict=False)
            # The only keys a non-SALU checkpoint may legitimately lack are the Said router's:
            # canonical CLS retrieval uses encode_image()/encode_text() and never touches the
            # router, so its absence cannot affect the measurement. Anything else is an error.
            missing = [key for key in incompatible.missing_keys
                       if not key.startswith('said_router.')]
            if missing or list(incompatible.unexpected_keys):
                raise RuntimeError(
                    'checkpoint %s did not load cleanly: missing %r, unexpected %r'
                    % (path, missing, list(incompatible.unexpected_keys)))
            router_absent = sorted(key for key in incompatible.missing_keys
                                   if key.startswith('said_router.'))
            key = labels.get(tag, tag)
            payload['canonical'][key] = evaluate_canonical(core, args, preprocess)
            payload['canonical'][key]['checkpoint'] = path
            payload['canonical'][key]['skipped_keys'] = skipped
            payload['canonical'][key]['absent_said_router_keys'] = router_absent
            payload['canonical'][key]['loaded_tensors'] = len(state)
            print('CANONICAL %-12s done (loaded %d tensors, skipped %d, router-absent %d)'
                  % (key, len(state), len(skipped), len(router_absent)), flush=True)

    output_path = args.output if os.path.isabs(args.output) else os.path.join(REPO, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
    print('WROTE %s' % output_path, flush=True)


if __name__ == '__main__':
    main()
