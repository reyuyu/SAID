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

import torch
import torch.nn.functional as F
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, 'train')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval.phase30a_fixed_cohort import (  # noqa: E402
    PHASE3_SCORERS,
    cohort_mean,
    common_mode_diagnosis,
    common_mode_metrics,
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
def evaluate_cohort_gap(model, samples, image_root, preprocess, device, batch_size,
                        gap_anti_temperature):
    """Per-sample internal gap + geometry metrics on one frozen cohort (weights fixed)."""
    core = model.module if hasattr(model, 'module') else model
    was_training = core.training
    core.eval()
    rows = []
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
    finally:
        if was_training:
            core.train()
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
    parser.add_argument('--checkpoints', required=True, help='comma-separated "tag:path"')
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
    for tag, path in parse_checkpoints(args.checkpoints):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        missing, unexpected = core.load_state_dict(checkpoint['model'], strict=False)
        if list(missing) or list(unexpected):
            raise RuntimeError('checkpoint %s does not match the model' % path)
        started = time.time()
        rows = evaluate_cohort_gap(core, samples, image_root, preprocess, args.device,
                                   args.image_batch_size, args.gap_anti_temperature)
        internal = aggregate_cohort(rows)
        internal['common_mode_diagnosis'] = common_mode_diagnosis(internal)
        usr = evaluate_cohort_usr(core, samples, image_root, preprocess, args.device,
                                  args.image_batch_size, args.gap_anti_temperature)
        payload['checkpoints'][tag] = {
            'checkpoint': path,
            'checkpoint_sha256': file_sha256(path),
            'checkpoint_step': int(checkpoint.get('step', 0)),
            'checkpoint_phase': checkpoint.get('phase'),
            'checkpoint_objective_mode': checkpoint.get('objective_mode'),
            'internal': internal,
            'usr': usr,
            'wall_sec': time.time() - started,
        }
        summary = usr['reports']
        print('EVAL %-10s %-10s closure=%+.6f gap_after=%.6f cos(zS,zU)=%.6f | '
              'USR global R@1=%.4f MRR=%.4f  unsaid R@1=%.4f MRR=%.4f  complete R@1=%.4f MRR=%.4f'
              % (args.label, tag, internal['gap_closure_ratio_mean'],
                 internal['gap_after_mean'],
                 internal['cos_said_unsaid'], summary['global']['R@1'], summary['global']['MRR'],
                 summary['unsaid']['R@1'], summary['unsaid']['MRR'],
                 summary['complete']['R@1'], summary['complete']['MRR']), flush=True)

    if args.canonical:
        payload['canonical'] = {}
        for tag, path in parse_checkpoints(args.checkpoints):
            if tag not in ('initial', 'step100', 'final'):
                continue
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            core.load_state_dict(checkpoint['model'], strict=False)
            payload['canonical'][tag] = evaluate_canonical(core, args, preprocess)
            print('CANONICAL %-10s done' % tag, flush=True)

    output_path = args.output if os.path.isabs(args.output) else os.path.join(REPO, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
    print('WROTE %s' % output_path, flush=True)


if __name__ == '__main__':
    main()
