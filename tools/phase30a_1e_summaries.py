#!/usr/bin/env python3
"""Phase 3.0A.1e: merge the combined decomposition run into the committed summaries.

The paired bootstrap needs both endpoints of a comparison inside one process, so the
comparisons come from a single combined run over all five checkpoints. The proxy
correlations are recomputed here from the per-query semantic outcomes that the run stores,
which keeps them available per checkpoint.
"""
import json
import os
import sys

REPO = '/root/SAID-gap-completion'
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'train'))

import torch  # noqa: E402

from eval.paired_statistics import spearman  # noqa: E402
from eval.unsaid_retrieval import rank_of  # noqa: E402

EVAL_DIR = os.path.join(REPO, 'outputs', 'phase30a_common_mode_semantics')
OUT_DIR = os.path.join(REPO, 'docs', 'phase30a_common_mode_semantics')
os.makedirs(OUT_DIR, exist_ok=True)

COMBINED = os.path.join(EVAL_DIR, 'combined_all_checkpoints.json')
SCORERS = ('global', 'said_raw', 'unsaid_raw', 'centroid', 'said_centered',
           'unsaid_centered', 'anti_minus_said', 'complete_existing')
REPORT_KEYS = ('R@1', 'R@5', 'R@10', 'MRR', 'mean_rank', 'median_rank',
               'positive_score_mean', 'positive_minus_mean_negative',
               'positive_minus_best_negative', 'candidate_pool', 'query_count')
INTERNAL_KEYS = ('gap_before_mean', 'gap_after_mean', 'gap_reduction_mean',
                 'gap_closure_ratio_mean', 'gap_closure_positive_fraction',
                 'cos_said_unsaid', 'unsaid_novel_component_norm',
                 'raw_pool_cosine', 'centered_pool_cosine', 'common_mode_ratio',
                 'common_mode_norm_ratio_said', 'common_mode_norm_ratio_unsaid',
                 'common_mode_norm_ratio_said_max', 'common_mode_norm_ratio_unsaid_max',
                 'centered_said_norm', 'centered_unsaid_norm', 'patch_centroid_norm')
ORDER = ('initial', 'A100', 'B100', 'C100', 'D100')
DESCRIPTIONS = {
    'initial': 'frozen initial checkpoint (shared by every arm)',
    'A100': 'L_S only (Said-only control)',
    'B100': 'L_S + L_gap_discover',
    'C100': 'L_S + L_gap_discover + L_global_absorb',
    'D100': 'L_S + L_gap + L_absorb at tau = 0.5',
}

if not os.path.exists(COMBINED):
    raise SystemExit('missing %s' % COMBINED)
payload = json.load(open(COMBINED))
checkpoints = payload['checkpoints']
present = [tag for tag in ORDER if tag in checkpoints]
print('combined run checkpoints: %s' % present)

common = {
    'protocol': 'phase30a-1e-common-mode-semantic-decomposition',
    'usr_protocol': payload['usr_protocol'],
    'usr_manifest': payload['usr_manifest'],
    'usr_manifest_sha256': payload['usr_manifest_sha256'],
    'cohort_sha256': payload['cohort_sha256'],
    'cohort_query_count': payload['cohort_query_count'],
    'cohort_candidate_pool': payload['cohort_candidate_pool'],
    'optimizer_steps': 0,
    'training_performed': False,
    'c_u_is_evaluation_target_only': True,
    'notes': ('Read-only evaluation of existing checkpoints on one frozen cohort '
              '(sharegpt4v1k-usr-v1, Q=868, candidate pool 868). Every scorer feature is '
              'precomputed from (I, C_S) before the candidate pool exists, and the score '
              'matrix is a plain inner product over the full pool. C_U never enters A_U, z_U, '
              'g or any feature construction. No Gap loss or core math change.'),
    'scorer_definitions': {
        'global': 'normalize(g) -- the CLS baseline',
        'said_raw': 'normalize(p_S)',
        'unsaid_raw': 'normalize(p_U) -- the current z_U',
        'centroid': 'normalize(mu), mu = mean_p h_p -- the shared common mode alone',
        'said_centered': 'normalize(delta_S), delta_S = sum_p A_S,p (h_p - mu)',
        'unsaid_centered': 'normalize(delta_U), delta_U = sum_p A_U,p (h_p - mu)',
        'anti_minus_said': 'normalize(delta_U - delta_S) = normalize(p_U - p_S) -- the contrast',
        'complete_existing': 'normalize(s_ref + u_new) -- the existing completion',
    },
}

summary = dict(common)
summary['checkpoints'] = {}
for tag in present:
    checkpoint = checkpoints[tag]
    reports = checkpoint['decomposition']['reports']
    summary['checkpoints'][tag] = {
        'description': DESCRIPTIONS.get(tag, tag),
        'checkpoint_path': checkpoint['checkpoint'],
        'checkpoint_sha256': checkpoint['checkpoint_sha256'],
        'checkpoint_step': checkpoint['checkpoint_step'],
        'checkpoint_phase': checkpoint['checkpoint_phase'],
        'checkpoint_objective_mode': checkpoint['checkpoint_objective_mode'],
        'query_count': checkpoint['decomposition']['query_count'],
        'candidate_pool': checkpoint['decomposition']['candidate_pool'],
        'reports': {name: {key: reports[name][key] for key in REPORT_KEYS
                           if key in reports[name]} for name in SCORERS},
        'internal': {key: checkpoint['internal'][key] for key in INTERNAL_KEYS
                     if key in checkpoint['internal']},
    }

n = common['cohort_query_count']
summary['random_chance'] = {'R@1': 1.0 / n, 'R@5': min(5.0 / n, 1.0),
                            'R@10': min(10.0 / n, 1.0)}
summary['unsaid_raw_chance_multiple_R@1'] = {
    tag: summary['checkpoints'][tag]['reports']['unsaid_raw']['R@1'] / (1.0 / n)
    for tag in present}
# how much of the raw z_U signal the centroid alone already provides
summary['centroid_share_of_unsaid_raw_R@1'] = {
    tag: (summary['checkpoints'][tag]['reports']['centroid']['R@1']
          / summary['checkpoints'][tag]['reports']['unsaid_raw']['R@1'])
    for tag in present}
with open(os.path.join(OUT_DIR, 'semantic_decomposition_summary.json'), 'w') as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)

paired = {
    'protocol': 'phase30a-1e-paired-statistics',
    'cohort': {'cohort_sha256': common['cohort_sha256'],
               'query_count': common['cohort_query_count'],
               'candidate_pool': common['cohort_candidate_pool']},
    'method': ('deterministic paired bootstrap over query indices (one shared resample '
               'applied to both methods), percentile 95% CI; exact two-sided McNemar on the '
               'R@1 hits. Independent binomial error bars are NOT used.'),
    'replicates': payload.get('paired_statistics', {}).get('replicates'),
    'seed': payload.get('paired_statistics', {}).get('seed'),
    'comparisons': payload.get('paired_statistics', {}).get('comparisons', {}),
}
with open(os.path.join(OUT_DIR, 'paired_statistics.json'), 'w') as handle:
    json.dump(paired, handle, indent=2, sort_keys=True)

proxy = {'protocol': 'phase30a-1e-proxy-correlation',
         'question': ('per query, does the internal gap track withheld-semantic quality? '
                      'closure_i vs -rank_i should be positive and gap_after_i vs rank_i '
                      'negative if internal closure is a useful proxy.'),
         'method': 'Spearman rank correlation with average ranks for ties',
         'scorer': 'unsaid_raw',
         'checkpoints': payload.get('proxy_correlation', {}),
         'rank_statistics': payload.get('per_query_rank_statistics', {})}
with open(os.path.join(OUT_DIR, 'proxy_correlation.json'), 'w') as handle:
    json.dump(proxy, handle, indent=2, sort_keys=True)

print()
print('wrote:')
for name in sorted(os.listdir(OUT_DIR)):
    print('  %s (%d bytes)' % (name, os.path.getsize(os.path.join(OUT_DIR, name))))
print('paired comparisons: %d' % len(paired['comparisons']))
print('proxy checkpoints: %d' % len(proxy['checkpoints']))
