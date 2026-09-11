#!/usr/bin/env python3
"""Phase 3.0A.2: merge the full 3-epoch fixed-cohort evaluation into the committed summaries.

Writes ``full3epoch_fixed_cohort.json``, ``full3epoch_paired_stats.json`` and (when present)
``canonical_retrieval.json``. Checkpoints, raw logs and score matrices stay uncommitted.
"""
import json
import os
import sys

REPO = '/root/SAID-gap-completion'
sys.path.insert(0, REPO)

EVAL_DIR = os.path.join(REPO, 'outputs', 'phase30a_full3epoch')
OUT_DIR = os.path.join(REPO, 'docs', 'phase30a_full3epoch')
os.makedirs(OUT_DIR, exist_ok=True)

RAW = os.path.join(EVAL_DIR, 'full3epoch_fixed_cohort_raw.json')
CANONICAL_RAW = os.path.join(EVAL_DIR, 'canonical_raw.json')

# matched checkpoints: label -> (A tag, C tag, recorded step)
MATCHED = (
    ('step0', None, None, 0),
    ('step100', 'A100', 'C100', 100),
    ('step200', 'A200', 'C200', 200),
    ('step500', 'A500', 'C500', 500),
    ('step1216', 'A1216', 'C1216', 1216),
    ('step2432', 'A2432', 'C2432', 2432),
    ('step3000', 'A3000', 'C3000', 3000),
    ('step_end', 'Aend', 'Cend', None),
)
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
                 'centered_said_norm', 'centered_unsaid_norm', 'patch_centroid_norm',
                 'route_top1_acc', 'evidence_top1_acc', 'route_margin', 'evidence_margin',
                 'said_attention_entropy', 'unsaid_attention_entropy')
CROSS_ARM_SCORERS = ('unsaid_raw', 'unsaid_centered', 'global', 'centroid', 'said_raw',
                     'anti_minus_said')

if not os.path.exists(RAW):
    raise SystemExit('missing %s' % RAW)
payload = json.load(open(RAW))
checkpoints = payload['checkpoints']
print('checkpoints in the evaluation: %s' % sorted(checkpoints))

common = {
    'protocol': 'phase30a-2-full-3epoch-fixed-cohort',
    'usr_protocol': payload['usr_protocol'],
    'usr_manifest': payload['usr_manifest'],
    'usr_manifest_sha256': payload['usr_manifest_sha256'],
    'cohort_sha256': payload['cohort_sha256'],
    'cohort_query_count': payload['cohort_query_count'],
    'cohort_candidate_pool': payload['cohort_candidate_pool'],
    'optimizer_steps': 0,
    'training_performed_during_evaluation': False,
    'c_u_is_evaluation_target_only': True,
    'arms': {
        'A': 'Said-only control: lambda_said=1, lambda_gap_discover=0, lambda_global_absorb=0',
        'C': 'Full Base: lambda_said=1, lambda_gap_discover=1, lambda_global_absorb=1',
    },
    'notes': ('Both arms trained the full 3 epochs (steps_per_epoch 1216, stop 3648) from the '
              'same frozen initial state with a matched data stream; evaluation is read-only '
              'on one frozen cohort (sharegpt4v1k-usr-v1, Q=868, candidate pool 868). Every '
              'scorer feature is built from (I, C_S) before the candidate pool exists.'),
    'scorer_definitions': {
        'global': 'normalize(g) -- CLS baseline',
        'said_raw': 'normalize(p_S)',
        'unsaid_raw': 'normalize(p_U) -- z_U',
        'centroid': 'normalize(mu), mu = mean_p h_p',
        'said_centered': 'normalize(delta_S)',
        'unsaid_centered': 'normalize(delta_U)',
        'anti_minus_said': 'normalize(delta_U - delta_S)',
        'complete_existing': 'normalize(s_ref + u_new)',
    },
}

summary = dict(common)
summary['checkpoints'] = {}
for label, a_tag, c_tag, step in MATCHED:
    entry = {'recorded_step': step, 'arms': {}}
    for arm, tag in (('A', a_tag), ('C', c_tag)):
        key = tag if tag is not None else 'initial'
        checkpoint = checkpoints.get(key)
        if checkpoint is None:
            continue
        reports = checkpoint['decomposition']['reports']
        entry['arms'][arm] = {
            'checkpoint': checkpoint['checkpoint'],
            'checkpoint_sha256': checkpoint['checkpoint_sha256'],
            'checkpoint_step': checkpoint['checkpoint_step'],
            'checkpoint_phase': checkpoint['checkpoint_phase'],
            'checkpoint_objective_mode': checkpoint['checkpoint_objective_mode'],
            'reports': {name: {key: reports[name][key] for key in REPORT_KEYS
                               if key in reports[name]} for name in SCORERS},
            'internal': {key: checkpoint['internal'][key] for key in INTERNAL_KEYS
                         if key in checkpoint['internal']},
        }
    summary['checkpoints'][label] = entry

n = common['cohort_query_count']
summary['random_chance'] = {'R@1': 1.0 / n, 'R@5': min(5.0 / n, 1.0),
                            'R@10': min(10.0 / n, 1.0)}
with open(os.path.join(OUT_DIR, 'full3epoch_fixed_cohort.json'), 'w') as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)

# ---------------------------------------------------------------- paired statistics
paired = {
    'protocol': 'phase30a-2-full-3epoch-paired-statistics',
    'cohort': {'cohort_sha256': common['cohort_sha256'],
               'query_count': common['cohort_query_count'],
               'candidate_pool': common['cohort_candidate_pool']},
    'method': ('deterministic paired bootstrap over query indices (one shared resample for '
               'both arms), percentile 95% CI, 10000 replicates; exact two-sided McNemar on '
               'R@1 hits. A and C are matched by construction (same initial state, same '
               'sampler order, same C_S stream), so every step is an exact paired comparison.'),
    'replicates': payload.get('cross_arm_paired_statistics', {}).get('replicates'),
    'seed': payload.get('cross_arm_paired_statistics', {}).get('seed'),
    'by_step': {},
    'within_C_diagnostic': payload.get('paired_statistics', {}).get('comparisons', {}),
}
by_scorer = {}
for name, comparison in payload.get('cross_arm_paired_statistics', {}).get(
        'comparisons', {}).items():
    by_scorer.setdefault(comparison['scorer'], {})[comparison['step']] = {
        'left': comparison['left'], 'right': comparison['right'],
        'R@1': comparison['R@1'], 'MRR': comparison['MRR'],
        'mcnemar_R@1': comparison['mcnemar_R@1'],
    }
paired['by_scorer_and_step'] = by_scorer
# an ordered trajectory view: for each scorer, the delta and CI across matched steps
ordered_steps = ('100', '200', '500', '1216', '2432', '3000', 'end')
paired['effect_size_trajectory'] = {
    scorer: [{'step': step,
              'delta_R@1': by_scorer[scorer][step]['R@1']['delta'],
              'ci_R@1': [by_scorer[scorer][step]['R@1']['ci_low'],
                         by_scorer[scorer][step]['R@1']['ci_high']],
              'excludes_zero_R@1': by_scorer[scorer][step]['R@1']['excludes_zero'],
              'delta_MRR': by_scorer[scorer][step]['MRR']['delta'],
              'ci_MRR': [by_scorer[scorer][step]['MRR']['ci_low'],
                         by_scorer[scorer][step]['MRR']['ci_high']],
              'excludes_zero_MRR': by_scorer[scorer][step]['MRR']['excludes_zero'],
              'mcnemar_p': by_scorer[scorer][step]['mcnemar_R@1']['p_value_exact_two_sided']}
             for step in ordered_steps if step in by_scorer[scorer]]
    for scorer in CROSS_ARM_SCORERS if scorer in by_scorer}
with open(os.path.join(OUT_DIR, 'full3epoch_paired_stats.json'), 'w') as handle:
    json.dump(paired, handle, indent=2, sort_keys=True)

# ---------------------------------------------------------------- canonical retrieval
if os.path.exists(CANONICAL_RAW):
    canonical = json.load(open(CANONICAL_RAW))
    with open(os.path.join(OUT_DIR, 'canonical_retrieval.json'), 'w') as handle:
        json.dump({'protocol': 'phase30a-2-full-3epoch-canonical-retrieval',
                   'note': ('Standard CLIP CLS retrieval (encode_image) after full training. '
                            'The base objective has no Global-text InfoNCE, so this checks it '
                            'does not damage canonical retrieval.'),
                   'raw': canonical.get('canonical', {})},
                  handle, indent=2, sort_keys=True)
    print('canonical retrieval merged')
else:
    print('WARNING: %s not found; canonical_retrieval.json not written' % CANONICAL_RAW)

print()
print('wrote:')
for name in sorted(os.listdir(OUT_DIR)):
    print('  %s (%d bytes)' % (name, os.path.getsize(os.path.join(OUT_DIR, name))))
print('paired scorers: %s' % sorted(by_scorer))
