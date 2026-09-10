#!/usr/bin/env python3
"""Phase 3.0A.1d: merge the fixed-cohort evaluations into the committed summaries.

Reads the per-arm evaluation JSONs and writes the small tracked summaries:
``fixed_cohort_gap_summary.json``, ``fixed_cohort_usr_summary.json`` and
``loss_attribution_summary.json``. Large score matrices and checkpoints stay uncommitted.
"""
import json
import os

REPO = '/root/SAID-gap-completion'
EVAL_DIR = os.path.join(REPO, 'outputs', 'phase30a_fixed_semantic_eval')
OUT_DIR = os.path.join(REPO, 'docs', 'phase30a_fixed_semantic_eval')
os.makedirs(OUT_DIR, exist_ok=True)

ARMS = {
    'A_tau1.0_S_only': {'losses': 'L_S', 'summands': 'S'},
    'B_tau1.0_S+D': {'losses': 'L_S + L_gap', 'summands': 'S+D'},
    'C_tau1.0_S+D+A': {'losses': 'L_S + L_gap + L_absorb', 'summands': 'S+D+A'},
    'D_tau0.5_S+D+A': {'losses': 'L_S + L_gap + L_absorb', 'summands': 'S+D+A'},
}
CHECKPOINTS = ('initial', 'step20', 'step50', 'step100')

GAP_KEYS = (
    'said_attention_entropy', 'unsaid_attention_entropy',
    'said_unsaid_attention_overlap', 'said_unsaid_attention_jsd',
    'cos_said_unsaid', 'unsaid_novel_component_norm',
    'gap_before_mean', 'gap_after_mean', 'gap_reduction_mean',
    'gap_closure_ratio_mean', 'gap_closure_positive_fraction',
    'cos_global_said', 'cos_global_unsaid',
    'patch_pair_cosine_mean', 'patch_pair_cosine_std',
    'patch_centered_energy',
    'said_raw_pool_norm', 'unsaid_raw_pool_norm', 'raw_pool_cosine',
    'patch_centroid_norm', 'common_mode_ratio',
    'centered_said_norm', 'centered_unsaid_norm',
    'centered_pool_cosine', 'common_mode_fraction_said', 'common_mode_fraction_unsaid',
)
USR_SCORERS = ('global', 'unsaid', 'complete')
USR_KEYS = ('R@1', 'R@5', 'R@10', 'MRR', 'mean_rank', 'median_rank')

loaded = {}
for arm in ARMS:
    path = os.path.join(EVAL_DIR, '%s.json' % arm)
    if not os.path.exists(path):
        print('MISSING %s' % path)
        continue
    payload = json.load(open(path))
    loaded[arm] = payload
print('loaded arms: %s' % sorted(loaded))

common = {}
if loaded:
    first = next(iter(loaded.values()))
    common = {
        'protocol': first['protocol'],
        'usr_protocol': first['usr_protocol'],
        'usr_manifest': first['usr_manifest'],
        'usr_manifest_sha256': first['usr_manifest_sha256'],
        'cohort_sha256': first['cohort_sha256'],
        'cohort_query_count': first['cohort_query_count'],
        'cohort_candidate_pool': first['cohort_candidate_pool'],
        'c_u_is_evaluation_target_only': True,
        'notes': ('Every checkpoint is evaluated on the identical frozen cohort: same images, '
                  'same C_S, same C_U targets, same 868-candidate pool. No optimizer step is '
                  'taken during evaluation. z_U is built from (I, C_S) before any candidate '
                  'exists, and the score matrices are plain inner products over the full pool. '
                  'The Gap core loss and the Gap core math are unchanged.'),
    }

# ---------------------------------------------------------------- fixed cohort gap
gap_summary = dict(common)
gap_summary['trajectory'] = {}
for arm, payload in loaded.items():
    entry = {'losses': ARMS[arm]['losses'], 'gap_anti_temperature':
             payload['gap_anti_temperature'], 'checkpoints': {}}
    for tag in CHECKPOINTS:
        checkpoint = payload['checkpoints'].get(tag)
        if checkpoint is None:
            continue
        internal = checkpoint['internal']
        entry['checkpoints'][tag] = {
            'checkpoint_step': checkpoint['checkpoint_step'],
            'checkpoint_phase': checkpoint['checkpoint_phase'],
            'checkpoint_objective_mode': checkpoint['checkpoint_objective_mode'],
            'metrics': {key: internal[key] for key in GAP_KEYS if key in internal},
            'common_mode_diagnosis': internal['common_mode_diagnosis'],
        }
    gap_summary['trajectory'][arm] = entry
gap_summary['checkpoint_order'] = list(CHECKPOINTS)
with open(os.path.join(OUT_DIR, 'fixed_cohort_gap_summary.json'), 'w') as handle:
    json.dump(gap_summary, handle, indent=2, sort_keys=True)

# ---------------------------------------------------------------- fixed cohort USR
usr_summary = dict(common)
usr_summary['trajectory'] = {}
for arm, payload in loaded.items():
    entry = {'losses': ARMS[arm]['losses'], 'gap_anti_temperature':
             payload['gap_anti_temperature'], 'checkpoints': {}}
    for tag in CHECKPOINTS:
        checkpoint = payload['checkpoints'].get(tag)
        if checkpoint is None:
            continue
        entry['checkpoints'][tag] = {
            'checkpoint_step': checkpoint['checkpoint_step'],
            'reports': {name: {key: checkpoint['usr']['reports'][name][key] for key in USR_KEYS}
                        for name in USR_SCORERS},
            'cross_scorer_rank_delta': checkpoint['usr']['cross_scorer_rank_delta'],
            'candidate_pool': checkpoint['usr']['candidate_pool'],
            'query_count': checkpoint['usr']['query_count'],
        }
    usr_summary['trajectory'][arm] = entry
n_queries = usr_summary.get('cohort_query_count') or 0
usr_summary['random_chance'] = {
    'R@1': 1.0 / n_queries if n_queries else None,
    'R@5': min(5.0 / n_queries, 1.0) if n_queries else None,
    'R@10': min(10.0 / n_queries, 1.0) if n_queries else None,
}
with open(os.path.join(OUT_DIR, 'fixed_cohort_usr_summary.json'), 'w') as handle:
    json.dump(usr_summary, handle, indent=2, sort_keys=True)


def value(payload, arm, tag, section, key, scorer=None):
    checkpoint = payload.get(arm, {}).get('checkpoints', {}).get(tag)
    if checkpoint is None:
        return None
    if section == 'internal':
        return checkpoint['internal'].get(key)
    return checkpoint['usr']['reports'][scorer].get(key)


# ---------------------------------------------------------------- loss attribution
attribution = {
    'protocol': 'phase30a-1d-loss-attribution',
    'design': ('Three arms identical in every respect except the two gap weights, all at '
               'tau=1.0: A = L_S (0,0), B = L_S + L_gap_discover (1,0), '
               'C = L_S + L_gap_discover + L_global_absorb (1,1). C is the existing '
               'runs_salu/phase30a_1c_tau1 arm, reused rather than re-run.'),
    'arm_losses': {arm: ARMS[arm]['losses'] for arm in ARMS},
    'cohort': {key: common.get(key) for key in
               ('cohort_sha256', 'cohort_query_count', 'cohort_candidate_pool')},
    'stream_identity': {},
    'checkpoints': list(CHECKPOINTS),
    'by_checkpoint': {},
    'headline': {},
}
# stream identity from the training logs (already verified for A/B/C by the run scripts)
for arm, run in (('A_tau1.0_S_only', 'phase30a_1d_A_said_only'),
                 ('B_tau1.0_S+D', 'phase30a_1d_B_said_discover'),
                 ('C_tau1.0_S+D+A', 'phase30a_1c_tau1')):
    audit_path = os.path.join(REPO, 'runs_salu', run, 'reproducibility_rank0.json')
    if os.path.exists(audit_path):
        audit = json.load(open(audit_path))
        attribution['stream_identity'][arm] = {
            'initial_state_sha256': audit['initial_state_sha256'],
            'sampler_order_sha256': audit['sampler_order_sha256'],
            'caption_stream_sha256': audit['caption_stream_sha256'],
            'full_caption_stream_sha256': audit['full_caption_stream_sha256'],
            'unsaid_caption_stream_sha256': audit['unsaid_caption_stream_sha256'],
        }
if attribution['stream_identity']:
    keys = ('initial_state_sha256', 'sampler_order_sha256', 'caption_stream_sha256')
    attribution['stream_identity_all_matched'] = all(
        len({entry[key] for entry in attribution['stream_identity'].values()}) == 1
        for key in keys)

for tag in CHECKPOINTS:
    row = {}
    for arm in ('A_tau1.0_S_only', 'B_tau1.0_S+D', 'C_tau1.0_S+D+A', 'D_tau0.5_S+D+A'):
        if arm not in loaded:
            continue
        row[arm] = {
            'z_unsaid_usr': {key: value(loaded, arm, tag, 'usr', key, scorer='unsaid')
                             for key in USR_KEYS},
            'global_usr': {key: value(loaded, arm, tag, 'usr', key, scorer='global')
                           for key in USR_KEYS},
            'complete_usr': {key: value(loaded, arm, tag, 'usr', key, scorer='complete')
                             for key in USR_KEYS},
            'internal_closure_ratio_mean': value(loaded, arm, tag, 'internal',
                                                 'gap_closure_ratio_mean'),
            'internal_gap_after_mean': value(loaded, arm, tag, 'internal', 'gap_after_mean'),
            'internal_gap_before_mean': value(loaded, arm, tag, 'internal', 'gap_before_mean'),
            'internal_closure_positive_fraction': value(loaded, arm, tag, 'internal',
                                                        'gap_closure_positive_fraction'),
            'cos_said_unsaid': value(loaded, arm, tag, 'internal', 'cos_said_unsaid'),
            'unsaid_novel_component_norm': value(loaded, arm, tag, 'internal',
                                                 'unsaid_novel_component_norm'),
        }
    attribution['by_checkpoint'][tag] = row

# headline contrasts at step100 on the fixed cohort
for tag in ('step100',):
    row = attribution['by_checkpoint'].get(tag, {})
    def pick(arm, path, key, scorer=None):
        entry = row.get(arm)
        if not entry:
            return None
        if path == 'usr':
            return entry['%s_usr' % scorer][key]
        return entry.get(path)
    attribution['headline'][tag] = {
        'z_unsaid_R@1_A': pick('A_tau1.0_S_only', 'usr', 'R@1', 'z_unsaid'),
        'z_unsaid_R@1_B': pick('B_tau1.0_S+D', 'usr', 'R@1', 'z_unsaid'),
        'z_unsaid_R@1_C': pick('C_tau1.0_S+D+A', 'usr', 'R@1', 'z_unsaid'),
        'global_R@1_A': pick('A_tau1.0_S_only', 'usr', 'R@1', 'global'),
        'global_R@1_B': pick('B_tau1.0_S+D', 'usr', 'R@1', 'global'),
        'global_R@1_C': pick('C_tau1.0_S+D+A', 'usr', 'R@1', 'global'),
        'closure_A': pick('A_tau1.0_S_only', 'internal_closure_ratio_mean', None),
        'closure_B': pick('B_tau1.0_S+D', 'internal_closure_ratio_mean', None),
        'closure_C': pick('C_tau1.0_S+D+A', 'internal_closure_ratio_mean', None),
    }
with open(os.path.join(OUT_DIR, 'loss_attribution_summary.json'), 'w') as handle:
    json.dump(attribution, handle, indent=2, sort_keys=True)

# ---------------------------------------------------------------- canonical retrieval
CANONICAL_VARIANTS = ('first_sentence', 'fixed_sparse', 'full_dense')
canonical_summary = {
    'protocol': 'phase30a-1d-canonical-retrieval',
    'note': ('Standard CLIP CLS retrieval via encode_image(). The base objective has no '
             'Global-text InfoNCE, so this only checks that it does not break canonical '
             'retrieval. Measured on the initial and step-100 checkpoints only.'),
    'sharegpt4v1k': {},
    'coco_val2017': {},
}
for arm in ARMS:
    path = os.path.join(EVAL_DIR, '%s_canonical.json' % arm)
    if not os.path.exists(path):
        continue
    payload = json.load(open(path)).get('canonical', {})
    canonical_summary['sharegpt4v1k'][arm] = {}
    canonical_summary['coco_val2017'][arm] = {}
    for tag in ('initial', 'step100'):
        entry = payload.get(tag) or {}
        one_k = entry.get('sharegpt4v1k') or {}
        canonical_summary['sharegpt4v1k'][arm][tag] = {
            variant: one_k[variant]['retrieval'] for variant in CANONICAL_VARIANTS
            if variant in one_k}
        if entry.get('coco_val2017'):
            canonical_summary['coco_val2017'][arm][tag] = entry['coco_val2017']
with open(os.path.join(OUT_DIR, 'canonical_retrieval_summary.json'), 'w') as handle:
    json.dump(canonical_summary, handle, indent=2, sort_keys=True)

print()
print('wrote:')
for name in sorted(os.listdir(OUT_DIR)):
    print('  %s (%d bytes)' % (name, os.path.getsize(os.path.join(OUT_DIR, name))))
