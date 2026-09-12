"""SAID-Token v1 (T1): training trajectory, fixed-cohort diagnostics and the retrieval table.

    python tools/diag/t1_summary.py                 # trajectory + diagnostics of the 500-step run
    python tools/diag/t1_summary.py --retrieval     # canonical COCO + Urban-1k table and the gate

The retrieval numbers of Initial / S0 / C0 / C1 are read from the existing committed result files, so
this tool never re-runs an old evaluation and never invents a number.
"""
import argparse
import json
import os

REPO = '/root/SAID-token-v1'
C0 = '/root/SAID-gap-completion'
C1 = '/root/SAID-c1-tcr'
RUN = os.path.join(REPO, 'runs_salu/said_token_v1/t1_500step')
OUT = os.path.join(REPO, 'outputs/said_token_v1')

# the frozen 500-step gate: both floors must hold and at least one must be strictly beaten
GATE_I2T_R1 = 0.6058000000
GATE_T2I_R1 = 0.4123600000

TRAJECTORY = ['loss_said_global_mean', 'loss_said_i2t', 'loss_said_t2i', 'loss_rec_global_mean',
              'loss_total_global_mean', 'positive_said_score', 'negative_said_score',
              'margin_active_fraction', 'rec_valid_fraction', 'router_logit_std', 'soft_gate_mean',
              'selected_router_score', 'excluded_router_score', 'u_raw_norm',
              'cos_prediction_reference', 'said_token_count', 'unsaid_token_count',
              'said_scaling_abs_diff', 'rec_scaling_abs_diff', 'said_rank_spread',
              'native_cls_pairwise_cos', 'aggregation_token_pairwise_cos', 'empty_text_count',
              'sec_per_step', 'peak_memory_gb']
DIAGNOSTICS = ['diag_cos_error_normal', 'diag_delta_zero_text', 'diag_delta_zero_u',
               'diag_delta_permuted_text', 'diag_delta_wrong_u', 'diag_constant_baseline_error',
               'diag_pred_r1_teacher_normal', 'diag_pred_variance_normal',
               'diag_gate_overlap_with_permuted', 'diag_router_logit_std', 'diag_empty_text']


def read_log(path):
    records = []
    if not os.path.exists(path):
        return records
    with open(path) as handle:
        for line in handle:
            if line.startswith('LOG '):
                records.append(json.loads(line[4:]))
    return records


def print_table(records, keys, title):
    rows = [record for record in records if any(key in record for key in keys)]
    if not rows:
        print('%s: NOT RUN' % title)
        return
    print('=== %s ===' % title)
    print('| step | ' + ' | '.join(key.replace('diag_', '').replace('_global_mean', '')
                                   for key in keys) + ' |')
    print('|' + '---|' * (len(keys) + 1))
    for record in rows:
        cells = []
        for key in keys:
            value = record.get(key)
            cells.append('n/a' if value is None else '%.5f' % value)
        print('| %d | %s |' % (record['completed_steps'], ' | '.join(cells)))
    print()


def load_canonical(path, name):
    if not os.path.exists(path):
        return None
    payload = json.load(open(path))
    node = payload.get('canonical', {})
    key = name if name in node else (sorted(node)[0] if node else None)
    return node.get(key) if key else None


def coco_pair(entry):
    if entry is None:
        return None, None
    node = entry.get('coco_val2017') or {}
    if 'retrieval' in node:
        node = node['retrieval']
    return node.get('image2text_R1'), node.get('text2image_R1')


def urban_pair(path):
    """Urban-1k numbers live in ``urban1k.image2text.R1`` / ``urban1k.text2image.R1``."""
    if not os.path.exists(path):
        return None, None
    payload = json.load(open(path))
    node = payload.get('urban1k', payload)
    return (node.get('image2text', {}).get('R1'), node.get('text2image', {}).get('R1'))


def retrieval_table():
    sources = [
        ('Initial', os.path.join(C0, 'outputs/cvssl_screening/S0_canonical.json'), 'Initial',
         os.path.join(C0, 'outputs/cvssl_screening/baseline_urban1k/Initial_urban1k.json')),
        ('S0@500', os.path.join(C0, 'outputs/cvssl_screening/S0_canonical.json'), 'S0_step500',
         os.path.join(C0, 'outputs/cvssl_screening/baseline_urban1k/S0_step500_urban1k.json')),
        ('C0@500', os.path.join(C0, 'outputs/cvssl_screening/C0_canonical.json'), 'C0_step500',
         os.path.join(C0, 'outputs/cvssl_screening/baseline_urban1k/C0_step500_urban1k.json')),
        ('C1@500', os.path.join(C1, 'outputs/cvssl_screening/c1_tcr/C1_step500_canonical.json'),
         None, os.path.join(C0, 'outputs/cvssl_screening/baseline_urban1k/C1_step500_urban1k.json')),
        ('T1@500', os.path.join(OUT, 'T1_step500_canonical.json'), None,
         os.path.join(C0, 'outputs/cvssl_screening/baseline_urban1k/T1_step500_urban1k.json')),
    ]
    table = {}
    print('=== native student CLS retrieval (R@1, full precision) ===')
    print('| arm | COCO I2T R@1 | COCO T2I R@1 | Urban-1k I2T R@1 | Urban-1k T2I R@1 |')
    print('|---|---|---|---|---|')
    for label, coco_path, coco_name, urban_path in sources:
        i2t, t2i = coco_pair(load_canonical(coco_path, coco_name))
        u_i2t, u_t2i = urban_pair(urban_path)
        table[label] = {'coco_i2t_r1': i2t, 'coco_t2i_r1': t2i,
                        'urban_i2t_r1': u_i2t, 'urban_t2i_r1': u_t2i}
        cells = ['%.4f' % value if value is not None else 'n/a'
                 for value in (i2t, t2i, u_i2t, u_t2i)]
        print('| %s | %s |' % (label, ' | '.join(cells)))
    print()
    if table['S0@500']['coco_i2t_r1'] is not None and table['T1@500']['coco_i2t_r1'] is not None:
        base_i2t = table['S0@500']['coco_i2t_r1']
        base_t2i = table['S0@500']['coco_t2i_r1']
        t1_i2t = table['T1@500']['coco_i2t_r1']
        t1_t2i = table['T1@500']['coco_t2i_r1']
        floors = (t1_i2t >= GATE_I2T_R1 and t1_t2i >= GATE_T2I_R1)
        strict = (t1_i2t > base_i2t) or (t1_t2i > base_t2i)
        print('GATE floors_hold=%s strict_improvement=%s pass=%s' % (floors, strict, floors and strict))
        print('gate detail: I2T %.10f vs floor %.10f (S0 %.10f), T2I %.10f vs floor %.10f (S0 %.10f)'
              % (t1_i2t, GATE_I2T_R1, base_i2t, t1_t2i, GATE_T2I_R1, base_t2i))
    else:
        print('GATE: NOT RUN (a retrieval result is missing)')
    deltas = {}
    for label in ('C0@500', 'C1@500', 'T1@500'):
        for metric in ('coco_i2t_r1', 'coco_t2i_r1'):
            left, right = table[label][metric], table['S0@500'][metric]
            deltas['%s_%s_vs_S0' % (label, metric)] = (None if left is None or right is None
                                                       else left - right)
    return table, deltas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=RUN)
    parser.add_argument('--retrieval', action='store_true')
    parser.add_argument('--out', default=os.path.join(OUT, 't1_summary.json'))
    parsed = parser.parse_args()

    records = read_log(os.path.join(parsed.run, 'salu_log.jsonl'))
    print_table(records, TRAJECTORY, 'training trajectory (fixed cohort, rank 0)')
    print_table(records, DIAGNOSTICS, 'input-dependency diagnostics (fixed 64-image cohort)')
    payload = {'steps_logged': [record['completed_steps'] for record in records],
               'steps': records}
    summary_path = os.path.join(parsed.run, 'run_summary.json')
    if os.path.exists(summary_path):
        payload['run_summary'] = json.load(open(summary_path))
    if parsed.retrieval:
        table, deltas = retrieval_table()
        payload['retrieval'] = table
        payload['retrieval_deltas_vs_S0'] = deltas
    os.makedirs(os.path.dirname(parsed.out), exist_ok=True)
    with open(parsed.out, 'w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print('SUMMARY %s' % parsed.out)


if __name__ == '__main__':
    main()
