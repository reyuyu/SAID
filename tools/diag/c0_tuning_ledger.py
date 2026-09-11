"""C0 tuning ledger: score a candidate's canonical JSON, apply the frozen promotion rule, update
docs/said_cls_cvssl/c0_tuning/{results.csv,progress.json}.

    python tools/diag/c0_tuning_ledger.py score <run_tag> <canonical.json>
    python tools/diag/c0_tuning_ledger.py status

Full-precision comparison only; the markdown table's 4-decimal values are never used for the gate.
"""
import csv
import json
import os
import sys

REPO = '/root/SAID-gap-completion'
SCREEN = os.path.join(REPO, 'outputs/cvssl_screening')
TUNING_DIR = os.path.join(REPO, 'docs/said_cls_cvssl/c0_tuning')
PROGRESS = os.path.join(TUNING_DIR, 'progress.json')
RESULTS = os.path.join(TUNING_DIR, 'results.csv')
S0_JSON = os.path.join(SCREEN, 'S0_canonical.json')
C0_JSON = os.path.join(SCREEN, 'C0_canonical.json')

BASELINE_ENTRY = 'S0_step500'
EXISTING_C0_ENTRY = 'C0_step500'

SHARE_VARIANTS = [('first', 'first_sentence'), ('sparse', 'fixed_sparse'), ('full', 'full_dense')]
FIELDS = ['run', 'lambda_target', 'u_warmup', 'tau_U', 'coco_i2t_r1', 'coco_i2t_r5',
          'coco_i2t_r10', 'coco_t2i_r1', 'coco_t2i_r5', 'coco_t2i_r10',
          'k1_first_i2t_r1', 'k1_first_t2i_r1', 'k1_sparse_i2t_r1', 'k1_sparse_t2i_r1',
          'k1_full_i2t_r1', 'k1_full_t2i_r1', 'J', 'delta_J_vs_S0', 'larger_screening_gain',
          'passed', 'canonical_json']


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def entry(payload, name):
    if name not in payload['canonical']:
        raise KeyError('%s not in %s' % (name, sorted(payload['canonical'])))
    return payload['canonical'][name]


def score(payload, name):
    e = entry(payload, name)
    coco = e['coco_val2017']
    row = {
        'coco_i2t_r1': float(coco['image2text_R1']),
        'coco_i2t_r5': float(coco['image2text_R5']),
        'coco_i2t_r10': float(coco['image2text_R10']),
        'coco_t2i_r1': float(coco['text2image_R1']),
        'coco_t2i_r5': float(coco['text2image_R5']),
        'coco_t2i_r10': float(coco['text2image_R10']),
    }
    for tag, variant in SHARE_VARIANTS:
        retrieval = e['sharegpt4v1k'][variant]['retrieval']
        row['k1_%s_i2t_r1' % tag] = float(retrieval['image2text_R1'])
        row['k1_%s_t2i_r1' % tag] = float(retrieval['text2image_R1'])
        row['k1_%s_i2t_r5' % tag] = float(retrieval['image2text_R5'])
        row['k1_%s_t2i_r5' % tag] = float(retrieval['text2image_R5'])
        row['k1_%s_i2t_r10' % tag] = float(retrieval['image2text_R10'])
        row['k1_%s_t2i_r10' % tag] = float(retrieval['text2image_R10'])
    row['J'] = 0.5 * (row['coco_i2t_r1'] + row['coco_t2i_r1'])
    return row


def load_results():
    if not os.path.exists(RESULTS):
        return []
    with open(RESULTS, newline='') as handle:
        return list(csv.DictReader(handle))


def save_results(rows):
    os.makedirs(TUNING_DIR, exist_ok=True)
    with open(RESULTS, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in FIELDS})


def load_progress():
    if os.path.exists(PROGRESS):
        return read_json(PROGRESS)
    return {'completed': [], 'best': None, 'next_candidate': 'C_L03', 'verdict': 'IN_PROGRESS'}


def save_progress(payload):
    os.makedirs(TUNING_DIR, exist_ok=True)
    with open(PROGRESS, 'w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def baseline():
    s0 = score(read_json(S0_JSON), BASELINE_ENTRY)
    existing = score(read_json(C0_JSON), EXISTING_C0_ENTRY)
    return s0, existing


def evaluate_gate(row, s0):
    left = row['coco_i2t_r1'] >= s0['coco_i2t_r1']
    right = row['coco_t2i_r1'] >= s0['coco_t2i_r1']
    strict = (row['coco_i2t_r1'] > s0['coco_i2t_r1']) or (row['coco_t2i_r1'] > s0['coco_t2i_r1'])
    return left, right, strict, bool(left and right and strict)


def command_score(run_tag, canonical_path):
    s0, existing = baseline()
    payload = read_json(canonical_path)
    names = sorted(payload['canonical'])
    if len(names) != 1:
        raise SystemExit('expected exactly one checkpoint in %s, got %s' % (canonical_path, names))
    row = score(payload, names[0])
    row['run'] = run_tag
    row['canonical_json'] = canonical_path
    left, right, strict, passed = evaluate_gate(row, s0)
    row['delta_J_vs_S0'] = row['J'] - s0['J']
    row['larger_screening_gain'] = bool(row['delta_J_vs_S0'] >= 0.003)
    row['passed'] = passed

    print('== %s' % run_tag)
    print('   COCO I2T R@1 %.10f  (gate >= %.10f)  %s'
          % (row['coco_i2t_r1'], s0['coco_i2t_r1'], 'PASS' if left else 'FAIL'))
    print('   COCO T2I R@1 %.10f  (gate >= %.10f)  %s'
          % (row['coco_t2i_r1'], s0['coco_t2i_r1'], 'PASS' if right else 'FAIL'))
    print('   at least one strictly greater: %s' % strict)
    print('   J = %.10f   J_S0 = %.10f   delta = %+.10f   larger_screening_gain=%s'
          % (row['J'], s0['J'], row['delta_J_vs_S0'], row['larger_screening_gain']))
    print('   PASSED = %s' % passed)
    for key in ('k1_first_i2t_r1', 'k1_first_t2i_r1', 'k1_sparse_i2t_r1', 'k1_sparse_t2i_r1',
                'k1_full_i2t_r1', 'k1_full_t2i_r1'):
        print('   %-20s %.4f' % (key, row[key]))

    rows = [r for r in load_results() if r['run'] != run_tag]
    rows.append({key: row.get(key, '') for key in FIELDS})
    order = ['S0', 'C_existing_L10', 'C_L03', 'C_L01', 'C_BEST_W200', 'C_BEST_T02']
    rows.sort(key=lambda r: order.index(r['run']) if r['run'] in order else 99)
    save_results(rows)

    progress = load_progress()
    completed = [entry_ for entry_ in progress.get('completed', []) if entry_['run'] != run_tag]
    completed.append({'run': run_tag, 'checkpoint': payload['canonical'][names[0]]['checkpoint'],
                      'canonical_json': canonical_path, 'J': row['J'],
                      'delta_J_vs_S0': row['delta_J_vs_S0'], 'passed': passed})
    progress['completed'] = completed
    candidates = [entry_ for entry_ in completed if entry_['run'] != 'S0']
    if candidates:
        best = max(candidates, key=lambda entry_: entry_['J'])
        progress['best'] = best
    progress['verdict'] = ('CANDIDATE_PASSED_500STEP_SCREENING' if passed
                           else progress.get('verdict', 'IN_PROGRESS'))
    save_progress(progress)

    # what is the best fixed-weight C0 so far, and the best C0 overall
    fixed = {}
    for row_ in rows:
        if row_['run'] in ('C_existing_L10', 'C_L03', 'C_L01'):
            fixed[row_['run']] = float(row_['J'])
    print()
    print('   fixed-weight C0 J: %s' % fixed)
    if fixed:
        best_fixed = max(fixed.items(), key=lambda item: (item[1], -{'C_existing_L10': 1.0,
                                                                     'C_L03': 0.3,
                                                                     'C_L01': 0.1}[item[0]]))
        print('   best fixed-weight C0: %s (J=%.10f)' % best_fixed)
    return 0


def command_status():
    s0, existing = baseline()
    print('S0@500      J=%.10f  I2T R@1=%.10f  T2I R@1=%.10f'
          % (s0['J'], s0['coco_i2t_r1'], s0['coco_t2i_r1']))
    print('C0-L10@500  J=%.10f  I2T R@1=%.10f  T2I R@1=%.10f  delta=%+.10f'
          % (existing['J'], existing['coco_i2t_r1'], existing['coco_t2i_r1'],
             existing['J'] - s0['J']))
    print()
    rows = load_results()
    if not rows:
        print('no candidate scored yet')
    for row in rows:
        print('%-16s J=%-14s delta=%-14s passed=%s' % (row['run'], row['J'],
                                                       row.get('delta_J_vs_S0'),
                                                       row.get('passed')))
    print()
    print(json.dumps(load_progress(), indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    if sys.argv[1] == 'score':
        sys.exit(command_score(sys.argv[2], sys.argv[3]))
    sys.exit(command_status())
