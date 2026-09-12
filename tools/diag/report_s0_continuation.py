"""Read-only comparison report for the S0 / S0_TriMask_HS continuations (500 -> 1000 updates).

Collects the *standard* evaluator outputs only -- the canonical COCO val2017 table
(``tools/phase30a_fixed_cohort_eval.py --canonical --coco``) and the native-CLS Urban-1k table
(``/root/SAID-gap-completion/tools/eval_urban1k_cls.py``) -- for four checkpoints:

    S0_smartclip  @500 (frozen baseline) and @1000 (this continuation)
    S0_TriMask_HS @500 (frozen)          and @1000 (this continuation)

and reports the two comparisons that matter:

* across arms at the same budget (HS vs S0 at 500 and at 1000), i.e. "what does the hard text
  gate cost or buy", and
* within an arm across budgets (@500 -> @1000), i.e. "what did 500 more updates do".

Every number is the evaluator's own output, with the checkpoint sha256 carried through. The
frozen promotion gate is defined at 500 updates only: the report states the 500-step verdict as
a reference row and explicitly refuses to apply the gate to the 1000-step budget. When the paired
per-query files from ``tools/diag/coco_query_hits.py`` exist, their McNemar/bootstrap summary is
attached, because a 0.5 pp difference on 5,000 queries is inside the binomial noise and the point
estimates alone cannot decide it.

    python tools/diag/report_s0_continuation.py --out docs/s0_continuation/results.json \
        --markdown docs/s0_continuation/report.md
"""
import argparse
import json
import os
import subprocess

S0_500_CANONICAL = '/root/SAID-gap-completion/outputs/cvssl_screening/S0_canonical.json'
S0_500_URBAN = ('/root/SAID-gap-completion/outputs/cvssl_screening/baseline_urban1k/'
                'S0_step500_urban1k.json')
S0_1000_DIR = '/root/SAID-s0-continue-v01/runs_salu/said_cls_cvssl/s0_continue_1000'
HS_DIR = '/root/SAID-s0-trimask-hs-v02/runs_salu/said_s0_trimask_hs_v02'
HS_500_DIR = os.path.join(HS_DIR, 'step500')
HS_1000_DIR = os.path.join(HS_DIR, 'cont1000')
GATE = {'coco_i2t_r1': 0.6058, 'coco_t2i_r1': 0.41236}
GATE_NOTE = ('the frozen promotion gate is defined at 500 optimizer updates only; it is applied '
             'to the @500 rows as a reference and is NOT applied to the @1000 rows, which are a '
             '1000-step budget')
# binomial standard error of R@1, quoted so a reader can see the scale of "different"
BINOMIAL_N = {'coco': 5000, 'urban': 1000}


def read_json(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def sha256_of(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_metrics(path, label=None):
    """COCO val2017 R@K from a canonical-eval JSON; the label is auto-discovered when omitted."""
    payload = read_json(path)
    canonical = payload.get('canonical') or {}
    if label is None:
        keys = [key for key, value in canonical.items()
                if isinstance(value, dict) and 'coco_val2017' in value]
        if len(keys) != 1:
            raise SystemExit('%s holds %d evaluable canonical labels, pass one of %s'
                             % (path, len(keys), list(canonical)))
        label = keys[0]
    body = canonical[label]['coco_val2017']
    return label, {name: body[name] for name in sorted(body)}


def urban_metrics(path):
    payload = read_json(path)
    body = payload['urban1k']
    return {'i2t_r1': body['image2text']['R1'], 'i2t_r5': body['image2text']['R5'],
            'i2t_r10': body['image2text']['R10'], 't2i_r1': body['text2image']['R1'],
            't2i_r5': body['text2image']['R5'], 't2i_r10': body['text2image']['R10'],
            'n_images': body.get('n_images'), 'n_captions': body.get('n_captions')}


def optional_json(path):
    if path and os.path.isfile(path):
        return read_json(path)
    return None


def arm(name, checkpoint_step, canonical_path, urban_path, run_status_path=None,
        paired_path=None, canonical_label=None):
    entry = {'arm': name, 'checkpoint_step': checkpoint_step,
             'canonical_json': canonical_path, 'urban_json': urban_path}
    for key, path in (('canonical', canonical_path), ('urban', urban_path)):
        if not os.path.isfile(path):
            entry[key] = None
            entry['missing_%s' % key] = path
            continue
        entry['%s_sha256' % key] = sha256_of(path)
    if os.path.isfile(canonical_path):
        label, coco = canonical_metrics(canonical_path, canonical_label)
        entry['canonical_label'] = label
        entry['coco'] = coco
    if os.path.isfile(urban_path):
        entry['urban'] = urban_metrics(urban_path)
    if run_status_path and os.path.isfile(run_status_path):
        status = read_json(run_status_path)
        entry['run'] = {key: status.get(key) for key in
                        ('phase', 'attempt', 'max_steps', 'resume', 'save_steps',
                         'completed_steps', 'wall_seconds', 'exit_codes', 'conclusion',
                         'started_at_iso')}
    if paired_path:
        entry['paired'] = optional_json(paired_path)
    return entry


def delta(left, right, keys):
    out = {}
    for key in keys:
        if left is None or right is None or key not in left or key not in right:
            continue
        out[key] = {'left': left[key], 'right': right[key], 'delta': left[key] - right[key],
                    'delta_pp': 100.0 * (left[key] - right[key])}
    return out


def binomial_se(value, n):
    return (value * (1.0 - value) / float(n)) ** 0.5


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--s0-1000-dir', default=S0_1000_DIR)
    parser.add_argument('--hs-500-dir', default=HS_500_DIR)
    parser.add_argument('--hs-1000-dir', default=HS_1000_DIR)
    parser.add_argument('--paired', default=None,
                        help='combined paired-comparison JSON from tools/diag/coco_paired_report.py; '
                             'attached to both @1000 arms when given')
    parser.add_argument('--out', required=True)
    parser.add_argument('--markdown', default=None)
    args = parser.parse_args()

    s0k = os.path.join(args.s0_1000_dir, 'evaluation')
    h5 = os.path.join(args.hs_500_dir, 'evaluation')
    hk = os.path.join(args.hs_1000_dir, 'evaluation')

    arms = [
        # the frozen S0 file holds three labels (Initial / S0_step100 / S0_step500); the baseline is
        # named explicitly so the wrong row can never be picked up silently
        arm('S0_smartclip', 500, S0_500_CANONICAL, S0_500_URBAN, canonical_label='S0_step500'),
        arm('S0_smartclip', 1000,
            os.path.join(s0k, 'S0_smartclip_step001000_canonical.json'),
            os.path.join(s0k, 'S0_smartclip_step001000_urban1k.json'),
            run_status_path=os.path.join(args.s0_1000_dir, 'run_status.json'),
            paired_path=args.paired or os.path.join(s0k, 'paired_S0_smartclip_500_vs_1000.json')),
        arm('S0_TriMask_HS', 500,
            os.path.join(h5, 'S0_TriMask_HS_step000500_canonical.json'),
            os.path.join(h5, 'S0_TriMask_HS_step000500_urban1k.json'),
            run_status_path=os.path.join(args.hs_500_dir, 'run_status.json')),
        arm('S0_TriMask_HS', 1000,
            os.path.join(hk, 'S0_TriMask_HS_step001000_canonical.json'),
            os.path.join(hk, 'S0_TriMask_HS_step001000_urban1k.json'),
            run_status_path=os.path.join(args.hs_1000_dir, 'run_status.json'),
            paired_path=args.paired or os.path.join(hk, 'paired_S0_TriMask_HS_500_vs_1000.json')),
    ]
    by_key = {(entry['arm'], entry['checkpoint_step']): entry for entry in arms}
    # the frozen baseline must reproduce the gate values themselves; if it does not, the wrong row
    # of the frozen multi-label file was picked up and every delta below would be meaningless
    baseline = by_key[('S0_smartclip', 500)].get('coco')
    if baseline:
        for key, expected in (('image2text_R1', GATE['coco_i2t_r1']),
                              ('text2image_R1', GATE['coco_t2i_r1'])):
            if abs(baseline[key] - expected) > 1e-9:
                raise SystemExit('the frozen S0@500 row reads %s=%r but the gate definition says %r; '
                                 'refusing to report differences against the wrong baseline'
                                 % (key, baseline[key], expected))
    have = {key: (entry.get('coco') is not None and entry.get('urban') is not None)
            for key, entry in by_key.items()}

    coco_keys = ('image2text_R1', 'text2image_R1', 'image2text_R5', 'text2image_R5',
                 'image2text_R10', 'text2image_R10')
    urban_keys = ('i2t_r1', 't2i_r1', 'i2t_r5', 't2i_r5', 'i2t_r10', 't2i_r10')

    comparisons = {}
    if have[('S0_smartclip', 500)] and have[('S0_TriMask_HS', 500)]:
        comparisons['HS_minus_S0_at_500'] = {
            'coco': delta(by_key[('S0_TriMask_HS', 500)]['coco'],
                          by_key[('S0_smartclip', 500)]['coco'], coco_keys),
            'urban': delta(by_key[('S0_TriMask_HS', 500)]['urban'],
                           by_key[('S0_smartclip', 500)]['urban'], urban_keys)}
    if have[('S0_smartclip', 1000)] and have[('S0_TriMask_HS', 1000)]:
        comparisons['HS_minus_S0_at_1000'] = {
            'coco': delta(by_key[('S0_TriMask_HS', 1000)]['coco'],
                          by_key[('S0_smartclip', 1000)]['coco'], coco_keys),
            'urban': delta(by_key[('S0_TriMask_HS', 1000)]['urban'],
                           by_key[('S0_smartclip', 1000)]['urban'], urban_keys)}
    for name in ('S0_smartclip', 'S0_TriMask_HS'):
        if have[(name, 500)] and have[(name, 1000)]:
            comparisons['%s_1000_minus_500' % name] = {
                'coco': delta(by_key[(name, 1000)]['coco'], by_key[(name, 500)]['coco'],
                              coco_keys),
                'urban': delta(by_key[(name, 1000)]['urban'], by_key[(name, 500)]['urban'],
                               urban_keys)}

    noise = {}
    for key, entry in by_key.items():
        if entry.get('coco'):
            noise['%s@%d_coco' % key] = {
                'binomial_se_i2t_r1': binomial_se(entry['coco']['image2text_R1'],
                                                  BINOMIAL_N['coco']),
                'binomial_se_t2i_r1': binomial_se(entry['coco']['text2image_R1'],
                                                  BINOMIAL_N['coco']),
                'n_queries_i2t': BINOMIAL_N['coco'], 'n_queries_t2i': 5 * BINOMIAL_N['coco']}
        if entry.get('urban'):
            noise['%s@%d_urban' % key] = {
                'binomial_se_i2t_r1': binomial_se(entry['urban']['i2t_r1'], BINOMIAL_N['urban']),
                'binomial_se_t2i_r1': binomial_se(entry['urban']['t2i_r1'], BINOMIAL_N['urban']),
                'n_queries_i2t': BINOMIAL_N['urban'], 'n_queries_t2i': BINOMIAL_N['urban']}

    gate_rows = {}
    for name, entry in (('S0_smartclip', by_key[('S0_smartclip', 500)]),
                        ('S0_TriMask_HS', by_key[('S0_TriMask_HS', 500)])):
        coco = entry.get('coco')
        if not coco:
            continue
        i2t, t2i = coco['image2text_R1'], coco['text2image_R1']
        is_reference = (abs(i2t - GATE['coco_i2t_r1']) < 1e-9 and abs(t2i - GATE['coco_t2i_r1']) < 1e-9)
        body = {
            'coco_i2t_r1': i2t, 'coco_t2i_r1': t2i,
            'gate_coco_i2t_r1': GATE['coco_i2t_r1'], 'gate_coco_t2i_r1': GATE['coco_t2i_r1'],
            'i2t_at_or_above_gate': i2t >= GATE['coco_i2t_r1'],
            't2i_at_or_above_gate': t2i >= GATE['coco_t2i_r1'],
            'strictly_higher_count': int(i2t > GATE['coco_i2t_r1']) + int(t2i > GATE['coco_t2i_r1']),
            'is_gate_reference': is_reference,
        }
        if is_reference:
            # the baseline IS the gate definition: its two values are exactly the thresholds, so the
            # "at least one strictly higher" clause cannot apply to it. Applying the challenger rule
            # to the reference would report the baseline as failing its own gate.
            body['conclusion'] = ('the frozen baseline that defines the gate; the "at least one '
                                  'strictly higher" clause is a challenger rule and cannot apply to '
                                  'the reference itself')
        elif (i2t >= GATE['coco_i2t_r1'] and t2i >= GATE['coco_t2i_r1']
              and (i2t > GATE['coco_i2t_r1'] or t2i > GATE['coco_t2i_r1'])):
            body['conclusion'] = 'passed the frozen 500-step promotion gate'
        else:
            body['conclusion'] = 'did not pass the frozen 500-step promotion gate'
        gate_rows[name] = body

    git_heads = {}
    for directory in (args.s0_1000_dir, args.hs_1000_dir):
        if os.path.isdir(directory):
            head = subprocess.run(['git', '-C', directory, 'rev-parse', 'HEAD'],
                                  capture_output=True, text=True)
            if head.returncode == 0:
                git_heads[directory] = head.stdout.strip()

    payload = {
        'protocol': 's0-continuation-500-to-1000-comparison',
        'gate': dict(GATE, note=GATE_NOTE),
        'gate_check_at_500': gate_rows,
        'noise_scale': noise,
        'arms': arms,
        'comparisons': comparisons,
        'worktree_head': git_heads,
        'notes': [
            'Every metric is copied from the standard evaluators; nothing is recomputed here.',
            'coco = tools/phase30a_fixed_cohort_eval.py --canonical --coco (legacy_cls column, '
            '5 captions per image); urban = /root/SAID-gap-completion/tools/eval_urban1k_cls.py '
            '(native CLS, plain inner product over the full 1000-item pool).',
            GATE_NOTE,
            'binomial_se_* is the standard error of an R@1 estimate treated as an independent '
            'binomial sample: it is the scale of "a different-looking number", not a test. '
            'Paired per-query evidence, when present, is attached per arm.',
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1, sort_keys=True)

    lines = ['# S0 vs S0_TriMask_HS: 500 -> 1000 optimizer updates', '',
             'All numbers are the standard evaluators\' own output. ' + GATE_NOTE + '.', '',
             '## Metrics per checkpoint', '',
             '| arm | step | COCO I2T R@1 | COCO T2I R@1 | Urban I2T R@1 | Urban T2I R@1 |',
             '| --- | --- | --- | --- | --- | --- |']
    for entry in arms:
        coco = entry.get('coco') or {}
        urban = entry.get('urban') or {}
        def fmt(value):
            return 'n/a' if value is None else '%.4f' % value
        lines.append('| %s | %d | %s | %s | %s | %s |'
                     % (entry['arm'], entry['checkpoint_step'],
                        fmt(coco.get('image2text_R1')), fmt(coco.get('text2image_R1')),
                        fmt(urban.get('i2t_r1')), fmt(urban.get('t2i_r1'))))
    lines += ['', '## Differences (pp = percentage points)', '']
    for name, body in comparisons.items():
        lines.append('### %s' % name)
        lines.append('')
        lines.append('| metric | left | right | delta | delta (pp) |')
        lines.append('| --- | --- | --- | --- | --- |')
        for family in ('coco', 'urban'):
            for key, values in sorted(body.get(family, {}).items()):
                lines.append('| %s/%s | %.4f | %.4f | %+.4f | %+.2f |'
                             % (family, key, values['left'], values['right'],
                                values['delta'], values['delta_pp']))
        lines.append('')
    lines += ['## Frozen 500-step gate (reference row, not applied at 1000)', '',
              '| arm | COCO I2T R@1 | COCO T2I R@1 | verdict |', '| --- | --- | --- | --- |']
    for name, body in gate_rows.items():
        lines.append('| %s | %.4f | %.4f | %s |'
                     % (name, body['coco_i2t_r1'], body['coco_t2i_r1'], body['conclusion']))
    lines += ['', '## Paired per-query evidence (same queries, identical protocol)', '']
    paired_payload = None
    for entry in arms:
        if entry.get('paired'):
            paired_payload = entry['paired']
            break
    if paired_payload:
        lines.append('delta = left - right, positive means the left arm is better. Paired '
                     'bootstrap over the same query indices (%s replicates, seed %s); McNemar is '
                     'the exact two-sided test on the discordant queries.'
                     % (paired_payload.get('replicates'), paired_payload.get('seed')))
        lines.append('')
        lines.append('| comparison | direction | R@1 left | R@1 right | delta (pp) | 95% CI (pp) | '
                     'excludes 0 | left-only | right-only | McNemar p |')
        lines.append('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |')
        for row in paired_payload.get('comparisons', []):
            for direction in ('image2text', 'text2image'):
                body = ((row.get('directions') or {}).get(direction) or {}).get('R@1')
                if not body:
                    continue
                boot = body.get('paired_bootstrap') or {}
                mcn = body.get('mcnemar') or {}
                lines.append('| %s vs %s | %s | %.4f | %.4f | %+.2f | [%+.2f, %+.2f] | %s | %s | %s | %s |'
                             % (row.get('left'), row.get('right'), direction,
                                body['left'], body['right'], 100.0 * body['delta'],
                                100.0 * float(boot.get('ci_low', float('nan'))),
                                100.0 * float(boot.get('ci_high', float('nan'))),
                                boot.get('excludes_zero'),
                                mcn.get('n10_left_only'), mcn.get('n01_right_only'),
                                mcn.get('p_value_exact_two_sided')))
        lines.append('')
    else:
        lines.append('_not run for these checkpoints; the comparisons above are point estimates '
                     'and the binomial scale in results.json is the only noise reference._')
    lines.append('')
    if args.markdown:
        os.makedirs(os.path.dirname(os.path.abspath(args.markdown)), exist_ok=True)
        with open(args.markdown, 'w', encoding='utf-8') as handle:
            handle.write('\n'.join(lines) + '\n')
    print('WROTE %s' % args.out)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
