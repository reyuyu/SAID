"""Read-only comparison of the S0-TriMask-HS loss-weight arms at exactly 500 updates.

    S0_smartclip @500      the frozen gate baseline (defines the thresholds; not a challenger)
    S0_TriMask_HS @500     the frozen v0.2 weighting      lambda = 10 / 1 / 1  + 2 / 0.2
    S0_TriMask_HS_BAL @500 the balanced weighting         lambda = 10 / 10 / 10 + 2 / 2

Only the standard evaluator outputs are read (COCO canonical + Urban-1k), so nothing here can
change a metric: the frozen promotion gate is applied to the two challengers exactly as it was to
the original @500 run, and the baseline row is reported as the reference instead of being judged by
a rule whose "at least one strictly higher" clause cannot hold for the reference itself.

    python tools/diag/report_trimask_weights.py --out docs/said_trimask_hs/weight_balance_results.json \
        --markdown docs/said_trimask_hs/weight_balance_tables.md
"""
import argparse
import json
import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
S0_500_CANONICAL = '/root/SAID-gap-completion/outputs/cvssl_screening/S0_canonical.json'
S0_500_URBAN = ('/root/SAID-gap-completion/outputs/cvssl_screening/baseline_urban1k/'
                'S0_step500_urban1k.json')
HS_RUN = os.path.join(REPO, 'runs_salu', 'said_s0_trimask_hs_v02')
GATE = {'coco_i2t_r1': 0.6058, 'coco_t2i_r1': 0.41236}
GATE_RULE = ('COCO I2T R@1 >= 0.6058 and COCO T2I R@1 >= 0.41236, at least one strictly higher '
             '(raw full precision)')
BINOMIAL_N = {'coco': 5000, 'urban': 1000}
COCO_KEYS = ('image2text_R1', 'text2image_R1', 'image2text_R5', 'text2image_R5',
             'image2text_R10', 'text2image_R10')
URBAN_KEYS = ('i2t_r1', 't2i_r1', 'i2t_r5', 't2i_r5', 'i2t_r10', 't2i_r10')


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
    payload = read_json(path)
    canonical = payload.get('canonical') or {}
    if label is None:
        keys = [key for key, value in canonical.items()
                if isinstance(value, dict) and 'coco_val2017' in value]
        if len(keys) != 1:
            raise SystemExit('%s holds %d evaluable labels; name one of %s'
                             % (path, len(keys), list(canonical)))
        label = keys[0]
    body = canonical[label]['coco_val2017']
    return label, {name: body[name] for name in sorted(body)}


def urban_metrics(path):
    body = read_json(path)['urban1k']
    return {'i2t_r1': body['image2text']['R1'], 'i2t_r5': body['image2text']['R5'],
            'i2t_r10': body['image2text']['R10'], 't2i_r1': body['text2image']['R1'],
            't2i_r5': body['text2image']['R5'], 't2i_r10': body['text2image']['R10']}


def load_arm(name, canonical_path, urban_path, label=None, run_dir=None, weights=None):
    entry = {'arm': name, 'weights': weights, 'canonical_json': canonical_path,
             'urban_json': urban_path, 'run_dir': run_dir}
    for key, path in (('canonical', canonical_path), ('urban', urban_path)):
        if os.path.isfile(path):
            entry['%s_sha256' % key] = sha256_of(path)
        else:
            entry[key] = None
            entry['missing_%s' % key] = path
    if os.path.isfile(canonical_path):
        entry['canonical_label'], entry['coco'] = canonical_metrics(canonical_path, label)
    if os.path.isfile(urban_path):
        entry['urban'] = urban_metrics(urban_path)
    if run_dir:
        for key, filename in (('run_status', 'run_status.json'),
                              ('run_summary', 'run_summary.json')):
            path = os.path.join(run_dir, filename)
            if os.path.isfile(path):
                payload = read_json(path)
                if key == 'run_status':
                    entry[key] = {field: payload.get(field) for field in
                                  ('phase', 'attempt', 'arm', 'objective', 'loss_profile',
                                   'lambda_1', 'lambda_2', 'lambda_3', 'lambda_sparse_i',
                                   'lambda_sparse_t', 'completed_steps', 'exit_codes',
                                   'failure_reason', 'conclusion', 'wall_seconds')}
                else:
                    entry[key] = {field: payload.get(field) for field in
                                  ('completed_steps', 'mean_sec_per_step', 'peak_memory_gb',
                                   'wall_sec', 'lr_horizon_steps', 'lambda_1', 'lambda_2',
                                   'lambda_3', 'lambda_sparse_i', 'lambda_sparse_t',
                                   'loss_profile', 'caption_stream_sha256',
                                   'sample_stream_sha256')}
    if entry.get('coco'):
        coco = entry['coco']
        i2t, t2i = coco['image2text_R1'], coco['text2image_R1']
        reference = abs(i2t - GATE['coco_i2t_r1']) < 1e-9 and abs(t2i - GATE['coco_t2i_r1']) < 1e-9
        entry['gate'] = {
            'reference_row': reference,
            'i2t_at_or_above': i2t >= GATE['coco_i2t_r1'],
            't2i_at_or_above': t2i >= GATE['coco_t2i_r1'],
            'strictly_higher_count': int(i2t > GATE['coco_i2t_r1']) + int(t2i > GATE['coco_t2i_r1']),
            'i2t_delta_points': (i2t - GATE['coco_i2t_r1']) * 100.0,
            't2i_delta_points': (t2i - GATE['coco_t2i_r1']) * 100.0,
            'rule': GATE_RULE,
        }
        entry['noise_scale'] = {
            'coco_binomial_se_i2t_r1': (i2t * (1 - i2t) / BINOMIAL_N['coco']) ** 0.5,
            'coco_binomial_se_t2i_r1': (t2i * (1 - t2i) / BINOMIAL_N['coco']) ** 0.5,
            'note': 'standard error of an independent binomial R@1 estimate; the honest test is '
                    'paired over the same queries, which the query-hit tool provides',
        }
        if reference:
            entry['gate']['verdict'] = ('the frozen baseline that defines the gate; the challenger '
                                        'clause cannot apply to the reference itself')
        elif (entry['gate']['i2t_at_or_above'] and entry['gate']['t2i_at_or_above']
              and entry['gate']['strictly_higher_count'] > 0):
            entry['gate']['verdict'] = 'passed the frozen 500-step promotion gate'
        else:
            entry['gate']['verdict'] = 'did not pass the frozen 500-step promotion gate'
    return entry


def delta(left, right, keys):
    out = {}
    for key in keys:
        if not left or not right or key not in left or key not in right:
            continue
        out[key] = {'left': left[key], 'right': right[key], 'delta': left[key] - right[key],
                    'delta_pp': 100.0 * (left[key] - right[key])}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hs-run', default=HS_RUN)
    parser.add_argument('--paired', default=None,
                        help='optional paired-comparison JSON from coco_paired_report.py')
    parser.add_argument('--out', required=True)
    parser.add_argument('--markdown', default=None)
    args = parser.parse_args()

    hs500 = os.path.join(args.hs_run, 'step500')
    bal500 = os.path.join(args.hs_run, 'bal500')
    arms = [
        load_arm('S0_smartclip@500 (gate baseline)', S0_500_CANONICAL, S0_500_URBAN,
                 label='S0_step500', weights='10 / 1 / 1 + 2 / 0.0 (reference objective)'),
        load_arm('S0_TriMask_HS@500 (frozen v0.2)',
                 os.path.join(hs500, 'evaluation', 'S0_TriMask_HS_step000500_canonical.json'),
                 os.path.join(hs500, 'evaluation', 'S0_TriMask_HS_step000500_urban1k.json'),
                 run_dir=hs500, weights='10 / 1 / 1 + 2 / 0.2'),
        load_arm('S0_TriMask_HS_BAL@500 (balanced)',
                 os.path.join(bal500, 'evaluation', 'S0_TriMask_HS_BAL_step000500_canonical.json'),
                 os.path.join(bal500, 'evaluation', 'S0_TriMask_HS_BAL_step000500_urban1k.json'),
                 run_dir=bal500, weights='10 / 10 / 10 + 2 / 2'),
    ]
    by_name = {entry['arm']: entry for entry in arms}
    baseline = by_name['S0_smartclip@500 (gate baseline)']
    if baseline.get('coco'):
        for key, expected in (('image2text_R1', GATE['coco_i2t_r1']),
                              ('text2image_R1', GATE['coco_t2i_r1'])):
            if abs(baseline['coco'][key] - expected) > 1e-9:
                raise SystemExit('the frozen baseline row reads %s=%r, the gate definition says %r'
                                 % (key, baseline['coco'][key], expected))

    comparisons = {}
    for name, entry in by_name.items():
        if entry is baseline or not entry.get('coco'):
            continue
        short = name.split('@')[0]
        comparisons['%s_minus_S0@500' % short] = {
            'coco': delta(entry.get('coco'), baseline.get('coco'), COCO_KEYS),
            'urban': delta(entry.get('urban'), baseline.get('urban'), URBAN_KEYS)}
    if by_name['S0_TriMask_HS_BAL@500 (balanced)'].get('coco') \
            and by_name['S0_TriMask_HS@500 (frozen v0.2)'].get('coco'):
        comparisons['BAL_minus_HS_default'] = {
            'coco': delta(by_name['S0_TriMask_HS_BAL@500 (balanced)'].get('coco'),
                          by_name['S0_TriMask_HS@500 (frozen v0.2)'].get('coco'), COCO_KEYS),
            'urban': delta(by_name['S0_TriMask_HS_BAL@500 (balanced)'].get('urban'),
                           by_name['S0_TriMask_HS@500 (frozen v0.2)'].get('urban'), URBAN_KEYS)}

    paired = None
    if args.paired and os.path.isfile(args.paired):
        paired = read_json(args.paired)

    payload = {
        'protocol': 's0-trimask-hs-loss-weight-comparison-at-500-updates',
        'gate': dict(GATE, rule=GATE_RULE),
        'weights_compared': {
            'S0_TriMask_HS (v0.2)': 'lambda_1 = 10, lambda_2 = 1, lambda_3 = 1, '
                                    'lambda_sparse_i = 2, lambda_sparse_t = 0.2',
            'S0_TriMask_HS_BAL': 'lambda_1 = lambda_2 = lambda_3 = 10, '
                                 'lambda_sparse_i = lambda_sparse_t = 2',
        },
        'arms': arms,
        'comparisons': comparisons,
        'paired': paired,
        'notes': [
            'every metric is copied from the standard evaluators; nothing is recomputed here',
            'the gate is applied to both @500 challengers; the S0 row is the reference it is '
            'defined against and is therefore not judged by the challenger clause',
            'the binomial standard error is quoted only as a scale: at 5000 COCO queries a 0.5 pp '
            'difference is inside one standard error, so paired per-query evidence decides',
        ],
        'worktree_head': subprocess.run(['git', '-C', REPO, 'rev-parse', 'HEAD'],
                                        capture_output=True, text=True).stdout.strip(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1, sort_keys=True)

    lines = ['# S0-TriMask-HS 损失权重对比（恰好 500 次更新）', '',
             '新臂 `S0_TriMask_HS_BAL`：三项对齐权重都取 10、两项稀疏权重都取 2；对照组是冻结的 '
             '`S0_TriMask_HS`（10/1/1 + 2/0.2）与门槛基准 `S0_smartclip`。全部数字来自标准评估器。', '',
             '| 臂 | 权重 | COCO I2T R@1 | COCO T2I R@1 | Urban I2T R@1 | Urban T2I R@1 | 500 步门槛 |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for entry in arms:
        coco = entry.get('coco') or {}
        urban = entry.get('urban') or {}

        def fmt(value, digits=4):
            return 'n/a' if value is None else ('%.' + str(digits) + 'f') % value

        lines.append('| %s | %s | %s | %s | %s | %s | %s |'
                     % (entry['arm'], entry['weights'], fmt(coco.get('image2text_R1')),
                        fmt(coco.get('text2image_R1')), fmt(urban.get('i2t_r1')),
                        fmt(urban.get('t2i_r1')), (entry.get('gate') or {}).get('verdict', 'n/a')))
    lines += ['', '## 差值（百分点）', '']
    for name, body in comparisons.items():
        lines.append('### %s' % name)
        lines.append('')
        lines.append('| 指标 | 左 | 右 | 差值 | 差值(pp) |')
        lines.append('| --- | --- | --- | --- | --- |')
        for family in ('coco', 'urban'):
            for key, values in sorted(body.get(family, {}).items()):
                lines.append('| %s/%s | %.4f | %.4f | %+.4f | %+.2f |'
                             % (family, key, values['left'], values['right'], values['delta'],
                                values['delta_pp']))
        lines.append('')
    if paired:
        lines += ['## 配对逐查询证据（COCO，同一批查询）', '',
                  'delta = 左 − 右，负值表示左侧更差；配对 bootstrap %s 次重采样，seed %s。'
                  % (paired.get('replicates'), paired.get('seed')), '',
                  '| 比较 | 方向 | R@1 左 | R@1 右 | Δ(pp) | 95% CI (pp) | 排除 0 | 仅左对/仅右对 | McNemar p |',
                  '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for row in paired.get('comparisons', []):
            for direction in ('image2text', 'text2image'):
                body = ((row.get('directions') or {}).get(direction) or {}).get('R@1')
                if not body:
                    continue
                boot = body.get('paired_bootstrap') or {}
                mcn = body.get('mcnemar') or {}
                lines.append('| %s vs %s | %s | %.4f | %.4f | %+.2f | [%+.2f, %+.2f] | %s | %s/%s | %s |'
                             % (row.get('left'), row.get('right'), direction, body['left'],
                                body['right'], 100 * body['delta'],
                                100 * float(boot.get('ci_low', float('nan'))),
                                100 * float(boot.get('ci_high', float('nan'))),
                                boot.get('excludes_zero'), mcn.get('n10_left_only'),
                                mcn.get('n01_right_only'), mcn.get('p_value_exact_two_sided')))
        lines.append('')
    if args.markdown:
        os.makedirs(os.path.dirname(os.path.abspath(args.markdown)), exist_ok=True)
        with open(args.markdown, 'w', encoding='utf-8') as handle:
            handle.write('\n'.join(lines) + '\n')
    print('WROTE %s' % args.out)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
