"""Paired significance of the COCO val2017 R@K differences between query-hit files.

Consumes the files written by ``tools/diag/coco_query_hits.py`` and reports, for every requested
pair of arms and both retrieval directions, the per-query paired statistics that aggregate
numbers cannot give:

* R@K of each arm and their difference,
* ``eval.paired_statistics.paired_bootstrap`` -- deterministic paired bootstrap of the mean
  difference over the *same* queries (percentile CI + bootstrap SE),
* ``eval.paired_statistics.mcnemar_exact`` -- how many queries only one of the two arms gets
  right, with an exact p-value.

Pairing is only valid when both files scored the identical query set, so the caption and image
order digests must match; that is asserted, not assumed. The statistics functions are imported
from the repository instead of being reimplemented.

    python tools/diag/coco_paired_report.py \
        --hits S0@500=evaluation/query_hits_S0_step000500.json \
        --hits S0@1000=evaluation/query_hits_S0_step001000.json \
        --pair S0@500:S0@1000 --out evaluation/paired_S0_500_vs_1000.json \
        --markdown evaluation/paired_S0_500_vs_1000.md
"""
import argparse
import importlib.util
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAIRED_LIB = os.path.join(REPO, 'eval', 'paired_statistics.py')
DIRECTIONS = (('image2text', 'hits_i2t'), ('text2image', 'hits_t2i'))


def sha256_of(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_hits(values):
    table = {}
    for entry in values:
        if '=' not in entry:
            raise SystemExit('--hits wants label=path, got %r' % entry)
        label, path = entry.split('=', 1)
        table[label.strip()] = os.path.abspath(path.strip())
    return table


def _log_comb(n, k):
    import math
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def mcnemar(mcnemar_exact, left_hits, right_hits):
    """The exact two-sided McNemar test, with a log-space fallback.

    ``eval.paired_statistics.mcnemar_exact`` divides by ``float(2 ** discordant)``, which raises
    ``OverflowError`` once the number of discordant queries passes ~1023 — unreachable for the
    small cohorts that function was written for, but ordinary here (COCO T2I has 25,000 queries).
    The fallback computes the *same* definition -- ``2 * sum_{i<=min(n10,n01)} C(n, i) / 2**n``,
    capped at 1 -- in log space. ``verify_mcnemar_fallback`` checks the two agree wherever the
    library does not overflow.
    """
    try:
        return mcnemar_exact(left_hits, right_hits), None
    except OverflowError:
        n10 = sum(1 for a, b in zip(left_hits, right_hits) if a >= 0.5 and b < 0.5)
        n01 = sum(1 for a, b in zip(left_hits, right_hits) if a < 0.5 and b >= 0.5)
        discordant = n10 + n01
        smaller = min(n10, n01)
        if discordant == 0:
            value = 1.0
        else:
            import math
            log_terms = [_log_comb(discordant, i) for i in range(smaller + 1)]
            peak = max(log_terms)
            total = peak + math.log(sum(math.exp(term - peak) for term in log_terms))
            value = min(1.0, 2.0 * math.exp(total - discordant * math.log(2.0)))
        return ({'n10_left_only': n10, 'n01_right_only': n01, 'discordant': discordant,
                 'p_value_exact_two_sided': value,
                 'note': 'eval.paired_statistics.mcnemar_exact overflows above ~1023 discordant '
                         'pairs; the identical two-sided binomial tail was computed in log space'},
                'library OverflowError')


def verify_mcnemar_fallback(mcnemar_exact, cases=((30, 10), (120, 90), (500, 480), (0, 7))):
    """The fallback must reproduce the library exactly wherever the library can run at all."""
    report = []
    for n10, n01 in cases:
        left = [1] * n10 + [0] * n01
        right = [0] * n10 + [1] * n01
        try:
            library, _ = mcnemar(mcnemar_exact, left, right)
        except OverflowError:
            report.append({'n10': n10, 'n01': n01, 'library': 'overflow (unexpected)'})
            continue
        fallback_p = mcnemar(mcnemar_exact, left, right)[0]['p_value_exact_two_sided']
        # force the fallback path for this case as well
        import math
        smaller = min(n10, n01)
        n = n10 + n01
        log_terms = [_log_comb(n, i) for i in range(smaller + 1)]
        peak = max(log_terms)
        total = peak + math.log(sum(math.exp(t - peak) for t in log_terms))
        forced = min(1.0, 2.0 * math.exp(total - n * math.log(2.0))) if n else 1.0
        report.append({'n10': n10, 'n01': n01,
                       'library': library['p_value_exact_two_sided'],
                       'fallback': forced,
                       'agree': abs(library['p_value_exact_two_sided'] - forced) < 1e-12})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hits', action='append', required=True, metavar='LABEL=PATH')
    parser.add_argument('--pair', action='append', required=True, metavar='LEFT:RIGHT',
                        help='left is the arm the delta is measured from (delta = left - right)')
    parser.add_argument('--k', default='1,5,10')
    parser.add_argument('--replicates', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20260911)
    parser.add_argument('--out', required=True)
    parser.add_argument('--markdown', default=None)
    args = parser.parse_args()

    stats = load_module('paired_statistics_frozen', PAIRED_LIB)
    k_values = [int(part) for part in args.k.split(',') if part.strip()]
    table = parse_hits(args.hits)
    loaded = {}
    for label, path in table.items():
        with open(path, 'r', encoding='utf-8') as handle:
            loaded[label] = json.load(handle)

    comparisons = []
    for entry in args.pair:
        if ':' not in entry:
            raise SystemExit('--pair wants LEFT:RIGHT, got %r' % entry)
        left_label, right_label = [part.strip() for part in entry.split(':', 1)]
        for label in (left_label, right_label):
            if label not in loaded:
                raise SystemExit('--pair refers to %r which no --hits provided (%s)'
                                 % (label, sorted(loaded)))
        left, right = loaded[left_label], loaded[right_label]
        for field in ('caption_order_sha256', 'image_order_sha256'):
            if left[field] != right[field]:
                raise SystemExit('%s vs %s: %s differs (%s vs %s); the two files did not score '
                                 'the same queries, a paired test is invalid'
                                 % (left_label, right_label, field, left[field][:12],
                                    right[field][:12]))
        row = {
            'left': left_label, 'right': right_label,
            'left_checkpoint_sha256': left['checkpoint_sha256'],
            'right_checkpoint_sha256': right['checkpoint_sha256'],
            'caption_order_sha256': left['caption_order_sha256'],
            'image_order_sha256': left['image_order_sha256'],
            'directions': {},
        }
        for name, key in DIRECTIONS:
            per_k = {}
            for k in k_values:
                left_hits = left[key][str(k)]
                right_hits = right[key][str(k)]
                if len(left_hits) != len(right_hits):
                    raise SystemExit('%s vs %s: %s query count differs %d vs %d'
                                     % (left_label, right_label, name, len(left_hits),
                                        len(right_hits)))
                count = float(len(left_hits))
                delta = sum(left_hits) / count - sum(right_hits) / count
                mcnemar_body, mcnemar_fallback = mcnemar(stats.mcnemar_exact, left_hits, right_hits)
                per_k['R@%d' % k] = {
                    'n_queries': int(count),
                    'left': sum(left_hits) / count,
                    'right': sum(right_hits) / count,
                    'delta': delta,
                    'paired_bootstrap': stats.paired_bootstrap(left_hits, right_hits,
                                                               replicates=args.replicates,
                                                               seed=args.seed),
                    'mcnemar': mcnemar_body,
                    'mcnemar_fallback_reason': mcnemar_fallback,
                }
            row['directions'][name] = per_k
        comparisons.append(row)

    payload = {
        'protocol': 'coco-val2017-5caption-legacy-cls-paired',
        'replicates': args.replicates,
        'seed': args.seed,
        'delta_convention': 'delta = left - right, positive means the left arm is better',
        'paired_statistics_library': {'path': PAIRED_LIB, 'sha256': sha256_of(PAIRED_LIB)},
        'mcnemar_fallback_verification': verify_mcnemar_fallback(stats.mcnemar_exact),
        'hit_files': {label: {'path': path, 'sha256': sha256_of(path),
                              'checkpoint_sha256': loaded[label]['checkpoint_sha256'],
                              'completed_steps': loaded[label].get('completed_steps')}
                      for label, path in table.items()},
        'comparisons': comparisons,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, sort_keys=True, indent=1)

    lines = ['# Paired COCO val2017 comparison (identical queries, identical 512-row protocol)', '',
             'delta = left - right; positive means the left arm is better. '
             'Bootstrap: %d replicates, seed %d.' % (args.replicates, args.seed), '']
    used_fallback = False
    for row in comparisons:
        lines.append('## %s vs %s' % (row['left'], row['right']))
        lines.append('')
        lines.append('| direction | K | queries | left | right | delta | bootstrap SE | 95% CI | '
                     'excludes 0 | left-only | right-only | McNemar p |')
        lines.append('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |')
        for name, _ in DIRECTIONS:
            for k in k_values:
                body = row['directions'][name]['R@%d' % k]
                boot = body['paired_bootstrap']
                mcn = body['mcnemar']
                lines.append('| %s | %d | %d | %.4f | %.4f | %+.4f | %.5f | [%+.4f, %+.4f] | %s | %s | %s | %s |'
                             % (name, k, body['n_queries'], body['left'], body['right'],
                                body['delta'],
                                float(boot.get('bootstrap_se', float('nan'))),
                                float(boot.get('ci_low', float('nan'))),
                                float(boot.get('ci_high', float('nan'))),
                                boot.get('excludes_zero'),
                                mcn.get('n10_left_only'), mcn.get('n01_right_only'),
                                mcn.get('p_value_exact_two_sided')))
                if body.get('mcnemar_fallback_reason'):
                    used_fallback = True
        lines.append('')
    if used_fallback:
        checks = payload['mcnemar_fallback_verification']
        lines += ['The library `mcnemar_exact` overflows past ~1023 discordant pairs, so the p-values '
                  'above were computed by the log-space fallback in `tools/diag/coco_paired_report.py`. '
                  'The fallback reproduces the library exactly wherever the library runs: ' +
                  ', '.join('n10=%d/n01=%d %.12g vs %.12g agree=%s'
                            % (c['n10'], c['n01'], c['library'], c['fallback'], c['agree'])
                            for c in checks if 'library' in c and isinstance(c['library'], float)), '']
    if args.markdown:
        os.makedirs(os.path.dirname(os.path.abspath(args.markdown)), exist_ok=True)
        with open(args.markdown, 'w', encoding='utf-8') as handle:
            handle.write('\n'.join(lines) + '\n')
    print('WROTE %s' % args.out)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
