"""Build the S0/C0 screening performance table from the canonical retrieval JSONs.

Usage:  python tools/diag/build_cvssl_table.py <S0_canonical.json> <C0_canonical.json> [out.md]
"""
import json
import os
import sys

SHARE_VARIANTS = ['first_sentence', 'fixed_sparse', 'full_dense']
SHARE_LABEL = {'first_sentence': 'first', 'fixed_sparse': 'sparse', 'full_dense': 'full'}


def load(path):
    with open(path) as handle:
        return json.load(handle)


def metric(entry, variant, key):
    try:
        if variant is None:
            return float(entry['coco_val2017'][key])
        return float(entry['sharegpt4v1k'][variant]['retrieval'][key])
    except (KeyError, TypeError):
        return None


def main():
    s0_path, c0_path = sys.argv[1], sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else None
    s0, c0 = load(s0_path), load(c0_path)

    def entry(payload, name):
        canonical = payload['canonical']
        if name not in canonical:
            raise KeyError('%s missing in %s' % (name, sorted(canonical)))
        return canonical[name]

    rows = [
        ('COCO I2T R@1', None, 'image2text_R1'),
        ('COCO I2T R@5', None, 'image2text_R5'),
        ('COCO I2T R@10', None, 'image2text_R10'),
        ('COCO T2I R@1', None, 'text2image_R1'),
        ('COCO T2I R@5', None, 'text2image_R5'),
        ('COCO T2I R@10', None, 'text2image_R10'),
    ]
    for variant in SHARE_VARIANTS:
        tag = SHARE_LABEL[variant]
        for key, short in (('image2text_R1', 'I2T R@1'), ('image2text_R5', 'I2T R@5'),
                           ('image2text_R10', 'I2T R@10'), ('text2image_R1', 'T2I R@1'),
                           ('text2image_R5', 'T2I R@5'), ('text2image_R10', 'T2I R@10')):
            rows.append(('1K %s %s' % (tag, short), variant, key))

    columns = [('Initial', s0, 'Initial'), ('S0@100', s0, 'S0_step100'),
               ('C0@100', c0, 'C0_step100'), ('S0@500', s0, 'S0_step500'),
               ('C0@500', c0, 'C0_step500')]

    lines = []
    header = '| metric | ' + ' | '.join(name for name, _, _ in columns) + ' | C0-S0@500 |'
    lines.append(header)
    lines.append('|' + '---|' * (len(columns) + 2))
    for label, variant, key in rows:
        values = []
        for name, payload, entry_name in columns:
            value = metric(entry(payload, entry_name), variant, key)
            values.append(value)
        delta = (values[4] - values[3]) if (values[4] is not None and values[3] is not None) \
            else None
        cells = ['%.4f' % value if value is not None else 'n/a' for value in values]
        lines.append('| %s | %s | %s |' % (label, ' | '.join(cells),
                                           '%+.4f' % delta if delta is not None else 'n/a'))
    table = '\n'.join(lines)
    print(table)

    # S0 vs Initial and C0 vs S0 at step 500
    print()
    print('== deltas (percentage points)')
    print('%-22s %12s %12s' % ('metric', 'S0@500-Init', 'C0@500-S0@500'))
    for label, variant, key in rows:
        init = metric(entry(s0, 'Initial'), variant, key)
        s0_500 = metric(entry(s0, 'S0_step500'), variant, key)
        c0_500 = metric(entry(c0, 'C0_step500'), variant, key)
        left = (s0_500 - init) * 100 if None not in (init, s0_500) else None
        right = (c0_500 - s0_500) * 100 if None not in (c0_500, s0_500) else None
        print('%-22s %+12.2f %+12.2f'
              % (label, left if left is not None else float('nan'),
                 right if right is not None else float('nan')))

    # the 100 -> 500 trend
    print()
    print('== 100 -> 500 trend (percentage points)')
    print('%-22s %12s %12s' % ('metric', 'S0 500-100', 'C0 500-100'))
    for label, variant, key in rows:
        s0_100 = metric(entry(s0, 'S0_step100'), variant, key)
        s0_500 = metric(entry(s0, 'S0_step500'), variant, key)
        c0_100 = metric(entry(c0, 'C0_step100'), variant, key)
        c0_500 = metric(entry(c0, 'C0_step500'), variant, key)
        print('%-22s %+12.2f %+12.2f'
              % (label, (s0_500 - s0_100) * 100 if None not in (s0_100, s0_500) else float('nan'),
                 (c0_500 - c0_100) * 100 if None not in (c0_100, c0_500) else float('nan')))

    if out_path:
        with open(out_path, 'w') as handle:
            handle.write(table + '\n')
        print('\nwrote %s' % out_path)


if __name__ == '__main__':
    main()
