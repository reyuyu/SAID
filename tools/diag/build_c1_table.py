"""C1-TCR vs the frozen S0 / C0 500-step baselines, plus the reconstruction diagnostics table.

    python tools/diag/build_c1_table.py
"""
import json
import os

W = '/root/SAID-c1-tcr'
S = '/root/SAID-gap-completion'
FILES = [
    ('Initial', os.path.join(S, 'outputs/cvssl_screening/S0_canonical.json'), 'Initial'),
    ('S0@500', os.path.join(S, 'outputs/cvssl_screening/S0_canonical.json'), 'S0_step500'),
    ('C0-L10@500', os.path.join(S, 'outputs/cvssl_screening/C0_canonical.json'), 'C0_step500'),
    ('C_L03@500', os.path.join(S, 'outputs/cvssl_screening/c0_tuning/C_L03_canonical.json'), None),
    ('C1@500', os.path.join(W, 'outputs/cvssl_screening/c1_tcr/C1_step500_canonical.json'), None),
]
SHARE = [('first', 'first_sentence'), ('sparse', 'fixed_sparse'), ('full', 'full_dense')]
ROWS = [('COCO I2T R@1', None, 'image2text_R1'), ('COCO I2T R@5', None, 'image2text_R5'),
        ('COCO I2T R@10', None, 'image2text_R10'), ('COCO T2I R@1', None, 'text2image_R1'),
        ('COCO T2I R@5', None, 'text2image_R5'), ('COCO T2I R@10', None, 'text2image_R10')]
for _tag, _variant in SHARE:
    for key, short in (('image2text_R1', 'I2T R@1'), ('image2text_R5', 'I2T R@5'),
                       ('image2text_R10', 'I2T R@10'), ('text2image_R1', 'T2I R@1'),
                       ('text2image_R5', 'T2I R@5'), ('text2image_R10', 'T2I R@10')):
        ROWS.append(('1K %s %s' % (_tag, short), _variant, key))


def load(path, name):
    if not os.path.exists(path):
        return None
    payload = json.load(open(path))
    key = name or sorted(payload['canonical'])[0]
    return payload['canonical'][key]


def value(entry, variant, key):
    """Read one metric, tolerating the two shapes the evaluator has emitted over time."""
    if entry is None:
        return None
    try:
        node = entry['coco_val2017'] if variant is None else entry['sharegpt4v1k'][variant]
        if 'retrieval' in node:
            node = node['retrieval']
        return float(node[key])
    except (KeyError, TypeError):
        return None


entries = [(label, load(path, name)) for label, path, name in FILES]
print('| metric | ' + ' | '.join(label for label, _ in entries) + ' | C1 - S0@500 |')
print('|' + '---|' * (len(entries) + 2))
for label, variant, key in ROWS:
    values = [value(entry, variant, key) for _, entry in entries]
    cells = ['%.4f' % v if v is not None else 'n/a' for v in values]
    base, c1 = values[1], values[-1]
    delta = '%+.4f' % (c1 - base) if (base is not None and c1 is not None) else 'n/a'
    print('| %s | %s | %s |' % (label, ' | '.join(cells), delta))

print()
s0 = dict(entries)['S0@500']
c1 = dict(entries)['C1@500']
if c1 is None:
    print('C1 canonical json missing')
    raise SystemExit(0)
s0_i2t, s0_t2i = value(s0, None, 'image2text_R1'), value(s0, None, 'text2image_R1')
c1_i2t, c1_t2i = value(c1, None, 'image2text_R1'), value(c1, None, 'text2image_R1')
j_s0 = 0.5 * (s0_i2t + s0_t2i)
j_c1 = 0.5 * (c1_i2t + c1_t2i)
left = c1_i2t >= s0_i2t
right = c1_t2i >= s0_t2i
strict = (c1_i2t > s0_i2t) or (c1_t2i > s0_t2i)
print('== frozen gate (full precision)')
print('   COCO I2T R@1 %.10f vs S0 %.10f  %s' % (c1_i2t, s0_i2t, 'PASS' if left else 'FAIL'))
print('   COCO T2I R@1 %.10f vs S0 %.10f  %s' % (c1_t2i, s0_t2i, 'PASS' if right else 'FAIL'))
print('   J_C1 = %.10f  J_S0 = %.10f  delta = %+.10f' % (j_c1, j_s0, j_c1 - j_s0))
print('   at least one strict: %s' % strict)
print('   verdict: %s' % ('PROMISING_FOR_LONGER_RUN' if (left and right and strict)
                           else 'NOT_PROMISING'))
