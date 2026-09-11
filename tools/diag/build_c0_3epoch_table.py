"""Build the 3-epoch comparison table against S0@500 and C_L03@500.

    python tools/diag/build_c0_3epoch_table.py
"""
import json
import os

BASE = '/root/SAID-gap-completion/outputs'
FILES = [
    ('S0@500', os.path.join(BASE, 'cvssl_screening/S0_canonical.json'), 'S0_step500'),
    ('C0-L10@500', os.path.join(BASE, 'cvssl_screening/C0_canonical.json'), 'C0_step500'),
    ('C_L03@500', os.path.join(BASE, 'cvssl_screening/c0_tuning/C_L03_canonical.json'), None),
    ('3ep ep1', os.path.join(BASE, 'cvssl_screening/c0_3epoch/L03_ep1_canonical.json'), None),
    ('3ep ep2', os.path.join(BASE, 'cvssl_screening/c0_3epoch/L03_ep2_canonical.json'), None),
    ('3ep ep3', os.path.join(BASE, 'cvssl_screening/c0_3epoch/L03_ep3_canonical.json'), None),
]

ROWS = [
    ('COCO I2T R@1', None, 'image2text_R1'), ('COCO I2T R@5', None, 'image2text_R5'),
    ('COCO I2T R@10', None, 'image2text_R10'),
    ('COCO T2I R@1', None, 'text2image_R1'), ('COCO T2I R@5', None, 'text2image_R5'),
    ('COCO T2I R@10', None, 'text2image_R10'),
]
for variant, tag in (('first_sentence', 'first'), ('fixed_sparse', 'sparse'),
                     ('full_dense', 'full')):
    for key, short in (('image2text_R1', 'I2T R@1'), ('image2text_R5', 'I2T R@5'),
                       ('image2text_R10', 'I2T R@10'), ('text2image_R1', 'T2I R@1'),
                       ('text2image_R5', 'T2I R@5'), ('text2image_R10', 'T2I R@10')):
        ROWS.append(('1K %s %s' % (tag, short), variant, key))


def load(path, entry_name):
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        payload = json.load(handle)
    name = entry_name or sorted(payload['canonical'])[0]
    return payload['canonical'][name]


def value(entry, variant, key):
    if entry is None:
        return None
    if variant is None:
        return float(entry['coco_val2017'][key])
    return float(entry['sharegpt4v1k'][variant]['retrieval'][key])


def main():
    entries = [(label, load(path, name)) for label, path, name in FILES]
    header = '| metric | ' + ' | '.join(label for label, _ in entries) + ' | ep3 - S0@500 |'
    print(header)
    print('|' + '---|' * (len(entries) + 2))
    s0 = dict(entries)['S0@500']
    worst = []
    for label, variant, key in ROWS:
        values = [value(entry, variant, key) for _, entry in entries]
        cells = ['%.4f' % v if v is not None else 'n/a' for v in values]
        ep3 = values[-1]
        base = values[0]
        delta = ('%+.4f' % (ep3 - base)) if (ep3 is not None and base is not None) else 'n/a'
        print('| %s | %s | %s |' % (label, ' | '.join(cells), delta))
        if variant is None and key in ('image2text_R1', 'text2image_R1'):
            worst.append((label, base, ep3, (ep3 - base) if ep3 is not None else None))

    print()
    print('== gate check for the 3-epoch final checkpoint vs S0@500 (full precision)')
    s0_i2t = value(s0, None, 'image2text_R1')
    s0_t2i = value(s0, None, 'text2image_R1')
    final = entries[-1][1]
    if final is None:
        print('   ep3 canonical json not available yet')
        return
    ep3_i2t = value(final, None, 'image2text_R1')
    ep3_t2i = value(final, None, 'text2image_R1')
    j_s0 = 0.5 * (s0_i2t + s0_t2i)
    j_ep3 = 0.5 * (ep3_i2t + ep3_t2i)
    left = ep3_i2t >= s0_i2t
    right = ep3_t2i >= s0_t2i
    strict = (ep3_i2t > s0_i2t) or (ep3_t2i > s0_t2i)
    print('   COCO I2T R@1  %.10f vs gate %.10f  %s' % (ep3_i2t, s0_i2t,
                                                        'PASS' if left else 'FAIL'))
    print('   COCO T2I R@1  %.10f vs gate %.10f  %s' % (ep3_t2i, s0_t2i,
                                                        'PASS' if right else 'FAIL'))
    print('   at least one strict: %s' % strict)
    print('   J_S0 = %.10f   J_ep3 = %.10f   delta = %+.10f   larger_screening_gain=%s'
          % (j_s0, j_ep3, j_ep3 - j_s0, (j_ep3 - j_s0) >= 0.003))
    print('   PASSED = %s' % (left and right and strict))
    print()
    print('== per-epoch trajectory of the two primary directions (vs S0@500)')
    for label, entry in entries:
        i2t = value(entry, None, 'image2text_R1')
        t2i = value(entry, None, 'text2image_R1')
        if i2t is None:
            continue
        print('   %-12s I2T R@1 %.4f (%+.4f)   T2I R@1 %.4f (%+.4f)   J %.10f'
              % (label, i2t, i2t - s0_i2t, t2i, t2i - s0_t2i, 0.5 * (i2t + t2i)))


if __name__ == '__main__':
    main()
