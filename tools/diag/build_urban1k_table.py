"""Urban-1k + COCO comparison table (500-step SOTA table).

COCO numbers are read from the already-computed canonical JSONs (not re-evaluated); Urban-1k numbers
come from outputs/cvssl_screening/baseline_urban1k/*_urban1k.json.
"""
import json
import os

S = '/root/SAID-gap-completion'
U = os.path.join(S, 'outputs/cvssl_screening/baseline_urban1k')
COCO = {
    'Initial': (os.path.join(S, 'outputs/cvssl_screening/S0_canonical.json'), 'Initial'),
    'S0@500': (os.path.join(S, 'outputs/cvssl_screening/S0_canonical.json'), 'S0_step500'),
    'C0@500': (os.path.join(S, 'outputs/cvssl_screening/C0_canonical.json'), 'C0_step500'),
    'C1@500': ('/root/SAID-c1-tcr/outputs/cvssl_screening/c1_tcr/C1_step500_canonical.json', None),
}
URBAN = {'Initial': 'Initial_urban1k.json', 'S0@500': 'S0_step500_urban1k.json',
         'C0@500': 'C0_step500_urban1k.json', 'C1@500': 'C1_step500_urban1k.json'}


def coco_of(label, variant, key):
    path, name = COCO[label]
    if not os.path.exists(path):
        return None
    payload = json.load(open(path))
    entry = payload['canonical'][name or sorted(payload['canonical'])[0]]
    node = entry['coco_val2017']
    return float(node[key])


def urban_of(label, direction, key):
    path = os.path.join(U, URBAN[label])
    if not os.path.exists(path):
        return None
    payload = json.load(open(path))
    return float(payload['urban1k'][direction][key])


def fmt(value):
    return '%.4f' % value if value is not None else 'n/a'


def delta(a, b):
    return ('%+.4f' % (a - b)) if (a is not None and b is not None) else 'n/a'


LABELS = ['Initial', 'S0@500', 'C0@500', 'C1@500']
print('| dataset | metric | ' + ' | '.join(LABELS) + ' | C1 - S0 |')
print('|' + '---|' * (len(LABELS) + 3))
for direction, tag in (('image2text', 'I2T'), ('text2image', 'T2I')):
    for key in ('R1', 'R5', 'R10'):
        values = [urban_of(label, direction, key) for label in LABELS]
        print('| Urban-1k | %s %s | %s | %s |'
              % (tag, key, ' | '.join(fmt(v) for v in values), delta(values[3], values[1])))
for key, tag in (('image2text_R1', 'I2T R@1'), ('image2text_R5', 'I2T R@5'),
                 ('image2text_R10', 'I2T R@10'), ('text2image_R1', 'T2I R@1'),
                 ('text2image_R5', 'T2I R@5'), ('text2image_R10', 'T2I R@10')):
    values = [coco_of(label, None, key) for label in LABELS]
    print('| COCO val2017 | %s | %s | %s |'
          % (tag, ' | '.join(fmt(v) for v in values), delta(values[3], values[1])))

print()
print('== J = (I2T R@1 + T2I R@1) / 2')
for label in LABELS:
    u_i2t, u_t2i = urban_of(label, 'image2text', 'R1'), urban_of(label, 'text2image', 'R1')
    c_i2t, c_t2i = coco_of(label, None, 'image2text_R1'), coco_of(label, None, 'text2image_R1')
    if None in (u_i2t, u_t2i, c_i2t, c_t2i):
        print('   %-9s incomplete' % label)
        continue
    print('   %-9s Urban-1k J=%.4f   COCO J=%.4f' % (label, 0.5 * (u_i2t + u_t2i),
                                                     0.5 * (c_i2t + c_t2i)))

print()
print('== S0 -> C0 -> C1 on Urban-1k (percentage points, R@1)')
for direction, tag in (('image2text', 'I2T'), ('text2image', 'T2I')):
    s0 = urban_of('S0@500', direction, 'R1')
    c0 = urban_of('C0@500', direction, 'R1')
    c1 = urban_of('C1@500', direction, 'R1')
    print('   %s R@1: S0 %.4f | C0 %.4f (%+.2f pp vs S0) | C1 %.4f (%+.2f pp vs S0)'
          % (tag, s0, c0, (c0 - s0) * 100, c1, (c1 - s0) * 100))
