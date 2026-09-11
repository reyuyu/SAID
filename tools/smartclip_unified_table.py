#!/usr/bin/env python3
"""Phase 3.0A baseline validation: unified canonical retrieval table.

Merges the SmartCLIP reproduction results with the already-measured Initial / A / C numbers
(all produced by the SAME canonical evaluator) and prints the comparison table plus the
percentage-point deltas.
"""
import json
import os
import sys

REPO = '/root/SAID-gap-completion'
SC_RAW = os.path.join(REPO, 'outputs', 'smartclip_reproduction',
                      'smartclip_3epoch_canonical_raw.json')
AC_RAW = os.path.join(REPO, 'outputs', 'phase30a_full3epoch',
                      'canonical_stage1_priority.json')
OUT_DIR = os.path.join(REPO, 'docs', 'smartclip_reproduction')
os.makedirs(OUT_DIR, exist_ok=True)

VARIANTS = ('first_sentence', 'fixed_sparse', 'full_dense')
KEYS = ('image2text_R1', 'image2text_R5', 'image2text_R10',
        'text2image_R1', 'text2image_R5', 'text2image_R10')

sc = json.load(open(SC_RAW)).get('canonical', {})
ac = json.load(open(AC_RAW)).get('canonical', {})
print('SmartCLIP canonical keys:', sorted(sc))
print('A/C canonical keys       :', sorted(ac))


def block(entry, scope, variant=None):
    if entry is None:
        return None
    if scope == 'coco':
        return entry.get('coco_val2017')
    return ((entry.get('sharegpt4v1k') or {}).get(variant) or {}).get('retrieval')


def row(name, entry):
    out = {'name': name}
    coco = block(entry, 'coco')
    out['coco'] = (coco['image2text_R1'], coco['text2image_R1']) if coco else None
    for variant in VARIANTS:
        data = block(entry, '1k', variant)
        out[variant] = (data['image2text_R1'], data['text2image_R1']) if data else None
    out['entry'] = entry
    return out


rows = [
    row('Initial CLIP', ac.get('initial')),
    row('Original SmartCLIP 3ep', sc.get('SmartCLIP_epoch3')),
    row('A Said-only 3ep', ac.get('Aend')),
    row('C Full Base 3ep', ac.get('Cend')),
]

print()
print('=== Unified canonical retrieval (I2T R@1 / T2I R@1) ===')
header = '%-24s %16s %16s %16s %16s %16s' % (
    'model', 'COCO', '1K first_sentence', '1K fixed_sparse', '1K full_dense', '')
print(header)
print('-' * len(header))
for item in rows:
    cells = []
    for key in ('coco',) + VARIANTS:
        value = item.get(key)
        cells.append('n/a' if value is None else '%.4f / %.4f' % value)
    print('%-24s %16s %16s %16s %16s' % (item['name'], cells[0], cells[1], cells[2], cells[3]))


def delta(name, left, right):
    a = next((item for item in rows if item['name'] == left), None)
    b = next((item for item in rows if item['name'] == right), None)
    if a is None or b is None:
        return
    cells = []
    for key in ('coco',) + VARIANTS:
        va, vb = a.get(key), b.get(key)
        if va is None or vb is None:
            cells.append('n/a')
            continue
        cells.append('%+.2f / %+.2f' % (100.0 * (va[0] - vb[0]), 100.0 * (va[1] - vb[1])))
    print('%-24s %16s %16s %16s %16s' % (name, cells[0], cells[1], cells[2], cells[3]))


print()
print('=== deltas in percentage points (I2T / T2I) ===')
delta('SmartCLIP - Initial', 'Original SmartCLIP 3ep', 'Initial CLIP')
delta('A Said-only - Initial', 'A Said-only 3ep', 'Initial CLIP')
delta('A Said-only - SmartCLIP', 'A Said-only 3ep', 'Original SmartCLIP 3ep')
delta('C Full Base - Initial', 'C Full Base 3ep', 'Initial CLIP')
delta('C Full Base - A', 'C Full Base 3ep', 'A Said-only 3ep')

# ---------------------------------------------------------------- SmartCLIP trajectory
print()
print('=== SmartCLIP reproduction trajectory (I2T / T2I R@1) ===')
trajectory = {}
for label, key in (('initial', 'SmartCLIP_initial'), ('epoch1', 'SmartCLIP_epoch1'),
                   ('epoch2', 'SmartCLIP_epoch2'), ('epoch3', 'SmartCLIP_epoch3')):
    entry = sc.get(key)
    if entry is None:
        continue
    item = row(label, entry)
    trajectory[label] = {k: item[k] for k in ('coco',) + VARIANTS}
    cells = []
    for name in ('coco',) + VARIANTS:
        value = item[name]
        cells.append('n/a' if value is None else '%.4f / %.4f' % value)
    print('%-10s %16s %16s %16s %16s' % (label, cells[0], cells[1], cells[2], cells[3]))

# ---------------------------------------------------------------- full detail + summary
summary = {
    'protocol': 'smartclip-reproduction-canonical-retrieval',
    'note': ('Original SmartCLIP training path (train/train.py, CLIP_Clean_Train) reproduced '
             'for 3 epochs on full ShareGPT4V, evaluated with the SAME canonical evaluator used '
             'for the Said-only and Full-Base arms: ShareGPT4V-1K via eval/validation_protocol.py '
             '(3 caption variants) and COCO val2017 via eval/retrieval/coco_retrieval.py '
             '(5-caption). No optimizer step during evaluation. Legacy train_utils.eval_coco '
             'results are deliberately excluded from this comparison (different protocol).'),
    'checkpoints': {},
    'unified_table': {},
    'deltas_percentage_points': {},
}
for label, key in (('SmartCLIP_initial', 'SmartCLIP_initial'),
                   ('SmartCLIP_epoch1', 'SmartCLIP_epoch1'),
                   ('SmartCLIP_epoch2', 'SmartCLIP_epoch2'),
                   ('SmartCLIP_epoch3', 'SmartCLIP_epoch3')):
    entry = sc.get(key)
    if entry is None:
        continue
    summary['checkpoints'][label] = {
        'sharegpt4v1k': {variant: {k: block(entry, '1k', variant)[k] for k in KEYS}
                         for variant in VARIANTS if block(entry, '1k', variant)},
        'coco_val2017': {k: block(entry, 'coco')[k] for k in KEYS}
        if block(entry, 'coco') else None,
    }
for item in rows:
    summary['unified_table'][item['name']] = {
        'coco_I2T_R1': item['coco'][0] if item['coco'] else None,
        'coco_T2I_R1': item['coco'][1] if item['coco'] else None,
        **{variant: {'I2T_R1': item[variant][0], 'T2I_R1': item[variant][1]}
           for variant in VARIANTS if item[variant]},
    }


def pp(a, b):
    return None if a is None or b is None else 100.0 * (a - b)


def get(name):
    return next((item for item in rows if item['name'] == name), None)


for label, left, right in (('SmartCLIP - Initial', 'Original SmartCLIP 3ep', 'Initial CLIP'),
                           ('A Said-only - Initial', 'A Said-only 3ep', 'Initial CLIP'),
                           ('A Said-only - SmartCLIP', 'A Said-only 3ep',
                            'Original SmartCLIP 3ep'),
                           ('C Full Base - Initial', 'C Full Base 3ep', 'Initial CLIP'),
                           ('C Full Base - A', 'C Full Base 3ep', 'A Said-only 3ep')):
    a, b = get(left), get(right)
    if a is None or b is None:
        continue
    entry = {}
    if a['coco'] and b['coco']:
        entry['coco_I2T_R1'] = pp(a['coco'][0], b['coco'][0])
        entry['coco_T2I_R1'] = pp(a['coco'][1], b['coco'][1])
    for variant in VARIANTS:
        if a[variant] and b[variant]:
            entry[variant + '_I2T_R1'] = pp(a[variant][0], b[variant][0])
            entry[variant + '_T2I_R1'] = pp(a[variant][1], b[variant][1])
    summary['deltas_percentage_points'][label] = entry

with open(os.path.join(OUT_DIR, 'smartclip_3epoch_canonical.json'), 'w') as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
print()
print('wrote %s' % os.path.join(OUT_DIR, 'smartclip_3epoch_canonical.json'))
