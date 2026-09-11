#!/usr/bin/env python3
"""Phase 3.0A.2 canonical retrieval: merge results and print the main comparison table.

Reads the canonical stage JSONs and emits
``docs/phase30a_full3epoch/canonical_retrieval.json`` (small, tracked) plus the report table.
"""
import json
import os
import sys

REPO = '/root/SAID-gap-completion'
EVAL_DIR = os.path.join(REPO, 'outputs', 'phase30a_full3epoch')
OUT_DIR = os.path.join(REPO, 'docs', 'phase30a_full3epoch')
os.makedirs(OUT_DIR, exist_ok=True)

STAGES = ('canonical_stage1_priority.json', 'canonical_stage2_epochs.json')
VARIANTS = ('first_sentence', 'fixed_sparse', 'full_dense')

# checkpoints to report, in order, with the arm each belongs to
ORDER = (
    ('initial', 'initial'),
    ('A1216', 'A epoch1'),
    ('C1216', 'C epoch1'),
    ('A2432', 'A epoch2'),
    ('C2432', 'C epoch2'),
    ('Aend', 'A epoch3'),
    ('Cend', 'C epoch3'),
)

merged = {}
for name in STAGES:
    path = os.path.join(EVAL_DIR, name)
    if not os.path.exists(path):
        print('MISSING %s' % path)
        continue
    payload = json.load(open(path))
    for tag, entry in (payload.get('canonical') or {}).items():
        merged[tag] = entry
print('canonical checkpoints available: %s' % sorted(merged))


def metric(entry, scope, variant=None, key='image2text_R1'):
    if scope == 'coco':
        block = entry.get('coco_val2017') or {}
        return block.get(key)
    block = ((entry.get('sharegpt4v1k') or {}).get(variant) or {}).get('retrieval') or {}
    return block.get(key)


KEYS = ('image2text_R1', 'image2text_R5', 'image2text_R10',
        'text2image_R1', 'text2image_R5', 'text2image_R10')

summary = {
    'protocol': 'phase30a-2-canonical-retrieval',
    'note': ('Standard CLIP CLS image-text retrieval with the frozen protocols: '
             'ShareGPT4V-1K via eval/validation_protocol.py (3 caption variants) and COCO '
             'val2017 via eval/retrieval/coco_retrieval.py (5-caption). No optimizer step. '
             'The base objective contains no Global-text InfoNCE, so this measures whether '
             'full training damaged canonical retrieval.'),
    'checkpoints': {},
}
for tag, label in ORDER:
    entry = merged.get(tag)
    if entry is None:
        continue
    summary['checkpoints'][label] = {
        'tag': tag,
        'sharegpt4v1k': {variant: {key: metric(entry, '1k', variant, key) for key in KEYS}
                         for variant in VARIANTS
                         if metric(entry, '1k', variant, 'image2text_R1') is not None},
        'coco_val2017': {key: metric(entry, 'coco', None, key) for key in KEYS}
        if metric(entry, 'coco', None, 'image2text_R1') is not None else None,
    }
with open(os.path.join(OUT_DIR, 'canonical_retrieval.json'), 'w') as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)

# ---------------------------------------------------------------- main table
print('=== main comparison: I2T R@1 / T2I R@1 ===')
header = '%-16s %15s %15s %19s %19s %19s' % (
    'checkpoint', 'COCO I2T/T2I', '1K first_sentence', '1K fixed_sparse', '1K full_dense',
    '')
print(header)
print('-' * 100)


def pair(entry, scope, variant=None):
    i2t = metric(entry, scope, variant, 'image2text_R1')
    t2i = metric(entry, scope, variant, 'text2image_R1')
    if i2t is None or t2i is None:
        return None
    return i2t, t2i


table = {}
for tag, label in ORDER:
    entry = merged.get(tag)
    if entry is None:
        continue
    row = {}
    for name, scope, variant in (('coco', 'coco', None),
                                 ('first', '1k', 'first_sentence'),
                                 ('sparse', '1k', 'fixed_sparse'),
                                 ('dense', '1k', 'full_dense')):
        row[name] = pair(entry, scope, variant)
    table[label] = row
    print('%-16s %15s %15s %19s %19s' % (
        label,
        'n/a' if row['coco'] is None else '%.4f/%.4f' % row['coco'],
        'n/a' if row['first'] is None else '%.4f/%.4f' % row['first'],
        'n/a' if row['sparse'] is None else '%.4f/%.4f' % row['sparse'],
        'n/a' if row['dense'] is None else '%.4f/%.4f' % row['dense']))

print()
print('=== deltas in percentage points ===')


def delta_row(name, left, right):
    row_left, row_right = table.get(left), table.get(right)
    if not row_left or not row_right:
        return
    cells = []
    for key in ('coco', 'first', 'sparse', 'dense'):
        a, b = row_left.get(key), row_right.get(key)
        if a is None or b is None:
            cells.append('n/a')
            continue
        cells.append('%+.2f/%+.2f' % (100.0 * (a[0] - b[0]), 100.0 * (a[1] - b[1])))
    print('%-16s %15s %15s %19s %19s' % (name, cells[0], cells[1], cells[2], cells[3]))


delta_row('A(3ep) - Initial', 'A epoch3', 'initial')
delta_row('C(3ep) - Initial', 'C epoch3', 'initial')
delta_row('C(3ep) - A(3ep)', 'C epoch3', 'A epoch3')
if 'A epoch1' in table and 'C epoch1' in table:
    delta_row('C(1ep) - A(1ep)', 'C epoch1', 'A epoch1')
if 'A epoch2' in table and 'C epoch2' in table:
    delta_row('C(2ep) - A(2ep)', 'C epoch2', 'A epoch2')
print()
print('wrote %s' % os.path.join(OUT_DIR, 'canonical_retrieval.json'))
