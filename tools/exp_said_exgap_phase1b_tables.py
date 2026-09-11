"""Phase ExGAP-1B: build every report table from the two arms' logs and the retrieval JSON.

    python tools/exp_said_exgap_phase1b_tables.py \
        [--root runs_salu/said_exgap_phase1b] \
        [--retrieval outputs/said_exgap/phase1b_canonical_retrieval.json]

Prints markdown tables (and a JSON digest) built ONLY from measured values; a missing value is
printed as ``MISSING`` rather than being estimated.
"""
import argparse
import json
import os

ARMS = (('M0_said_only_masking', 'M0'), ('M1_full_exgap', 'M1'))
STEP_KEYS = ('completed_steps',)


def load_records(path):
    if not os.path.exists(path):
        return []
    records = []
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line.startswith('{'):
                records.append(json.loads(line))
    return records


def at_step(records, completed):
    for record in records:
        if record.get('completed_steps') == completed:
            return record
    return None


def fmt(value, digits=4):
    if value is None:
        return 'MISSING'
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if value != 0 and abs(value) < 1e-3:
            return '%.3g' % value
        return ('%.' + str(digits) + 'f') % value
    return str(value)


def delta(a, b):
    if a is None or b is None:
        return None
    return b - a


def causal_table(records, steps):
    metrics = (
        ('route_top1_acc', 'route_top1'),
        ('evidence_top1_acc', 'evidence_top1'),
        ('loss_said', 'loss_said'),
        ('S_sc_mean', 'S_sc'),
        ('S_gc_mean', 'S_gc'),
        ('S_uc_mean', 'S_uc'),
        ('S_sc_minus_S_gc_mean', 'S_sc - S_gc'),
        ('S_uc_minus_S_gc_mean', 'S_uc - S_gc'),
        ('D_U_mean', 'D_U = S_gc - S_uc'),
        ('explanatory_gap_raw_mean', 'gap_raw_mean'),
        ('gap_weight_mean', 'gap_weight_mean'),
        ('explanatory_gap_positive_fraction', 'gap_positive_fraction'),
        ('mask_keep_ratio', 'mask_keep_ratio'),
        ('mask_drop_ratio', 'mask_drop_ratio'),
        ('said_above_threshold_fraction', 'said_above_threshold_fraction'),
        ('loss_exgap', 'loss_exgap'),
    )
    lines = []
    for step in steps:
        m0 = at_step(records['M0'], step)
        m1 = at_step(records['M1'], step)
        lines.append('### completed step %d\n' % step)
        lines.append('| metric | M0 Said-only | M1 ExGAP | M1-M0 |')
        lines.append('| --- | ---: | ---: | ---: |')
        for key, label in metrics:
            a = None if m0 is None else m0.get(key)
            b = None if m1 is None else m1.get(key)
            lines.append('| %s | %s | %s | %s |' % (label, fmt(a), fmt(b), fmt(delta(a, b))))
        lines.append('')
    return '\n'.join(lines)


def geometry_table(records, steps):
    keys = (('global_pairwise_cos', 'global_pairwise_cos'),
            ('global_pairwise_cos_max', 'global_pairwise_cos_max'),
            ('said_pairwise_cos', 'said_pairwise_cos'),
            ('unsaid_pairwise_cos', 'unsaid_pairwise_cos'),
            ('unsaid_pairwise_cos_max', 'unsaid_pairwise_cos_max'),
            ('global_embedding_std', 'global_embedding_std'),
            ('said_embedding_std', 'said_embedding_std'),
            ('unsaid_embedding_std', 'unsaid_embedding_std'),
            ('global_feature_norm', 'global_feature_norm'),
            ('said_feature_norm', 'said_feature_norm'),
            ('unsaid_feature_norm', 'unsaid_feature_norm'))
    steps = [step for step in steps
             if at_step(records['M0'], step) is not None
             and at_step(records['M1'], step) is not None]
    lines = ['| metric | M0 @%s | M1 @%s | M0 @%s | M1 @%s |' % (
        steps[0], steps[0], steps[-1], steps[-1])]
    lines.append('| --- | ---: | ---: | ---: | ---: |')
    first = {label: at_step(records[label], steps[0]) for _, label in ARMS}
    last = {label: at_step(records[label], steps[-1]) for _, label in ARMS}
    for key, label in keys:
        lines.append('| %s | %s | %s | %s | %s |' % (
            label, fmt(None if first['M0'] is None else first['M0'].get(key)),
            fmt(None if first['M1'] is None else first['M1'].get(key)),
            fmt(None if last['M0'] is None else last['M0'].get(key)),
            fmt(None if last['M1'] is None else last['M1'].get(key))))
    return '\n'.join(lines)


def gradient_table(records, steps):
    groups = ('visual_backbone', 'patch_pathway', 'text_encoder', 'said_router', 'exgap_pooling')
    lines = []
    for step in steps:
        lines.append('### completed step %d\n' % step)
        lines.append('| group | M0 G_S | M0 G_ExGAP | M0 R_grad | M1 G_S | M1 G_ExGAP | M1 R_grad |')
        lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: |')
        for group in groups:
            row = []
            for _arm, label in ARMS:
                record = at_step(records[label], step) or {}
                row.extend([fmt(record.get('grad_attr_gS_%s' % group)),
                            fmt(record.get('grad_attr_gE_%s' % group)),
                            fmt(record.get('grad_attr_R_%s' % group), digits=6)])
            lines.append('| %s | %s |' % (group, ' | '.join(row)))
        lines.append('')
    return '\n'.join(lines)


def retrieval_rows(payload):
    """(name, representation, variant, i2t_r1, t2i_r1, i2t_r5, t2i_r5, i2t_r10, t2i_r10)."""
    rows = []
    for name, entry in sorted(payload.get('canonical', {}).items()):
        coco = entry.get('coco_val2017_by_representation') or {}
        for representation, metrics in sorted(coco.items()):
            rows.append((name, representation, 'coco_val2017',
                         metrics.get('image2text_R1'), metrics.get('text2image_R1'),
                         metrics.get('image2text_R5'), metrics.get('text2image_R5'),
                         metrics.get('image2text_R10'), metrics.get('text2image_R10')))
        for variant, result in sorted((entry.get('sharegpt4v1k') or {}).items()):
            for representation, metrics in sorted(
                    (result.get('retrieval_by_representation') or {}).items()):
                rows.append((name, representation, variant,
                             metrics.get('image2text_R1'), metrics.get('text2image_R1'),
                             metrics.get('image2text_R5'), metrics.get('text2image_R5'),
                             metrics.get('image2text_R10'), metrics.get('text2image_R10')))
    return rows


def retrieval_table(payload):
    rows = retrieval_rows(payload)
    lines = ['| model | image repr | dataset / variant | I2T R@1 | T2I R@1 | I2T R@5 | T2I R@5 '
             '| I2T R@10 | T2I R@10 |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, representation, variant, *values in rows:
        marker = '**PRIMARY**' if representation == 'patch_global' else 'diagnostic'
        lines.append('| %s | %s (%s) | %s | %s |' % (
            name, representation, marker, variant, ' | '.join(fmt(v) for v in values)))
    return '\n'.join(lines)


def source_table(payload):
    lines = ['| checkpoint | legacy_cls COCO I2T R@1 | patch_global COCO I2T R@1 | delta | '
             'legacy_cls COCO T2I R@1 | patch_global COCO T2I R@1 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, entry in sorted(payload.get('canonical', {}).items()):
        coco = entry.get('coco_val2017_by_representation') or {}
        cls = coco.get('legacy_cls') or {}
        patch = coco.get('patch_global') or {}
        lines.append('| %s | %s | %s | %s | %s | %s |' % (
            name, fmt(cls.get('image2text_R1')), fmt(patch.get('image2text_R1')),
            fmt(delta(cls.get('image2text_R1'), patch.get('image2text_R1'))),
            fmt(cls.get('text2image_R1')), fmt(patch.get('text2image_R1'))))
    return '\n'.join(lines)


def provenance_table(payload):
    keys = ('checkpoint_sha256', 'checkpoint_step', 'checkpoint_objective_mode',
            'checkpoint_global_pool', 'git_head', 'image_representation', 'global_pool',
            'loaded_tensors', 'missing_keys', 'unexpected_keys', 'dataset_manifest_sha256')
    lines = ['| checkpoint | ' + ' | '.join(keys) + ' |',
             '| --- | ' + ' | '.join('---' for _ in keys) + ' |']
    for name, entry in sorted(payload.get('canonical', {}).items()):
        cells = []
        for key in keys:
            value = entry.get(key)
            if isinstance(value, str) and len(value) > 24:
                value = value[:24] + '...'
            cells.append(fmt(value))
        lines.append('| %s | %s |' % (name, ' | '.join(cells)))
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='runs_salu/said_exgap_phase1b')
    parser.add_argument('--retrieval',
                        default='outputs/said_exgap/phase1b_canonical_retrieval.json')
    parser.add_argument('--json_out', default='outputs/said_exgap/phase1b_tables.json')
    args = parser.parse_args()

    records = {}
    for arm, label in ARMS:
        records[label] = load_records(os.path.join(args.root, arm, 'salu_log.jsonl'))
    steps = [20, 100, 500]
    print('## Causal table (completed steps 20/100/500)\n')
    print(causal_table(records, steps))
    print('\n## Representation geometry\n')
    # the collapse cohort is computed on ``step % exgap_collapse_every == 0``, i.e. at completed
    # steps 21 / 101 / 481 for this run; those are the steps that actually carry the geometry
    print(geometry_table(records, (21, 101, 481)))
    print('\n## Gradient attribution\n')
    print(gradient_table(records, steps))

    payload = {}
    if os.path.exists(args.retrieval):
        with open(args.retrieval, encoding='utf-8') as handle:
            payload = json.load(handle)
        print('\n## Canonical retrieval\n')
        print(retrieval_table(payload))
        print('\n## Representation source\n')
        print(source_table(payload))
        print('\n## Provenance\n')
        print(provenance_table(payload))
    else:
        print('\nMISSING retrieval payload at %s' % args.retrieval)

    digest = {
        'steps': steps,
        'records_present': {arm: sorted({r.get('completed_steps') for r in recs})
                            for arm, recs in records.items()},
        'retrieval_present': bool(payload),
    }
    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, 'w', encoding='utf-8') as handle:
        json.dump(digest, handle, indent=2, sort_keys=True)
    print('\nWROTE %s' % args.json_out)


if __name__ == '__main__':
    main()
