"""SAID-ExGAP v1.5 F0 3-epoch trajectory tables + verdict inputs.

    python tools/exp_said_exgap_finalcls_f0_trajectory.py

Prints (a) the training-side trajectory at the checkpoint steps, (b) the native-CLS cohort
geometry, (c) the canonical retrieval trajectory (PRIMARY = legacy_cls), and (d) the explicit
booleans the RECOVERS / PERSISTENT TRADE-OFF / PEAKS EARLY / COLLAPSES judgement is made from.
Nothing is estimated: an unavailable value prints as MISSING.
"""
import argparse
import json
import os

TARGETS = (0, 500, 1216, 2432, 3648)
TRAIN_KEYS = (
    ('loss_said', 'loss_said'),
    ('route_top1_acc', 'route_top1'),
    ('evidence_top1_acc', 'evidence_top1'),
    ('S_gc_mean', 'S_gc'),
    ('S_sc_mean', 'S_sc'),
    ('S_sc_minus_S_gc_mean', 'S_sc - S_gc'),
    ('said_to_global_cos', 'cos(z_S, g)'),
    ('said_relevance_mean', 'relevance mean'),
    ('said_relevance_std', 'relevance std'),
    ('said_relevance_p10', 'relevance p10'),
    ('said_relevance_p90', 'relevance p90'),
    ('said_fraction_gt_0_6', 'relevance > 0.6 (diag)'),
)
GEOMETRY_KEYS = (
    ('global_cls_pairwise_cos', 'global_cls_pairwise_cos'),
    ('global_cls_std', 'global_cls_std'),
    ('64way_i2t_at1', '64way_i2t@1'),
    ('64way_t2i_at1', '64way_t2i@1'),
)
RETRIEVAL_CELLS = (
    ('coco_val2017', 'i2t_r1', 'COCO I2T R@1'),
    ('coco_val2017', 't2i_r1', 'COCO T2I R@1'),
    ('coco_val2017', 'i2t_r5', 'COCO I2T R@5'),
    ('coco_val2017', 't2i_r5', 'COCO T2I R@5'),
    ('coco_val2017', 'i2t_r10', 'COCO I2T R@10'),
    ('coco_val2017', 't2i_r10', 'COCO T2I R@10'),
)
VARIANTS = (('first_sentence', '1K first'), ('fixed_sparse', '1K sparse'),
            ('full_dense', '1K full'))
ORDER = ('initial', 'step500', 'epoch1', 'epoch2', 'epoch3')


def load(path):
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path, encoding='utf-8') if line.startswith('{')]


def fmt(value, digits=4):
    if value is None:
        return 'MISSING'
    if isinstance(value, float):
        if value != 0 and abs(value) < 1e-3:
            return '%.3g' % value
        return ('%.' + str(digits) + 'f') % value
    return str(value)


def nearest(records, completed):
    """The logged record closest to ``completed`` (returns the record and its real step)."""
    if not records:
        return None, None
    best = min(records, key=lambda record: abs(record.get('completed_steps', -1) - completed))
    return best, best.get('completed_steps')


def metric(entry, dataset, cell):
    if dataset == 'coco_val2017':
        metrics = ((entry.get('coco_val2017_by_representation') or {}).get('legacy_cls') or {})
        mapping = {'i2t_r1': 'image2text_R1', 't2i_r1': 'text2image_R1',
                   'i2t_r5': 'image2text_R5', 't2i_r5': 'text2image_R5',
                   'i2t_r10': 'image2text_R10', 't2i_r10': 'text2image_R10'}
        return metrics.get(mapping[cell])
    result = (entry.get('sharegpt4v1k') or {}).get(dataset) or {}
    metrics = (result.get('retrieval_by_representation') or {}).get('legacy_cls') or {}
    mapping = {'i2t_r1': 'image2text_R1', 't2i_r1': 'text2image_R1',
               'i2t_r5': 'image2text_R5', 't2i_r5': 'text2image_R5',
               'i2t_r10': 'image2text_R10', 't2i_r10': 'text2image_R10'}
    return metrics.get(mapping[cell])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='runs_salu/said_exgap_finalcls_3ep/F0_said_only_3ep')
    parser.add_argument('--retrieval',
                        default='outputs/said_exgap/f0_3epoch_canonical_retrieval.json')
    args = parser.parse_args()

    records = load(os.path.join(args.root, 'salu_log.jsonl'))
    payload = json.load(open(args.retrieval, encoding='utf-8')) if os.path.exists(
        args.retrieval) else {}
    entries = payload.get('canonical', {})

    print('logged records: %d (completed steps %s .. %s)'
          % (len(records), records[0].get('completed_steps') if records else 'MISSING',
             records[-1].get('completed_steps') if records else 'MISSING'))

    print('\n## 1. training trajectory (nearest logged record to each checkpoint step)\n')
    header = '| metric | ' + ' | '.join('step %d' % t for t in TARGETS) + ' |'
    print(header)
    print('| --- | ' + ' | '.join('---:' for _ in TARGETS) + ' |')
    print('| (logged completed step) | '
          + ' | '.join(str(nearest(records, t)[1]) for t in TARGETS) + ' |')
    for key, label in TRAIN_KEYS:
        cells = [fmt((nearest(records, t)[0] or {}).get(key)) for t in TARGETS]
        print('| %s | %s |' % (label, ' | '.join(cells)))
    print()
    print('## 2. native-CLS cohort geometry (logged on step %% 50 == 0)\n')
    print(header)
    print('| --- | ' + ' | '.join('---:' for _ in TARGETS) + ' |')
    for key, label in GEOMETRY_KEYS:
        cells = []
        for target in TARGETS:
            record = None
            for candidate in records:
                if candidate.get(key) is None:
                    continue
                if record is None or abs(candidate['completed_steps'] - target) < \
                        abs(record['completed_steps'] - target):
                    record = candidate
            cells.append(fmt((record or {}).get(key)))
        print('| %s | %s |' % (label, ' | '.join(cells)))

    print('\n## 3. canonical retrieval trajectory (PRIMARY = legacy_cls / native CLS)\n')
    if not entries:
        print('MISSING retrieval payload at %s' % args.retrieval)
    rows = list(RETRIEVAL_CELLS) + [
        (variant, 'i2t_r1', '%s I2T R@1' % label) for variant, label in VARIANTS] + [
        (variant, 't2i_r1', '%s T2I R@1' % label) for variant, label in VARIANTS]
    print('| metric | ' + ' | '.join(ORDER) + ' |')
    print('| --- | ' + ' | '.join('---:' for _ in ORDER) + ' |')
    for dataset, cell, label in rows:
        cells = []
        for name in ORDER:
            entry = entries.get(name)
            cells.append(fmt(None if entry is None else metric(entry, dataset, cell)))
        print('| %s | %s |' % (label, ' | '.join(cells)))

    print('\n## 4. verdict inputs (measured booleans, no interpretation)\n')
    def value(name, dataset, cell):
        entry = entries.get(name)
        return None if entry is None else metric(entry, dataset, cell)

    initial_i2t = value('initial', 'coco_val2017', 'i2t_r1')
    final_i2t = value('epoch3', 'coco_val2017', 'i2t_r1')
    step500_i2t = value('step500', 'coco_val2017', 'i2t_r1')
    trajectory = [value(name, 'coco_val2017', 'i2t_r1') for name in ORDER]
    known = [v for v in trajectory if v is not None]
    print('COCO I2T R@1 trajectory:', trajectory)
    print('COCO T2I R@1 trajectory:', [value(name, 'coco_val2017', 't2i_r1') for name in ORDER])
    print('1K full I2T R@1 trajectory:',
          [value(name, 'full_dense', 'i2t_r1') for name in ORDER])
    print('1K sparse I2T R@1 trajectory:',
          [value(name, 'fixed_sparse', 'i2t_r1') for name in ORDER])
    print('1K first I2T R@1 trajectory:',
          [value(name, 'first_sentence', 'i2t_r1') for name in ORDER])
    cosines = [record.get('global_cls_pairwise_cos') for record in records
               if record.get('global_cls_pairwise_cos') is not None]
    print('max global_cls_pairwise_cos over training:', fmt(max(cosines) if cosines else None))
    if initial_i2t is not None and final_i2t is not None and step500_i2t is not None:
        print('COCO I2T recovered to Initial by epoch3?          ', final_i2t >= initial_i2t)
        print('COCO I2T still below Initial at epoch3?           ', final_i2t < initial_i2t)
        print('COCO I2T best checkpoint is step500 (not later)?  ',
              step500_i2t >= max(v for v in known[1:]))
        print('COCO I2T monotone non-decreasing after step500?   ',
              all(b >= a for a, b in zip(known[1:], known[2:])))
        print('COCO T2I above Initial at epoch3?                 ',
              (value('epoch3', 'coco_val2017', 't2i_r1') or 0) >
              (value('initial', 'coco_val2017', 't2i_r1') or 1))


if __name__ == '__main__':
    main()
