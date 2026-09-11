"""SAID-ExGAP v1.5 report tables (F0 vs F1) built only from measured values.

    python tools/exp_said_exgap_finalcls_tables.py --tag step500 \
        [--retrieval outputs/said_exgap/finalcls_v15_canonical_retrieval.json]

Prints markdown tables; a value that was not measured is printed as ``MISSING`` -- never as 0.
"""
import argparse
import json
import os

ARMS = (('F0_said_only', 'F0 Said-only'), ('F1_full_exgap', 'F1 ExGAP'))
CORE = (
    ('loss_said', 'loss_said'),
    ('loss_route', 'loss_route'),
    ('loss_evidence', 'loss_evidence'),
    ('loss_exgap', 'loss_exgap'),
    ('route_top1_acc', 'route_top1'),
    ('evidence_top1_acc', 'evidence_top1'),
    ('S_gc_mean', 'S_gc'),
    ('S_sc_mean', 'S_sc'),
    ('S_uc_mean', 'S_uc'),
    ('S_sc_minus_S_gc_mean', 'S_sc - S_gc'),
    ('D_U_mean', 'D_U = S_gc - S_uc'),
    ('gap_weight_mean', 'G_exp (gap_weight)'),
    ('explanatory_gap_positive_fraction', 'gap_positive_fraction'),
    ('said_relevance_mean', 'said_relevance_mean'),
    ('said_relevance_std', 'said_relevance_std'),
    ('said_fraction_gt_0_6', 'said_fraction_gt_0.6 (diagnostic)'),
    ('said_effective_patch_count', 'N_eff Said (soft)'),
    ('unsaid_effective_patch_count', 'N_eff Unsaid (soft)'),
    ('said_to_global_cos', 'cos(z_S, g)'),
    ('unsaid_to_global_cos', 'cos(z_U, g)'),
    ('said_to_unsaid_cos', 'cos(z_S, z_U)'),
    ('said_minus_global_l2', '||z_S - g||'),
    ('unsaid_minus_global_l2', '||z_U - g||'),
)
GEOMETRY = (
    ('global_cls_pairwise_cos', 'global_cls_pairwise_cos'),
    ('global_cls_pairwise_cos_max', 'global_cls_pairwise_cos_max'),
    ('global_cls_std', 'global_cls_std'),
    ('64way_i2t_at1', '64way_i2t@1'),
    ('64way_t2i_at1', '64way_t2i@1'),
)
RATE = (
    ('compute_sec_per_step', 'compute_sec_per_step'),
    ('compute_samples_per_sec', 'compute_samples_per_sec'),
    ('peak_gpu_mem_gb', 'peak_gpu_mem_gb'),
)


def load(path):
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path, encoding='utf-8') if line.startswith('{')]


def fmt(value):
    if value is None:
        return 'MISSING'
    if isinstance(value, float):
        if value != 0 and abs(value) < 1e-3:
            return '%.3g' % value
        return '%.4f' % value
    return str(value)


def last_with(records, key):
    for record in reversed(records):
        if record.get(key) is not None:
            return record
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='runs_salu/said_exgap_finalcls')
    parser.add_argument('--tag', default='step500')
    parser.add_argument('--retrieval',
                        default='outputs/said_exgap/finalcls_v15_canonical_retrieval.json')
    args = parser.parse_args()

    records, final = {}, {}
    for arm, _ in ARMS:
        path = os.path.join(args.root, '%s_%s' % (args.tag, arm), 'salu_log.jsonl')
        records[arm] = load(path)
        final[arm] = records[arm][-1] if records[arm] else {}

    print('## %s -- training geometry (last logged step)\n' % args.tag)
    print('| metric | %s | %s | F1-F0 |' % (ARMS[0][1], ARMS[1][1]))
    print('| --- | ---: | ---: | ---: |')
    for key, label in CORE:
        a, b = final[ARMS[0][0]].get(key), final[ARMS[1][0]].get(key)
        delta = (b - a) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
        print('| %s | %s | %s | %s |' % (label, fmt(a), fmt(b), fmt(delta)))
    print()

    print('## %s -- native CLS geometry (fixed 64-image cohort)\n' % args.tag)
    print('| metric | %s | %s |' % (ARMS[0][1], ARMS[1][1]))
    print('| --- | ---: | ---: |')
    for key, label in GEOMETRY:
        values = []
        for arm, _ in ARMS:
            record = last_with(records[arm], key)
            values.append(fmt(None if record is None else record.get(key)))
        print('| %s | %s | %s |' % (label, values[0], values[1]))
    print()

    print('## %s -- throughput\n' % args.tag)
    print('| metric | %s | %s |' % (ARMS[0][1], ARMS[1][1]))
    print('| --- | ---: | ---: |')
    for key, label in RATE:
        print('| %s | %s | %s |' % (label, fmt(final[ARMS[0][0]].get(key)),
                                    fmt(final[ARMS[1][0]].get(key))))
    print()

    print('## %s -- gradient attribution (last attributed step)\n' % args.tag)
    print('| group | G_S | G_ExGAP | R_grad | arm |')
    print('| --- | ---: | ---: | ---: | --- |')
    for arm, label in ARMS:
        record = None
        for candidate in reversed(records[arm]):
            if any(key.startswith('grad_attr_gS_') for key in candidate):
                record = candidate
                break
        if record is None:
            print('| MISSING | MISSING | MISSING | MISSING | %s |' % label)
            continue
        for group in ('visual_backbone', 'final_block', 'text_encoder', 'said_router', 'other'):
            print('| %s | %s | %s | %s | %s |'
                  % (group, fmt(record.get('grad_attr_gS_%s' % group)),
                     fmt(record.get('grad_attr_gE_%s' % group)),
                     fmt(record.get('grad_attr_R_%s' % group)), label))
    print()

    if os.path.exists(args.retrieval):
        payload = json.load(open(args.retrieval, encoding='utf-8'))
        print('## canonical retrieval (PRIMARY = legacy_cls / native CLS)\n')
        print('| model | image repr | dataset / variant | I2T R@1 | T2I R@1 | I2T R@5 | T2I R@5 '
              '| I2T R@10 | T2I R@10 |')
        print('| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |')
        for name, entry in sorted(payload.get('canonical', {}).items()):
            rows = []
            coco = entry.get('coco_val2017_by_representation') or {}
            for representation in ('legacy_cls', 'patch_global'):
                if representation in coco:
                    rows.append((representation, 'coco_val2017', coco[representation]))
            for variant, result in sorted((entry.get('sharegpt4v1k') or {}).items()):
                for representation in ('legacy_cls', 'patch_global'):
                    metrics = (result.get('retrieval_by_representation') or {}).get(representation)
                    if metrics:
                        rows.append((representation, variant, metrics))
            for representation, dataset, metrics in rows:
                marker = '**PRIMARY**' if representation == 'legacy_cls' else 'diagnostic'
                print('| %s | %s (%s) | %s | %s |' % (
                    name, representation, marker, dataset,
                    ' | '.join(fmt(metrics.get(key)) for key in (
                        'image2text_R1', 'text2image_R1', 'image2text_R5', 'text2image_R5',
                        'image2text_R10', 'text2image_R10'))))
        print('\n## provenance\n')
        for name, entry in sorted(payload.get('canonical', {}).items()):
            print('- **%s**: sha256 `%s`, step %s, objective %s, loaded %s tensors, '
                  'missing %s, unexpected %s, git %s'
                  % (name, entry.get('checkpoint_sha256'), entry.get('checkpoint_step'),
                     entry.get('checkpoint_objective_mode'), entry.get('loaded_tensors'),
                     entry.get('missing_keys'), entry.get('unexpected_keys'),
                     entry.get('git_head')))
    else:
        print('MISSING retrieval payload at %s' % args.retrieval)


if __name__ == '__main__':
    main()
