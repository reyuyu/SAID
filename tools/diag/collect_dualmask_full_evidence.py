#!/usr/bin/env python3
"""Collect the Dual-Mask-Full v0.1 evidence JSON from the run directory and the frozen references.

Reads only files; writes one JSON next to the report. Every number records where it came from, and
anything not measured stays null.
"""
import hashlib
import json
import os

REPO = '/root/SAID-s0-dualmask-full-v01'
RUN = os.path.join(REPO, 'runs_salu/dualmask_full_masked_formal500_v01')
OLD_REPORT = '/root/old_masked_formal500_report.json'   # extracted from commit 3b67bce
OLD_SHA = '3b67bcec222691d06f0443f8e8129b197b800e82'
OUT = os.path.join(REPO, 'docs/dual_mask_suffix_full_v01/masked_full_formal500_report.json')
ARM = 'S0_DUALMASK_FULL_V01'


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_log():
    rows = []
    with open(os.path.join(RUN, 'salu_log.jsonl')) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def gate_trajectory(rows):
    keys = ('m_u_keep_ratio', 'm_u_probability_mean', 'm_u_positive_keep_ratio',
            'm_u_positive_probability_mean', 'm_u_positive_all_open_fraction',
            'm_u_positive_all_closed_fraction', 'm_u_positive_cos_to_g')
    series = {key: [[row['completed_steps'], row.get(key)] for row in rows] for key in keys}
    summary = {}
    for key in keys:
        values = [(step, value) for step, value in series[key] if value is not None]
        if not values:
            summary[key] = None
            continue
        summary[key] = {'first': values[0][1], 'last': values[-1][1],
                        'min': min(v for _, v in values), 'max': max(v for _, v in values)}
    return series, summary


def loss_trajectory(rows):
    keys = ('loss_total', 'loss_s0', 'L_S', 'S_S', 'loss_u_align_global', 'loss_u_sparse_global',
            'weighted_s0_align', 'weighted_s0_sparse', 'weighted_u_align', 'weighted_u_sparse')
    return {key: [[row['completed_steps'], row.get(key)] for row in rows] for key in keys}


def main():
    rows = read_log()
    steps = [row['completed_steps'] for row in rows]
    config = json.load(open(os.path.join(RUN, 'config.json')))
    status = json.load(open(os.path.join(RUN, 'run_status.json')))
    old = json.load(open(OLD_REPORT))
    series, summary = gate_trajectory(rows)
    memory = [None] * 4
    for row in rows:
        for entry in row.get('rank_health') or []:
            index = int(entry['rank'])
            value = entry.get('peak_allocated_mb')
            if value is not None and (memory[index] is None or value > memory[index]):
                memory[index] = value
    reserved = [None] * 4
    for row in rows:
        for entry in row.get('rank_health') or []:
            index = int(entry['rank'])
            value = entry.get('peak_reserved_mb')
            if value is not None and (reserved[index] is None or value > reserved[index]):
                reserved[index] = value
    step_seconds = sorted(row['synchronized_step_seconds'] for row in rows
                          if row.get('synchronized_step_seconds'))
    median = step_seconds[len(step_seconds) // 2] if step_seconds else None
    consumed = [row['rank_health'][0]['consumed_samples'] for row in rows]
    stream = {entry['rank']: entry['stream_sha256'] for entry in config['stream_summary']}
    old_stream = {entry['rank']: entry['stream_sha256'] for entry in old['stream_summary']}
    stream_match = {rank: (stream.get(rank) == old_stream.get(rank)) for rank in sorted(stream)}
    coco = status.get('coco', {}).get('metrics')
    urban = (status.get('urban1k') or {}).get('metrics') or None

    def pp(new, ref):
        return None if new is None or ref is None else round(100.0 * (new - ref), 4)

    s0 = {'coco_i2t_r1': 0.60580, 'coco_t2i_r1': 0.41236,
          'urban_i2t_r1': 0.87000, 'urban_t2i_r1': 0.84200}
    weak = {'coco': old['coco']['metrics'],
            'urban': old['urban1k']['urban1k'],
            'checkpoint_sha256': old['strict_export']['checkpoint_sha256'],
            'bare_student_sha256': old['strict_export']['bare_student_sha256'],
            'training_sha': old['training_sha'],
            'suffix_lambda': old['configuration']['suffix_lambda'],
            'u_sparsity': old['configuration']['u_sparsity']}
    report = {
        'arm': ARM,
        'scope': 'masked only; exactly 500 optimizer updates; combined ablation',
        'objective': '10*L_S + 2*S_S + 10*L_U + 2*S_U',
        'old_objective': '10*L_S + 2*S_S + 1*L_U',
        'combination_note': ('the suffix alignment weight and the new U-gate sparsity term changed '
                             'together; nothing here attributes the result to either one alone'),
        'training_sha': config.get('training_sha'),
        'base_sha': 'ff5ad1d4b918d56c6bfa48a2870dc5223e757237',
        'run_directory': RUN,
        'actual_optimizer_updates': max(steps) if steps else 0,
        'logged_steps': len(rows),
        'sample_occurrences': ((consumed[-1] if consumed else None) * int(config.get('world_size', 1)))
        if consumed else None,
        'sample_occurrences_definition': ('rank 0 consumed samples at the last logged step, times '
                                          'the world size; the old weak-suffix run recorded 512000'),
        'gradient_groups_at_logged_steps': {
            str(row['completed_steps']): (row.get('rank_health') or [{}])[0].get(
                'gradient_norms_before_update')
            for row in rows
            if (row.get('rank_health') or [{}])[0].get('gradient_norms_before_update')},
        'configuration': config,
        'exit_codes': status.get('exit_codes'),
        'stage_walls': status.get('stage_walls'),
        'evaluator_sha256': status.get('evaluator_sha256'),
        'protocols': status.get('protocols'),
        'checkpoint': status.get('checkpoint'),
        'checkpoint_sha256': status.get('checkpoint_sha256'),
        'student': status.get('student'),
        'student_metadata': status.get('student_metadata'),
        'coco': status.get('coco'),
        'urban1k': status.get('urban1k'),
        'step_seconds_median': median,
        'step_seconds_max': max(step_seconds) if step_seconds else None,
        'per_gpu_peak_allocated_mb': memory,
        'per_gpu_peak_reserved_mb': reserved,
        'global_valid_min_max': [min(row['valid_global'] for row in rows),
                                 max(row['valid_global'] for row in rows)],
        'gate_trajectory_summary': summary,
        'gate_trajectory': series,
        'loss_trajectory': loss_trajectory(rows),
        'first3': rows[:3],
        'checkpoint_steps': [row['completed_steps'] for row in rows
                             if row['completed_steps'] in (100, 200, 300, 400, 500)],
        'per_rank_stream_digests': stream,
        'per_rank_stream_digests_old_weak_suffix': old_stream,
        'per_rank_stream_digest_match': stream_match,
        'per_rank_stream_digests_all_match': all(stream_match.values()),
        'reference_S0_at_500': s0,
        'reference_weak_suffix_masked_at_500': weak,
        'reference_weak_suffix_report_commit': OLD_SHA,
        'comparison': {
            'coco_i2t_r1_pp_vs_S0': pp(coco['image2text_R1'], s0['coco_i2t_r1']) if coco else None,
            'coco_t2i_r1_pp_vs_S0': pp(coco['text2image_R1'], s0['coco_t2i_r1']) if coco else None,
            'urban_i2t_r1_pp_vs_S0': pp(urban['image2text']['R1'], s0['urban_i2t_r1']) if urban else None,
            'urban_t2i_r1_pp_vs_S0': pp(urban['text2image']['R1'], s0['urban_t2i_r1']) if urban else None,
            'coco_i2t_r1_pp_vs_weak': pp(coco['image2text_R1'], weak['coco']['image2text_R1']) if coco else None,
            'coco_t2i_r1_pp_vs_weak': pp(coco['text2image_R1'], weak['coco']['text2image_R1']) if coco else None,
            'urban_i2t_r1_pp_vs_weak': pp(urban['image2text']['R1'], weak['urban']['image2text']['R1']) if urban else None,
            'urban_t2i_r1_pp_vs_weak': pp(urban['text2image']['R1'], weak['urban']['text2image']['R1']) if urban else None,
        },
        'coco_query_counts': {
            'i2t_1pp_queries': 50, 't2i_1pp_queries': 250,
            'definition': '5000 image queries (0.02 pp each) and 25000 text queries (0.004 pp each)',
        },
        'gate_verdict': None,
        'limits': [
            'No native arm was trained and no existing baseline was retrained.',
            'Combined ablation: suffix alignment weight and U-gate sparsity changed together.',
            'Equal numeric weights do not imply equal gradient norms.',
            'More closed coordinates are not evidence of a purer Unsaid extract.',
            'Only native CLS/EOS student embeddings are evaluated: no gate, fusion or reranking.',
            'Single seed; no cross-seed stability claim, and the 0.04-0.28 pp run-to-run spread of '
            'the reference arms is not used as a significance threshold.',
            'The old weak-suffix evaluator scripts lived outside the repository on the other '
            'machine; the canonical COCO and Urban protocols in use here are the ones the frozen '
            'S0/GlobalOnly results were produced with on this machine.',
        ],
    }
    if coco and urban:
        passed_coco = (coco['image2text_R1'] >= s0['coco_i2t_r1']
                       and coco['text2image_R1'] >= s0['coco_t2i_r1']
                       and (coco['image2text_R1'] > s0['coco_i2t_r1']
                            or coco['text2image_R1'] > s0['coco_t2i_r1']))
        report['gate_verdict'] = {
            'rule': 'COCO both directions R@1 not below S0@500, and at least one strictly above',
            'coco_i2t_not_below_S0': coco['image2text_R1'] >= s0['coco_i2t_r1'],
            'coco_t2i_not_below_S0': coco['text2image_R1'] >= s0['coco_t2i_r1'],
            'coco_i2t_strictly_above': coco['image2text_R1'] > s0['coco_i2t_r1'],
            'coco_t2i_strictly_above': coco['text2image_R1'] > s0['coco_t2i_r1'],
            'verdict': 'PASS' if passed_coco else 'FAIL',
            'urban_note': 'Urban-1k is reported separately and the threshold is not changed now',
        }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print('WROTE ' + OUT)
    print(json.dumps({k: report[k] for k in ('actual_optimizer_updates', 'sample_occurrences',
                                             'per_rank_stream_digests_all_match',
                                             'comparison', 'gate_verdict', 'coco', 'urban1k')},
                     ensure_ascii=False, indent=1)[:4000])


if __name__ == '__main__':
    main()
