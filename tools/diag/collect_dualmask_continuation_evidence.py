#!/usr/bin/env python3
"""Collect the Dual-Mask-Full v0.1 continuation evidence JSON (500 -> 3651, full 3 epochs).

Reads the continuation run directory, the original 500-step run and the frozen references; writes one
JSON next to the report. Unmeasured values stay null, and every comparison records the reference it
used.
"""
import hashlib
import json
import os

REPO = '/root/SAID-s0-dualmask-full-v01'
RUN = os.path.join(REPO, 'runs_salu/dualmask_full_masked_3epoch_v01')
ORIGINAL = os.path.join(REPO, 'runs_salu/dualmask_full_masked_formal500_v01')
OUT = os.path.join(REPO, 'docs/dual_mask_suffix_full_v01/continuation_3epoch_report.json')
ARM = 'S0_DUALMASK_FULL_V01'
EVAL_STATUS = os.path.join(RUN, 'evaluation_cont_status.json')

FROZEN = {
    'S0@500': {'coco_i2t_r1': 0.60580, 'coco_t2i_r1': 0.41236,
               'urban_i2t_r1': 0.87000, 'urban_t2i_r1': 0.84200},
    'S0@1000': {'coco_i2t_r1': 0.61720, 'coco_t2i_r1': 0.41992,
                'urban_i2t_r1': 0.89000, 'urban_t2i_r1': 0.85500},
    'S0_TriMask_HS@1000': {'coco_i2t_r1': 0.61480, 'coco_t2i_r1': 0.41796,
                           'urban_i2t_r1': 0.89000, 'urban_t2i_r1': 0.85500},
    'PG_CLIP@1000': {'coco_i2t_r1': 0.60360, 'coco_t2i_r1': 0.41560,
                     'urban_i2t_r1': 0.89100, 'urban_t2i_r1': 0.85200},
    'S0_DUALMASK_FULL@500': {'coco_i2t_r1': 0.58780, 'coco_t2i_r1': 0.40216,
                             'urban_i2t_r1': 0.88400, 'urban_t2i_r1': 0.87800},
    'weak_suffix_masked@500': {'coco_i2t_r1': 0.60540, 'coco_t2i_r1': 0.41464,
                               'urban_i2t_r1': 0.88200, 'urban_t2i_r1': 0.85800},
}


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_log(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pp(new, ref):
    return None if new is None or ref is None else round(100.0 * (new - ref), 4)


def main():
    rows = read_log(os.path.join(RUN, 'salu_log.jsonl'))
    config = json.load(open(os.path.join(RUN, 'config.json')))
    original_config = json.load(open(os.path.join(ORIGINAL, 'config.json')))
    status = json.load(open(EVAL_STATUS)) if os.path.isfile(EVAL_STATUS) else {'results': {}}
    steps = [row['completed_steps'] for row in rows]
    assert steps == list(range(501, 3652)), 'the continuation log is not a gapless 501..3651 range'

    epoch_layout = []
    for epoch in sorted({row['epoch'] for row in rows}):
        subset = [row for row in rows if row['epoch'] == epoch]
        epoch_layout.append({'epoch': epoch, 'first_step': subset[0]['completed_steps'],
                             'last_step': subset[-1]['completed_steps'],
                             'first_step_in_epoch': subset[0]['step_in_epoch'],
                             'last_step_in_epoch': subset[-1]['step_in_epoch'],
                             'batches': len(subset)})

    def series(key):
        return [[row['completed_steps'], row.get(key)] for row in rows]

    memory = [None] * 4
    for row in rows:
        for entry in row.get('rank_health') or []:
            index = int(entry['rank'])
            value = entry.get('peak_allocated_mb')
            if value is not None and (memory[index] is None or value > memory[index]):
                memory[index] = value
    step_seconds = sorted(row['synchronized_step_seconds'] for row in rows
                          if row.get('synchronized_step_seconds'))
    checkpoints = {}
    for name in sorted(os.listdir(RUN)):
        if name.endswith('.pt'):
            path = os.path.join(RUN, name)
            checkpoints[name] = {'size_bytes': os.path.getsize(path), 'sha256': sha256_of(path)}

    evaluations = {}
    for step, entry in sorted(status.get('results', {}).items(), key=lambda kv: int(kv[0])):
        coco = entry['coco']
        urban = entry['urban1k']
        line = {
            'checkpoint_sha256': entry['checkpoint_sha256'],
            'student_sha256': entry['student_sha256'],
            'coco': {'i2t_r1': coco['image2text_R1'], 'i2t_r5': coco['image2text_R5'],
                     'i2t_r10': coco['image2text_R10'], 't2i_r1': coco['text2image_R1'],
                     't2i_r5': coco['text2image_R5'], 't2i_r10': coco['text2image_R10']},
            'urban1k': {'i2t_r1': urban['image2text']['R1'], 'i2t_r5': urban['image2text']['R5'],
                        'i2t_r10': urban['image2text']['R10'],
                        't2i_r1': urban['text2image']['R1'], 't2i_r5': urban['text2image']['R5'],
                        't2i_r10': urban['text2image']['R10']},
        }
        for name, ref in FROZEN.items():
            line['vs_' + name] = {
                'coco_i2t_r1_pp': pp(coco['image2text_R1'], ref['coco_i2t_r1']),
                'coco_t2i_r1_pp': pp(coco['text2image_R1'], ref['coco_t2i_r1']),
                'urban_i2t_r1_pp': pp(urban['image2text']['R1'], ref['urban_i2t_r1']),
                'urban_t2i_r1_pp': pp(urban['text2image']['R1'], ref['urban_t2i_r1']),
            }
        evaluations[step] = line

    report = {
        'arm': ARM,
        'scope': ('strict continuation of the 500-step masked run to the full cosine horizon '
                  '(3 epochs = 3651 updates), then export and the two frozen native evaluations'),
        'objective': config.get('total_objective'),
        'continuation_sha': config.get('training_sha'),
        'resumed_from_training_sha': config.get('resumed_from_training_sha'),
        'resumed_from_checkpoint': config.get('resumed_from'),
        'objective_code_sha256': config.get('objective_code_sha256'),
        'run_directory': RUN,
        'original_run_directory': ORIGINAL,
        'total_optimizer_updates': max(steps),
        'continuation_optimizer_updates': len(rows),
        'full_3_epochs': True,
        'lr_horizon_steps': config.get('lr_horizon_steps'),
        'loader_batches': config.get('loader_batches'),
        'final_lr': rows[-1]['lr'],
        'resume_checks': config.get('resume_checks'),
        'resume_skip_rule': config.get('resume_skip_rule'),
        'skipped_batches': rows[0].get('skipped_batches'),
        'resume_stream_batch_index': config.get('resume_stream_batch_index'),
        'epoch_layout': epoch_layout,
        'stream_digest_scope': rows[-1].get('stream_digest_scope'),
        'per_rank_stream_digests_continuation': config.get('stream_summary'),
        'per_rank_stream_digests_original_500': original_config.get('stream_summary'),
        'step_seconds_median': step_seconds[len(step_seconds) // 2] if step_seconds else None,
        'step_seconds_max': max(step_seconds) if step_seconds else None,
        'training_seconds': config.get('training_seconds'),
        'per_gpu_peak_allocated_mb': memory,
        'global_valid_min_max': [min(row['valid_global'] for row in rows),
                                 max(row['valid_global'] for row in rows)],
        'checkpoints': checkpoints,
        'evaluations': evaluations,
        'frozen_references': FROZEN,
        'loss_trajectory': {key: series(key) for key in
                            ('loss_total', 'loss_s0', 'L_S', 'S_S', 'loss_u_align_global',
                             'loss_u_sparse_global', 'weighted_u_align', 'weighted_u_sparse')},
        'gate_trajectory': {key: series(key) for key in
                            ('m_u_keep_ratio', 'm_u_probability_mean', 'm_u_positive_keep_ratio',
                             'm_u_positive_probability_mean', 'm_u_positive_all_open_fraction',
                             'm_u_positive_all_closed_fraction', 'm_u_positive_cos_to_g')},
        'limits': [
            'Single seed; no cross-seed stability claim.',
            'Combined ablation: the suffix weight and the U-gate sparsity term changed together.',
            'The frozen promotion gate was defined for the 500-step budget; it is not re-applied to a '
            'different budget here, and no gate verdict is claimed for the 3-epoch result.',
            'Longer training is not a repair for the 500-step comparison; both budgets are reported.',
            'Only native CLS/EOS student embeddings are evaluated: no gate, fusion or reranking.',
            'The continuation skip consumes the already-seen batches through the real data pipeline, '
            'which costs about one 500-step data pass; that overhead is inside training_seconds.',
            'Four of the seven saved checkpoints were evaluated; the others were saved but not scored.',
        ],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print('WROTE ' + OUT)
    print(json.dumps({'total_updates': report['total_optimizer_updates'],
                      'skipped_batches': report['skipped_batches'],
                      'epoch_layout': epoch_layout,
                      'evaluations': {k: {'coco': v['coco'], 'urban': v['urban1k'],
                                          'vs S0@500': v['vs_S0@500'],
                                          'vs S0@1000': v['vs_S0@1000'],
                                          'vs own@500': v['vs_S0_DUALMASK_FULL@500']}
                                      for k, v in evaluations.items()}},
                     ensure_ascii=False, indent=1)[:6000])


if __name__ == '__main__':
    main()
