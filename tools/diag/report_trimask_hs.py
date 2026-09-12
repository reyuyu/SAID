"""Compile the S0-TriMask-HS v0.2 evidence into docs/said_trimask_hs/{results.json,report.md}.

Every number is read from a produced artifact (the run directory, the two evaluation JSONs, the two
frozen baselines). The script refuses to report unless the run completed exactly the authorised 500
updates, the checkpoint records the hard-gate objective with lambda_sparse_t = 0.2 and the three path
weights 10/1/1, and both evaluations used the exported bare student.
"""
import argparse
import hashlib
import json
import os

ARM = 'S0_TriMask_HS'
OBJECTIVE = 'smartclip_trimask_hs'
GATE_MODE = 'hard_st'
GATE_COCO_I2T = 0.6058
GATE_COCO_T2I = 0.41236
CURVE_STEPS = (1, 20, 100, 250, 500)
CURVE_KEYS = (
    'loss_total', 'loss_1', 'loss_1_i2t', 'loss_1_t2i', 'loss_2', 'loss_2_i2t', 'loss_2_t2i',
    'loss_3', 'loss_3_i2t', 'loss_3_t2i', 'loss_sparse_i', 'loss_sparse_t',
    'weighted_loss_1', 'weighted_loss_2', 'weighted_loss_3', 'weighted_loss_sparse_i',
    'weighted_loss_sparse_t', 'ell1_mean', 'ell2_mean', 'ell3_mean', 'adv_gap_positive_part',
    'third_not_worse_fraction', 'third_better_fraction',
    'path1_margin_max_negative_i2t', 'path1_margin_logsumexp_i2t',
    'path2_margin_max_negative_i2t', 'path2_margin_logsumexp_i2t',
    'path3_margin_max_negative_i2t', 'path3_margin_logsumexp_i2t',
    'path1_top1_i2t', 'path1_top1_t2i', 'path2_top1_i2t', 'path2_top1_t2i',
    'path3_top1_i2t', 'path3_top1_t2i',
    'mask_i_keep_ratio', 'mask_i_empty_fraction', 'mask_i_full_fraction',
    'mask_i_retained_energy_fraction', 'mask_i_soft_saturation_fraction',
    'mask_t_mean', 'mask_t_std', 'mask_t_within_sample_std', 'mask_t_empty_fraction',
    'mask_t_full_fraction', 'mask_t_cosine_with_unmasked', 'mask_t_retained_energy_fraction',
    'text_direction_change_fraction', 'text_gate_pT_mean', 'text_gate_pT_min',
    'text_gate_pT_max', 'text_gate_pT_near_threshold_fraction',
    'text_gate_hT_zero_fraction', 'text_gate_hT_one_fraction', 'text_gate_hT_is_binary',
    'text_gate_produces_zeros',
    'mask_intersection_intersection_count_mean', 'mask_intersection_jaccard_mean',
    'mask_intersection_intersection_empty_fraction',
    'mask_intersection_both_nonempty_but_intersection_empty_fraction',
    'mask_intersection_union_empty_fraction', 'mask_intersection_jaccard_defined_count',
    'mask_t_variance_within_caption', 'mask_t_variance_between_captions',
    'mask_t_between_over_total', 'mask_t_cross_caption_std_mean',
    'mask_t_profile_min', 'mask_t_profile_max', 'mask_t_profile_std',
    'sec_per_step', 'peak_memory_gb', 'grad_norm_backbone', 'grad_norm_visual_mask_net',
    'grad_norm_text_stem', 'grad_norm_text_gate_projection', 'grads_finite')


def read(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def log_records(path):
    records = {}
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            if line.startswith('LOG '):
                payload = json.loads(line[4:])
                records[int(payload['completed_steps'])] = payload
    return records


def coco_block(payload, name):
    inner = payload['canonical'][name]['coco_val2017']
    return {'i2t_r1': inner['image2text_R1'], 'i2t_r5': inner['image2text_R5'],
            'i2t_r10': inner['image2text_R10'], 't2i_r1': inner['text2image_R1'],
            't2i_r5': inner['text2image_R5'], 't2i_r10': inner['text2image_R10']}


def urban_block(payload):
    block = payload['urban1k']
    return {'i2t_r1': block['image2text']['R1'], 'i2t_r5': block['image2text']['R5'],
            'i2t_r10': block['image2text']['R10'], 't2i_r1': block['text2image']['R1'],
            't2i_r5': block['text2image']['R5'], 't2i_r10': block['text2image']['R10']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--step', type=int, default=500)
    parser.add_argument('--docs', required=True)
    parser.add_argument('--s0',
                        default='/root/SAID-token-v1/docs/said_token_v1/fix_v2_results.json')
    parser.add_argument('--soft',
                        default='/root/SAID-s0-trimask-v01/docs/said_trimask/results.json')
    parsed = parser.parse_args()
    run, step, tag = parsed.run, parsed.step, '%06d' % parsed.step

    config = read(os.path.join(run, 'config.json'))
    summary = read(os.path.join(run, 'run_summary.json'))
    for key, expected in (('objective', OBJECTIVE), ('arm', ARM), ('text_gate_mode', GATE_MODE)):
        if config.get(key) != expected:
            raise SystemExit('config %s is %r, expected %r' % (key, config.get(key), expected))
    if float(config.get('lambda_sparse_t', -1)) != 0.2:
        raise SystemExit('config lambda_sparse_t is %r, expected 0.2'
                         % config.get('lambda_sparse_t'))
    if (config.get('lambda_1'), config.get('lambda_2'), config.get('lambda_3')) != (10.0, 1.0, 1.0):
        raise SystemExit('config path weights are not 10/1/1: %r'
                         % [config.get('lambda_1'), config.get('lambda_2'), config.get('lambda_3')])
    if int(summary['completed_steps']) != step:
        raise SystemExit('the run completed %r updates, expected exactly %d'
                         % (summary['completed_steps'], step))
    if int(summary['lr_horizon_steps']) != int(config['lr_horizon_steps']):
        raise SystemExit('run summary and config disagree about the LR horizon')

    checkpoint = os.path.join(run, 'trimask_%s_step%s.pt' % (ARM, tag))
    student = os.path.join(run, 'student_%s.pt' % tag)
    for path in (checkpoint, student):
        if not os.path.exists(path):
            raise SystemExit('missing %s' % path)
    student_sha = file_sha256(student)
    canonical_path = os.path.join(run, 'evaluation', '%s_step%s_canonical.json' % (ARM, tag))
    urban_path = os.path.join(run, 'evaluation', '%s_step%s_urban1k.json' % (ARM, tag))
    canonical = read(canonical_path)
    urban = read(urban_path)
    name = '%s@%d' % (ARM, step)
    if canonical['canonical'][name].get('checkpoint_sha256') != student_sha:
        raise SystemExit('the COCO evaluation did not use the exported student')
    if urban.get('checkpoint_sha256') != student_sha:
        raise SystemExit('the Urban-1k evaluation did not use the exported student')
    coco = coco_block(canonical, name)
    urban_metrics = urban_block(urban)

    s0_source = read(parsed.s0)['retrieval']['S0@500']
    s0 = {'coco': {key.replace('image2text_', 'i2t_').replace('text2image_', 't2i_'):
                   value for key, value in s0_source['coco'].items()},
          'urban': {'i2t_r1': s0_source['urban1k']['image2text']['R1'],
                    'i2t_r5': s0_source['urban1k']['image2text']['R5'],
                    'i2t_r10': s0_source['urban1k']['image2text']['R10'],
                    't2i_r1': s0_source['urban1k']['text2image']['R1'],
                    't2i_r5': s0_source['urban1k']['text2image']['R5'],
                    't2i_r10': s0_source['urban1k']['text2image']['R10']}}
    soft_results = read(parsed.soft)
    soft_retrieval = soft_results['retrieval'][
        next(key for key in soft_results['retrieval'] if key.startswith('S0_TriMask@'))]
    soft = {'coco': {key.replace('image2text_', 'i2t_').replace('text2image_', 't2i_'):
                     value for key, value in soft_retrieval['coco'].items()},
            'urban': {'i2t_r1': soft_retrieval['urban1k']['image2text']['R1'],
                      'i2t_r5': soft_retrieval['urban1k']['image2text']['R5'],
                      'i2t_r10': soft_retrieval['urban1k']['image2text']['R10'],
                      't2i_r1': soft_retrieval['urban1k']['text2image']['R1'],
                      't2i_r5': soft_retrieval['urban1k']['text2image']['R5'],
                      't2i_r10': soft_retrieval['urban1k']['text2image']['R10']}}

    records = log_records(os.path.join(run, 'salu_log.jsonl'))
    curve = []
    for index in CURVE_STEPS:
        record = records.get(index)
        if record is None:
            raise SystemExit('no log record for step %d' % index)
        curve.append({key: record.get(key) for key in ('completed_steps',) + CURVE_KEYS})

    passed = (coco['i2t_r1'] >= GATE_COCO_I2T and coco['t2i_r1'] >= GATE_COCO_T2I
              and (coco['i2t_r1'] > GATE_COCO_I2T or coco['t2i_r1'] > GATE_COCO_T2I))
    deltas = {
        'vs_s0': {key: round(coco[key] - s0['coco'][key], 6) for key in ('i2t_r1', 't2i_r1')},
        'vs_soft': {key: round(coco[key] - soft['coco'][key], 6) for key in ('i2t_r1', 't2i_r1')},
        'urban_vs_s0': {key: round(urban_metrics[key] - s0['urban'][key], 6)
                        for key in ('i2t_r1', 't2i_r1')},
        'urban_vs_soft': {key: round(urban_metrics[key] - soft['urban'][key], 6)
                          for key in ('i2t_r1', 't2i_r1')},
    }
    result = {
        'arm': ARM, 'objective': OBJECTIVE, 'text_gate_mode': GATE_MODE,
        'phase': config.get('phase'), 'run': run,
        'implementation_git_head': config.get('git_head'),
        'completed_steps': int(summary['completed_steps']),
        'lr_horizon_steps': int(summary['lr_horizon_steps']),
        'steps_per_epoch': int(summary['steps_per_epoch']),
        'wall_sec': float(summary['wall_sec']),
        'mean_sec_per_step': float(summary['mean_sec_per_step']),
        'peak_memory_gb': float(summary['peak_memory_gb']),
        'initial_state_sha256': summary['initial_state_sha256'],
        'checkpoint_sha256': file_sha256(checkpoint),
        'student_sha256': student_sha,
        'caption_stream_sha256': summary['caption_stream_sha256'],
        'sample_stream_sha256': summary['sample_stream_sha256'],
        'lambdas': {'lambda_1': config['lambda_1'], 'lambda_2': config['lambda_2'],
                    'lambda_3': config['lambda_3'],
                    'lambda_sparse_i': config['lambda_sparse_i'],
                    'lambda_sparse_t': config['lambda_sparse_t'], 'lambda_adv': 0.0},
        'retrieval': {
            name: {'coco': coco, 'urban1k': urban_metrics, 'student_sha256': student_sha,
                   'source': canonical_path},
            'S0@500': {'coco': s0['coco'], 'urban1k': s0['urban'], 'source': parsed.s0},
            'S0_TriMask@500_soft': {'coco': soft['coco'], 'urban1k': soft['urban'],
                                    'source': parsed.soft},
        },
        'deltas': deltas,
        'gate': {'rule': 'COCO I2T R@1 >= %.5f and COCO T2I R@1 >= %.5f, at least one strictly '
                         'higher (raw full precision)' % (GATE_COCO_I2T, GATE_COCO_T2I),
                 'coco_i2t_r1': coco['i2t_r1'], 'coco_t2i_r1': coco['t2i_r1'],
                 'i2t_pass': bool(coco['i2t_r1'] >= GATE_COCO_I2T),
                 't2i_pass': bool(coco['t2i_r1'] >= GATE_COCO_T2I),
                 'verdict': 'PROMISING_AT_500' if passed else 'FAIL', 'pass': bool(passed)},
        'training_curve': curve,
        'not_run': [
            '没有 hard-only 对照（本轮同时改了门型和加入了文本稀疏项）',
            '没有优势 loss（lambda_adv = 0），第三路优势只做诊断',
            '没有第四路未 mask 全局对齐，没有交叉池化、U 或重建',
            '没有固定文本门对照臂，无法把差异单独归因于硬门或稀疏项',
            '没有 250 步选型、没有超参搜索、没有延长到 3 epoch',
        ],
    }
    os.makedirs(parsed.docs, exist_ok=True)
    with open(os.path.join(parsed.docs, 'results.json'), 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    last = curve[-1]
    lines = []
    lines.append('# S0-TriMask-HS v0.2 -- 结果\n')
    lines.append('硬前向 + 直通梯度的文本门，并加入文本稀疏项（λ_sparse_T = 0.2）。'
                 '唯一正式运行恰好 %d 次 optimizer update；实现提交 `%s`。\n'
                 % (result['completed_steps'], result['implementation_git_head']))
    lines.append('## 训练\n')
    lines.append('- 恰好 %d 次更新；LR horizon %d（三 epoch 计划，未被压缩），每 epoch %d 步\n'
                 % (result['completed_steps'], result['lr_horizon_steps'],
                    result['steps_per_epoch']))
    lines.append('- 总墙钟 %.1f 秒，平均 %.4f 秒/步，峰值显存 %.3f GiB（4 卡 × 256 对，'
                 'global batch 1024，未做累积）\n'
                 % (result['wall_sec'], result['mean_sec_per_step'], result['peak_memory_gb']))
    lines.append('- 目标：10·L1 + 1·L2 + 1·L3 + 2·L_sparse_I + 0.2·L_sparse_T；'
                 '共同初始化 SHA256 `%s`\n' % result['initial_state_sha256'])
    lines.append('- checkpoint `%s`；裸学生 `%s`\n'
                 % (result['checkpoint_sha256'], result['student_sha256']))

    lines.append('\n## 原生 CLS/EOS 检索（R@1/5/10）\n')
    lines.append('| 模型 | COCO I2T | COCO T2I | Urban I2T | Urban T2I |')
    lines.append('|---|---|---|---|---|')
    for label, coco_row, urban_row in (
            ('S0@500（门槛）', s0['coco'], s0['urban']),
            ('S0_TriMask@500（soft, v0.1）', soft['coco'], soft['urban']),
            ('**%s**' % name, coco, urban_metrics)):
        lines.append('| %s | %.4f/%.4f/%.4f | %.4f/%.4f/%.4f | %.4f/%.4f/%.4f | %.4f/%.4f/%.4f |'
                     % ((label,) + tuple(coco_row[k] for k in
                                         ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5',
                                          't2i_r10'))
                        + tuple(urban_row[k] for k in
                                ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10'))))
    lines.append('\nR@1 差值（百分点）：相对 S0 COCO 图→文 %+.2f、文→图 %+.2f；'
                 '相对 soft TriMask COCO 图→文 %+.2f、文→图 %+.2f。'
                 'Urban 相对 S0 图→文 %+.2f、文→图 %+.2f。\n'
                 % (deltas['vs_s0']['i2t_r1'] * 100, deltas['vs_s0']['t2i_r1'] * 100,
                    deltas['vs_soft']['i2t_r1'] * 100, deltas['vs_soft']['t2i_r1'] * 100,
                    deltas['urban_vs_s0']['i2t_r1'] * 100, deltas['urban_vs_s0']['t2i_r1'] * 100))

    lines.append('## 晋级门\n')
    lines.append('%s\n' % result['gate']['rule'])
    lines.append('实测 COCO I2T R@1 = %.5f（%s）、COCO T2I R@1 = %.5f（%s）→ **%s**\n'
                 % (coco['i2t_r1'], '达标' if result['gate']['i2t_pass'] else '未达标',
                    coco['t2i_r1'], '达标' if result['gate']['t2i_pass'] else '未达标',
                    result['gate']['verdict']))

    lines.append('\n## 三路与稀疏项\n')
    lines.append('| step | L1 | L2 | L3 | 2·L_sparse_I | 0.2·L_sparse_T | loss_total | '
                 'adv_gap | 第三路不劣比例 |')
    lines.append('|---|---|---|---|---|---|---|---|---|')
    for row_ in curve:
        lines.append('| %d | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.3f |'
                     % (row_['completed_steps'], row_['loss_1'], row_['loss_2'], row_['loss_3'],
                        row_['weighted_loss_sparse_i'], row_['weighted_loss_sparse_t'],
                        row_['loss_total'], row_['adv_gap_positive_part'],
                        row_['third_not_worse_fraction']))

    lines.append('\n## 文本硬门与两侧 mask\n')
    lines.append('| step | mT 保留率 | mT 为 0 的比例 | mT 全关样本 | pT 均值 | pT min/max | '
                 'mT 空/全开样本 | mI 保留率 | 交集数量 | Jaccard | 交集为空比例 |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for row_ in curve:
        lines.append('| %d | %.4f | %.4f | %.4f | %.4f | %.3f/%.4f | %.4f/%.4f | %.4f | %.1f | '
                     '%s | %.4f |'
                     % (row_['completed_steps'], row_['mask_t_mean'],
                        row_['text_gate_hT_zero_fraction'], row_['mask_t_empty_fraction'],
                        row_['text_gate_pT_mean'] if row_['text_gate_pT_mean'] is not None else -1,
                        row_['text_gate_pT_min'] if row_['text_gate_pT_min'] is not None else -1,
                        row_['text_gate_pT_max'] if row_['text_gate_pT_max'] is not None else -1,
                        row_['mask_t_empty_fraction'], row_['mask_t_full_fraction'],
                        row_['mask_i_keep_ratio'],
                        row_['mask_intersection_intersection_count_mean']
                        if row_['mask_intersection_intersection_count_mean'] is not None else -1,
                        ('null' if row_['mask_intersection_jaccard_mean'] is None
                         else '%.4f' % row_['mask_intersection_jaccard_mean']),
                        row_['mask_intersection_intersection_empty_fraction']
                        if row_['mask_intersection_intersection_empty_fraction'] is not None
                        else -1))
    lines.append('\n末步：mT 保留率 %.4f、其中为 0 的比例 %.4f、全关样本比例 %.4f；'
                 'pT 均值 %.4f（min %.4f / max %.4f，距阈值 0.05 内的比例 %.4f）；'
                 'mT 与未 mask 文本的余弦均值 %.5f；文本方向改变比例 %.4f；'
                 '两侧 mask 交集均值 %.1f、并集为空比例 %.4f。\n'
                 % (last['mask_t_mean'], last['text_gate_hT_zero_fraction'],
                    last['mask_t_empty_fraction'],
                    last['text_gate_pT_mean'], last['text_gate_pT_min'], last['text_gate_pT_max'],
                    last['text_gate_pT_near_threshold_fraction'],
                    last['mask_t_cosine_with_unmasked'], last['text_direction_change_fraction'],
                    last['mask_intersection_intersection_count_mean'],
                    last['mask_intersection_union_empty_fraction']))
    lines.append('\n注意：`adv_gap` 是 relu(ℓ3 − best_single) 的**正部平均**，不是有正负抵消的'
                 '平均差；两个 margin（最强负例 vs 负例 logsumexp）含义不同，报告与日志中都分开命名。\n')

    lines.append('\n## 评估协议\n')
    lines.append('COCO canonical：`phase30a_fixed_cohort_eval.py --canonical --canonical_only '
                 '--coco`，全量 val2017 5000 图 × 25000 文本；runner 在启动前显式检查 '
                 '`COCO_DATA_ROOT` 与注释文件绝对路径。Urban-1k：'
                 '`eval_urban1k_cls.py`，1000 图 / 1000 文本，`--expect-steps 500` 已断言。'
                 '两个评估的 checkpoint_sha256 都等于导出裸学生的 SHA256。\n')
    lines.append('仅使用 normalize(encode_image(I)) 与 normalize(encode_text(C))；'
                 '没有 mask reranking、三路分数融合、教师或 decoder。\n')

    lines.append('\n## 归因边界与限制\n')
    lines.append('- 本轮**同时**把文本门从软重加权换成硬前向+直通梯度，并加入了 λ_sparse_T = 0.2，'
                 '因此结果只能先归因为这套组合，不能拆成单项因果结论。\n')
    lines.append('- 硬门产生 0 不等于删掉了正确的语义；稀疏度提高不等于检索提高。\n')
    lines.append('- 没有证明单个 512 维坐标对应任何情感或概念；没有证明两个 mask 承担了不同语义。\n')
    lines.append('- 单 seed、单配置、500 步，未做 250 步选型，也未延长到 3 epoch。\n')
    lines.append('\n## NOT RUN\n')
    for item in result['not_run']:
        lines.append('- %s' % item)

    with open(os.path.join(parsed.docs, 'report.md'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))
    print('REPORT ' + json.dumps({'gate': result['gate'], 'deltas': deltas,
                                  'timing': {'wall_sec': result['wall_sec'],
                                             'mean_sec_per_step': result['mean_sec_per_step'],
                                             'peak_memory_gb': result['peak_memory_gb']}},
                                 ensure_ascii=False, sort_keys=True))
    print('WROTE %s/results.json and report.md' % parsed.docs)


if __name__ == '__main__':
    main()
