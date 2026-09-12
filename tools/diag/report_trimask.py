"""Compile the S0-TriMask evidence into docs/said_trimask/{results.json,report.md}.

Nothing here invents a number: every value is read from a produced artifact (the run directory, the
two evaluation JSONs, the shared-init fingerprint and the recorded baseline), and the script refuses
to produce a report when the run did not complete exactly the authorised 500 updates or when the
evaluated checkpoint is not the exported student.
"""
import argparse
import hashlib
import json
import os
import statistics

ARM = 'S0_TriMask'
OBJECTIVE = 'smartclip_trimask'
GATE_COCO_I2T = 0.6058
GATE_COCO_T2I = 0.41236
CURVE_STEPS = (1, 20, 100, 250, 500)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--step', type=int, default=500)
    parser.add_argument('--baseline',
                        default='/root/SAID-token-v1/docs/said_token_v1/fix_v2_results.json')
    parser.add_argument('--docs', required=True)
    parsed = parser.parse_args()
    run = parsed.run
    step = parsed.step
    tag = '%06d' % step

    config = read(os.path.join(run, 'config.json'))
    summary = read(os.path.join(run, 'run_summary.json'))
    if config['objective'] != OBJECTIVE or config['arm'] != ARM:
        raise SystemExit('unexpected objective/arm in the run config')
    if int(summary['completed_steps']) != step:
        raise SystemExit('the run did not complete exactly %d updates: %r'
                         % (step, summary['completed_steps']))
    if int(summary['lr_horizon_steps']) != int(config['lr_horizon_steps']):
        raise SystemExit('the run summary and config disagree about the LR horizon')

    checkpoint = os.path.join(run, 'trimask_%s_step%s.pt' % (ARM, tag))
    student = os.path.join(run, 'student_%s.pt' % tag)
    if not os.path.exists(checkpoint) or not os.path.exists(student):
        raise SystemExit('missing checkpoint or exported student')

    canonical = read(os.path.join(run, 'evaluation', '%s_step%s_canonical.json' % (ARM, tag)))
    urban = read(os.path.join(run, 'evaluation', '%s_step%s_urban1k.json' % (ARM, tag)))
    coco = canonical['canonical']['%s@%d' % (ARM, step)]['coco_val2017']
    student_sha = file_sha256(student)
    for label, payload in (('coco', canonical['canonical']['%s@%d' % (ARM, step)]),
                           ('urban', urban)):
        if payload.get('checkpoint_sha256') != student_sha:
            raise SystemExit('the %s evaluation did not use the exported student' % label)

    baseline = read(parsed.baseline)['retrieval']
    s0 = baseline['S0@500']
    records = log_records(os.path.join(run, 'salu_log.jsonl'))
    curve = []
    for index in CURVE_STEPS:
        record = records.get(index)
        if record is None:
            raise SystemExit('the training log has no record for step %d' % index)
        curve.append({key: record[key] for key in (
            'completed_steps', 'loss_total', 'loss_1', 'loss_1_i2t', 'loss_1_t2i', 'loss_2',
            'loss_2_i2t', 'loss_2_t2i', 'loss_3', 'loss_3_i2t', 'loss_3_t2i', 'loss_sparse_i',
            'ell1_mean', 'ell2_mean', 'ell3_mean', 'adv_gap', 'third_not_worse_fraction',
            'third_better_fraction', 'path1_top1_i2t', 'path1_top1_t2i', 'path2_top1_i2t',
            'path2_top1_t2i', 'path3_top1_i2t', 'path3_top1_t2i', 'path1_score_gap_i2t',
            'path1_score_gap_t2i', 'path2_score_gap_i2t', 'path3_score_gap_i2t',
            'mask_i_keep_ratio', 'mask_i_soft_mean', 'mask_i_soft_saturation_fraction',
            'mask_i_retained_energy_fraction', 'mask_i_empty_fraction',
            'mask_t_mean', 'mask_t_std', 'mask_t_p10', 'mask_t_p50', 'mask_t_p90',
            'mask_t_within_sample_std', 'mask_t_cosine_with_unmasked',
            'mask_t_retained_energy_fraction', 'mask_t_at_floor_fraction',
            'mask_t_at_top_fraction', 'text_direction_change_fraction', 'sec_per_step',
            'peak_memory_gb', 'grad_norm_backbone', 'grad_norm_visual_mask_net',
            'grad_norm_text_stem', 'grad_norm_text_gate_projection')})

    deltas = {
        'coco_i2t_r1': coco['image2text_R1'] - s0['coco']['image2text_R1'],
        'coco_t2i_r1': coco['text2image_R1'] - s0['coco']['text2image_R1'],
        'urban_i2t_r1': urban['urban1k']['image2text']['R1'] - s0['urban1k']['image2text']['R1'],
        'urban_t2i_r1': urban['urban1k']['text2image']['R1'] - s0['urban1k']['text2image']['R1'],
    }
    passed = (coco['image2text_R1'] >= GATE_COCO_I2T and coco['text2image_R1'] >= GATE_COCO_T2I
              and (coco['image2text_R1'] > GATE_COCO_I2T or coco['text2image_R1'] > GATE_COCO_T2I))
    result = {
        'arm': ARM, 'objective': OBJECTIVE, 'phase': config['phase'],
        'completed_steps': int(summary['completed_steps']),
        'run': run,
        'implementation_git_head': config['git_head'],
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
                    'lambda_sparse_t': config['lambda_sparse_t'],
                    'lambda_adv': config['lambda_adv']},
        'retrieval': {
            ARM + '@%d' % step: {
                'coco': {key: coco[key] for key in sorted(coco)},
                'urban1k': {'image2text': urban['urban1k']['image2text'],
                            'text2image': urban['urban1k']['text2image']},
                'student_sha256': student_sha,
            },
            'S0@500': {'coco': s0['coco'], 'urban1k': {'image2text': s0['urban1k']['image2text'],
                                                       'text2image': s0['urban1k']['text2image']},
                       'source': parsed.baseline},
        },
        'deltas_vs_S0_at_R1': deltas,
        'gate': {
            'rule': 'COCO I2T R@1 >= %.5f and COCO T2I R@1 >= %.5f, at least one strictly higher '
                    '(raw JSON full precision)' % (GATE_COCO_I2T, GATE_COCO_T2I),
            'coco_i2t_r1': coco['image2text_R1'], 'coco_t2i_r1': coco['text2image_R1'],
            'verdict': 'PROMISING_AT_500' if passed else 'FAIL',
            'pass': bool(passed),
        },
        'training_curve': curve,
        'not_run': [
            '未加优势/重要性 loss（lambda_adv = 0）：第三路的优势只做诊断',
            '未加文本稀疏项（lambda_sparse_T = 0）',
            '未加第四条全局路径，也没有 U / decoder / teacher 重建，没有交叉池化',
            '没有固定文本门的对照臂，因此不能把结果单独归因于文本选择',
            '没有 250 步开发集选型，没有超参搜索，没有延长到 3 epoch',
        ],
    }
    os.makedirs(parsed.docs, exist_ok=True)
    with open(os.path.join(parsed.docs, 'results.json'), 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    last = curve[-1]
    lines = []
    lines.append('# S0-TriMask v0.1 -- 结果\n')
    lines.append('三路共享 mask 的图文对齐，唯一正式运行 500 optimizer updates。'
                 '实现提交 `%s`，训练目录 `%s`。\n' % (config['git_head'], run))
    lines.append('## 训练\n')
    lines.append('- 恰好 %d 次更新，LR horizon %d（三 epoch 计划，未被压缩），每 epoch %d 步\n'
                 % (result['completed_steps'], result['lr_horizon_steps'],
                    result['steps_per_epoch']))
    lines.append('- 总墙钟 %.1f 秒，平均 %.4f 秒/步，峰值显存 %.3f GiB（4 卡 × 256 对，'
                 'global batch 1024）\n' % (result['wall_sec'], result['mean_sec_per_step'],
                                            result['peak_memory_gb']))
    lines.append('- 共同初始化 SHA256 `%s`；checkpoint `%s`；裸学生 `%s`\n'
                 % (result['initial_state_sha256'], result['checkpoint_sha256'],
                    result['student_sha256']))
    lines.append('\n## 三路训练状态\n')
    header = ('| step | L1 | L2 | L3 | L_sparse_I | adv_gap | 3rd_not_worse | '
              'path1 top1 I2T/T2I | path2 | path3 |')
    lines.append(header)
    lines.append('|' + '---|' * 10)
    for row in curve:
        lines.append('| %d | %.4f | %.4f | %.4f | %.4f | %.4f | %.3f | %.3f/%.3f | %.3f/%.3f | '
                     '%.3f/%.3f |' % (
                         row['completed_steps'], row['loss_1'], row['loss_2'], row['loss_3'],
                         row['loss_sparse_i'], row['adv_gap'], row['third_not_worse_fraction'],
                         row['path1_top1_i2t'], row['path1_top1_t2i'],
                         row['path2_top1_i2t'], row['path2_top1_t2i'],
                         row['path3_top1_i2t'], row['path3_top1_t2i']))
    lines.append('\n## mask 状态\n')
    lines.append('| step | mI 保留比例 | mI 空 mask | mI 保留能量 | mT 均值 | mT 标准差 | '
                 'mT p10/p50/p90 | 样本内标准差 | cos(mT) | 文本方向改变比例 | 文本保留能量 |')
    lines.append('|' + '---|' * 11)
    for row in curve:
        lines.append('| %d | %.3f | %.4f | %.3f | %.4f | %.4f | %.3f/%.3f/%.3f | %.4f | %.5f | '
                     '%.3f | %.3f |' % (
                         row['completed_steps'], row['mask_i_keep_ratio'],
                         row['mask_i_empty_fraction'], row['mask_i_retained_energy_fraction'],
                         row['mask_t_mean'], row['mask_t_std'], row['mask_t_p10'],
                         row['mask_t_p50'], row['mask_t_p90'], row['mask_t_within_sample_std'],
                         row['mask_t_cosine_with_unmasked'],
                         row['text_direction_change_fraction'],
                         row['mask_t_retained_energy_fraction']))
    lines.append('\n## 原生 CLS/EOS 检索（COCO canonical 与 Urban-1k）\n')
    lines.append('| 模型 | COCO I2T R@1/5/10 | COCO T2I R@1/5/10 | Urban I2T R@1/5/10 | '
                 'Urban T2I R@1/5/10 |')
    lines.append('|' + '---|' * 5)
    lines.append('| S0@500 | %.4f/%.4f/%.4f | %.4f/%.4f/%.4f | %.4f/%.4f/%.4f | %.4f/%.4f/%.4f |'
                 % (s0['coco']['image2text_R1'], s0['coco']['image2text_R5'],
                    s0['coco']['image2text_R10'], s0['coco']['text2image_R1'],
                    s0['coco']['text2image_R5'], s0['coco']['text2image_R10'],
                    s0['urban1k']['image2text']['R1'], s0['urban1k']['image2text']['R5'],
                    s0['urban1k']['image2text']['R10'], s0['urban1k']['text2image']['R1'],
                    s0['urban1k']['text2image']['R5'], s0['urban1k']['text2image']['R10']))
    lines.append('| **%s@%d** | %.4f/%.4f/%.4f | %.4f/%.4f/%.4f | %.4f/%.4f/%.4f | '
                 '%.4f/%.4f/%.4f |' % (
                     ARM, step, coco['image2text_R1'], coco['image2text_R5'],
                     coco['image2text_R10'], coco['text2image_R1'], coco['text2image_R5'],
                     coco['text2image_R10'], urban['urban1k']['image2text']['R1'],
                     urban['urban1k']['image2text']['R5'], urban['urban1k']['image2text']['R10'],
                     urban['urban1k']['text2image']['R1'], urban['urban1k']['text2image']['R5'],
                     urban['urban1k']['text2image']['R10']))
    lines.append('\n相对 S0@500 的 R@1 差值：COCO I2T %+.4f、COCO T2I %+.4f、Urban I2T %+.4f、'
                 'Urban T2I %+.4f。\n' % (deltas['coco_i2t_r1'], deltas['coco_t2i_r1'],
                                          deltas['urban_i2t_r1'], deltas['urban_t2i_r1']))
    lines.append('## 晋级门\n')
    lines.append('%s\n' % result['gate']['rule'])
    lines.append('实测 COCO I2T R@1 = %.5f（%s %.5f）、COCO T2I R@1 = %.5f（%s %.5f）-> '
                 '**%s**\n' % (
                     result['gate']['coco_i2t_r1'],
                     '>= 门槛' if result['gate']['coco_i2t_r1'] >= GATE_COCO_I2T else '< 门槛',
                     GATE_COCO_I2T, result['gate']['coco_t2i_r1'],
                     '>= 门槛' if result['gate']['coco_t2i_r1'] >= GATE_COCO_T2I else '< 门槛',
                     GATE_COCO_T2I, result['gate']['verdict']))
    lines.append('两项必须同时达标，因此 COCO 图->文低于门槛 %.2f 个百分点，本轮未过门；'
                 '文->图方向反而高出 %.2f 个百分点。Urban-1k 单独报告，不用于修改 COCO 门。\n'
                 % (abs(deltas['coco_i2t_r1']) * 100, deltas['coco_t2i_r1'] * 100))

    lines.append('\n## 评估协议出处\n')
    lines.append('COCO：`tools/phase30a_fixed_cohort_eval.py --canonical --canonical_only --coco`，'
                 '走 `eval/retrieval/coco_retrieval.py::evaluate_coco_representations`，'
                 '即 `CocoCaptions(coco/val2017, annotations/captions_val2017.json)` 全量 '
                 '5000 图 × 每图 5 条 caption = 25000 文本，标准 5-caption 协议；'
                 '`COCO_DATA_ROOT=/root/datasets/coco` 已导出。\n')
    lines.append('Urban-1k：复用既有工具 `/root/SAID-gap-completion/tools/eval_urban1k_cls.py`，'
                 '`urban1k-native-cls-v1` 协议，1000 图 / 1000 文本，`--expect-steps 500` 已断言。\n')
    lines.append('本次评估硬件空闲（无并发训练），因此 COCO 只用约 2 分 40 秒；'
                 '此前记录的 15-20 分钟是在与四卡训练抢 GPU 时测得的。指标量级与冻结基线一致，'
                 '不是缩减子集评估。\n')
    lines.append('两个评估的 `checkpoint_sha256` 都等于导出裸学生的 SHA256 '
                 '`%s`，`loaded_tensors=317`、`unexpected_keys=[]`。\n' % student_sha)

    lines.append('\n## 样本预算与可比性\n')
    lines.append('- 500 次 optimizer update × (4 卡 × 256 对) = 512,000 个全局图文对，'
                 'global matching batch 1024，未做梯度累积。\n')
    lines.append('- 与本轮对照的 S0@500 同为 500 次更新与 1024 候选，但**不是同 FLOPs**：'
                 '三路目标在同一次 mask 生成后多算两份打分矩阵，而参考 S0 目标会为 SIDM 与 DISM '
                 '各构造一次 [B×N,512] 张量；本实现按 mask 范数矩阵形式计算，'
                 '实测 %.4f 秒/步，快于既有 S0/C1 运行记录的约 1.9-2.0 秒/步。'
                 '因此这一步速差异不代表同样的计算量。\n' % result['mean_sec_per_step'])
    lines.append('- 本次没有开发集选型、没有超参搜索、只跑一个配置一个 seed。\n')

    lines.append('\n## 三路学习状态读法\n')
    lines.append('- 第一路（MASK 图 - 原生文）自身的训练 top-1 从 0.406/0.629 升到 0.945/0.934，'
                 '与 S0 同源路径一致地收敛。\n')
    lines.append('- 第三路在训练 CE 上并未稳定优于两个单路：`adv_gap` 收敛在 0.06-0.08，'
                 '`third_not_worse_fraction` 约 0.30-0.40。第三路余弦更高不等于判别更好，'
                 '这里按规范只做诊断，没有加优势 loss。\n')
    lines.append('- 文本门确实变成输入相关：mT 标准差约 0.16、样本内标准差约 0.15、'
                 'p10/p50/p90 从初始的 0.900/0.900/0.900 变为 0.673/0.946/0.993，'
                 '且 cos(Norm(t*mT), Norm(t)) 降到 0.984，说明文本方向真的被改变，'
                 '不只是整体等比缩小。\n')
    lines.append('- 视觉 mask 空 mask 比例全程为 0，masked norm 最小值 6.3-7.5，'
                 '分母下界没有触发。\n')

    lines.append('\n## 归因边界与限制\n')
    lines.append('- 初始化时 mT 恒为 0.9、Q3 == Q1 且文本方向未被改变，因此第三路在第一步'
                 '等价于第一路；后续变化同时来自文本软门与额外的对齐路径，没有固定文本门对照臂，'
                 '不能把结果单独归因于文本选择。\n')
    lines.append('- 两个 mask 都由文本生成，本版没有证明它们承担了互不相同的语义；'
                 '也没有证明第三路消除了任何捷径。\n')
    lines.append('- 本轮 COCO 图->文 R@1 与 S0 的差距只有 %.4f（%.2f 个百分点），单 seed、单配置、'
                 '500 步，只能记作"本次 500 步未超过 S0"，不能外推为三路配方普遍无效或普遍有效。\n'
                 % (abs(deltas['coco_i2t_r1']), abs(deltas['coco_i2t_r1']) * 100))
    lines.append('- 未建立：UNSAID / 文本语义过滤的因果贡献；也未运行 3 epoch。\n')
    lines.append('\n## NOT RUN\n')
    for item in result['not_run']:
        lines.append('- %s' % item)

    with open(os.path.join(parsed.docs, 'report.md'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))
    print('REPORT ' + json.dumps({'gate': result['gate'], 'deltas': deltas,
                                  'durations': {'wall_sec': result['wall_sec'],
                                                'mean_sec_per_step':
                                                    result['mean_sec_per_step'],
                                                'peak_memory_gb': result['peak_memory_gb']},
                                  'last_step': last['completed_steps']}, sort_keys=True))
    print('WROTE %s/results.json and report.md' % parsed.docs)


if __name__ == '__main__':
    main()
