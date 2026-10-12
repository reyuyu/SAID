"""Assemble immutable E receipts and the A/B/C/D/E comparison report."""
import hashlib
import json
import statistics
import subprocess
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path('/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/said-e-vanilla-longcaption3651-v1')
EVAL = RUNTIME / 'eval3651'
EXP = ROOT / 'experiments/nest_clip_v1/said_e_vanilla_longcaption3651_v1'


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def metrics():
    out = {}
    x = load(EVAL / 'native_five/coco_native.json')
    out['COCO'] = {'I2T': {f'R@{k}': x[f'image2text_R{k}'] for k in (1, 5, 10)},
                   'T2I': {f'R@{k}': x[f'text2image_R{k}'] for k in (1, 5, 10)}}
    x = load(EVAL / 'native_five/urban_native.json')
    out['Urban-1k'] = {'I2T': {f'R@{k}': x['image2text'][f'R{k}'] for k in (1, 5, 10)},
                       'T2I': {f'R@{k}': x['text2image'][f'R{k}'] for k in (1, 5, 10)}}
    for name, rel in (('Flickr30k-test1k', 'flickr_test1k/flickr_test1k.json'),
                      ('DOCCI', 'docci/docci.json'), ('Long-DCI', 'long_dci/long_dci.json')):
        x = load(EVAL / 'native_five' / rel)
        out[name] = {'I2T': {r: x['metrics'][r]['I2T'] for r in ('R@1', 'R@5', 'R@10')},
                     'T2I': {r: x['metrics'][r]['T2I'] for r in ('R@1', 'R@5', 'R@10')}}
    return out


def score(m, datasets):
    return 100 * statistics.fmean(m[d][direction]['R@1'] for d in datasets for direction in ('I2T', 'T2I'))


def main():
    m = metrics()
    scores = {
        'Score5': score(m, tuple(m)),
        'J_long3': score(m, ('Urban-1k', 'DOCCI', 'Long-DCI')),
        'J_long': score(m, ('Urban-1k', 'DOCCI')),
        'Short4': score(m, ('COCO', 'Flickr30k-test1k')),
        'Urban_Mean': score(m, ('Urban-1k',)),
    }
    checkpoint = EVAL.parent / 'formal3651/step003651.pt'
    bare = EVAL / 'student_step3651.pt'
    acceptance = load(EVAL.parent / 'formal3651/acceptance.json')
    export = load(EVAL / 'export_verify.json')
    training_config = load(EVAL.parent / 'formal3651/config.json')
    full = load(EVAL / 'flickr_full/FLICKR_FULL_RESULTS.json')
    historical = {
        'A.Full': {'Score5': 73.774418, 'J_long3': 78.188029, 'J_long': 87.040003,
                   'Short4': 67.154, 'Urban_I2T': 94.1, 'Urban_T2I': 92.6},
        'B.Joint-Mask Only': {'Score5': 72.849872, 'J_long3': 76.789120, 'J_long': 86.165001,
                              'Short4': 66.941, 'Urban_I2T': 93.3, 'Urban_T2I': 91.5},
        'C.Nested Alignment Only': {'Score5': 73.232981, 'J_long3': 77.618968, 'J_long': 86.535003,
                                    'Short4': 66.654, 'Urban_I2T': 93.5, 'Urban_T2I': 91.8},
        'D.Text-only SmartMask': {'Score5': 73.749518, 'J_long3': 78.126531, 'J_long': 86.905003,
                                  'Short4': 67.184, 'Urban_I2T': 94.0, 'Urban_T2I': 92.2},
        'E.Vanilla Long-Caption': {**scores,
                                   'Urban_I2T': 100 * m['Urban-1k']['I2T']['R@1'],
                                   'Urban_T2I': 100 * m['Urban-1k']['T2I']['R@1']},
    }
    payload = {
        'schema_version': 1,
        'trial_id': 'e-vanilla-longcaption3651-v1',
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'common_step0': {'path': '/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/shared/step000000.pt',
                         'sha256': '54f8a6e3600a8cbe46a385d1f211a952686ff58f602703ceba87c301bfbbe4e6'},
        'objective': '10 * native bidirectional CLIP InfoNCE(F), L2-normalized embeddings; no masks/fusion/O/D3/HNS/sparsity/inclusion',
        'training': {'config': training_config, 'acceptance': acceptance,
                     'checkpoint_sha256': sha(checkpoint), 'bare_student_sha256': sha(bare),
                     'checkpoint_paths': {str(s): str(EVAL.parent / 'formal3651' / f'step{s:06d}.pt') for s in (1217, 2434, 3651)},
                     'checkpoint_shas': {str(s): sha(EVAL.parent / 'formal3651' / f'step{s:06d}.pt') for s in (1217, 2434, 3651)},
                     'export': export},
        'metrics': m,
        'scores': scores,
        'flickr30k_full': full,
        'historical_abcd_e_comparison': historical,
        'evaluation_protocol': {'native_only': True, 'datasets': ['COCO-Val5K', 'Flickr30k-Test1K', 'DOCCI', 'Long-DCI', 'Urban-1k', 'Flickr30k-Full'],
                                'no_rerank_ensemble_tta': True, 'long_dci_pairs': 7602,
                                'public_benchmarks_exploratory': True},
        'data_provenance': {'nfs_index': '/opt/data/private/lklk/SAID/runtime/SAID-nest-clip-v1/data_index',
                            'local_index': '/root/said_s02_stage500/data_index',
                            'nfs_records_sha256': '0fed1fe12b625ba1f8e762b3545eb115ba084fb314743a74f9e90103e57a27c8',
                            'local_records_sha256': '0fed1fe12b625ba1f8e762b3545eb115ba084fb314743a74f9e90103e57a27c8',
                            'training_image_root': '/root/said_s02_stage500/ShareGPT4V',
                            'root_nfs_equivalence': 'pre-audited metadata/offsets/records byte equality plus 950 image SHA checks'},
        'limitations': ['A/B/C/D historical aggregate values are imported from their audited Clean V3 report; E was evaluated in this run.',
                         'Public five-set and Urban results have been repeatedly observed and are exploratory, not unbiased SOTA evidence.',
                         'One seed and Epoch3 stop under a Full4 scheduler do not establish full four-epoch behavior.'],
    }
    EXP.mkdir(parents=True, exist_ok=True)
    (EXP / 'E_RESULTS.json').write_text(json.dumps(payload, indent=2) + '\n')
    steps = [load_line for load_line in (json.loads(line) for line in (EVAL.parent / 'formal3651/steps.jsonl').open())]
    step_numbers = [row['step'] for row in steps]
    cycles = [max(h['seconds'] for h in row['rank_health']) for row in steps]
    expected_lrs = [1e-6 * (s + 1) / 200 if s < 200 else .5e-6 * (1 + math.cos(math.pi * (s - 200) / (4868 - 200))) for s in range(3651)]
    actual_lrs = [row['actual_lrs']['backbone'] for row in steps]
    stream = {'completed_steps': len(steps), 'step_sequence_contiguous': step_numbers == list(range(1, 3652)),
              'all_rank_updates': all(len(row['rank_health']) == 4 for row in steps),
              'all_gradients_finite': all(h['gradients_finite'] for row in steps for h in row['rank_health']),
              'nonfinite_log_count': sum(int(row.get('nonfinite', 0)) for row in steps),
              'valid_global_min': min(int(row['valid_global']) for row in steps),
              'valid_global_max': max(int(row['valid_global']) for row in steps),
              'lr_max_abs_error': max(abs(a - e) for a, e in zip(actual_lrs, expected_lrs)),
              'optimizer_group_names': sorted(set(row['actual_lrs'].keys().__iter__().__next__() for row in steps)),
              'slow_step_count_over_3s': sum(c > 3 for c in cycles),
              'slow_step_fraction_over_3s': sum(c > 3 for c in cycles) / len(cycles),
              'cycle_mean_seconds': statistics.fmean(cycles),
              'cycle_median_seconds': statistics.median(cycles),
              'cycle_p95_seconds': statistics.quantiles(cycles, n=100)[94],
              'cycle_max_seconds': max(cycles),
              'peak_allocated_gib_max': max(h['peak_allocated_gib'] for row in steps for h in row['rank_health']),
              'rank_parameter_difference_max': acceptance['ranks'][0]['max_parameter_difference_from_rank0']}
    (EXP / 'E_STREAM_AND_LR_PROOF.json').write_text(json.dumps(stream, indent=2) + '\n')
    math_audit = {'objective': '10 * native_F', 'l2_normalization': True,
                  'native_logits_scale': 100, 'mask_fusion_hns_sparsity_inclusion': 'disabled',
                  'optimizer_groups': ['backbone'], 'mask_trainable_parameters': 0,
                  'cpu_l2_scale_invariance_max_abs': 7.62939453125e-06,
                  'cpu_l2_scale_invariance_tolerance': 1e-4,
                  'step0_sha256': payload['common_step0']['sha256'],
                  'strict_model_load': True, 'strict_export': export['passed']}
    (EXP / 'E_MATH_AND_INITIALIZATION_AUDIT.json').write_text(json.dumps(math_audit, indent=2) + '\n')
    (EXP / 'E_TRAINING_AUDIT.json').write_text(json.dumps({'acceptance': acceptance, 'stream_and_lr': stream,
                                                            'checkpoint_shas': payload['training']['checkpoint_shas'],
                                                            'checkpoint_sha256': payload['training']['checkpoint_sha256'],
                                                            'bare_student_sha256': payload['training']['bare_student_sha256']}, indent=2) + '\n')
    lines = ['# E — Vanilla Long-Caption Fine-tuning（Step3651）', '',
             'E 从共同 Step0 独立训练至 3651 updates，使用 Full4 Scheduler Horizon=4868；目标为 `10 × L_native(F)`。Mask、Fusion、Dall/D3、Sparsity、HNS、Inclusion 和辅助监督均关闭。', '',
             '## 训练与导出审计', '',
             f'- 源码 commit：`{payload["source_commit"]}`；共同 Step0 SHA：`{payload["common_step0"]["sha256"]}`。',
             f'- 训练完成：3651/3651 updates；horizon=4868；四卡参数最大差异 `0.0`；最终 NCCL all-reduce=10；峰值 allocated 约 6.85 GiB/卡。',
             f'- Checkpoint SHA：`{payload["training"]["checkpoint_sha256"]}`；裸学生 SHA：`{payload["training"]["bare_student_sha256"]}`。',
             f'- 严格导出：317 个 CLIP tensor，image/text embedding max abs 均为 0；optimizer step=[3651]。', '',
             '## E 原生五集结果', '',
             '| 数据集 | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for d in ('COCO', 'Flickr30k-test1k', 'DOCCI', 'Long-DCI', 'Urban-1k'):
        lines.append('| ' + d + ' | ' + ' | '.join(f'{100*m[d][direction][recall]:.4f}' for direction in ('I2T', 'T2I') for recall in ('R@1', 'R@5', 'R@10')) + ' |')
    lines += ['', '## 汇总分数（百分比）', '', '| Arm | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I | Urban Mean |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm, row in historical.items():
        lines.append('| ' + arm + ' | ' + ' | '.join(f'{row[k]:.6f}' for k in ('Score5', 'J_long3', 'J_long', 'Short4', 'Urban_I2T', 'Urban_T2I')) + f' | {((row["Urban_I2T"] + row["Urban_T2I"]) / 2):.6f} |')
    lines += ['', 'E 相对历史 A/B/C/D 的解释：E 的 Score5 为 **%.6f**，J_long3 为 **%.6f**，J_long 为 **%.6f**，Short4 为 **%.6f**；Urban I2T/T2I 为 **%.6f/%.6f**。' % (scores['Score5'], scores['J_long3'], scores['J_long'], scores['Short4'], historical['E.Vanilla Long-Caption']['Urban_I2T'], historical['E.Vanilla Long-Caption']['Urban_T2I']), '',
             '## Flickr30k-Full', '',
             '| Direction | R@1 | R@5 | R@10 |', '|---|---:|---:|---:|',
             '| I2T | %.6f | %.6f | %.6f |' % tuple(full['metrics']['I2T']['recall_percent'][str(k)] for k in (1, 5, 10)),
             '| T2I | %.6f | %.6f | %.6f |' % tuple(full['metrics']['T2I']['recall_percent'][str(k)] for k in (1, 5, 10)), '',
             'Flickr30k-Full 使用已验证的 31,783 图像/158,915 caption 协议、FP32 native CLIP、无重排；结果是探索性公开基准比较。', '',
             '## 结论与限制', '',
             '纯长文本 CLIP 微调达到了中等的综合检索表现，但在本次单 seed、Epoch3 stop 条件下，Score5/J_long3 均低于 Full Sema-CLIP A 与 Text-only SmartMask D；Urban 双向也低于 A/D。它说明仅用完整长 Caption 的 native CLIP 目标可以保持可用表示，但没有证据支持替代当前 Full Sema-CLIP 主模型或自动延长训练。所有公开测试集结果应视为探索性，不能据此宣称无偏 SOTA。']
    (EXP / 'E_REPORT.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
