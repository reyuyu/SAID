"""Add independently checked completion evidence to the existing final report."""
import datetime
import hashlib
import json
import math
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROTOCOLS = ['coco', 'urban', 'flickr_test1k', 'docci', 'dci', 'long_dci']


def sha(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


assert json.loads((ROOT / 'status.json').read_text())['phase'] == 'COMPLETE'
result = json.loads((ROOT / 'reports/results.json').read_text())
export = json.loads((ROOT / 'exports/export.json').read_text())
stream = json.loads((ROOT / 'validation/continuation_data_stream.json').read_text())
assert result['status'] == 'COMPLETE' and len(result['rows']) == 36
assert stream['status'] == 'PASS' and stream['compared_updates'] == 3151
assert export['status'] == 'PASS' and export['completed_steps'] == 3651
assert set(export['optimizer_steps']) == {'clip', 'mask', 'suffix'}
for name, count in [('clip', 301), ('mask', 14), ('suffix', 4)]:
    assert export['optimizer_steps'][name] == {'min': 3651, 'max': 3651, 'count': count}
for file_key, hash_key in [('checkpoint', 'checkpoint_sha256'), ('bare_student', 'bare_sha256')]:
    assert sha(export[file_key]) == export[hash_key]
pre = json.loads((ROOT / 'validation/preflight.json').read_text())
assert sha(pre['parent_checkpoint']) == pre['parent_sha256']
for filename, expected in json.loads((ROOT / 'validation/launch_files.json').read_text()).items():
    assert sha(ROOT / filename) == expected, filename
for filename, expected in json.loads((ROOT / 'validation/reference_files.json').read_text()).items():
    assert sha(filename) == expected, filename
repos = {}
for path, expected in [('/root/lk_projects/SAID-reproduction/full-resume', pre['fixed_training_sha']),
                       ('/root/lk_projects/SAID-publish-suffix3', '4d66d7a3768c97b41e44f3be2db31b10253422f5')]:
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=path, text=True).strip()
    assert head == expected and not dirty, (path, head, dirty)
    repos[path] = head
exit_paths = ['run/exitcode'] + ['logs/' + name + '.py.exitcode' for name in ['export_final', 'evaluate', 'compare']] + ['evaluation/' + name + '/exitcode' for name in PROTOCOLS]
codes = {path: int((ROOT / path).read_text()) for path in exit_paths}
assert all(code == 0 for code in codes.values())
for protocol in PROTOCOLS:
    raw = json.loads((ROOT / 'evaluation' / protocol / (protocol + '.json')).read_text())
    assert raw['checkpoint_sha256'] == export['bare_sha256']
    for rank in ['R@1', 'R@5', 'R@10']:
        for direction in ['I2T', 'T2I']:
            value = result['metrics'][protocol][rank][direction]
            assert math.isfinite(value) and 0 <= value <= 1
summary = {}
for name in result['comparators']:
    deltas = [row['delta_vs_' + name + '_pp'] for row in result['rows']]
    summary[name] = {
        'improved_36': sum(value > 1e-5 for value in deltas),
        'decreased_36': sum(value < -1e-5 for value in deltas),
        'tied_36': sum(abs(value) <= 1e-5 for value in deltas),
        'mean_delta_36_pp': sum(deltas) / 36,
        **result['summary'][name],
    }
cfg = export['training_config']
evidence = {
    'status': 'PASS', 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'exit_codes': codes, 'repositories': repos, 'original_launch_files_unchanged': True,
    'comparison_references_unchanged': True, 'parent_checkpoint_hash_reverified': True,
    'final_checkpoint_hash_reverified': True, 'bare_student_hash_reverified': True,
    'stream_compared_updates': 3151, 'metric_count': 36, 'summary': summary,
    'hashes': {key: export[key] for key in ['checkpoint_sha256', 'bare_sha256', 'tensor_digest']},
    'postprocess_completion': 'COMPLETE status plus successful export/evaluate/compare exit records; wrapper OS exit code not recorded by original launcher',
    'training_config': cfg,
}
(ROOT / 'validation/final_supervision.json').write_text(json.dumps(evidence, indent=2) + '\n')
lines = ['\n## 最终监督验收', '', '以下为训练与六项评测完成后的独立复核。全部36项指标见上表和 comparison.csv。', '', '### 实际训练参数', '',
         f"- 总更新3651，父checkpoint更新500，新增3151；epochs={cfg['epochs']}，每epoch {cfg['loader_batches']}批。",
         f"- 模型ViT-B/16；lambda_suffix={cfg['suffix_lambda']}，lambda_u_sparse={cfg['u_sparsity_lambda']}；4×A100 80GB，batch/rank={cfg['batch_size_per_gpu']}，总batch=1024（epoch尾批除外），accumulation={cfg['accumulation']}，seed={cfg['seed']}。",
         f"- CLIP / S0门 / U门初始LR：{cfg['arguments']['lr']} / {cfg['arguments']['mask_lr']} / {cfg['arguments']['suffix_lr']}；cosine horizon={cfg['lr_horizon_steps']}，warmup={cfg['arguments']['warmup']}，严格接续，无重新warmup。",
         f"- AdamW，CLIP weight_decay={cfg['arguments']['weight_decay']}，两门weight_decay=0；精度{cfg['precision']}；image/text chunk={cfg['image_chunk']}/{cfg['text_chunk']}；workers/rank={cfg['arguments']['num_workers']}，total_len={cfg['total_len']}。",
         '- 三组优化器最终计数均为3651（CLIP 301、S0门14、U门4个状态）；裸学生317张量严格加载且逐张量相等。',
         '- 501–3651全部3151行连续，四个rank的累计样本/文本摘要和rank0逐批摘要均匹配此前Full复现；摘要不包含像素字节，也不构成原500步的独立内容级重放验证。',
         '', '### 比较结论', '', '| 比较对象 | R@1胜/负/平（12项） | R@1平均差（百分点） | 全36项胜/负/平 |', '|---|---:|---:|---:|']
labels = {'suffix3_500': '本模型500步（训练长度比较）', 'clean3651': '本机Clean3651（等预算）', 'full3651': '本机Full3651（等预算）', 'historical_clean3651': '历史A800 Clean3651', 'historical_full3651': '历史A800 Full3651'}
for name, value in summary.items():
    lines.append(f"| {labels[name]} | {value['R1_improved']}/{value['R1_decreased']}/{value['R1_tied']} | {value['mean_R1_delta_pp']:+.4f} | {value['improved_36']}/{value['decreased_36']}/{value['tied_36']} |")
lines += ['', '同机、同训练预算的Clean/Full为主要对照。原500步仅表示延长训练的变化；历史A800另列，不能据单seed小差异宣称统计显著。', '', '### 原GitHub历史A800单列', '', 'R@1，单位%，每格图→文 / 文→图。', '', '| 协议 | 本次3/0@3651 A100 | 历史Clean3651 A800 | 历史Full3651 A800 |', '|---|---:|---:|---:|']
for protocol in PROTOCOLS:
    values = []
    for source in [result['metrics'], result['comparators']['historical_clean3651'], result['comparators']['historical_full3651']]:
        values.append(' / '.join(f"{source[protocol]['R@1'][direction] * 100:.3f}" for direction in ['I2T', 'T2I']))
    lines.append('| ' + protocol + ' | ' + ' | '.join(values) + ' |')
lines += ['', '### 退出码', '', '| 进程 | 实际记录退出码 |', '|---|---:|']
for path, code in codes.items():
    lines.append(f'| {path} | {code} |')
lines += ['', '后处理包装进程原实现未单独记录OS退出码；status.json为COMPLETE，内部三个子脚本退出码均为0。', '', '### 权重与固定代码', '',
          f"- 完整checkpoint：`{export['checkpoint']}`", f"- checkpoint SHA256：`{export['checkpoint_sha256']}`",
          f"- 裸学生：`{export['bare_student']}`", f"- 裸学生 SHA256：`{export['bare_sha256']}`",
          f"- 张量内容摘要：`{export['tensor_digest']}`", f"- 父checkpoint SHA256：`{pre['parent_sha256']}`",
          f"- 训练Git SHA：`{pre['fixed_training_sha']}`", f"- 模型代码SHA256：`{pre['objective_code_sha256']}`",
          '- 冻结评测Git SHA：`4d66d7a3768c97b41e44f3be2db31b10253422f5`。',
          '- [最终监督证据](../validation/final_supervision.json) · [全3151行数据流校验](../validation/continuation_data_stream.json) · [监督日志](../logs/supervision.jsonl)。',
          '- 原训练/导出/评测/比较脚本SHA与启动时完全一致，两个固定checkout均无修改；未推送GitHub。']
report = ROOT / 'reports/FINAL_REPORT.md'
original = report.read_text()
assert '\n## 最终监督验收' not in original, 'Refusing to duplicate final supplement'
report.write_text(original + '\n'.join(lines) + '\n')
print(json.dumps({'status': 'PASS', 'exit_codes': codes, 'summary': summary, 'hashes': evidence['hashes']}, indent=2), flush=True)
