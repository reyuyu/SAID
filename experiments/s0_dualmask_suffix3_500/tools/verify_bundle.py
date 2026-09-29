"""Validate the published small-evidence bundle without loading models or private data."""
import json
import math
from pathlib import Path

from common import BUNDLE, sha


def normalized(protocol, raw):
    if protocol == 'coco':
        return {f'R@{k}': {'I2T': raw['metrics'][f'image2text_R{k}'],
                          'T2I': raw['metrics'][f'text2image_R{k}']} for k in [1, 5, 10]}
    if protocol == 'urban':
        return {f'R@{k}': {'I2T': raw['urban1k']['image2text'][f'R{k}'],
                          'T2I': raw['urban1k']['text2image'][f'R{k}']} for k in [1, 5, 10]}
    return raw['metrics']


def main():
    manifest = json.loads((BUNDLE / 'manifests/bundle_sha256.json').read_text())
    for relative, expected in manifest['files'].items():
        if sha(BUNDLE / relative) != expected:
            raise RuntimeError('Bundle file mismatch: ' + relative)
    results = json.loads((BUNDLE / 'results.json').read_text())
    comparison = json.loads((BUNDLE / 'comparison.json').read_text())
    config = json.loads((BUNDLE / 'evidence/training/config.json').read_text())
    export = json.loads((BUNDLE / 'evidence/export.json').read_text())
    if config['formal_optimizer_updates'] != 500 or config['suffix_lambda'] != 3 or config['u_sparsity_lambda'] != 0:
        raise RuntimeError('Wrong training result')
    if config['lr_horizon_steps'] != 3651 or config['world_size'] != 4 or config['batch_size_per_gpu'] != 256:
        raise RuntimeError('Wrong training protocol')
    rows = [json.loads(line) for line in (BUNDLE / 'evidence/training/training_steps_000001_000500.jsonl').read_text().splitlines() if line.strip()]
    if [row['completed_steps'] for row in rows] != list(range(1, 501)):
        raise RuntimeError('Incomplete training history')
    if not all(math.isfinite(row['loss_total']) for row in rows):
        raise RuntimeError('Non-finite training loss')
    if export['bare_sha256'] != results['bare_student_sha256'] or export['checkpoint_sha256'] != results['checkpoint_sha256']:
        raise RuntimeError('Export identity mismatch')
    metrics_checked = 0
    for item in results['results']:
        protocol = item['protocol']
        raw = json.loads((BUNDLE / item['raw_result']).read_text())
        if raw['checkpoint_sha256'] != export['bare_sha256'] or normalized(protocol, raw) != item['metrics']:
            raise RuntimeError('Result identity/value mismatch: ' + protocol)
        if item['metrics'] != comparison['metrics'][protocol]:
            raise RuntimeError('Comparison uses different scores: ' + protocol)
        if (BUNDLE / 'evidence/evaluation' / protocol / 'exitcode.txt').read_text().strip() != '0':
            raise RuntimeError('Evaluation failed: ' + protocol)
        for baseline, metrics in comparison['baselines'].items():
            original = json.loads((BUNDLE / 'evidence/baselines' / baseline / protocol / (protocol + '.json')).read_text())
            if normalized(protocol, original) != metrics[protocol]:
                raise RuntimeError('Baseline score mismatch: ' + baseline + '/' + protocol)
        metrics_checked += 6
    if len(results['results']) != 6 or metrics_checked != 36:
        raise RuntimeError('Incomplete six-protocol score table')
    if (BUNDLE / 'evidence/training/exitcode.txt').read_text().strip() != '0':
        raise RuntimeError('Training did not exit successfully')
    print(json.dumps({'status': 'PASS', 'hashed_files': len(manifest['files']),
                      'training_rows': len(rows), 'protocols': 6, 'candidate_metrics': 36,
                      'baseline_models_checked': len(comparison['baselines'])}, indent=2))


if __name__ == '__main__':
    main()
