"""Check the published 3-epoch evidence without loading weights or datasets."""
import csv
import json
import math
from pathlib import Path
from common import BUNDLE, sha


def normalize(protocol, raw):
    if protocol == 'coco':
        return {f'R@{k}': {'I2T': raw['metrics'][f'image2text_R{k}'], 'T2I': raw['metrics'][f'text2image_R{k}']} for k in [1, 5, 10]}
    if protocol == 'urban':
        return {f'R@{k}': {'I2T': raw['urban1k']['image2text'][f'R{k}'], 'T2I': raw['urban1k']['text2image'][f'R{k}']} for k in [1, 5, 10]}
    return raw['metrics']


def main():
    manifest = json.loads((BUNDLE / 'manifests/bundle_sha256.json').read_text())
    for relative, expected in {**manifest['files'], **manifest['parent_files']}.items():
        if sha(BUNDLE / relative) != expected:
            raise RuntimeError('File identity mismatch: ' + relative)
    candidate = json.loads((BUNDLE / 'results.json').read_text())
    comparison = json.loads((BUNDLE / 'comparison.json').read_text())
    export = json.loads((BUNDLE / 'evidence/export.json').read_text())
    config = json.loads((BUNDLE / 'evidence/training/config.json').read_text())
    assert candidate['completed_steps'] == export['completed_steps'] == config['formal_optimizer_updates'] == 3651
    assert config['suffix_lambda'] == 3 and config['u_sparsity_lambda'] == 0
    assert config['lr_horizon_steps'] == 3651 and config['resume_completed_steps'] == 500
    assert candidate['bare_student_sha256'] == export['bare_sha256']
    parent = BUNDLE / '../s0_dualmask_suffix3_500/evidence/training/training_steps_000001_000500.jsonl'
    old_rows = [json.loads(x) for x in parent.read_text().splitlines() if x.strip()]
    new_rows = [json.loads(x) for x in (BUNDLE / 'evidence/training/training_steps_000501_003651.jsonl').read_text().splitlines() if x.strip()]
    assert [r['completed_steps'] for r in old_rows] == list(range(1, 501))
    assert [r['completed_steps'] for r in new_rows] == list(range(501, 3652))
    assert all(math.isfinite(r['loss_total']) for r in new_rows)
    assert (new_rows[0]['epoch'], new_rows[0]['step_in_epoch']) == (0, 500)
    assert (new_rows[-1]['epoch'], new_rows[-1]['step_in_epoch']) == (2, 1216)
    assert (BUNDLE / 'evidence/training/exitcode.txt').read_text().strip() == '0'
    for state in export['optimizer_steps'].values():
        assert state['min'] == state['max'] == 3651
    for item in candidate['results']:
        protocol = item['protocol']
        raw = json.loads((BUNDLE / item['raw_result']).read_text())
        assert raw['checkpoint_sha256'] == export['bare_sha256']
        assert normalize(protocol, raw) == item['metrics'] == comparison['metrics'][protocol]
        assert (BUNDLE / 'evidence/evaluation' / protocol / 'exitcode.txt').read_text().strip() == '0'
        for baseline in ['clean3651', 'full3651']:
            reference = json.loads((BUNDLE / 'evidence/baselines' / baseline / protocol / (protocol + '.json')).read_text())
            assert normalize(protocol, reference) == comparison['comparators'][baseline][protocol]
    parent_results = json.loads((BUNDLE / '../s0_dualmask_suffix3_500/results.json').read_text())
    for item in parent_results['results']:
        assert item['metrics'] == comparison['comparators']['suffix3_500'][item['protocol']]
    registry = json.loads((BUNDLE.parent / 'results.json').read_text())
    indexed = [e for e in registry['experiments'] if e.get('source') == 's0_dualmask_suffix3_3epoch/results.json']
    assert len(indexed) == 1
    assert indexed[0]['results'] == [{k: v for k, v in item.items() if k != 'raw_result'} for item in candidate['results']]
    with (BUNDLE / 'comparison.csv').open(encoding='utf-8-sig', newline='') as stream:
        assert len(list(csv.DictReader(stream))) == 36
    stream_check = json.loads((BUNDLE / 'validation/continuation_data_stream.json').read_text())
    assert stream_check['status'] == 'PASS' and stream_check['compared_updates'] == 3151
    print(json.dumps({'status': 'PASS', 'hashed_files': len(manifest['files']), 'parent_rows': len(old_rows),
                      'continuation_rows': len(new_rows), 'protocols': 6, 'candidate_metrics': 36,
                      'same_machine_baselines': 2, 'registry_matches': True}, indent=2))


if __name__ == '__main__':
    main()
