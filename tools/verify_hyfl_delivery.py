"""Final CPU-only evidence validation; does not run a model or overwrite artifacts."""
import argparse
import json
from pathlib import Path
import subprocess
from tools.retrieval_bounded import atomic_json, sha


def run(output):
    out = Path(output)
    required = ['PROTOCOL_AUDIT.md', 'DATA_PROVENANCE.json', 'EVAL_SCHEDULER_REPORT.md',
                'EVAL_TESTS.json', 'RESULTS_HYFL_PROTOCOL.json', 'RESULTS_HYFL_PROTOCOL.md',
                'LEGACY_COMPARISON.md', 'RUN_STATE.json', 'FINAL_REPORT.md', 'CANONICAL_COCO_REPROOF.json',
                'NATIVE_PRECISION_RESTORE_AUDIT.json', 'DATA_TEXT_AUDIT.json']
    assert all((out / n).is_file() for n in required)
    result = json.loads((out / 'RESULTS_HYFL_PROTOCOL.json').read_text())
    state = json.loads((out / 'RUN_STATE.json').read_text())
    tests = json.loads((out / 'EVAL_TESTS.json').read_text())
    assert state['status'] == 'COMPLETED' and len(state['jobs']) == 6
    assert all(r['status'] == 'COMPLETED' and r['returncode'] == 0 for r in state['jobs'].values())
    assert tests['cpu']['pass'] and tests['cpu']['tests'] == 14 and tests['gpu']['status'] == 'PASS'
    assert tests['gpu']['state_sha256_before'] == tests['gpu']['state_sha256_after']
    assert result['datasets']['Flickr30k-Full']['status'] == 'BLOCKED_PROTOCOL_DATA'
    assert 'metrics' not in result['datasets']['Flickr30k-Full']
    metrics = 0; regressions = 0
    for name, r in {**result['datasets'], **result['legacy_only']}.items():
        if 'metrics' not in r:
            continue
        for d in ['I2T', 'T2I']:
            m = r['metrics'][d]; assert set(m['correct']) == {'1', '5', '10'}
            assert m['correct']['1'] <= m['correct']['5'] <= m['correct']['10'] <= m['query_count']
            for k, v in m['correct'].items():
                assert abs(m['recall_percent'][k] - 100*v/m['query_count']) < 1e-10
                metrics += 1
        if name != 'DCI':
            assert r['legacy_regression']['all_counts_match']
            regressions += 6
        assert r['checkpoint_sha256'] == result['checkpoint_sha256']
        assert r['identity']['encoder_sources']['tools/eval_hyfl_native.py'] == sha(out / 'WORKER_EXECUTED_NATIVE_V3.py')
    assert metrics == 36 and regressions == 30  # Five available main sets + separate Flickr1K, NOT six main sets.
    assert 'UNVERIFIED' in result['datasets']['DCI']['protocol_status']
    assert 'UNVERIFIED' in result['datasets']['Long-DCI']['protocol_status']
    resource = json.loads((out / 'FINAL_RESOURCE_CHECK.json').read_text()); assert resource['idle']
    assert not subprocess.check_output(['git', 'diff', '6dcae270f8e754323512648e1e09ed903ab698fc', '--', 'model', 'train', 'eval'], text=True)
    atomic_json(out / 'DELIVERY_VALIDATION.json', {
        'status': 'PASS_WITH_DECLARED_PROTOCOL_BLOCKS_AND_RETAINED_DIAGNOSTIC_FAILURES',
        'main_recall_values_computed': 30, 'blocked_main_recall_values': 6,
        'legacy_only_recall_values': 6, 'exact_historical_regressions': 30,
        'required_artifacts_present': required, 'fixed_model_sha256': result['checkpoint_sha256'],
        'source_snapshot_and_executed_receipts_match': True, 'production_diff': 'EMPTY',
        'unapproved_native_batch3_test': 'FAIL_PRESERVED', 'native_batch64_crossGPU': 'PASS',
        'full_flickr': 'BLOCKED_PROTOCOL_DATA', 'DCI': 'PROTOCOL_UNVERIFIED',
        'Long-DCI_official_CSV': 'UNVERIFIED', 'gpu_idle': True}, exclusive=True)
    print('Delivery validated: 30 main recall values, 6 legacy-only, 6 blocked; 30 exact regressions')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True)
    a = p.parse_args(); run(a.output)
