"""Documented evaluation-only recovery after independent DOCCI numerical diagnosis."""
import argparse
import difflib
import json
from pathlib import Path
from tools.retrieval_bounded import atomic_json, sha

ROOT = Path(__file__).resolve().parents[1]


def run(output, evidence):
    out, evidence = Path(output), Path(evidence)
    plan = json.loads((out / 'FROZEN_PLAN.json').read_text())
    diag = json.loads((evidence / 'docci_numeric_diagnosis/DOCCI_NUMERICAL_DIAGNOSIS.json').read_text())
    tf32 = json.loads((evidence / 'docci_numeric_diagnosis/TF32_ONE_VARIABLE_PROOF.json').read_text())
    failure = json.loads((evidence / 'evaluation/DOCCI/attempt1/RESULT.failure.json').read_text())
    original_state = json.loads((evidence / 'evaluation/RUN_STATE.json').read_text())
    cpu = (evidence / 'cpu_tests_final14.log').read_text()
    assert 'Ran 14 tests' in cpu and cpu.rstrip().endswith('OK')
    assert original_state['status'] == 'FAILED'
    assert diag['historical_exact_counts_match'] and tf32['numerical_effect_supported']
    assert all(diag['counts'][d]['1'] == failure['legacy_regression']['expected_counts'][d]['1'] for d in ('I2T', 'T2I'))
    assert failure['legacy_regression']['actual_counts']['I2T']['5'] - failure['legacy_regression']['expected_counts']['I2T']['5'] == 1
    assert sha(plan['checkpoint']) == failure['checkpoint_sha256'] == diag['checkpoint_sha256']
    diffs = {}
    for name in ['eval_hyfl_native', 'eval_resource_queue']:
        before = evidence / f'source_snapshots/{name}_v1.py'; after = ROOT / f'tools/{name}.py'
        diffs[name] = {'before_sha256': sha(before), 'after_sha256': sha(after),
                       'diff': ''.join(difflib.unified_diff(before.read_text().splitlines(keepends=True),
                                                         after.read_text().splitlines(keepends=True),
                                                         fromfile=f'{name}_v1.py', tofile=f'{name}.py'))}
    recovery = {'original_state': original_state, 'original_failure': failure, 'independent_diagnosis': diag,
                'tf32_one_variable_proof': tf32,
                'code_diffs': diffs, 'model_and_protocol_math_changed': False,
                'regression_contract': 'User section5.3: exact R1 correct counts; compare/disclose R5/R10',
                'first_attempt_bug': 'Controller incorrectly required all R5/R10 counts to be exact, beyond stated R1 hard gate.',
                'remaining_regression_discrepancies_not_marked_pass': True,
                'cache_handling': 'No old cache identity is bypassed. Fresh v2 namespace; v1 shards/receipts retained unchanged.',
                'FP32_settings': 'Unchanged: autocast off, TF32 off; no score rounding or protocol adjustment.'}
    atomic_json(out / 'EVALUATION_RECOVERY_AUDIT.json', recovery, exclusive=True)
    plan['recovery'] = {'source_plan_sha256': sha(out / 'FROZEN_PLAN.json'),
                         'audit_sha256': sha(out / 'EVALUATION_RECOVERY_AUDIT.json'),
                         'new_runtime': str(evidence / 'evaluation_v2'),
                         'old_runtime': str(evidence / 'evaluation')}
    plan['frozen_prerequisites']['cpu_log_sha256'] = sha(evidence / 'cpu_tests_final14.log')
    atomic_json(out / 'FROZEN_PLAN_V2.json', plan, exclusive=True)
    # Retain the preliminary test report and make the canonical report current.
    (out / 'EVAL_TESTS.json').rename(out / 'EVAL_TESTS_INITIAL.json')
    gpu = json.loads((evidence / 'gpu_tests/GPU_INFERENCE_TEST.json').read_text())
    atomic_json(out / 'EVAL_TESTS.json', {'cpu': {'pass': True, 'tests': 14, 'log': cpu}, 'gpu': gpu,
                                        'independent_legacy_diagnosis': diag,
                                        'legacy_default_precision_vs_strict_FP32': {
                                            'embedding_tolerance_5e_6_pass': False,
                                            'image_max_abs': diag['image_max_abs'],
                                            'cause_supported_by_single_variable_test': 'cuDNN TF32 default True versus formal False',
                                            'formal_flags_changed_to_match_score': False, 'proof': tf32},
                                        'long_dci_tokens': 'PASS',
                                        'note': 'R5/R10 regression deviations are separately reported; not full exact-count PASS.'}, exclusive=True)
    print('RECOVERY PREPARED; original evidence retained')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); p.add_argument('--evidence', required=True)
    a = p.parse_args(); run(a.output, a.evidence)
