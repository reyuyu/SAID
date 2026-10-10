"""Restore original FP32 backend profile, retaining both incompatible-run receipts."""
import argparse
import difflib
import json
from pathlib import Path
from tools.retrieval_bounded import atomic_json, sha

ROOT = Path(__file__).resolve().parents[1]


def run(output, evidence):
    out, evidence = Path(output), Path(evidence)
    gpu = json.loads((evidence / 'gpu_native_batch64/GPU_NATIVE_BATCH64_TEST.json').read_text())
    batch_variation = json.loads((evidence / 'gpu_native_tests/GPU_INFERENCE_TEST.json').read_text())
    assert gpu['status'] == 'PASS'
    old_plan = out / 'FROZEN_PLAN_V2.json'
    plan = json.loads(old_plan.read_text())
    second = json.loads((evidence / 'evaluation_v2/RUN_STATE.json').read_text())
    assert second['status'] == 'FAILED'
    failure = json.loads((evidence / 'evaluation_v2/COCO/attempt1/RESULT.failure.json').read_text())
    proof = json.loads((evidence / 'docci_numeric_diagnosis/TF32_ONE_VARIABLE_PROOF.json').read_text())
    assert proof['numerical_effect_supported']
    before = evidence / 'source_snapshots/eval_hyfl_native_v2.py'
    after = ROOT / 'tools/eval_hyfl_native.py'
    audit = {'root_cause': 'New controller incorrectly overrode native cuDNN FP32 backend default (True) with False.',
             'production_sources_changed': False, 'weights_dtype_features_dtype': 'FP32', 'autocast': False,
             'native_profile_restored': {'cuda_matmul_allow_tf32': False, 'cudnn_allow_tf32': True},
             'no_adjustment_of_loss_model_tokenizer_similarity_data': True,
             'second_state': second, 'second_failure': failure, 'tf32_single_variable_proof': proof,
             'new_native_gpu_test': gpu, 'different_batch_native_diagnostic': batch_variation,
             'batch3_optimization_approved': False, 'formal_batch_frozen': 64,
             'before_source_sha256': sha(before), 'after_source_sha256': sha(after),
             'exact_diff': ''.join(difflib.unified_diff(before.read_text().splitlines(keepends=True),
                                                      after.read_text().splitlines(keepends=True),
                                                      fromfile='worker_v2.py', tofile='worker_native_v3.py')),
             'fresh_namespace': str(evidence / 'evaluation_native_v3'),
             'older_outputs_and_caches_preserved': True,
             'note': 'Original-native FP32 includes cuDNN permitted TF32 convolution, not pure IEEE32 contractions everywhere.'}
    atomic_json(out / 'NATIVE_PRECISION_RESTORE_AUDIT.json', audit, exclusive=True)
    plan['native_profile'] = audit['native_profile_restored']
    plan['recovery'] = {'source_plan_sha256': sha(old_plan), 'audit_sha256': sha(out / 'NATIVE_PRECISION_RESTORE_AUDIT.json'),
                         'new_runtime': str(evidence / 'evaluation_native_v3')}
    plan['frozen_prerequisites']['gpu_receipt_sha256'] = sha(evidence / 'gpu_native_batch64/GPU_NATIVE_BATCH64_TEST.json')
    atomic_json(out / 'FROZEN_NATIVE_PLAN.json', plan, exclusive=True)
    (out / 'EVAL_TESTS.json').rename(out / 'EVAL_TESTS_PRE_NATIVE.json')
    tests = json.loads((out / 'EVAL_TESTS_PRE_NATIVE.json').read_text())
    tests['gpu_strict_fp32_diagnostic'] = tests['gpu']; tests['gpu'] = gpu
    tests['native_precision_restore'] = 'NATIVE_PRECISION_RESTORE_AUDIT.json'
    atomic_json(out / 'EVAL_TESTS.json', tests, exclusive=True)
    print('Native FP32 profile restored and validated; no model/protocol-math change')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); p.add_argument('--evidence', required=True)
    a = p.parse_args(); run(a.output, a.evidence)
