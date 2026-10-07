"""Fail closed on oversized, forbidden or credential-bearing staged Git files."""
import json
import re
import subprocess

ALLOWED = {
    'recovery/nested_d3_local_search.py',
    'recovery/nested_d3_local_search_evidence.py',
    'tests/test_nested_d3_local_search.py',
    'recovery/nested_detail_d3_balanced_full.py',
    'recovery/nested_detail_d3_balanced_full_evidence.py',
    'tests/test_nested_detail_d3_balanced_full.py',
    'experiments/nest_clip_v1/nested_detail_d3_balanced_full_v1/config.json',
    'experiments/nest_clip_v1/nested_detail_d3_balanced_full_v1/CPU_TESTS.json',
    'experiments/nest_clip_v1/nested_detail_d3_balanced_full_v1/DEBIAS_REFERENCE.json',
    'experiments/nest_clip_v1/nested_detail_d3_balanced_full_v1/FULL_RESULTS.md',
    'experiments/nest_clip_v1/nested_detail_d3_balanced_full_v1/RESULTS.json',
    'experiments/nest_clip_v1/nested_detail_d3_balanced_full_v1/TRAINING_DIAGNOSTICS.json',
    'experiments/nest_clip_v1/nested_detail_d3_balanced_full_v1/SAMPLING_AUDIT.json',
    'experiments/nest_clip_v1/nested_detail_d3_balanced_full_v1/MASK_HIERARCHY_AUDIT.json',
    'experiments/nest_clip_v1/nested_detail_d3_balanced_full_v1/RUNTIME_STATS.json',
    'experiments/nest_clip_v1/nested_detail_d3_balanced_full_v1/EXPORT_AUDIT.json',
    'experiments/nest_clip_v1/nested_detail_d3_balanced_full_v1/VALIDATION.json',
    'recovery/nested_detail_d3_balanced500.py',
    'recovery/nested_detail_d3_balanced_gradients.py',
    'recovery/review_nested_detail_d3_balanced500.py',
    'tests/test_nested_detail_d3_balanced.py',
    'experiments/nest_clip_v1/nested_detail_d3_balanced_500_v1/config.json',
    'experiments/nest_clip_v1/nested_detail_d3_balanced_500_v1/REPORT.md',
    'experiments/nest_clip_v1/nested_detail_d3_balanced_500_v1/RESULTS.json',
    'experiments/nest_clip_v1/nested_detail_d3_balanced_500_v1/TRAINING_DIAGNOSTICS.json',
    'experiments/nest_clip_v1/nested_detail_d3_balanced_500_v1/SAMPLING_AUDIT.json',
    'experiments/nest_clip_v1/nested_detail_d3_balanced_500_v1/MASK_HIERARCHY_AUDIT.json',
    'experiments/nest_clip_v1/nested_detail_d3_balanced_500_v1/GRADIENT_SPOTCHECK.json',
    'experiments/nest_clip_v1/nested_detail_d3_balanced_500_v1/RUNTIME_STATS.json',
    'experiments/nest_clip_v1/nested_detail_d3_balanced_500_v1/VALIDATION.json',
    'recovery/nested_detail_d3_equal500.py',
    'recovery/nested_detail_d3_gradients.py',
    'recovery/review_nested_detail_d3_equal500.py',
    'tests/test_nested_detail_d3.py',
    'experiments/nest_clip_v1/nested_detail_d3_equal_500_v1/config.json',
    'experiments/nest_clip_v1/nested_detail_d3_equal_500_v1/REPORT.md',
    'experiments/nest_clip_v1/nested_detail_d3_equal_500_v1/RESULTS.json',
    'experiments/nest_clip_v1/nested_detail_d3_equal_500_v1/TRAINING_DIAGNOSTICS.json',
    'experiments/nest_clip_v1/nested_detail_d3_equal_500_v1/SAMPLING_AUDIT.json',
    'experiments/nest_clip_v1/nested_detail_d3_equal_500_v1/MASK_HIERARCHY_AUDIT.json',
    'experiments/nest_clip_v1/nested_detail_d3_equal_500_v1/GRADIENT_SPOTCHECK.json',
    'experiments/nest_clip_v1/nested_detail_d3_equal_500_v1/RUNTIME_STATS.json',
    'experiments/nest_clip_v1/nested_detail_d3_equal_500_v1/VALIDATION.json',
    'recovery/nested_detail_equal_weight500.py',
    'recovery/nested_detail_equal_weight_gradients.py',
    'recovery/review_nested_detail_equal_weight500.py',
    'tests/test_nested_detail_equal_weight.py',
    'experiments/nest_clip_v1/nested_detail_equal_weight_500_v1/config.json',
    'experiments/nest_clip_v1/nested_detail_equal_weight_500_v1/REPORT.md',
    'experiments/nest_clip_v1/nested_detail_equal_weight_500_v1/RESULTS.json',
    'experiments/nest_clip_v1/nested_detail_equal_weight_500_v1/TRAINING_DIAGNOSTICS.json',
    'experiments/nest_clip_v1/nested_detail_equal_weight_500_v1/SAMPLING_AUDIT.json',
    'experiments/nest_clip_v1/nested_detail_equal_weight_500_v1/MASK_HIERARCHY_AUDIT.json',
    'experiments/nest_clip_v1/nested_detail_equal_weight_500_v1/GRADIENT_SPOTCHECK.json',
    'experiments/nest_clip_v1/nested_detail_equal_weight_500_v1/RUNTIME_STATS.json',
    'experiments/nest_clip_v1/nested_detail_equal_weight_500_v1/VALIDATION.json',
    'model/balanced_hparam_search.py','tools/nest_clip.py','tests/test_nested_detail.py',
    'recovery/nested_detail500.py','recovery/nested_detail_gradients.py','recovery/test_nested_detail500.py',
    'recovery/review_nested_detail500.py',
    'experiments/nest_clip_v1/nested_detail_500_v1/config.json',
    'experiments/nest_clip_v1/nested_detail_500_v1/REPORT.md',
    'experiments/nest_clip_v1/nested_detail_500_v1/RESULTS.json',
    'experiments/nest_clip_v1/nested_detail_500_v1/TRAINING_DIAGNOSTICS.json',
    'experiments/nest_clip_v1/nested_detail_500_v1/SAMPLING_AUDIT.json',
    'experiments/nest_clip_v1/nested_detail_500_v1/MASK_HIERARCHY_AUDIT.json',
    'experiments/nest_clip_v1/nested_detail_500_v1/GRADIENT_SPOTCHECK.json',
    'experiments/nest_clip_v1/nested_detail_500_v1/RUNTIME_STATS.json',
    'experiments/nest_clip_v1/nested_detail_500_v1/VALIDATION.json',
    'recovery/review_all_detail500.py',
    'train/nested_semantic_data.py', 'tests/test_summary_all_detail.py',
    'recovery/s02_all_detail500.py', 'recovery/test_all_detail500.py',
    'recovery/configs/summary02_all_detail500.json',
    'recovery/ALL_DETAIL500_RESULTS.md', 'recovery/ALL_DETAIL500_RESULTS.json',
    'recovery/ALL_DETAIL500_DIAGNOSTICS.json', 'recovery/ALL_DETAIL500_RUNTIME_STATS.json',
    'recovery/evaluate_epoch3.py', 'recovery/EPOCH3_RESULTS.md', 'recovery/EPOCH3_RESULTS.json',
    'recovery/s02_local_full.py', 'recovery/test_s02_local_full.py',
    'recovery/full_evidence.py', 'recovery/FULL_RESULTS.md', 'recovery/FULL_RESULTS.json',
    'recovery/full_monitor.py',
    'recovery/FULL_RUNTIME_STATS.json', 'recovery/RESUME_PROVENANCE.json',
    'recovery/FULL_DIAGNOSTICS.json', 'recovery/FULL_GITHUB_RECEIPT.json',
    'recovery/s02_local500.py', 'recovery/local500_policy.py', 'recovery/test_local500_policy.py',
    'recovery/local500_evidence.py',
    'recovery/configs/summary02_local500.json', 'recovery/STEP500_LOCAL_REPRODUCTION.md',
    'recovery/LOCAL_RUNTIME_STATS.json', 'recovery/LOCAL_PER_STEP_SUMMARY.json',
    'recovery/s02_full_stage_report.py',
    'recovery/s02_full_stage.py', 'recovery/s02_full_local_data.py',
    'recovery/test_s02_full_stage.py', 'recovery/configs/s02_local_full_paths.json',
    'recovery/S02_LOCAL_FULL_DATA.md', 'recovery/S02_LOCAL_FULL_DATA.json',
    'recovery/S02_LOCAL_FULL_RESOURCE_SUMMARY.json',
    'recovery/nfs500_evidence.py',
    'recovery/nfs500_policy.py', 'recovery/s02_nfs500.py', 'recovery/test_nfs500_policy.py',
    'recovery/configs/summary02_nfs500.json', 'train/train_nested_semantic_mask.py',
    'experiments/nest_clip_v1/armb_summary02_4epoch_v1/reproduction_train_gate.py',
    'recovery/STEP500_NFS_REPRODUCTION.md', 'recovery/STEP500_RESULTS.json',
    'recovery/NFS_RUNTIME_STATS.json', 'recovery/NFS_PER_STEP_SUMMARY.json', 'recovery/CONTINUATION_GATE.json',
    '.gitignore', 'recovery/s02_stage500.py', 'recovery/s02_train500.py',
    'recovery/s02_stage500_report.py', 'recovery/check_stage500_publish.py',
    'recovery/test_s02_stage500.py', 'recovery/S02_LOCAL_STAGE500.md',
    'recovery/S02_LOCAL_STAGE500.json', 'recovery/S02_LOCAL_STAGE500_RESOURCE_SUMMARY.json',
    'recovery/SERVER_REBOOT_FORENSICS.md',
}
SEARCH_ROOT='experiments/nest_clip_v1/nested_d3_local_search500_v1/'
ALLOWED.update(SEARCH_ROOT+name for name in (
    'CPU_TESTS.json','SEARCH_PLAN.json','SEARCH_STATE.json','SEARCH_SUMMARY.md','REPORT.md','RESULTS.json','GITHUB_RECEIPT.json'))
ALLOWED.update(SEARCH_ROOT+arm+'/'+name for arm in ('W20','W25','W35','W40','S25','S30','KR234') for name in (
    'config.json','REPORT.md','RESULTS.json','TRAINING_DIAGNOSTICS.json','GRADIENT_SPOTCHECK.json',
    'MASK_HIERARCHY_AUDIT.json','SAMPLING_AUDIT.json','RUNTIME_STATS.json','EXPORT_AUDIT.json','VALIDATION.json'))
ALLOWED.update(('recovery/nested_d3_followup500.py','tests/test_nested_d3_followup500.py'))
FOLLOWUP_ROOT='experiments/nest_clip_v1/nested_d3_followup500_v1/'
ALLOWED.update(FOLLOWUP_ROOT+name for name in (
    'CPU_TESTS.json','SEARCH_PLAN.json','REPORT.md','SEARCH_SUMMARY.md','RESULTS.json'))
ALLOWED.update('experiments/nest_clip_v1/'+arm+'/'+name
    for arm in ('kr2m1_500_v1','weaksparse_051015_500_v1') for name in (
    'config.json','REPORT.md','RESULTS.json','TRAINING_DIAGNOSTICS.json','GRADIENT_SPOTCHECK.json',
    'MASK_HIERARCHY_AUDIT.json','SAMPLING_AUDIT.json','RUNTIME_STATS.json','EXPORT_AUDIT.json','VALIDATION.json'))
ALLOWED.update(('recovery/kr234_sparsity123_500.py','tests/test_kr234_sparsity123_500.py'))
ALLOWED.update('experiments/nest_clip_v1/kr234_sparsity123_500_v1/'+name for name in (
    'config.json','CPU_TESTS.json','SEARCH_PLAN.json','REPORT.md','SEARCH_SUMMARY.md','RESULTS.json',
    'TRAINING_DIAGNOSTICS.json','GRADIENT_SPOTCHECK.json','MASK_HIERARCHY_AUDIT.json',
    'SAMPLING_AUDIT.json','RUNTIME_STATS.json','EXPORT_AUDIT.json','VALIDATION.json'))
ALLOWED.update(('recovery/nested_d3_inc0_500.py','tests/test_nested_d3_inc0_500.py'))
ALLOWED.update('experiments/nest_clip_v1/nested_d3_inc0_500_v1/'+name for name in (
    'config.json','CPU_TESTS.json','SEARCH_PLAN.json','REPORT.md','SEARCH_SUMMARY.md','RESULTS.json',
    'TRAINING_DIAGNOSTICS.json','GRADIENT_SPOTCHECK.json','MASK_HIERARCHY_AUDIT.json',
    'SAMPLING_AUDIT.json','RUNTIME_STATS.json','EXPORT_AUDIT.json','VALIDATION.json'))
ALLOWED.update(('recovery/nested_d3_inc05_500.py','tests/test_nested_d3_inc05_500.py'))
ALLOWED.update('experiments/nest_clip_v1/nested_d3_inc05_500_v1/'+name for name in (
    'config.json','CPU_TESTS.json','SEARCH_PLAN.json','REPORT.md','SEARCH_SUMMARY.md','RESULTS.json',
    'TRAINING_DIAGNOSTICS.json','GRADIENT_SPOTCHECK.json','MASK_HIERARCHY_AUDIT.json',
    'SAMPLING_AUDIT.json','RUNTIME_STATS.json','EXPORT_AUDIT.json','VALIDATION.json'))
ALLOWED.update(('recovery/nested_d3_coupled_reg500.py','tests/test_nested_d3_coupled_reg500.py'))
ALLOWED.update('experiments/nest_clip_v1/nested_d3_coupled_reg500_v1/'+name for name in (
    'config.json','CPU_TESTS.json','SEARCH_PLAN.json','REPORT.md','SEARCH_SUMMARY.md','RESULTS.json',
    'TRAINING_DIAGNOSTICS.json','GRADIENT_SPOTCHECK.json','MASK_HIERARCHY_AUDIT.json',
    'SAMPLING_AUDIT.json','RUNTIME_STATS.json','EXPORT_AUDIT.json','VALIDATION.json','REGULARIZER_AUDIT.json'))
ALLOWED.update(('model/nested_support_band.py','recovery/nested_d3_bbns500.py',
    'recovery/nested_d3_support_audit.py','tests/test_nested_d3_bbns500.py','configs/nested_d3_bbns500.json'))
ALLOWED.update('experiments/nest_clip_v1/nested_d3_bbns500_v1/'+name for name in (
    'config.json','CPU_TESTS.json','SEARCH_PLAN.json','REPORT.md','SEARCH_SUMMARY.md','RESULTS.json',
    'TRAINING_DIAGNOSTICS.json','GRADIENT_SPOTCHECK.json','MASK_HIERARCHY_AUDIT.json',
    'SAMPLING_AUDIT.json','RUNTIME_STATS.json','EXPORT_AUDIT.json','VALIDATION.json',
    'ANCHOR_SUPPORT_AUDIT.json','ANCHOR_SUPPORT_AUDIT.md','ANCHOR_SUPPORT_COHORT.json',
    'SUPPORT_BAND_AUDIT.json','GRADIENT_ROUTING_AUDIT.json'))
PATTERNS = [
    re.compile(rb'-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----'),
    re.compile(rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b'),
    re.compile(rb'\bhf_[A-Za-z0-9]{20,}\b'),
    re.compile(rb'(?i)(?:https?://[^\s"<>]+[?&](?:x-amz-signature|signature|sig|token|access_token)=)'),
    re.compile(rb'(?i)(?:authorization\s*[:=]\s*["\x27]?(?:bearer|basic)\s+[a-z0-9+/=._-]{12,})'),
    re.compile(rb'(?i)(?:secret_access_key|aws_secret_access_key|access_token|api_key|password|cookie)\s*["\x27]?\s*[:=]\s*["\x27][^"\x27\s]{12,}["\x27]'),
]


def inspect():
    raw = subprocess.check_output(['git', 'diff', '--cached', '--name-only', '-z'])
    paths = [p.decode() for p in raw.split(b'\0') if p]
    if not paths:
        raise RuntimeError('No staged files to review')
    stats, failures = [], []
    for path in paths:
        if path not in ALLOWED:
            failures.append(dict(path=path, reason='Outside explicit small-artifact allowlist'))
            continue
        payload = subprocess.check_output(['git', 'show', ':' + path])
        stats.append(dict(path=path, bytes=len(payload)))
        if len(payload) > 1024 * 1024:
            failures.append(dict(path=path, reason='File exceeds 1MiB publication cap'))
        if any(pattern.search(payload) for pattern in PATTERNS):
            failures.append(dict(path=path, reason='Credential/signed-URL pattern; contents suppressed'))
        if b'\0' in payload:
            failures.append(dict(path=path, reason='Unexpected binary file'))
    result = dict(passed=not failures, files=stats, total_bytes=sum(s['bytes'] for s in stats), failures=failures,
                  review='staged blobs only, allowlist, 1MiB cap, private keys/AK/SK/token/cookie/signed URL sanity patterns')
    print(json.dumps(result, indent=2))
    if failures:
        raise RuntimeError('Publication sanity check failed')
    return result


if __name__ == '__main__':
    inspect()
