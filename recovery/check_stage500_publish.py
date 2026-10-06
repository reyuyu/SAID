"""Fail closed on oversized, forbidden or credential-bearing staged Git files."""
import json
import re
import subprocess

ALLOWED = {
    'recovery/s02_local500.py', 'recovery/local500_policy.py', 'recovery/test_local500_policy.py',
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
