"""Local per-file cache advice and reviewed full-stage publication summary."""
import argparse
from collections import Counter
import datetime
import json
import os
from pathlib import Path
import sqlite3
import time

from recovery.s02_full_stage import (ROOT,RAW,LOCAL,IMAGES,DB,RESULT,REPORT,LEDGER,OLD_LEDGER,
    TOTAL,MANIFEST_SHA,local_path,advise,sha,rows,now)
from recovery.resource_stall_v2 import dump


def cache_advisor(pid):
    # Completed audit flags are set only after PIL closes the local JPEG.
    # This helper never reads image bytes or touches the NFS source.
    args=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
    if b'recovery.s02_full_stage' not in args or b'stage' not in args:
        raise RuntimeError('Cache advisor requires the exact active stage supervisor')
    connection=sqlite3.connect(DB,timeout=30)
    advised=set()
    started=now()
    last_emit=time.monotonic()
    while (Path('/proc')/str(pid)/'cmdline').exists():
        try:
            args=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
        except FileNotFoundError:
            break
        if b'recovery.s02_full_stage' not in args or b'stage' not in args:
            break
        for ordinal,name in connection.execute('SELECT ordinal,relative FROM images WHERE decoded=1'):
            if ordinal in advised:
                continue
            with local_path(name).open('rb') as handle:
                advise(handle.fileno())
            advised.add(ordinal)
        if time.monotonic()-last_emit>=60:
            event=dict(utc=now(),local_completed_files_advised=len(advised),image_bytes_read=0,
                local_root=str(IMAGES),global_drop_caches=False,NFS_touched=False)
            with (RAW/'local-cache-advice.jsonl').open('a') as handle:
                handle.write(json.dumps(event)+'\n')
            last_emit=time.monotonic()
        time.sleep(10)
    # Drain completed audit flags once after supervisor exit.
    for ordinal,name in connection.execute('SELECT ordinal,relative FROM images WHERE decoded=1'):
        if ordinal not in advised:
            with local_path(name).open('rb') as handle:
                advise(handle.fileno())
            advised.add(ordinal)
    connection.close()
    result=dict(started_utc=started,finished_utc=now(),completed_files_advised=len(advised),
        image_bytes_read=0,NFS_touched=False,global_drop_caches=False,
        scope='DONTNEED hints only for closed local JPEG files after successful local-only audit')
    dump(RAW/'local-cache-advice-result.json',result)
    print(json.dumps(result),flush=True)


def distribution(values):
    if not values:
        return dict(count=0,min=None,max=None,median=None,p95=None)
    values=sorted(values)
    def q(x):
        pos=(len(values)-1)*x
        lo=int(pos)
        hi=min(lo+1,len(values)-1)
        return values[lo]+(values[hi]-values[lo])*(pos-lo)
    return dict(count=len(values),min=values[0],max=values[-1],median=q(.5),p95=q(.95))


def finalize():
    result=json.loads(RESULT.read_text())
    telemetry=list(rows(RAW/'resource-progress.jsonl'))
    if not telemetry:
        raise RuntimeError('No resource telemetry; cannot finalize evidence')
    systems=[r['system'] for r in telemetry]
    summary=dict(status=result['status'],finished_utc=now(),samples=len(telemetry),
        phase_counts=dict(Counter(r['phase'] for r in telemetry)),
        peak_memory_current=max(result.get('peak_cgroup_memory',0),max(s['memory_current'] for s in systems)),
        peak_file_cache=max(s['file'] for s in systems),peak_inactive_file=max(s['inactive_file'] for s in systems),
        memory_oom_kill=max(s['memory_events'].get('oom_kill',0) for s in systems),
        memory_oom=max(s['memory_events'].get('oom',0) for s in systems),
        PSI_scope='host /proc/pressure; cgroup v1 has no per-cgroup PSI',
        io_PSI_full_avg10=distribution([s['io_PSI'].get('full',{}).get('avg10',0) for s in systems]),
        memory_PSI_full_avg10=distribution([s['memory_PSI'].get('full',{}).get('avg10',0) for s in systems]),
        copy_rolling5min_MiB_s=distribution([r['rolling5min_MiB_s'] for r in telemetry if r['phase']=='COPY']),
        pod_anomaly=result.get('copy',{}).get('pod_anomaly'),
        pause_reason=result.get('pause_reason'),errors=result.get('copy',{}).get('errors',[]),
        no_training_started=True,NFS_source_preserved=True)
    artifacts=[]
    for path in sorted(RAW.iterdir()):
        if path.is_file() and path.name!='task.lock':
            artifacts.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path),
                time_range_utc=[result.get('copy',{}).get('started_utc',result['prepared_utc']),result.get('finished_utc',now())],
                uploaded=False))
            artifacts[-1]['time_scope']='Associated copy/validation run; not an inference from filesystem mtimes'
            if path.name=='required_training_images.jsonl':
                artifacts[-1]['time_range_utc']=[None,result['prepared_utc']]
                artifacts[-1]['time_scope']='Byte-exact reconstruction finished before preflight; precise start was not separately logged'
    for path in (OLD_LEDGER,DB):
        if path.exists():
            artifacts.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path),uploaded=False))
    result['local_raw_artifacts']=artifacts
    result['resource_summary']=summary
    result['local_only_assets_not_uploaded']=[str(IMAGES),str(LOCAL/'data_index'),str(LOCAL/'full-ready.json'),str(DB)]
    result['training_path_config']='recovery/configs/s02_local_full_paths.json'
    result['source_of_truth']=str(ROOT/'local_assets/training/ShareGPT4V')
    result['risk_record']='Ephemeral Docker overlay cache. Pod rebuild may discard it; restore from persistent manifest/NFS. Never delete NFS images because a local cache exists.'
    result['git_source_commit_launch']='ed9ab013d126316d6af87e306e4213adfd87c5d6'
    result['audit_cache_advice_mode']='Original launch uses separate local per-file advisor; final code also advises immediately after each closed local decode. No image bytes or decode semantics changed.'
    result['cache_advisor_summary']=json.loads((RAW/'local-cache-advice-result.json').read_text()) if (RAW/'local-cache-advice-result.json').exists() else None
    result['benchmark_comparison_limit']='Sequential real missing sorted ranges, not repeated identical payloads; family counts recorded. No cached duplicate source passes used for copy benchmarking.'
    if result['ready']:
        v=result['verification']
        if result['status']!='LOCAL_FULL_TRAINING_DATA_READY' or not v['passed'] or v['exact_paths']!=TOTAL or v['local_only_decode_passed']!=TOTAL or v['resolved_sample_paths_checked']<5000:
            raise RuntimeError('Readiness/evidence inconsistent')
        connection=sqlite3.connect(DB)
        for query in ('SELECT COUNT(*) FROM images WHERE sha IS NULL','SELECT COUNT(*) FROM images WHERE decoded=0','SELECT COUNT(*) FROM images WHERE state="pending"'):
            if connection.execute(query).fetchone()[0]:
                raise RuntimeError('Unhashed/undecoded/missing training images')
        connection.close()
        if result['manifest']['sha256']!=MANIFEST_SHA or summary['memory_oom_kill']:
            raise RuntimeError('Manifest/OOM admission failed')
    dump(RESULT,result)
    dump(ROOT/'recovery/S02_LOCAL_FULL_RESOURCE_SUMMARY.json',summary)
    copy=result.get('copy',{})
    copy_seconds=sum(b['elapsed_s'] for b in copy.get('benchmarks',[]))+copy.get('sustained_copy',{}).get('elapsed_s',0)
    if 'sustained_copy' not in copy:
        copy_seconds=copy.get('elapsed_s',copy_seconds)
    result['copy_elapsed_s']=copy_seconds
    result['copy_mean_MiB_s']=copy.get('newly_copied_bytes',0)/max(copy_seconds,.001)/2**20
    result['validation_elapsed_s']=max(0,copy.get('elapsed_s',0)-copy_seconds)
    result['elapsed_scope']='Copy benchmarks + sustained copy + validation; preflight reconstruction/inventory time is separate'
    dump(RESULT,result)
    lines=['# S=0.2 full local training cache','',f"Status: `{result['status']}`. Training started: false.",'',
        f'Local images: `{IMAGES}`. Ephemeral Docker overlay; the user explicitly accepts cache loss after pod rebuild.',
        f'NFS source of truth: `{result["source_of_truth"]}`. Original images remain intact. Never treat `/root` as the only copy.',
        'GitHub backs up reproducible code/config/reports. After cache loss, reconstruct/reuse the persistent manifest and stage again.', '',
        f"Frozen manifest SHA256: `{MANIFEST_SHA}`; family counts: `{result['family_counts']}`.",
        f"Payload: {result['total_bytes']} bytes = {result['total_GB']:.6f} GB = {result['total_GiB']:.6f} GiB. Earlier568 referred to GiB, not decimalGB.",
        f"Initial verified local images: {result['local_existing_images']} ({result['local_existing_bytes']} bytes). Exact initial missing: {result['remaining_images']} / {result['remaining_bytes']} bytes.",
        f"Reused old images: {result.get('reused_old_files')}; newly copied: {result.get('newly_copied_files')}; elapsed seconds: {copy.get('elapsed_s')}.",
        f"Copy elapsed seconds / mean MiB/s: {copy_seconds:.3f} / {result['copy_mean_MiB_s']:.3f}. Validation elapsed seconds: {result['validation_elapsed_s']:.3f}. Preflight time is separate.",
        'Queue: family COCO/LLaVA/SAM, source parent, filename. Each new file has one streamed NFS read with SHA256, preserved partial prefix, atomic rename; destination filesystem flush and persistent ledger fsync are batched every2000 files.', '',
        '| Workers | Duration s | MiB/s | Files/s | Files | Healthy |', '|---:|---:|---:|---:|---:|---|']
    for b in copy.get('benchmarks',[]):
        lines.append(f"| {b['workers']} | {b['elapsed_s']:.3f} | {b['MiB_s']:.3f} | {b['files_s']:.3f} | {b['files']} | {b['healthy']} |")
    if not any(b['workers']==8 for b in copy.get('benchmarks',[])):
        lines.append('| 8 | not run | N/A | N/A | N/A | Conditional6->8 promotion not met, or earlier pause |')
    lines += ['', f"Selected workers: {copy.get('final_workers')}. Sustained copy: `{copy.get('sustained_copy')}`.",
        'Benchmark ranges are different real remaining files; family mix is recorded in JSON. A higher concurrency is selected only with at least15% throughput improvement and healthy resources.', '',
        f"Peak cgroup memory: {summary['peak_memory_current']/2**30:.3f} GiB. oom_kill: {summary['memory_oom_kill']}. Pod anomaly: {summary['pod_anomaly']}.",
        f"Host IO/memory PSI summaries: `{summary['io_PSI_full_avg10']}` / `{summary['memory_PSI_full_avg10']}`. File cache is not RSS.",
        f"Integrity: `{result.get('verification')}`. Full decode audit reads local images only. `.part` is never a training input; missing/symlink paths fail-fast with no NFS fallback.",
        f"Local free bytes after this stage: {result['destination_free_bytes']}.",
        'Local-only formal data-path config: `recovery/configs/s02_local_full_paths.json`; native Dataset semantics and relative paths remain unchanged. No500/4868 training is launched.', '',
        'Raw logs/manifests/full hash ledger/path proofs remain local. Reviewed inventory follows; GitHub contains only summaries.', '']
    for item in artifacts:
        lines.append(f"- `{item['path']}`: {item['bytes']} bytes; SHA256 `{item['sha256']}`; UTC scope `{item.get('time_range_utc','existing retained artifact')}`.")
    lines += ['', 'GitHub sync: pending final commit/push/fetch verification; final phase completion remains false until synchronization succeeds.']
    REPORT.write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'],summary=summary,raw_artifacts=len(artifacts)),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['cache-advisor','finalize'])
    parser.add_argument('--pid',type=int)
    args=parser.parse_args()
    if args.command=='cache-advisor':
        if not args.pid:
            parser.error('--pid required for cache advisor')
        cache_advisor(args.pid)
    else:
        finalize()


if __name__=='__main__':
    main()
