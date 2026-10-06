"""Review local500 evidence and record verified Git sync; never train."""
import argparse
import datetime
import json
import math
from pathlib import Path
import re
import subprocess
import time

from recovery.s02_local500 import RUN,PHASE,ROOT,OUT,IMAGES,LOCAL,dump,rows,sha,distribution


def review():
    result=json.loads((OUT/'STEP500_RESULTS.json').read_text())
    stats=json.loads((OUT/'LOCAL_RUNTIME_STATS.json').read_text())
    prelaunch=json.loads((RUN/'prelaunch-local-path-proof-5000.json').read_text())
    assert prelaunch['passed'] and prelaunch['count']==5000 and prelaunch['all_local']
    assert all(Path(r['actual_path']).is_relative_to(IMAGES) for r in prelaunch['rows'])
    result['prelaunch_local_path_proof']=dict(passed=True,checked=5000,seed=0,
        selection=prelaunch['selection'],training_RNG_untouched=True,NFS_fallback=False)
    # Reuse read-only frozen sample-ID comparison, without the old launch/report flow.
    from recovery import nfs500_evidence as historical
    historical.RUN=RUN
    historical.PHASE=PHASE
    if result['stopped_at_500']:
        result['full500_sample_stream_proof']=historical.full_stream_check()
        assert result['checkpoint_proof']['passed'] and result['first_five_gate']['passed']
        assert stats['actual_local_path_proof']['passed']
        steps=rows(RUN/'step500/steps.jsonl')
        assert [r['step'] for r in steps]==list(range(1,501))
        assert all(math.isfinite(r['loss']) for r in steps)
        assert all(h['batch']==256 and h['gradients_finite'] for r in steps for h in r['rank_health'])
        config=json.loads((RUN/'step500/config.json').read_text())
        assert config['image_root']==str(IMAGES) and config['resume'] is None and config['start_updates']==0
        assert config['init_sha256']==result['step0_sha256'] and config['horizon']==4868
        result['completed_training_invariants']=dict(passed=True,exact_step_sequence='1..500',
            losses_and_gradients_finite=True,all_rank_batches=256,global_batch=1024,accumulation=1,
            image_root=config['image_root'],NFS_fallback=False,start_updates=0,resume=None,horizon=4868)
        if result.get('metrics'):
            from experiments.nest_clip_v1.balanced_hparam_search_v1.search import native_metrics
            _,raw_metrics,sources=native_metrics(RUN/'step500')
            result['native_evaluation_provenance']={name:dict(
                **{k:data[k] for k in ('protocol','n_images','n_captions','manifest_sha256',
                    'checkpoint_sha256','native_only','native_student_only') if k in data},
                local_result_path=str(sources[name]),local_result_sha256=sha(sources[name]),
                evaluation_uses_strict_bare_student=True) for name,data in raw_metrics.items()}
    else:
        result['full500_sample_stream_proof']=dict(passed=False,reason='Incomplete trajectory; no full500 claim')
    provenance=result['launch_provenance']
    end=datetime.datetime.fromisoformat(json.loads((RUN/'commands.json').read_text())[-1]['ended_utc'])
    start=datetime.datetime.fromisoformat(provenance['started_utc'])
    boot_start=provenance['started_monotonic']
    boot_end=boot_start+(end-start).total_seconds()
    kernel=subprocess.run(['dmesg','--color=never'],capture_output=True,text=True)
    if kernel.returncode:
        kernel_summary=dict(available=False,reason='dmesg unavailable',exitcode=kernel.returncode)
    else:
        selected=[]
        for line in kernel.stdout.splitlines():
            match=re.match(r'\[\s*([\d.]+)\]',line)
            if match and boot_start<=float(match.group(1))<=boot_end:
                selected.append(line)
        raw=RUN/'kernel-during-run.raw.log'
        raw.write_text('\n'.join(selected)+'\n')
        patterns=dict(IO_FILESYSTEM=r'I/O error|Input/output error|EXT4-fs error|XFS.*(?:error|corrupt)|nvme.*(?:error|reset)',
            GPU_DRIVER=r'NVRM.*(?:Xid|SXid)|fallen off the bus|AER.*(?:error|fatal)',
            OOM_HANG=r'oom-killer|Out of memory|Killed process|hung task|blocked for more than|lockup|panic',
            NFS_RPC=r'NFS.*(?:not responding|error)|RPC.*(?:timeout|timed out)|rpc_check_timeout')
        kernel_summary=dict(available=True,scope='Host ringbuffer during run; host-wide, attribution not inferred',
            time_range_utc=[provenance['started_utc'],end.isoformat()],line_count=len(selected),
            keyword_counts={k:sum(bool(re.search(pattern,line,re.I)) for line in selected) for k,pattern in patterns.items()},
            raw_path=str(raw),bytes=raw.stat().st_size,sha256=sha(raw),uploaded=False)
        stats['raw_local_artifacts'].append(dict(path=str(raw),bytes=raw.stat().st_size,sha256=sha(raw),
            time_range_utc=kernel_summary['time_range_utc'],uploaded=False))
    stats['kernel_during_run']=kernel_summary
    stats['true_training_io_error_observed']=bool(stats['io_error_count'])
    stats['pod_or_supervisor_anomaly_observed']=bool(stats['stop_reason'] and 'supervisor' in stats['stop_reason'].lower())
    stats['excluded_assets']=[str(LOCAL),str(RUN/'step500/step000005.pt'),str(RUN/'step500/step000500.pt'),
        str(RUN/'step500/student_step500.pt'),str(ROOT/'local_assets'),str(ROOT/'runtime/SAID-nest-clip-v1/shared'),
        str(ROOT/'recovery/evidence/s02-local-full-data-local')]
    cycles=rows(RUN/'step500/cycle_timing.jsonl')
    stats['steady_steps7_plus_full_cycle_seconds']=distribution([r['four_rank_max_seconds'] for r in cycles if r['step']>=7])
    lookup={rank:{p['step']:p for p in rows(PHASE/f'rank{rank}.jsonl')} for rank in range(4)}
    stats['steady_steps7_plus_data_wait_slowest_rank_seconds']=distribution([
        max(lookup[r][c['step']]['data_wait_s'] for r in range(4)) for c in cycles if c['step']>=7])
    if result['stopped_at_500']:
        log=(RUN/'train500.log').read_text()
        if re.search(r'CUDA error|CUDA out of memory|Nonfinite|Image failure|Missing local sample|Traceback',log):
            raise RuntimeError('Completed-run log contains unexpected failure; review before publication')
    dump(OUT/'STEP500_RESULTS.json',result)
    dump(OUT/'LOCAL_RUNTIME_STATS.json',stats)
    report=OUT/'STEP500_LOCAL_REPRODUCTION.md'
    lines=['','Reviewed evidence:',
        f'Prelaunch frozen random sample paths:5000 local-only. Full sample stream records verified:{result["full500_sample_stream_proof"].get("records_checked",0)}.',
        'Steady steps7+ full-cycle seconds: `'+json.dumps(stats['steady_steps7_plus_full_cycle_seconds'])+'`.',
        'Steady steps7+ slowest-rank wait seconds: `'+json.dumps(stats['steady_steps7_plus_data_wait_slowest_rank_seconds'])+'`.',
        f'Peak cgroup:{stats["peak_cgroup_memory_bytes"]/2**30:.6f} GiB; GPU peaks:`{stats["GPU_peak_memory"]}`.',
        'PSI summaries are host-scoped on cgroup v1. Kernel summary: `'+json.dumps(kernel_summary)+'`.',
        'Excluded assets:`'+json.dumps(stats['excluded_assets'])+'`. No checkpoint/dataset/cache/raw log uploaded.']
    report.write_text(report.read_text()+'\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'],full500_stream=result['full500_sample_stream_proof'],
        full_cycle=stats['full_cycle_seconds'],data_wait=stats['data_wait_seconds_slowest_rank'],kernel=kernel_summary),indent=2))


def record_sync():
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    branch=git('branch','--show-current')
    assert branch=='recovery/s02-local500'
    head=git('rev-parse','HEAD')
    remote=git('rev-parse','refs/remotes/origin/'+branch)
    assert head==remote
    paths=[OUT/name for name in ('STEP500_RESULTS.json','LOCAL_RUNTIME_STATS.json','CONTINUATION_GATE.json','STEP500_LOCAL_REPRODUCTION.md')]
    for p in paths:
        assert subprocess.check_output(['git','show','HEAD:'+str(p.relative_to(ROOT))],cwd=ROOT)==p.read_bytes()
    receipt=dict(branch=branch,verified_evidence_commit=head,remote_evidence_HEAD=remote,HEAD_equal=True,
        verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Evidence commit verified; receipt commit also requires final push/fetch HEAD comparison')
    for p in paths[:3]:
        value=json.loads(p.read_text())
        value['github_sync']=receipt
        value['github_sync_complete']=True
        dump(p,value)
    report=paths[3]
    text=report.read_text()
    assert 'GitHub synchronization: PENDING.' in text
    report.write_text(text.replace('GitHub synchronization: PENDING.',
        f'GitHub synchronization: evidence `{head}` pushed/fetched, remote HEAD matched on `{branch}`. Receipt commit is pushed/fetched and checked separately.'))
    print(json.dumps(receipt))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('review','record-sync'))
    args=parser.parse_args()
    review() if args.command=='review' else record_sync()
