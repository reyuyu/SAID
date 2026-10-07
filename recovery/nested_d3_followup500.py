"""Detached KR2M1 -> native evaluation/review -> fresh WeakSparse -> publish."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from recovery import nested_d3_local_search as search
from recovery.s02_nfs500 import ROOT, dump, now, rows, sha

EXP = ROOT/'experiments/nest_clip_v1/nested_d3_followup500_v1'
RUN_ROOT = ROOT/'runtime/SAID-nest-clip-v1/nested-d3-kr2m1-weaksparse500-20261007'
BRANCH = 'experiment/nested-d3-kr2m1-weaksparse500-v1'
MAIN_LOG = RUN_ROOT.parent/(RUN_ROOT.name+'.runner.log')
IDENTITY = RUN_ROOT.parent/(RUN_ROOT.name+'.runner.json')
ENTRY = 'recovery.nested_d3_followup500'
ARMS = {
    'KR2M1': dict(axis='granularity',weights=[1.35,1.35,.30],r=2.,
        mode='nested_detail_kr2m1',experiment_dir=str(ROOT/'experiments/nest_clip_v1/kr2m1_500_v1')),
    'WeakSparse-051015': dict(axis='sparsity',weights=[1.35,1.35,.30],r=2.,
        mode='nested_detail_d3',sparsity_weights=[.5,1.,1.5],
        experiment_dir=str(ROOT/'experiments/nest_clip_v1/weaksparse_051015_500_v1')),
}
REPORT_NAMES = ('config.json','REPORT.md','RESULTS.json','TRAINING_DIAGNOSTICS.json',
    'GRADIENT_SPOTCHECK.json','MASK_HIERARCHY_AUDIT.json','SAMPLING_AUDIT.json',
    'RUNTIME_STATS.json','EXPORT_AUDIT.json','VALIDATION.json')


def configure():
    search.EXP, search.RUN_ROOT, search.BRANCH = EXP, RUN_ROOT, BRANCH
    search.ARMS = ARMS
    search.ENTRY_MODULE = ENTRY
    search.PHASE_PREFIX = 'formal-nested-d3-kr2m1-weaksparse500-20261007-'
    search.EXTRA_SOURCES = {'recovery/nested_d3_followup500.py','tests/test_nested_d3_followup500.py',
        'recovery/check_stage500_publish.py'}


def frequency_median(hist):
    total = sum(hist.values())
    if not total:
        return None
    positions = [(total-1)//2,total//2]
    found = []; cumulative = 0
    for k,n in sorted((int(k),n) for k,n in hist.items()):
        for position in positions:
            if cumulative <= position < cumulative+n:
                found.append(k)
        cumulative += n
    assert len(found)==2
    return sum(found)/2


def augment_diagnostics(arm):
    """Derive K/m and medians from all actual batches, with no image reread."""
    exp=search.experiment_dir(arm)
    path=exp/'TRAINING_DIAGNOSTICS.json'; value=json.loads(path.read_text())
    sampling=value['sampling']; by_m=sampling['K_histogram_by_m']
    count=sum(sum(h.values()) for h in by_m.values())
    assert count==sampling['valid_records']
    ratios=sum(int(k)/int(m)*n for m,h in by_m.items() for k,n in h.items())
    selected=sum(int(k)*n for h in by_m.values() for k,n in h.items())
    pool=sum(int(m)*sum(h.values()) for m,h in by_m.items())
    sampling.update(median_K_valid=frequency_median(sampling['K_histogram_valid']),
        median_K_all=frequency_median(sampling['K_histogram_all']),
        mean_K_over_m_valid=ratios/count,pooled_K_over_m=selected/pool,
        coverage_population='Valid complete visible detail pools; padding-only fallback excluded')
    if arm=='KR2M1':
        for m,hist in by_m.items():
            m=int(m); allowed=range(2,m) if m>=3 else [1]
            assert all(int(k) in allowed for k in hist)
        sampling['K_rule']='m>=3: Uniform{2,...,m-1}; m=2:1; m<=1: unchanged legal fallback'
    dump(path,value)
    with (exp/'REPORT.md').open('a') as stream:
        stream.write('\nActual sampling: median valid K='+str(sampling['median_K_valid'])+
            '; mean K/m='+str(sampling['mean_K_over_m_valid'])+
            '; pooled K/m='+str(sampling['pooled_K_over_m'])+'.\n')
    result=json.loads((exp/'RESULTS.json').read_text())
    result['classification']='COMPLETED_ISOLATED_ARM'
    result['sequential_queue']=list(ARMS)
    dump(exp/'RESULTS.json',result)


def summarize():
    from recovery.nested_d3_local_search_evidence import quality
    anchor=json.loads((search.ANCHOR_EXP/'RESULTS.json').read_text())
    leaderboard={'Anchor':quality(anchor)}; details={}
    for arm in ARMS:
        exp=search.experiment_dir(arm)
        assert json.loads((exp/'VALIDATION.json').read_text())['passed']
        r=json.loads((exp/'RESULTS.json').read_text()); assert r['completed_steps']==500
        leaderboard[arm]=quality(r)
        details[arm]=dict(quality_delta_vs_anchor_pp=r['quality_delta_vs_anchor_pp'],
            recall_delta_vs_anchor_pp=r['recall_delta_vs_anchor_pp'],
            diagnostics=json.loads((exp/'TRAINING_DIAGNOSTICS.json').read_text()),
            gradient=json.loads((exp/'GRADIENT_SPOTCHECK.json').read_text()),
            masks=json.loads((exp/'MASK_HIERARCHY_AUDIT.json').read_text()))
    dump(EXP/'RESULTS.json',dict(status='BOTH_ARMS_REVIEWED',leaderboard=leaderboard,arms=details,
        queue=list(ARMS),both_fresh_common0=True,local_only=True,automatic_full=False,
        automatic_combinations=False,automatic_third_arm=False))
    lines=['# KR2M1 and WeakSparse: two independent local500 experiments','',
        'Queue: KR2M1 -> five native benchmarks -> complete report/review -> fresh WeakSparse -> same evaluation/report/review.',
        'Both fresh common0, exactly500 optimizer updates, local-only; no full/combined/third experiment.',
        'KR2M1 changes only K sampling. WeakSparse restores fixed K3 and changes only absolute sparsity coefficients to [.5,1.,1.5], mass3, without normalization.',
        '', '| Arm | Score5 | J_long3 | J_long | Short4 | Urban I2T | Urban T2I |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm,q in leaderboard.items():
        lines.append('| '+arm+' | '+' | '.join(f'{q[k]:.6f}' for k in
            ('Score5','J_long3','J_long','Short4','Urban_I2T','Urban_T2I'))+' |')
    lines+=['','Every directional R@1/5/10, all anchor deltas, K histogram/median/coverage, CE/shares, gradients, keep/IoU/violations and immutable checkpoint/bare identities are in each arm REPORT.md/JSON.',
        'Raw logs and binary artifacts stay local. Each RUNTIME_STATS.json records local paths,size,SHA256,time windows.',
        'No automatic follow-up beyond these two experiments.']
    for name in ('REPORT.md','SEARCH_SUMMARY.md'):
        (EXP/name).write_text('\n'.join(lines)+'\n')


def publication_paths():
    return [EXP/n for n in ('REPORT.md','SEARCH_SUMMARY.md','RESULTS.json')]+[
        search.experiment_dir(a)/n for a in ARMS for n in REPORT_NAMES]


def publish():
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=ROOT,text=True,timeout=120).strip()
    assert git('branch','--show-current')==BRANCH,'Branch changed; refuse publication'
    assert not git('diff','--cached','--name-only'),'Unrelated staged files; refuse publication'
    for arm in ARMS:
        provenance=json.loads((RUN_ROOT/arm/'launch-provenance.json').read_text())
        for path,digest in provenance['source_sha256'].items():
            assert sha(ROOT/path)==digest,('Source changed after launch',path)
    paths=publication_paths()
    assert all(p.is_file() and p.stat().st_size<1024*1024 for p in paths)
    subprocess.run(['git','status','--short'],cwd=ROOT,check=True)
    subprocess.run(['git','add','--',*[str(p.relative_to(ROOT)) for p in paths]],cwd=ROOT,check=True)
    from recovery.check_stage500_publish import inspect
    review=inspect();assert review['passed']
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    assert not git('diff','--name-only','--',*[str(p.relative_to(ROOT)) for p in paths])
    subprocess.run(['git','commit','-m','Report independent KR2M1 and WeakSparse local500 experiments'],cwd=ROOT,check=True)
    head=git('rev-parse','HEAD')
    subprocess.run(['git','push','origin','HEAD:refs/heads/'+BRANCH],cwd=ROOT,check=True,timeout=120)
    subprocess.run(['git','fetch','origin','refs/heads/'+BRANCH+':refs/remotes/origin/'+BRANCH],cwd=ROOT,check=True,timeout=120)
    remote=git('rev-parse','refs/remotes/origin/'+BRANCH)
    assert head==remote==git('rev-parse','FETCH_HEAD')
    for path in paths:
        assert subprocess.check_output(['git','show',head+':'+str(path.relative_to(ROOT))],cwd=ROOT)==path.read_bytes()
    receipt=dict(passed=True,branch=BRANCH,commit=head,remote_HEAD=remote,push_success=True,
        fetch_success=True,remote_HEAD_matches_local=True,checked_utc=now(),
        publication_check=review,local_only_receipt=True)
    dump(EXP/'GITHUB_RECEIPT.json',receipt)
    return receipt


def state(status,active=None,completed=(),**extra):
    dump(EXP/'QUEUE_STATE.json',dict(status=status,active_arm=active,completed_arms=list(completed),
        queue=list(ARMS),updated_utc=now(),runner_pid=os.getpid(),runner_session=os.getsid(0),
        main_log=str(MAIN_LOG),automatic_full=False,automatic_combinations=False,
        automatic_third_arm=False,**extra))


def detached_launch():
    search.prepare()
    assert not RUN_ROOT.exists() and not IDENTITY.exists(),'Never overwrite/restart a trajectory'
    assert not json.loads((EXP/'CPU_TESTS.json').read_text())['failed_count']
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()==BRANCH
    MAIN_LOG.parent.mkdir(parents=True,exist_ok=True)
    with MAIN_LOG.open('ab',buffering=0) as output:
        process=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m',ENTRY],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True,
            close_fds=True,env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4'))
    receipt=dict(pid=process.pid,session=process.pid,started_utc=now(),entry=ENTRY,
        main_log=str(MAIN_LOG),queue=list(ARMS),stdin='/dev/null',start_new_session=True,
        stdout_stderr_to_persistent_log=True,shell_session_independent=True)
    dump(IDENTITY,receipt);print(json.dumps(receipt),flush=True)


def main():
    configure()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=list(ARMS));parser.add_argument('--worker',action='store_true')
    parser.add_argument('--gradient',action='store_true');parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--detach',action='store_true')
    args,remaining=parser.parse_known_args()
    if args.worker or args.gradient:
        search.activate(args.arm);sys.argv=[sys.argv[0],*remaining]
        search.worker() if args.worker else search.gradient();return
    assert not remaining and args.arm is None
    if args.prepare:search.prepare();return
    if args.detach:detached_launch();return
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    def stop(sig,frame):raise RuntimeError('HARD_STOP supervisor signal '+str(sig))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    lock_path=RUN_ROOT.parent/(RUN_ROOT.name+'.runner.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not RUN_ROOT.exists(),'No automatic resume/retry/overwrite'
        search.prepare();completed=[]
        try:
            for arm in ARMS:
                # Successful review/report is required before the next fresh arm.
                if completed:
                    assert json.loads((search.experiment_dir(completed[-1])/'VALIDATION.json').read_text())['passed']
                    assert (search.experiment_dir(completed[-1])/'REPORT.md').is_file()
                search.activate(arm);state('RUNNING',arm,completed)
                search.Supervisor().run();augment_diagnostics(arm);completed.append(arm)
            summarize();state('PUBLISHING',completed=completed)
            receipt=publish();state('COMPLETED_AND_SYNCED',completed=completed,github=receipt)
        except BaseException as error:
            state('STOPPED_WITH_EVIDENCE',search.ARM,completed,error=type(error).__name__+': '+str(error),
                automatic_retry=False,checkpoints_and_progress_retained=True)
            raise


if __name__=='__main__':main()
