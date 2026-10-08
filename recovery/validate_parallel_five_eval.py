"""Real same-bare serial/parallel validation; existing metric readers/aggregates."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from tools import eval_five_parallel as ev

ROOT=ev.ROOT
EXP=ROOT/'engineering/parallel_five_eval_v1'
RUN=ROOT/'runtime/SAID-nest-clip-v1/engineering-parallel-five-eval-v1'
BASE=ROOT/'runtime/SAID-nest-clip-v1/nested-d3-hns-sg500-v1/HNS-SG/step500'
BARE=BASE/'student_step500.pt'
CHECKPOINT=BASE/'step000500.pt'
BARE_SHA='f9bf1daef60e0fba326eccece40f7e6c24225c20fb16f6b2cf80675ad74f1d09'
CHECKPOINT_SHA='485b9d043142e06d0c5b690db86479bfc2df311adfa58aa2f26f1ba49c92d45e'
BRANCH='engineering/parallel-five-eval-v1'


def read(p):return json.loads(Path(p).read_text())


def save(name,value):
    path=EXP/name
    path.parent.mkdir(parents=True,exist_ok=True);ev.save(path,value)


def state(status,**kw):
    save('STATE.json',dict(status=status,pid=os.getpid(),updated_utc=ev.utc(),**kw))


def run_phase(name):
    out=RUN/name
    assert not out.exists(),'Never overwrite completed/partial validation results'
    out.mkdir()
    shutil.copy2(BASE/'export-check.json',out/'export-check.json')
    cmd=[sys.executable,'-m','tools.eval_five_parallel','--checkpoint',str(BARE),
        '--training-checkpoint',str(CHECKPOINT),'--output-dir',str(out)]
    if name=='serial':cmd.append('--serial')
    state(name.upper()+'_RUNNING',command=cmd,output_dir=str(out))
    with (RUN/(name+'-scheduler.log')).open('xb') as log:
        p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1'))
        try:rc=p.wait()
        except BaseException:
            p.terminate()
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            raise
    if rc:raise RuntimeError(name+' scheduling failed; '+str(log.name))
    receipt=read(out/'EVAL_PARALLEL_RUN.json');assert receipt['status']=='COMPLETED'
    save(name.upper()+'_RUN.json',receipt)


def compare():
    from experiments.nest_clip_v1.balanced_hparam_search_v1.search import native_metrics,scores
    from experiments.nest_clip_v1.armb_summary02_4epoch_v1.report import with_short
    a,ar,ap=native_metrics(RUN/'serial');b,br,bp=native_metrics(RUN/'parallel')
    sa=with_short(dict(metrics=a,scores=scores(a)));sb=with_short(dict(metrics=b,scores=scores(b)))
    metrics_exact=a==b;scores_exact=sa==sb;details={}
    # Ignore only volatile scheduling metadata; compare every other raw field.
    for ds in a:
        aa={k:v for k,v in ar[ds].items() if k not in ('elapsed_seconds','device')}
        bb={k:v for k,v in br[ds].items() if k not in ('elapsed_seconds','device')}
        counts={k:ar[ds][k] for k in ('n_images','n_captions') if k in ar[ds]}
        details[ds]=dict(raw_semantic_JSON_exact=aa==bb,counts=counts,
            counts_exact=all(ar[ds].get(k)==br[ds].get(k) for k in ('n_images','n_captions')),
            max_recall_abs_error=max(abs(a[ds][dr][k]-b[ds][dr][k]) for dr in a[ds] for k in a[ds][dr]))
        save('results/serial/'+ds+'.json',ar[ds]);save('results/parallel/'+ds+'.json',br[ds])
    # Canonical COCO JSON omits counts. Freeze original annotations and command
    # roots; unchanged evaluator asserts5000 images and takes five/image.
    ann=ev.ASSETS/'evaluation/coco/annotations/captions_val2017.json'
    meta=read(ann)
    assert len(meta['images'])==5000 and len({r['image_id'] for r in meta['annotations']})==5000
    details['COCO'].update(counts=dict(n_images=5000,n_captions=25000),
        count_basis='Unchanged canonical evaluator asserts5000 and selects first five/image; native JSON omits counts',
        annotation_sha256=ev.sha(ann))
    if not metrics_exact or not scores_exact or not all(d['raw_semantic_JSON_exact'] and d['counts_exact'] for d in details.values()):
        save('COMPARISON.json',dict(passed=False,metrics_exact=metrics_exact,scores_exact=scores_exact,datasets=details,
            serial_metrics=a,parallel_metrics=b,serial_scores=sa,parallel_scores=sb))
        raise RuntimeError('NUMERICAL_MISMATCH: do not integrate parallel default')
    serial=read(RUN/'serial/EVAL_PARALLEL_RUN.json');parallel=read(RUN/'parallel/EVAL_PARALLEL_RUN.json')
    assert serial['evaluator_source_sha256']==parallel['evaluator_source_sha256']
    assert ev.sha(BARE)==BARE_SHA and ev.sha(CHECKPOINT)==CHECKPOINT_SHA
    result=dict(passed=True,metrics_exact=True,scores_exact=True,max_recall_abs_error=0.,datasets=details,
        serial_metrics=a,parallel_metrics=b,serial_scores=sa,parallel_scores=sb,
        bare_sha256=BARE_SHA,training_checkpoint_sha256=CHECKPOINT_SHA,
        unchanged_evaluator_source_sha256=parallel['evaluator_source_sha256'],
        raw_float_scope='Every semantic JSON field exact; device/elapsed_seconds are intentionally volatile',
        aggregates='Calls existing search.scores and existing report.with_short; no new formulas')
    runtime=dict(serial_wall_seconds=serial['wall_seconds'],parallel_wall_seconds=parallel['wall_seconds'],
        speedup=serial['wall_seconds']/parallel['wall_seconds'],gpu_mapping=parallel['gpu_mapping'],
        serial_durations=serial['durations'],parallel_durations=parallel['durations'],
        definition='Start first child evaluator through end last child; includes unchanged per-process model loading',
        OMP_NUM_THREADS=4,order='Serial first, parallel second; one timing pair, filesystem caching not controlled')
    inventory=[p for p in RUN.rglob('*.log') if p.name!='runner.log']
    result['local_logs_not_uploaded']=[dict(path=str(p),bytes=p.stat().st_size,sha256=ev.sha(p),
        time_range_utc=[serial['started_utc'],parallel['ended_utc']]) for p in inventory]
    save('COMPARISON.json',result);save('RUNTIME_COMPARISON.json',runtime)
    lines=['# Parallel five native evaluation validation','',
        'Same pre-existing strict bare HNS-SG@500, serial GPU0 in historical order vs four independent evaluator processes. No training or gradient audit.',
        'All30 directional R@1/5/10 values and all four existing aggregates are exact. Every semantic raw JSON field is exact; only device/elapsed_seconds differ. COCO JSON omits counts; canonical5000/25000 count basis and annotation SHA are recorded.',
        '',f"Serial wall:{serial['wall_seconds']:.3f}s; parallel wall:{parallel['wall_seconds']:.3f}s; speedup:{runtime['speedup']:.3f}x.",
        '', '| Dataset | GPU | Serial duration(s) | Parallel duration(s) |','|---|---:|---:|---:|']
    for n in ev.ORDER:lines.append(f"| {n} | {parallel['gpu_mapping'][n]} | {serial['durations'][n]:.3f} | {parallel['durations'][n]:.3f} |")
    lines.extend(['','COCO GPU0; DOCCI GPU1; Long-DCI GPU2; Flickr then Urban onGPU3. No concurrent tasks share a GPU.',
        'Timing includes each original Python process model load. One serial-then-parallel pair; filesystem cache/order can influence measured speedup. No claim that individual benchmarks speed up.',
        'Evaluator/training/export sources unchanged. Formal shared scheduling integration occurs only after this numerical gate passes. Metrics/aggregation are reused unchanged.',
        'Artifacts:SERIAL_RUN.json, PARALLEL_RUN.json (exact commands/UTC/returncodes/source hashes), COMPARISON.json, RUNTIME_COMPARISON.json, results/{serial,parallel}/*.json.',
        'Checkpoint/bare/datasets and raw logs remain local; paths/sizes/SHA/time ranges are in COMPARISON.json.'])
    (EXP/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


def run():
    RUN.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    def stop(sig,frame):raise RuntimeError('Validation interrupted; preserve all outputs')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (RUN/'validation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            assert ev.sha(BARE)==BARE_SHA and ev.sha(CHECKPOINT)==CHECKPOINT_SHA
            check=read(BASE/'export-check.json')
            assert check['passed'] and check['bare_sha256']==BARE_SHA and check['checkpoint_sha256']==CHECKPOINT_SHA
            run_phase('serial');run_phase('parallel');compare()
            state('NUMERICAL_AND_TIMING_VALIDATION_PASSED',default_integration_pending=True)
        except BaseException as error:
            state('STOPPED_WITH_EVIDENCE',error=type(error).__name__+': '+str(error),default_integration_allowed=False)
            raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--detach',action='store_true');args=p.parse_args()
    if not args.detach:run();return
    RUN.mkdir(parents=True,exist_ok=True)
    assert not (RUN/'runner.json').exists()
    with (RUN/'runner.log').open('ab',buffering=0) as log:
        child=subprocess.Popen([sys.executable,'-u','-m','recovery.validate_parallel_five_eval'],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    ev.save(RUN/'runner.json',dict(pid=child.pid,log=str(RUN/'runner.log'),queue=['serial','parallel','compare'],training=False))
    print(child.pid)


if __name__=='__main__':main()
