"""Schedule unchanged native evaluators; no model loading or metric arithmetic."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT/'local_assets'
ORDER = ('coco', 'urban', 'flickr', 'docci', 'long_dci')
EXTENDED = 'experiments.s0_dualmask_full_v01.evidence.step2000.new_evaluations.eval_extended_real'
EVALUATOR_SOURCES = (
    'tools/eval_nest_native.py', 'tools/urban1k_retrieval.py', 'eval/retrieval/coco_retrieval.py',
    'experiments/s0_dualmask_full_v01/evidence/step2000/new_evaluations/eval_extended_real.py',
    'tools/eval_urban1k_cls.py', 'model/longclip.py', 'model/model_longclip.py', 'model/simple_tokenizer.py',
)


def utc(): return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''): digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    tmp=path.with_suffix(path.suffix+'.pending')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


@dataclass(frozen=True)
class Job:
    name: str
    gpu: int
    command: tuple
    output: Path
    log: Path


def gpu_mapping(gpus=(0,1,2,3)):
    if len(gpus)!=4 or len(set(gpus))!=4 or any(not isinstance(g,int) or g<0 for g in gpus):
        raise ValueError('Exactly four distinct nonnegative GPU indices required')
    return dict(coco=gpus[0],docci=gpus[1],long_dci=gpus[2],flickr=gpus[3],urban=gpus[3])


def build_jobs(checkpoint, output_dir, *, gpus=(0,1,2,3), assets=ASSETS, python=sys.executable, serial=False):
    checkpoint,output_dir,assets=map(lambda p:Path(p).resolve(),(checkpoint,output_dir,assets))
    mapping={name:gpus[0] for name in ORDER} if serial else gpu_mapping(gpus)
    bench=assets/'retrieval_benchmarks'; jobs={}
    for name in ORDER:
        common=[str(python),'-m', 'tools.eval_nest_native' if name in ('coco','urban') else EXTENDED,
                '--checkpoint',str(checkpoint)]
        if name in ('coco','urban'):
            folder=assets/('evaluation/coco/val2017' if name=='coco' else 'evaluation/Urban1k/Urban1k')
            output=output_dir/(name+'_native.json')
            command=common+['--dataset',name,'--root',str(folder),'--device',f'cuda:{mapping[name]}',
                            '--batch-size','64','--output',str(output)]
        else:
            protocol,manifest,folder={
                'flickr':('flickr_test1k','flickr30k_test1k.jsonl','flickr30k'),
                'docci':('docci','docci_test.jsonl','docci'),
                'long_dci':('long_dci','long_dci_reconstructed.jsonl','dci'),
            }[name]
            output=output_dir/protocol/(protocol+'.json')
            command=common+['--device',f'cuda:{mapping[name]}','--batch-size','64','--output-dir',str(output.parent),
                            f"{protocol}:{bench/'manifests'/manifest}:{bench/folder/'images'}"]
        jobs[name]=Job(name,mapping[name],tuple(command),output,output_dir/'logs'/(name+'.log'))
    validate_jobs(jobs)
    return jobs


def validate_jobs(jobs):
    if set(jobs)!=set(ORDER): raise ValueError('Exactly five benchmark jobs required')
    writes=[]
    for job in jobs.values():
        writes.extend((job.output.resolve(),job.log.resolve()))
        if '--output-dir' in job.command:
            writes.append(job.output.parent/'extended_summary.json')
    if len(writes)!=len(set(writes)): raise ValueError('Duplicate output/log path')


def require_gpu_idle(gpus):
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None and visible != ','.join(str(i) for i in range(4)):
        raise RuntimeError('Ambiguous CUDA_VISIBLE_DEVICES; unset it or use identity0,1,2,3. Explicit cuda:N mode only')
    listing=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    uuid_to_gpu={line.split(',')[1].strip():int(line.split(',')[0]) for line in listing.splitlines()}
    if not set(gpus)<=set(uuid_to_gpu.values()): raise RuntimeError('Requested GPUs unavailable')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    if any(uuid_to_gpu.get(line.split(',')[0].strip()) in gpus for line in apps.splitlines() if ',' in line):
        raise RuntimeError('Assigned GPUs occupied; training/audit/evaluation must exit first')
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid(): continue
        try: args=[a.decode(errors='replace') for a in (p/'cmdline').read_bytes().split(b'\0') if a]
        except OSError: continue
        if any(Path(a).name=='torchrun' or a.startswith('train.train_') or
               (a.startswith('recovery.') and ('gradient' in a or a.endswith('_gradients'))) for a in args):
            raise RuntimeError('Training/gradient audit process still present: PID'+p.name)


def check_output(job):
    if not job.output.is_file(): raise RuntimeError('Missing evaluator JSON: '+str(job.output))
    result=json.loads(job.output.read_text())
    if not isinstance(result,dict) or not result: raise RuntimeError('Empty/invalid evaluator JSON')
    if '--output-dir' in job.command:
        summary=job.output.parent/'extended_summary.json'
        if not summary.is_file() or not isinstance(json.loads(summary.read_text()),dict):
            raise RuntimeError('Missing/invalid extended_summary.json: '+str(summary))


class Scheduler:
    def __init__(self,jobs,output_dir,*,serial=False,popen=subprocess.Popen):
        validate_jobs(jobs)
        self.jobs,self.output_dir,self.serial,self.popen=jobs,Path(output_dir),serial,popen
        self.lock=threading.RLock();self.active={};self.stop=threading.Event()
        self.records={};self.run_record={}

    def persist(self):
        with self.lock:
            save(self.output_dir/'EVAL_PARALLEL_RUN.json',dict(self.run_record,
                jobs=dict(self.records),commands={n:list(j.command) for n,j in self.jobs.items()},
                gpu_mapping={n:j.gpu for n,j in self.jobs.items()},
                durations={n:r.get('duration_seconds') for n,r in self.records.items()},
                returncodes={n:r.get('returncode') for n,r in self.records.items()},
                output_paths={n:str(j.output) for n,j in self.jobs.items()}))

    def job(self,job):
        if self.stop.is_set(): return
        record=dict(gpu=job.gpu,command=list(job.command),output=str(job.output),log=str(job.log),
                    started_utc=utc(),started_monotonic=time.monotonic(),status='RUNNING')
        with self.lock:self.records[job.name]=record;self.persist()
        process=None
        try:
            with job.log.open('xb') as handle:
                # Inherit environment unchanged. No CUDA_VISIBLE_DEVICES remap,
                # batch changes, DDP, model/metric code or evaluation retry.
                process=self.popen(list(job.command),cwd=ROOT,stdin=subprocess.DEVNULL,
                    stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                with self.lock:
                    self.active[job.name]=process;record['pid']=process.pid
                    if self.stop.is_set() and process.poll() is None:
                        os.killpg(process.pid,signal.SIGTERM)
                rc=process.wait()
            record['returncode']=rc
            if rc!=0: raise RuntimeError('Evaluator returncode='+str(rc))
            check_output(job)
            record['status']='COMPLETED'
        except Exception as error:
            record.update(status='FAILED',error=type(error).__name__+': '+str(error))
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            if process is not None:record['returncode']=process.returncode
            record.update(ended_utc=utc(),ended_monotonic=time.monotonic())
            record['duration_seconds']=record['ended_monotonic']-record['started_monotonic']
            with self.lock:self.active.pop(job.name,None);self.persist()

    def lane(self,names):
        for name in names:
            self.job(self.jobs[name])
            if self.records.get(name,{}).get('status')!='COMPLETED':
                # Flickr failure must not launch Urban, but other GPU lanes
                # finish normally and retain their outputs.
                for skipped in names[names.index(name)+1:]:
                    with self.lock:self.records[skipped]=dict(status='BLOCKED_BY_PREDECESSOR',
                        gpu=self.jobs[skipped].gpu,log=str(self.jobs[skipped].log),returncode=None)
                self.persist();break

    def cancel(self):
        self.stop.set()
        with self.lock:
            for process in self.active.values():
                if process.poll() is None:
                    try:os.killpg(process.pid,signal.SIGTERM)
                    except ProcessLookupError:pass

    def run(self,provenance,*,preflight=require_gpu_idle):
        for job in self.jobs.values():
            if job.output.exists() or job.log.exists() or ('--output-dir' in job.command and job.output.parent.exists()):
                raise FileExistsError('Refusing stale/overlapping evaluator output: '+str(job.output))
        self.output_dir.mkdir(parents=True,exist_ok=True)
        receipt=self.output_dir/'EVAL_PARALLEL_RUN.json'
        if receipt.exists():raise FileExistsError(receipt)
        (self.output_dir/'logs').mkdir(exist_ok=False)
        locks=ExitStack()
        try:
            for gpu in sorted({j.gpu for j in self.jobs.values()}):
                f=locks.enter_context(Path(f'/tmp/said-eval-gpu-{gpu}.lock').open('a'))
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            preflight({j.gpu for j in self.jobs.values()})
            self.run_record=dict(provenance,status='RUNNING',mode='serial' if self.serial else 'parallel',
                started_utc=utc(),scheduler_started_monotonic=time.monotonic(),metrics_implemented_here=False)
            self.persist()
            lanes=[list(ORDER)] if self.serial else [['coco'],['docci'],['long_dci'],['flickr','urban']]
            with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
                futures=[pool.submit(self.lane,lane) for lane in lanes]
                for future in futures:future.result()
            failures={n:r for n,r in self.records.items() if r['status']!='COMPLETED'}
            if self.stop.is_set():failures['scheduler']=dict(status='CANCELLED')
            started=[r['started_monotonic'] for r in self.records.values() if 'started_monotonic' in r]
            ended=[r['ended_monotonic'] for r in self.records.values() if 'ended_monotonic' in r]
            self.run_record.update(status='FAILED' if failures else 'COMPLETED',ended_utc=utc(),
                wall_seconds=max(ended)-min(started) if started and ended else None,
                scheduler_wall_seconds=time.monotonic()-self.run_record['scheduler_started_monotonic'],
                failures={n:dict(status=r['status'],error=r.get('error'),log=r.get('log')) for n,r in failures.items()})
            self.persist()
            return 1 if failures else 0
        except Exception as error:
            self.cancel();self.run_record.update(status='FAILED',ended_utc=utc(),error=type(error).__name__+': '+str(error))
            self.persist();raise
        finally:locks.close()


def provenance(bare,training_checkpoint=None):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    return dict(checkpoint=str(bare),checkpoint_sha256=sha(bare),bare_student=str(bare),bare_sha256=sha(bare),
        training_checkpoint=str(training_checkpoint) if training_checkpoint else None,
        training_checkpoint_sha256=sha(training_checkpoint) if training_checkpoint else None,
        evaluator_git_commit=commit,evaluator_source_sha256={p:sha(ROOT/p) for p in EVALUATOR_SOURCES},
        scheduler_source_sha256=sha(Path(__file__)),batch_size=64,device_binding='explicit cuda:N; no visibility remapping')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True,help='Existing strict bare student')
    p.add_argument('--training-checkpoint',type=Path,help='Optional resumable checkpoint for provenance only')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--assets-root',type=Path,default=ASSETS)
    p.add_argument('--gpus',default='0,1,2,3')
    p.add_argument('--serial',action='store_true',help='Validation only: historical order on first GPU')
    args=p.parse_args();gpus=tuple(int(x) for x in args.gpus.split(','));gpu_mapping(gpus)
    args.checkpoint=args.checkpoint.resolve(strict=True)
    jobs=build_jobs(args.checkpoint,args.output_dir,gpus=gpus,assets=args.assets_root,serial=args.serial)
    scheduler=Scheduler(jobs,args.output_dir,serial=args.serial)
    def stop(sig,frame):scheduler.cancel()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    result=scheduler.run(provenance(args.checkpoint,args.training_checkpoint))
    print(json.dumps(scheduler.run_record),flush=True)
    raise SystemExit(result)


if __name__=='__main__':main()
